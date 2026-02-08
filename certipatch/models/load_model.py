"""certipatch.models.load_model

This module defines the model adapter contract.

Hard requirement:
- The rest of the project MUST NOT depend on a specific model class.
- Any local model is allowed if it satisfies the adapter contract.

Backends supported by design (not implemented in this scaffold):
- `transformer_lens` for HookedTransformer-style models.
- `huggingface` for AutoModelForCausalLM-style models.

Fail-closed behavior is mandatory:
- If the requested hookpoint cannot be found, abort.
- If answer tokens are not single tokens under the tokenizer, enforce fallback or abort
  (as specified by config).
- If the model cannot report a stable fingerprint/revision, record a deterministic
  local fingerprint (e.g., SHA256 of checkpoint file list) and mark it in the certificate.

Codex MUST implement this module end-to-end because it is the foundation of
"any model" flexibility.

Key concepts:
- "hookpoint kind" is an abstract name such as `resid_post`.
- The adapter maps `(kind, layer_index)` to a concrete hook implementation.

The primary experiments assume the ability to intercept and modify the residual stream
at the answer position for a chosen layer set.

Answer position p(x) MUST be computed per-example:
    p = attention_mask.sum(dim=1) - 1

Never use p = seq_len - 1 unless the code enforces left padding with explicit checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, cast

import torch

BackendName = Literal["transformer_lens", "huggingface"]

HookpointKind = Literal["resid_post"]


@dataclass(frozen=True)
class ModelInfo:
    """Stable model identification captured in run_record and certificate."""

    backend: BackendName
    model_path_or_id: str
    revision: (
        str  # may be "unknown" for local models; verifier treats unknown as valid but records it.
    )
    tokenizer_path_or_id: str
    d_model: int
    n_layers: int


class HookHandle(Protocol):
    """A handle returned by hook registration to enable removal."""

    def remove(self) -> None: ...


class ModelAdapter(Protocol):
    """Minimal contract that downstream modules rely on.

    Required capabilities:
    - Tokenize prompts and provide attention masks.
    - Run forward pass to obtain logits.
    - Register a forward hook at an internal hookpoint.
    - Provide model metadata (layers, hidden size, fingerprint).

    The adapter MUST operate in eval mode and MUST disable dropout.

    Hooking semantics:
    - Hook functions receive (activation_tensor, batch_indices, position_indices) and return
      a modified activation tensor.
    - The hook MUST be applied only at specified (batch, position) indices; other tokens
      MUST remain unchanged.
    """

    info: ModelInfo
    tokenizer: Any

    def tokenize(self, prompts: Sequence[str]) -> Dict[str, Any]:
        """Tokenize prompts.

        Must return:
            - input_ids: int tensor [B, T]
            - attention_mask: int/bool tensor [B, T]

        Fail-closed:
            - If tokenizer does not provide attention_mask, compute it deterministically.
        """
        ...

    def forward_logits(self, input_ids: Any, attention_mask: Any) -> Any:
        """Return logits [B, T, V] for the given batch."""
        ...

    def resolve_candidate_layers(
        self, mode: str, explicit: Optional[List[int]] = None
    ) -> List[int]:
        """Resolve candidate layers from config.

        mode:
            - "quartiles": use floor(n/4), floor(n/2), floor(3n/4), n-1
            - "explicit": use the explicit list

        Fail-closed:
            - Ensure all indices are within [0, n_layers-1] and unique.
        """
        ...

    def register_hook(
        self,
        kind: HookpointKind,
        layer: int,
        hook_fn: Callable[[Any, Any, Any], Any],
    ) -> HookHandle:
        """Register a forward hook at (kind, layer).

        hook_fn signature:
            hook_fn(activation, batch_idx, pos_idx) -> activation_modified

        The adapter MUST map (kind, layer) to the correct internal tensor.

        Fail-closed:
            - If the mapping is not available for the backend/model, raise ValueError.

        """
        ...


def _parse_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).strip().lower()
    if s in {"float32", "fp32"}:
        return torch.float32
    if s in {"float16", "fp16"}:
        return torch.float16
    if s in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _require_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"Requested CUDA device '{device_str}' but torch.cuda.is_available() is False."
        )
    return device


def _local_path_fingerprint(path: str) -> str:
    """Deterministic fingerprint for a local path (names-only, not content)."""
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.is_file():
        return sha256(p.name.encode("utf-8")).hexdigest()

    files: list[str] = []
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        try:
            rel = f.relative_to(p).as_posix()
        except Exception:  # noqa: BLE001
            rel = f.as_posix()
        files.append(rel)
    return sha256(("\n".join(files) + "\n").encode("utf-8")).hexdigest()


def _resolve_candidate_layers(n_layers: int, mode: str, explicit: Optional[List[int]]) -> List[int]:
    if n_layers <= 0:
        raise ValueError("n_layers must be > 0")
    mode_s = str(mode)
    if mode_s == "quartiles":
        idxs = [n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1]
    elif mode_s == "explicit":
        if explicit is None:
            raise ValueError("explicit layer list must be provided when mode=='explicit'")
        idxs = [int(x) for x in explicit]
    else:
        raise ValueError(f"Unknown candidate layer mode: {mode}")

    uniq: list[int] = []
    for x in idxs:
        if x < 0 or x >= n_layers:
            raise ValueError(f"Layer index out of range: {x} for n_layers={n_layers}")
        if x not in uniq:
            uniq.append(x)
    return uniq


@dataclass(frozen=True)
class _TLHookHandle:
    model: Any
    hook_name: str

    def remove(self) -> None:
        hp = self.model.hook_dict.get(self.hook_name)
        if hp is None:
            return
        hp.remove_hooks(dir="fwd", including_permanent=False)


@dataclass(frozen=True)
class _HFHookHandle:
    handle: Any

    def remove(self) -> None:
        self.handle.remove()


class TransformerLensAdapter:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        info: ModelInfo,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self._device = device

    def tokenize(self, prompts: Sequence[str]) -> Dict[str, Any]:
        self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            # GPT-style tokenizers often omit pad_token; use eos as a deterministic fallback.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = enc.get("input_ids")
        if input_ids is None:
            raise ValueError("Tokenizer did not return input_ids.")

        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                raise ValueError("Tokenizer did not return attention_mask and has no pad_token_id.")
            attention_mask = (input_ids != pad_id).to(dtype=torch.int64)

        return {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention_mask.to(self._device),
        }

    def forward_logits(self, input_ids: Any, attention_mask: Any) -> Any:
        toks = torch.as_tensor(input_ids, device=self._device)
        mask = torch.as_tensor(attention_mask, device=self._device)
        return self.model(toks, attention_mask=mask, return_type="logits")

    def resolve_candidate_layers(
        self, mode: str, explicit: Optional[List[int]] = None
    ) -> List[int]:
        return _resolve_candidate_layers(self.info.n_layers, mode, explicit)

    def register_hook(
        self,
        kind: HookpointKind,
        layer: int,
        hook_fn: Callable[[Any, Any, Any], Any],
    ) -> HookHandle:
        if kind != "resid_post":
            raise ValueError(f"Unsupported hookpoint kind for transformer_lens: {kind}")

        hook_name = f"blocks.{int(layer)}.hook_resid_post"
        if hook_name not in getattr(self.model, "hook_dict", {}):
            raise ValueError(f"Hookpoint not found: {hook_name}")

        def tl_hook(activation: Any, _hook: Any) -> Any:
            return hook_fn(activation, None, None)

        self.model.add_hook(hook_name, tl_hook, dir="fwd")
        return _TLHookHandle(model=self.model, hook_name=hook_name)


class HuggingFaceAdapter:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        info: ModelInfo,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self._device = device

    def tokenize(self, prompts: Sequence[str]) -> Dict[str, Any]:
        self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            truncation=False,
        )
        input_ids = enc.get("input_ids")
        if input_ids is None:
            raise ValueError("Tokenizer did not return input_ids.")

        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                raise ValueError("Tokenizer did not return attention_mask and has no pad_token_id.")
            attention_mask = (input_ids != pad_id).to(dtype=torch.int64)

        return {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention_mask.to(self._device),
        }

    def forward_logits(self, input_ids: Any, attention_mask: Any) -> Any:
        toks = torch.as_tensor(input_ids, device=self._device)
        mask = torch.as_tensor(attention_mask, device=self._device)
        out = self.model(input_ids=toks, attention_mask=mask, use_cache=False)
        logits = getattr(out, "logits", None)
        if logits is None:
            raise ValueError("Model forward did not return logits.")
        return logits

    def resolve_candidate_layers(
        self, mode: str, explicit: Optional[List[int]] = None
    ) -> List[int]:
        return _resolve_candidate_layers(self.info.n_layers, mode, explicit)

    def register_hook(
        self,
        kind: HookpointKind,
        layer: int,
        hook_fn: Callable[[Any, Any, Any], Any],
    ) -> HookHandle:
        if kind != "resid_post":
            raise ValueError(f"Unsupported hookpoint kind for huggingface: {kind}")

        layers = None
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            layers = self.model.gpt_neox.layers

        if layers is None:
            raise ValueError(
                "Unsupported HF model: cannot locate transformer block list for hooking."
            )
        if layer < 0 or layer >= len(layers):
            raise ValueError(f"Layer index out of range: {layer}")

        block = layers[int(layer)]

        def hf_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                if not output:
                    raise ValueError("Unexpected empty tuple output from transformer block.")
                hidden = output[0]
                new_hidden = hook_fn(hidden, None, None)
                return (new_hidden,) + output[1:]
            if torch.is_tensor(output):
                return hook_fn(output, None, None)
            raise ValueError(f"Unsupported block output type for hooking: {type(output).__name__}")

        handle = block.register_forward_hook(hf_hook)
        return _HFHookHandle(handle=handle)


def load_model_from_config(cfg: Mapping[str, Any]) -> ModelAdapter:
    """Instantiate and return a ModelAdapter based on cfg.

    Required cfg keys (enforced by schema):
        cfg['model']['backend']
        cfg['model']['model_path_or_id']
        cfg['model']['revision'] (may be null; adapter records resolved revision)
        cfg['model']['tokenizer_path_or_id'] (may be null; defaults to model_path_or_id)
        cfg['run']['device'], cfg['run']['dtype']

    Pseudocode:
        backend = cfg['model']['backend']
        if backend == 'transformer_lens':
            model = HookedTransformer.from_pretrained(model_path_or_id, revision=revision)
            tokenizer = model.tokenizer or AutoTokenizer.from_pretrained(...)
            adapter = TransformerLensAdapter(model, tokenizer, device, dtype)
        elif backend == 'huggingface':
            model = AutoModelForCausalLM.from_pretrained(model_path_or_id, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_id or model_path_or_id)
            adapter = HuggingFaceAdapter(model, tokenizer, device, dtype)
        else:
            raise ValueError

        adapter MUST:
          - set model.eval()
          - disable dropout
          - record ModelInfo with resolved revision or local fingerprint

    """
    model_cfg = cfg.get("model", {})
    run_cfg = cfg.get("run", {})
    if not isinstance(model_cfg, Mapping) or not isinstance(run_cfg, Mapping):
        raise ValueError("cfg must include 'model' and 'run' mappings.")

    backend = str(model_cfg["backend"])
    model_path_or_id = str(model_cfg["model_path_or_id"])
    revision = model_cfg.get("revision")
    tokenizer_path_or_id = model_cfg.get("tokenizer_path_or_id") or model_path_or_id
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))

    device = _require_device(str(run_cfg.get("device", "cpu")))
    dtype = _parse_dtype(str(run_cfg.get("dtype", "float32")))

    if backend == "transformer_lens":
        try:
            from transformer_lens import HookedTransformer  # type: ignore[import-untyped]
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "transformer_lens backend requested but transformer_lens is not available."
            ) from e

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            str(tokenizer_path_or_id),
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        tok.padding_side = "right"
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token

        model = HookedTransformer.from_pretrained(
            model_path_or_id,
            device=str(device),
            dtype=str(run_cfg.get("dtype", "float32")),
            tokenizer=tok,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model.eval()

        resolved_revision = str(revision) if revision is not None else "unknown"
        info = ModelInfo(
            backend="transformer_lens",
            model_path_or_id=model_path_or_id,
            revision=resolved_revision,
            tokenizer_path_or_id=str(tokenizer_path_or_id),
            d_model=int(model.cfg.d_model),
            n_layers=int(model.cfg.n_layers),
        )
        return TransformerLensAdapter(model=model, tokenizer=tok, info=info, device=device)

    if backend == "huggingface":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "huggingface backend requested but transformers is not available."
            ) from e

        tok = AutoTokenizer.from_pretrained(
            str(tokenizer_path_or_id),
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        tok.padding_side = "right"
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token

        model_load_kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path_or_id, **model_load_kwargs)
        except Exception as e:  # noqa: BLE001
            err_txt = f"{type(e).__name__}: {e}".lower()
            # HF cache can occasionally contain a truncated .safetensors file.
            # Fall back to PyTorch weights instead of failing the whole run.
            if "safetensor" not in err_txt:
                raise
            print(
                "[warn] huggingface safetensors load failed; retrying with "
                f"use_safetensors=False for model={model_path_or_id}"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path_or_id,
                **model_load_kwargs,
                use_safetensors=False,
            )
        model = cast(Any, model)
        model.to(device)
        model.eval()

        hf_resolved_revision: str | None = (
            str(revision)
            if revision is not None
            else getattr(getattr(model, "config", None), "_commit_hash", None)
        )
        if not hf_resolved_revision:
            if os.path.exists(model_path_or_id):
                hf_resolved_revision = (
                    f"local_list_sha256:{_local_path_fingerprint(model_path_or_id)}"
                )
            else:
                hf_resolved_revision = "unknown"

        cfg_obj = getattr(model, "config", None)
        d_model = int(getattr(cfg_obj, "n_embd", getattr(cfg_obj, "hidden_size", 0)))
        n_layers = int(getattr(cfg_obj, "n_layer", getattr(cfg_obj, "num_hidden_layers", 0)))
        if d_model <= 0 or n_layers <= 0:
            raise ValueError("Could not infer d_model/n_layers from HF model config.")

        info = ModelInfo(
            backend="huggingface",
            model_path_or_id=model_path_or_id,
            revision=str(hf_resolved_revision),
            tokenizer_path_or_id=str(tokenizer_path_or_id),
            d_model=d_model,
            n_layers=n_layers,
        )
        return HuggingFaceAdapter(model=model, tokenizer=tok, info=info, device=device)

    raise ValueError(f"Unknown backend: {backend}")


def assert_or_select_answer_tokens(adapter: ModelAdapter, cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Select answer tokens (primary or fallback) in a deterministic, fail-closed way.

    The tokenizer MUST map both yes/no strings to single tokens.

    Behavior (MUST):
      1) Try primary tokens from cfg['answer_tokens']['primary'].
      2) If either is not a single token, try fallback tokens.
      3) If fallback also fails, abort.

    Return:
      {'yes': yes_token_str, 'no': no_token_str, 'mode': 'primary'|'fallback'}

    The selected mode MUST be written to run_record and certificate.

    """
    answer_cfg = cfg.get("answer_tokens", {})
    if not isinstance(answer_cfg, Mapping):
        raise ValueError("cfg['answer_tokens'] must be provided.")

    primary = answer_cfg.get("primary", {})
    fallback = answer_cfg.get("fallback", {})
    if not isinstance(primary, Mapping) or not isinstance(fallback, Mapping):
        raise ValueError("cfg['answer_tokens']['primary'/'fallback'] must be mappings.")

    def encode_one(token_str: str) -> int:
        ids = adapter.tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token string is not single-token: {token_str!r} -> {ids}")
        decoded = adapter.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        if decoded != token_str:
            raise ValueError(f"Tokenizer decode mismatch: {token_str!r} -> {decoded!r}")
        return int(ids[0])

    def try_pair(pair: Mapping[str, Any]) -> Optional[Dict[str, str]]:
        yes = pair.get("yes")
        no = pair.get("no")
        if not isinstance(yes, str) or not isinstance(no, str):
            raise ValueError("Answer token pair must contain string keys 'yes' and 'no'.")
        try:
            yes_id = encode_one(yes)
            no_id = encode_one(no)
        except ValueError:
            return None
        return {
            "yes": yes,
            "no": no,
            "yes_id": str(yes_id),
            "no_id": str(no_id),
        }

    primary_res = try_pair(primary)
    if primary_res is not None:
        primary_res["mode"] = "primary"
        return primary_res

    fallback_res = try_pair(fallback)
    if fallback_res is not None:
        fallback_res["mode"] = "fallback"
        return fallback_res

    raise ValueError(
        "Neither primary nor fallback answer token pairs are single-token under the tokenizer."
    )


def discover_supported_hookpoints(adapter: ModelAdapter) -> List[str]:
    """Return a list of supported hookpoint kinds for this adapter.

    Used only for debugging and fail-closed error messaging.
    """
    # v1 SSOT hookpoint kind
    return ["resid_post"]
