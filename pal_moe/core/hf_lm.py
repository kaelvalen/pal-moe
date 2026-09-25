"""HuggingFace causal-LM backend: frozen, hooked, content-hashed (v3 phase 3).

Infrastructure only - nothing here is a result. The adapter exposes three things the
v3 paths need, and changes no weight of the base model:

- `keys(texts, layer)`       hidden state at a configurable decoder layer (the FIXED
                             ADDRESS for the fast path), pooled at the last token;
- `down_proj_io(texts)`      the input and output of one MLP down-projection at the last
                             token (keys and values for the medium path's closed-form edit);
- `set_delta(layer, D)`      a forward hook that adds `x @ D` to that down-projection's
                             output. The base weight is never written (a 4-bit bitsandbytes
                             weight cannot be edited in place anyway), so removing the delta
                             restores the base model bitwise.

`transformers` / `bitsandbytes` are imported lazily; `from_pretrained(..., load_in_4bit=
True)` is the 7-8B path, and `HFCausalLM(model, tokenizer)` wraps any already-built
model (the tests use a tiny randomly initialised Llama).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .hashing import digest, module_digest


def decoder_layers(model) -> list:
    for path in (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError(f"cannot find decoder layers on {type(model).__name__}")


def down_projection(layer) -> torch.nn.Module:
    mlp = getattr(layer, "mlp", None)
    for name in ("down_proj", "c_proj", "dense_4h_to_h", "fc2"):
        if mlp is not None and hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError(f"cannot find an MLP down-projection on {type(layer).__name__}")


class HFCausalLM:
    def __init__(
        self,
        model,
        tokenizer,
        key_layers=(-8,),
        edit_layer: int = -8,
        base_hash: str | None = None,
        hash_weights: bool = False,
    ):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.layers = decoder_layers(model)
        self.key_layers = tuple(key_layers)
        self.edit_layer = edit_layer
        self._deltas: dict[int, torch.Tensor] = {}
        self._delta_handles: dict[int, object] = {}
        cfg = getattr(model, "config", None)
        if base_hash is None:
            ident = [
                type(model).__name__,
                getattr(cfg, "_name_or_path", ""),
                getattr(cfg, "_commit_hash", ""),
                repr(getattr(cfg, "quantization_config", None)),
            ]
            base_hash = module_digest(model) if hash_weights else digest(ident)
        self.base_hash = base_hash

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        load_in_4bit: bool = True,
        device_map="auto",
        dtype=torch.bfloat16,
        **kw,
    ) -> HFCausalLM:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        qcfg = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=qcfg, device_map=device_map, dtype=dtype
        )
        return cls(model, tok, **kw)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def _encode(self, texts: list[str]) -> dict:
        batch = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in batch.items()}

    @staticmethod
    def _last_index(batch) -> torch.Tensor:
        mask = batch.get("attention_mask")
        if mask is None:
            n = batch["input_ids"].size(1)
            return torch.full(
                (batch["input_ids"].size(0),), n - 1, device=batch["input_ids"].device
            )
        # left or right padding: the last position whose mask is 1
        pos = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
        return (pos * mask).argmax(-1)

    # -- keys (fast path address) -------------------------------------------------

    @torch.no_grad()
    def keys(self, texts: list[str], layer: int | None = None) -> torch.Tensor:
        """[B, hidden] float32 hidden state after decoder `layer`, at the last token."""
        layer = self.key_layers[0] if layer is None else layer
        batch = self._encode(texts)
        out = self.model(**batch, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[layer if layer < 0 else layer + 1]
        idx = self._last_index(batch)
        return hs[torch.arange(hs.size(0), device=hs.device), idx].float()

    # -- medium path I/O -------------------------------------------------------------

    @contextmanager
    def _capture(self, layer: int):
        module = down_projection(self.layers[layer])
        store = {}

        def hook(mod, inputs, output):
            store["x"], store["y"] = inputs[0], output

        handle = module.register_forward_hook(hook)
        try:
            yield store
        finally:
            handle.remove()

    @torch.no_grad()
    def down_proj_io(self, texts: list[str], layer: int | None = None):
        """(input [B, d_ff], output [B, d_model]) of the edit layer's down-projection."""
        layer = self.edit_layer if layer is None else layer
        batch = self._encode(texts)
        with self._capture(layer) as store:
            self.model(**batch, use_cache=False)
        idx = self._last_index(batch)
        rows = torch.arange(idx.numel(), device=idx.device)
        return store["x"][rows, idx].float(), store["y"][rows, idx].float()

    def set_delta(self, layer: int, delta: torch.Tensor | None) -> None:
        """Install (or remove, with None) `y += x @ delta` on a down-projection."""
        layer = layer % len(self.layers)
        if layer in self._delta_handles:
            self._delta_handles.pop(layer).remove()
            self._deltas.pop(layer, None)
        if delta is None:
            return
        module = down_projection(self.layers[layer])
        self._deltas[layer] = delta

        def hook(mod, inputs, output):
            d = self._deltas[layer]
            return output + (inputs[0].to(d.dtype) @ d).to(output.dtype)

        self._delta_handles[layer] = module.register_forward_hook(hook)

    @contextmanager
    def base_only(self):
        """Temporarily remove every installed delta (edit contributions are computed
        against the base model, so they are independent of each other)."""
        saved = dict(self._deltas)
        for layer in list(self._delta_handles):
            self.set_delta(layer, None)
        try:
            yield
        finally:
            for layer, d in saved.items():
                self.set_delta(layer, d)

    def delta_digest(self) -> str:
        return digest({k: v for k, v in sorted(self._deltas.items())})

    # -- read path -------------------------------------------------------------------

    @torch.no_grad()
    def next_token_logits(self, texts: list[str]) -> torch.Tensor:
        batch = self._encode(texts)
        logits = self.model(**batch, use_cache=False).logits
        idx = self._last_index(batch)
        return logits[torch.arange(idx.numel(), device=idx.device), idx].float()

    @torch.no_grad()
    def target_logprob(self, prompts: list[str], targets: list[str]) -> torch.Tensor:
        """Sum log p(target | prompt) per pair (teacher-forced)."""
        out = []
        for p, t in zip(prompts, targets):
            ids_p = self.tokenizer(p, return_tensors="pt")["input_ids"].to(self.device)
            ids_t = self.tokenizer(t, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].to(self.device)
            ids = torch.cat([ids_p, ids_t], dim=1)
            logp = (
                self.model(input_ids=ids, use_cache=False)
                .logits.float()
                .log_softmax(-1)
            )
            n = ids_t.size(1)
            tgt = ids[0, -n:]
            out.append(logp[0, -n - 1 : -1].gather(-1, tgt[:, None]).sum())
        return torch.stack(out)

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 16) -> list[str]:
        batch = self._encode(prompts)
        pad = getattr(self.tokenizer, "pad_token_id", None)
        ids = self.model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad
        )
        return [
            self.tokenizer.decode(
                row[batch["input_ids"].size(1) :], skip_special_tokens=True
            )
            for row in ids
        ]

    def target_value(
        self,
        prompt: str,
        target: str,
        layer: int | None = None,
        steps: int = 20,
        lr: float = 0.5,
        weight_decay: float = 1e-3,
    ) -> torch.Tensor:
        """The down-projection output at the prompt's last token that makes `target` likely.

        ROME-style: optimise one activation-space vector `delta` added to that output
        (the base model stays frozen; no weight is trained here), return `y + delta`.
        The weight change itself is the medium path's closed-form solve.
        """
        layer = self.edit_layer if layer is None else layer
        module = down_projection(self.layers[layer])
        ids_p = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        ids_t = self.tokenizer(target, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(self.device)
        ids = torch.cat([ids_p, ids_t], dim=1)
        pos = ids_p.size(1) - 1
        _, y0 = self.down_proj_io([prompt], layer)
        delta = torch.zeros_like(y0[0], requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)

        def hook(mod, inputs, output):
            out = output.clone()
            out[:, pos] = out[:, pos] + delta.to(out.dtype)
            return out

        handle = module.register_forward_hook(hook)
        try:
            n = ids_t.size(1)
            for _ in range(steps):
                logp = (
                    self.model(input_ids=ids, use_cache=False)
                    .logits.float()
                    .log_softmax(-1)
                )
                loss = (
                    -logp[0, -n - 1 : -1].gather(-1, ids[0, -n:, None]).mean()
                    + weight_decay * delta.pow(2).sum()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        finally:
            handle.remove()
        return (y0[0] + delta.detach()).float()
