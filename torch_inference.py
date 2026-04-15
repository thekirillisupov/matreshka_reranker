"""Native torch inference for BAAI/bge-reranker-v2.5-gemma2-lightweight.

Supports both official cutoff_layers scoring and manual intermediate layer extraction
for matryoshka-style layer-wise evaluation.
"""

import torch
import numpy as np
from typing import Optional
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
DEFAULT_PROMPT = "Predict whether passage B contains an answer to query A."
SEP = "\n"


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def last_logit_pool(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract the last real token's logit for each sample in the batch."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return logits[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = logits.shape[0]
    return torch.stack(
        [logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0
    )


class TorchReranker:
    """Native torch reranker with intermediate layer extraction support.

    Args:
        model_name: HuggingFace model name or local path.
        device: torch device string (e.g. "cuda", "cuda:0", "cpu").
        dtype: torch dtype for model weights.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer.padding_side = "right"

        print(f"Loading model from {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

        self._read_config()
        print(
            f"Model loaded: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, "
            f"layer_wise={self.layer_wise}, "
            f"valid_cutoff_layers={self.valid_cutoff_layers}"
        )

    def _read_config(self):
        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        self.layer_wise = getattr(cfg, "layer_wise", False)
        self.start_layer = getattr(cfg, "start_layer", self.num_layers)
        self.layer_sep = getattr(cfg, "layer_sep", self.num_layers)
        self.final_logit_softcapping = getattr(cfg, "final_logit_softcapping", None)

        # Determine which layers have trained heads
        self.valid_cutoff_layers = list(
            range(self.start_layer, self.num_layers + 1, self.layer_sep)
        )

    def _get_head_weight(self) -> torch.Tensor:
        """Extract the classification head weight (shape: [1, hidden_size])."""
        if self.layer_wise and hasattr(self.model, "lm_head"):
            head = self.model.lm_head
            if isinstance(head, torch.nn.ModuleList):
                return head[0].linear_head.weight.data.clone()
        raise ValueError(
            "Cannot extract head weight: model is not layer_wise or has unexpected structure"
        )

    def _format_inputs(
        self,
        pairs: list[dict],
        query_max_length: int = 256,
        max_length: int = 512,
    ) -> tuple[dict, list[int], list[int]]:
        """Tokenize and format query-passage pairs into model inputs.

        Returns (batch_dict, query_lengths, prompt_lengths).
        """
        prompt_inputs = self.tokenizer(
            DEFAULT_PROMPT, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = self.tokenizer(SEP, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        encode_max_length = max_length + len(sep_inputs) + len(prompt_inputs)

        all_items = []
        query_lengths = []
        prompt_lengths = []

        for pair in pairs:
            query_text = f"A: {pair['query']}"
            passage_text = f"B: {pair['passage']}"

            query_tok = self.tokenizer(
                query_text,
                return_tensors=None,
                add_special_tokens=False,
                max_length=query_max_length,
                truncation=True,
            )
            passage_tok = self.tokenizer(
                passage_text,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )

            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_tok["input_ids"],
                sep_inputs + passage_tok["input_ids"],
                truncation="only_second",
                max_length=encode_max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            item.pop("token_type_ids", None)

            all_items.append(item)
            query_lengths.append(
                len([self.tokenizer.bos_token_id] + query_tok["input_ids"] + sep_inputs)
            )
            prompt_lengths.append(len(sep_inputs + prompt_inputs))

        # Pad batch
        batch = self.tokenizer.pad(
            [
                {"input_ids": it["input_ids"], "attention_mask": it["attention_mask"]}
                for it in all_items
            ],
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch, query_lengths, prompt_lengths

    @torch.no_grad()
    def score(
        self,
        pairs: list[dict],
        cutoff_layers: Optional[list[int]] = None,
        compress_layers: Optional[list[int]] = None,
        compress_ratio: int = 1,
        normalize: bool = True,
        batch_size: int = 8,
    ) -> dict[int, list[float]]:
        """Score pairs using the model's native forward with cutoff_layers.

        Uses the model's built-in layer_wise logic: only layers in
        valid_cutoff_layers are accepted (others are silently dropped).

        Args:
            pairs: List of dicts with 'query' and 'passage' keys.
            cutoff_layers: Which layers to score at. Defaults to model's valid layers.
            compress_layers: Layers at which to apply token compression.
            compress_ratio: Token compression ratio (1=no compression).
            normalize: If True, apply sigmoid to raw logits.
            batch_size: Batch size for inference.

        Returns:
            Dict mapping layer_index -> list of scores for each pair.
        """
        if cutoff_layers is None:
            cutoff_layers = self.valid_cutoff_layers

        all_scores = {layer: [] for layer in cutoff_layers}

        for start in trange(0, len(pairs), batch_size, desc="Scoring (native)"):
            batch_pairs = pairs[start : start + batch_size]
            batch, query_lengths, prompt_lengths = self._format_inputs(batch_pairs)

            outputs = self.model(
                **batch,
                output_hidden_states=True,
                cutoff_layers=cutoff_layers,
                compress_layer=compress_layers,
                compress_ratio=compress_ratio,
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths,
            )

            # outputs.logits is a tuple of tensors, one per valid cutoff layer
            # outputs.attention_masks is a tuple of attention masks
            logits_list = outputs.logits
            masks_list = outputs.attention_masks

            if not isinstance(logits_list, (tuple, list)):
                logits_list = (logits_list,)
            if not isinstance(masks_list, (tuple, list)):
                masks_list = (masks_list,)

            # Match logits to the cutoff layers that were actually used
            # The model filters cutoff_layers to valid ones internally
            used_layers = [l for l in cutoff_layers if l in self.valid_cutoff_layers]

            for i, layer in enumerate(used_layers):
                if i < len(logits_list):
                    logits = logits_list[i]
                    mask = (
                        masks_list[i]
                        if i < len(masks_list)
                        else batch["attention_mask"]
                    )
                    pooled = last_logit_pool(logits, mask)
                    scores = pooled.cpu().float().tolist()
                    if isinstance(scores[0], list):
                        scores = [s[0] for s in scores]
                    if normalize:
                        scores = [sigmoid(s) for s in scores]
                    all_scores[layer].extend(scores)

        # Remove empty layers (those not in valid_cutoff_layers)
        return {k: v for k, v in all_scores.items() if v}

    @torch.no_grad()
    def score_all_layers(
        self,
        pairs: list[dict],
        layers: Optional[list[int]] = None,
        normalize: bool = True,
        batch_size: int = 4,
    ) -> dict[int, list[float]]:
        """Score pairs by extracting hidden states from arbitrary layers.

        Bypasses the model's layer_wise filtering to test the matryoshka
        property: applying the same classification head to any layer's output.

        Args:
            pairs: List of dicts with 'query' and 'passage' keys.
            layers: Layer indices to extract scores from.
                    Defaults to a spread across the model depth.
            normalize: If True, apply sigmoid.
            batch_size: Batch size.

        Returns:
            Dict mapping layer_index -> list of scores.
        """
        if layers is None:
            # Sample layers across the model depth
            step = max(1, self.num_layers // 8)
            layers = list(range(step, self.num_layers, step))
            if self.num_layers not in layers:
                layers.append(self.num_layers)

        head_weight = self._get_head_weight()  # [1, hidden_size]
        norm_layer = self.model.model.norm  # Gemma2RMSNorm

        all_scores = {layer: [] for layer in layers}

        for start in trange(0, len(pairs), batch_size, desc="Scoring (all layers)"):
            batch_pairs = pairs[start : start + batch_size]
            batch, query_lengths, prompt_lengths = self._format_inputs(batch_pairs)

            # Call the base model directly (not CostWiseGemmaForCausalLM.forward)
            # to get ALL hidden states without layer_wise filtering
            base_model = self.model.model

            # Temporarily disable layer_wise to get standard hidden states
            orig_layer_wise = base_model.config.layer_wise
            base_model.config.layer_wise = False

            outputs = base_model(
                **batch,
                output_hidden_states=True,
            )

            base_model.config.layer_wise = orig_layer_wise

            # outputs.hidden_states: tuple of (num_layers + 1) tensors
            # Index 0 = embedding output, index i = output of layer i
            hidden_states = outputs.hidden_states
            attention_mask = batch["attention_mask"]

            for layer in layers:
                if layer > len(hidden_states) - 1:
                    continue

                hs = hidden_states[layer]  # [batch, seq_len, hidden_size]
                # Apply the final RMS norm (same as model does at cutoff)
                hs_normed = norm_layer(hs)
                # Pool last real token
                pooled = last_logit_pool(
                    hs_normed, attention_mask
                )  # [batch, hidden_size]
                # Apply classification head
                logits = torch.nn.functional.linear(pooled, head_weight)  # [batch, 1]

                # Apply softcapping if configured
                if self.final_logit_softcapping is not None:
                    logits = logits / self.final_logit_softcapping
                    logits = torch.tanh(logits)
                    logits = logits * self.final_logit_softcapping

                scores = logits.squeeze(-1).cpu().float().tolist()
                if not isinstance(scores, list):
                    scores = [scores]
                if normalize:
                    scores = [sigmoid(s) for s in scores]
                all_scores[layer].extend(scores)

        return {k: v for k, v in all_scores.items() if v}

    def get_head_weight(self) -> torch.Tensor:
        """Public accessor for the classification head weight tensor."""
        return self._get_head_weight()

    def get_norm_state_dict(self) -> dict:
        """Return the final norm layer's state dict."""
        return self.model.model.norm.state_dict()

    def get_num_layers(self) -> int:
        return self.num_layers

    def cleanup(self):
        """Release GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    from test_data import generate_test_pairs

    pairs = generate_test_pairs()
    reranker = TorchReranker()

    # Test native scoring with model's valid cutoff layers
    print("\n=== Native scoring (model's cutoff_layers) ===")
    native_scores = reranker.score(pairs)
    for layer, scores in native_scores.items():
        print(
            f"Layer {layer}: mean={np.mean(scores):.4f}, min={np.min(scores):.4f}, max={np.max(scores):.4f}"
        )

    # Test multi-layer extraction
    print("\n=== Multi-layer extraction (matryoshka test) ===")
    test_layers = [5, 10, 15, 20, 25, 28, 30, 35, 40, reranker.get_num_layers()]
    test_layers = [l for l in test_layers if l <= reranker.get_num_layers()]
    multi_scores = reranker.score_all_layers(pairs, layers=test_layers)
    for layer, scores in sorted(multi_scores.items()):
        print(
            f"Layer {layer}: mean={np.mean(scores):.4f}, min={np.min(scores):.4f}, max={np.max(scores):.4f}"
        )

    reranker.cleanup()
