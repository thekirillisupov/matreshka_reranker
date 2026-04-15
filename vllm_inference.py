"""vLLM-based reranker inference using truncated (surgically modified) models.

Uses vLLM's offline Python API with task="score" for cross-encoder scoring.
Falls back to generation-based scoring if the score API is not compatible.
"""

import gc
from typing import Optional

import numpy as np
import torch


DEFAULT_PROMPT = "Predict whether passage B contains an answer to query A."
SEP = "\n"


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


class VLLMReranker:
    """vLLM-based reranker for truncated models.

    Loads a truncated model (created by model_surgery.py) and scores
    query-passage pairs via vLLM's offline scoring API.

    Args:
        model_path: Path to truncated model directory.
        dtype: Model dtype string ("bfloat16", "float16", "auto").
        max_model_len: Maximum sequence length.
        gpu_memory_utilization: Fraction of GPU memory to use.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 1024,
        gpu_memory_utilization: float = 0.85,
    ):
        self.model_path = model_path
        self.llm = None
        self._mode = None

        self._init_scoring_mode(dtype, max_model_len, gpu_memory_utilization)

    def _init_scoring_mode(self, dtype, max_model_len, gpu_memory_utilization):
        """Try to initialize vLLM in scoring mode, fall back to generation."""
        from vllm import LLM

        # Try task="score" first (native cross-encoder support)
        try:
            print(f"Attempting vLLM scoring mode for {self.model_path}...")
            self.llm = LLM(
                model=self.model_path,
                task="score",
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            self._mode = "score"
            print("vLLM initialized in SCORE mode")
            return
        except Exception as e:
            print(f"Score mode failed: {e}")
            print("Falling back to generation mode...")

        # Fallback: load as generation model
        try:
            self.llm = LLM(
                model=self.model_path,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            self._mode = "generate"
            print("vLLM initialized in GENERATE mode (will extract logits)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model in both score and generate modes: {e}"
            )

    def score(
        self,
        pairs: list[dict],
        normalize: bool = True,
    ) -> list[float]:
        """Score query-passage pairs.

        Args:
            pairs: List of dicts with 'query' and 'passage' keys.
            normalize: If True, apply sigmoid to raw scores.

        Returns:
            List of relevance scores.
        """
        if self._mode == "score":
            return self._score_native(pairs, normalize)
        else:
            return self._score_via_generation(pairs, normalize)

    def _score_native(self, pairs: list[dict], normalize: bool) -> list[float]:
        """Score using vLLM's native score() API."""
        text_1 = [self._format_query(p["query"]) for p in pairs]
        text_2 = [self._format_passage(p["passage"]) for p in pairs]

        outputs = self.llm.score(text_1, text_2)

        scores = []
        for output in outputs:
            raw_score = output.outputs.score
            if normalize:
                scores.append(sigmoid(raw_score))
            else:
                scores.append(raw_score)
        return scores

    def _score_via_generation(self, pairs: list[dict], normalize: bool) -> list[float]:
        """Fallback: format input manually and extract the model's score logit."""
        from vllm import SamplingParams

        prompts = []
        for p in pairs:
            text = self._build_full_input(p["query"], p["passage"])
            prompts.append(text)

        # Generate with max_tokens=1 to just get the logit
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=1,
        )
        outputs = self.llm.generate(prompts, sampling_params)

        # In generation mode the raw score is not directly accessible,
        # so we return logprobs-based heuristic scores.
        # This is a fallback — native score mode is preferred.
        scores = []
        for output in outputs:
            # Use the first generated token's log probability as a proxy
            if output.outputs[0].logprobs:
                first_logprob = list(output.outputs[0].logprobs[0].values())[0]
                score = first_logprob.logprob
            else:
                score = 0.0
            if normalize:
                scores.append(sigmoid(score))
            else:
                scores.append(score)
        return scores

    def _format_query(self, query: str) -> str:
        return f"A: {query}"

    def _format_passage(self, passage: str) -> str:
        return f"B: {passage}"

    def _build_full_input(self, query: str, passage: str) -> str:
        """Build the full input string in the model's expected format."""
        q = self._format_query(query)
        p = self._format_passage(passage)
        return f"{q}{SEP}{p}{SEP}{DEFAULT_PROMPT}"

    @property
    def mode(self) -> str:
        return self._mode

    def cleanup(self):
        """Release GPU memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        torch.cuda.empty_cache()


def score_with_vllm(
    model_path: str,
    pairs: list[dict],
    normalize: bool = True,
    **kwargs,
) -> list[float]:
    """Convenience function: load model, score, cleanup.

    Args:
        model_path: Path to truncated model.
        pairs: Query-passage pairs.
        normalize: Apply sigmoid.
        **kwargs: Passed to VLLMReranker constructor.

    Returns:
        List of scores.
    """
    reranker = VLLMReranker(model_path, **kwargs)
    scores = reranker.score(pairs, normalize=normalize)
    mode = reranker.mode
    reranker.cleanup()
    return scores, mode


def score_multiple_models(
    model_dirs: dict[int, str],
    pairs: list[dict],
    normalize: bool = True,
    **kwargs,
) -> dict[int, list[float]]:
    """Score pairs with multiple truncated models (one at a time).

    Args:
        model_dirs: Dict mapping cutoff_layer -> model directory path.
        pairs: Query-passage pairs.
        normalize: Apply sigmoid.

    Returns:
        Dict mapping cutoff_layer -> list of scores.
    """
    all_scores = {}
    for layer, model_dir in sorted(model_dirs.items()):
        print(f"\n--- vLLM scoring: layer {layer} ---")
        scores, mode = score_with_vllm(model_dir, pairs, normalize=normalize, **kwargs)
        print(f"  Mode: {mode}, scored {len(scores)} pairs")
        all_scores[layer] = scores
    return all_scores


if __name__ == "__main__":
    import argparse
    from test_data import generate_test_pairs

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to truncated model directory")
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    pairs = generate_test_pairs()
    reranker = VLLMReranker(args.model_path, max_model_len=args.max_model_len)

    scores = reranker.score(pairs)
    print(f"\nMode: {reranker.mode}")
    print(f"Scores (first 5): {scores[:5]}")
    print(f"Mean: {np.mean(scores):.4f}")

    reranker.cleanup()
