"""Main benchmark: compare torch native vs vLLM reranker scores across layers.

Usage:
    python benchmark.py                          # Full pipeline
    python benchmark.py --torch-only             # Only torch inference (skip vLLM)
    python benchmark.py --cutoff-layers 20 28    # Specific layers
    python benchmark.py --skip-surgery           # Reuse existing truncated models
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tabulate import tabulate

from test_data import generate_test_pairs
from torch_inference import TorchReranker


def compute_metrics(
    scores_a: list[float], scores_b: list[float]
) -> dict[str, float]:
    """Compute comparison metrics between two score lists."""
    a, b = np.array(scores_a), np.array(scores_b)
    pearson_r, _ = pearsonr(a, b)
    spearman_rho, _ = spearmanr(a, b)
    mae = np.mean(np.abs(a - b))
    max_err = np.max(np.abs(a - b))

    # Rank agreement: how many pairs have the same ordering
    n = len(a)
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (a[i] - a[j]) * (b[i] - b[j]) > 0:
                concordant += 1
            elif a[i] == a[j] or b[i] == b[j]:
                concordant += 0.5
    rank_agreement = concordant / total if total > 0 else 1.0

    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "mae": mae,
        "max_error": max_err,
        "rank_agreement": rank_agreement,
    }


def print_scores_by_label(
    pairs: list[dict], scores: list[float], title: str
):
    """Print score statistics grouped by relevance label."""
    print(f"\n{title}")
    print("-" * 60)
    for label in ["high", "medium", "low", "none"]:
        label_scores = [s for p, s in zip(pairs, scores) if p["label"] == label]
        if label_scores:
            print(
                f"  {label:>8s}: mean={np.mean(label_scores):.4f}, "
                f"min={np.min(label_scores):.4f}, max={np.max(label_scores):.4f}"
            )


def run_torch_benchmark(
    pairs: list[dict],
    model_name: str,
    multi_layers: list[int],
    device: str,
    batch_size: int,
) -> tuple[dict, TorchReranker]:
    """Run torch inference and return scores + reranker (for head extraction)."""
    print("\n" + "=" * 70)
    print("TORCH NATIVE INFERENCE")
    print("=" * 70)

    reranker = TorchReranker(model_name=model_name, device=device)

    # Native scoring (model's valid cutoff layers)
    print("\n--- Native scoring (model's trained cutoff layers) ---")
    t0 = time.time()
    native_scores = reranker.score(pairs, batch_size=batch_size)
    native_time = time.time() - t0
    print(f"Time: {native_time:.2f}s")

    for layer, scores in native_scores.items():
        print_scores_by_label(pairs, scores, f"Layer {layer} (native)")

    # Multi-layer extraction
    print("\n--- Multi-layer extraction (matryoshka test) ---")
    t0 = time.time()
    multi_scores = reranker.score_all_layers(
        pairs, layers=multi_layers, batch_size=batch_size
    )
    multi_time = time.time() - t0
    print(f"Time: {multi_time:.2f}s")

    all_torch_scores = {**native_scores, **multi_scores}
    return all_torch_scores, reranker


def run_vllm_benchmark(
    pairs: list[dict],
    model_dirs: dict[int, str],
) -> dict[int, list[float]]:
    """Run vLLM inference on truncated models."""
    from vllm_inference import score_multiple_models

    print("\n" + "=" * 70)
    print("vLLM INFERENCE (truncated models)")
    print("=" * 70)

    t0 = time.time()
    vllm_scores = score_multiple_models(model_dirs, pairs)
    vllm_time = time.time() - t0
    print(f"\nTotal vLLM time: {vllm_time:.2f}s")

    for layer, scores in sorted(vllm_scores.items()):
        print_scores_by_label(pairs, scores, f"Layer {layer} (vLLM)")

    return vllm_scores


def print_layer_quality_table(
    pairs: list[dict],
    torch_scores: dict[int, list[float]],
    full_layer: int,
):
    """Print table comparing each layer's scores vs the full model."""
    print("\n" + "=" * 70)
    print(f"LAYER-WISE QUALITY (vs full model layer {full_layer})")
    print("=" * 70)

    if full_layer not in torch_scores:
        print(f"Warning: full model layer {full_layer} not in scores")
        return

    baseline = torch_scores[full_layer]
    rows = []

    for layer in sorted(torch_scores.keys()):
        scores = torch_scores[layer]
        if len(scores) != len(baseline):
            continue
        metrics = compute_metrics(baseline, scores)
        rows.append([
            layer,
            f"{metrics['pearson_r']:.4f}",
            f"{metrics['spearman_rho']:.4f}",
            f"{metrics['mae']:.4f}",
            f"{metrics['max_error']:.4f}",
            f"{np.mean(scores):.4f}",
            f"{metrics['rank_agreement']:.4f}",
        ])

    print(tabulate(
        rows,
        headers=["Layer", "Pearson r", "Spearman ρ", "MAE", "Max Error", "Mean Score", "Rank Agmt"],
        tablefmt="simple_outline",
    ))


def print_torch_vs_vllm_table(
    torch_scores: dict[int, list[float]],
    vllm_scores: dict[int, list[float]],
):
    """Print table comparing torch vs vLLM scores for the same layers."""
    print("\n" + "=" * 70)
    print("TORCH vs vLLM COMPARISON (same layer)")
    print("=" * 70)

    common_layers = sorted(set(torch_scores.keys()) & set(vllm_scores.keys()))
    if not common_layers:
        print("No common layers to compare")
        return

    rows = []
    for layer in common_layers:
        ts = torch_scores[layer]
        vs = vllm_scores[layer]
        if len(ts) != len(vs):
            rows.append([layer, "LENGTH MISMATCH", "", "", ""])
            continue
        metrics = compute_metrics(ts, vs)
        rows.append([
            layer,
            f"{metrics['pearson_r']:.6f}",
            f"{metrics['spearman_rho']:.6f}",
            f"{metrics['mae']:.6f}",
            f"{metrics['max_error']:.6f}",
        ])

    print(tabulate(
        rows,
        headers=["Layer", "Pearson r", "Spearman ρ", "MAE", "Max Error"],
        tablefmt="simple_outline",
    ))


def print_detailed_scores(
    pairs: list[dict],
    torch_scores: dict[int, list[float]],
    vllm_scores: dict[int, list[float]],
    max_pairs: int = 10,
):
    """Print per-pair scores for inspection."""
    print("\n" + "=" * 70)
    print("DETAILED PER-PAIR SCORES (first {} pairs)".format(max_pairs))
    print("=" * 70)

    # Pick one torch layer and one vllm layer for comparison
    torch_layers = sorted(torch_scores.keys())
    vllm_layers = sorted(vllm_scores.keys()) if vllm_scores else []

    headers = ["#", "Label", "Query (truncated)"]
    for l in torch_layers[-3:]:  # Last 3 torch layers
        headers.append(f"T-L{l}")
    for l in vllm_layers[-2:]:  # Last 2 vllm layers
        headers.append(f"V-L{l}")

    rows = []
    for i, pair in enumerate(pairs[:max_pairs]):
        row = [i, pair["label"], pair["query"][:30]]
        for l in torch_layers[-3:]:
            if i < len(torch_scores[l]):
                row.append(f"{torch_scores[l][i]:.4f}")
            else:
                row.append("-")
        for l in vllm_layers[-2:]:
            if i < len(vllm_scores[l]):
                row.append(f"{vllm_scores[l][i]:.4f}")
            else:
                row.append("-")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="simple_outline"))


def save_results(
    output_path: str,
    pairs: list[dict],
    torch_scores: dict[int, list[float]],
    vllm_scores: dict[int, list[float]],
):
    """Save all scores to a JSON file for later analysis."""
    results = {
        "pairs": [{"query": p["query"], "passage": p["passage"][:100], "label": p["label"]} for p in pairs],
        "torch_scores": {str(k): v for k, v in torch_scores.items()},
        "vllm_scores": {str(k): v for k, v in vllm_scores.items()},
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Reranker benchmark: torch vs vLLM across layers")
    parser.add_argument(
        "--model", default="BAAI/bge-reranker-v2.5-gemma2-lightweight",
        help="Source model name or path",
    )
    parser.add_argument(
        "--cutoff-layers", type=int, nargs="+", default=None,
        help="Layers for model surgery + vLLM (default: auto-detect from model)",
    )
    parser.add_argument(
        "--multi-layers", type=int, nargs="+", default=None,
        help="Layers for torch multi-layer extraction (default: spread across depth)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--surgery-dir", default="./truncated_models")
    parser.add_argument("--torch-only", action="store_true", help="Skip vLLM inference")
    parser.add_argument("--skip-surgery", action="store_true", help="Reuse existing truncated models")
    parser.add_argument("--output", default="benchmark_results.json", help="Results output file")
    args = parser.parse_args()

    # --- Generate test data ---
    print("Generating test data...")
    pairs = generate_test_pairs()
    print(f"Generated {len(pairs)} pairs")

    # --- Torch benchmark ---
    # Determine multi-layer list
    if args.multi_layers:
        multi_layers = args.multi_layers
    else:
        # Will be set after loading model to know num_layers
        multi_layers = None

    reranker = TorchReranker(model_name=args.model, device=args.device)
    num_layers = reranker.get_num_layers()

    if multi_layers is None:
        step = max(1, num_layers // 8)
        multi_layers = list(range(step, num_layers, step))
        if num_layers not in multi_layers:
            multi_layers.append(num_layers)

    # Native scores
    print("\n--- Native scoring ---")
    t0 = time.time()
    native_scores = reranker.score(pairs, batch_size=args.batch_size)
    native_time = time.time() - t0
    print(f"Native scoring: {native_time:.2f}s")

    for layer, scores in native_scores.items():
        print_scores_by_label(pairs, scores, f"Layer {layer} (native)")

    # Multi-layer extraction
    print("\n--- Multi-layer extraction ---")
    t0 = time.time()
    multi_scores = reranker.score_all_layers(
        pairs, layers=multi_layers, batch_size=args.batch_size
    )
    multi_time = time.time() - t0
    print(f"Multi-layer extraction: {multi_time:.2f}s")

    all_torch_scores = {**multi_scores, **native_scores}

    # Layer quality table
    full_layer = num_layers
    print_layer_quality_table(pairs, all_torch_scores, full_layer)

    # --- vLLM benchmark ---
    vllm_scores = {}
    if not args.torch_only:
        # Determine surgery layers
        if args.cutoff_layers:
            surgery_layers = args.cutoff_layers
        else:
            surgery_layers = reranker.valid_cutoff_layers

        # Cleanup torch model before vLLM to free GPU
        reranker.cleanup()

        # Model surgery
        if not args.skip_surgery:
            from model_surgery import create_multiple_truncated_models

            print("\n--- Model surgery ---")
            model_dirs = create_multiple_truncated_models(
                source_model_name=args.model,
                cutoff_layers=surgery_layers,
                output_dir=args.surgery_dir,
                device="cpu",
            )
        else:
            model_dirs = {
                layer: str(Path(args.surgery_dir) / f"layer_{layer}")
                for layer in surgery_layers
            }
            # Verify directories exist
            model_dirs = {
                layer: path
                for layer, path in model_dirs.items()
                if Path(path).exists()
            }
            if not model_dirs:
                print("No existing truncated models found. Run without --skip-surgery first.")
                return

        # vLLM scoring
        vllm_scores = run_vllm_benchmark(pairs, model_dirs)

        # Torch vs vLLM comparison
        print_torch_vs_vllm_table(all_torch_scores, vllm_scores)
    else:
        reranker.cleanup()

    # --- Detailed view ---
    print_detailed_scores(pairs, all_torch_scores, vllm_scores)

    # --- Save results ---
    save_results(args.output, pairs, all_torch_scores, vllm_scores)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
