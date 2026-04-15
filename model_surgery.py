"""Model surgery: truncate the Gemma2-based reranker to N layers.

Creates a self-contained model directory that can be loaded by both
torch (with trust_remote_code=True) and vLLM for scoring inference.

The truncated model keeps:
  - Embedding layer
  - First N decoder layers
  - Final RMS norm
  - Classification head (CostWiseHead)
  - Custom modeling files (gemma_model.py, gemma_config.py)
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "BAAI/bge-reranker-v2.5-gemma2-lightweight"


def create_truncated_model(
    source_model_name: str = DEFAULT_MODEL,
    cutoff_layer: int = 28,
    output_dir: str = "./truncated_models",
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Create a truncated model with only the first `cutoff_layer` layers.

    Args:
        source_model_name: HuggingFace model name or path to source model.
        cutoff_layer: Number of layers to keep (e.g. 28 keeps layers 0-27).
        output_dir: Base directory for truncated models.
        device: Device for loading (use "cpu" to save GPU memory).
        dtype: Model weights dtype.

    Returns:
        Path to the saved truncated model directory.
    """
    model_dir = os.path.join(output_dir, f"layer_{cutoff_layer}")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading source model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    layer_wise = getattr(config, "layer_wise", False)

    print(f"Source model: {num_layers} layers, hidden_size={hidden_size}, layer_wise={layer_wise}")
    assert cutoff_layer <= num_layers, (
        f"cutoff_layer={cutoff_layer} exceeds model's {num_layers} layers"
    )

    # --- Extract the classification head weight ---
    if layer_wise and isinstance(model.lm_head, torch.nn.ModuleList):
        # Use the first (and likely only) head
        head_weight = model.lm_head[0].linear_head.weight.data.clone()
        print(f"Extracted CostWiseHead weight: {head_weight.shape}")
    else:
        raise ValueError(
            "Model is not layer_wise or lm_head structure is unexpected. "
            "Cannot extract classification head."
        )

    # --- Build truncated state dict ---
    full_sd = model.state_dict()
    truncated_sd = {}

    for key, value in full_sd.items():
        # Keep embedding
        if key.startswith("model.embed_tokens."):
            truncated_sd[key] = value
        # Keep only first N layers
        elif key.startswith("model.layers."):
            parts = key.split(".")
            layer_idx = int(parts[2])
            if layer_idx < cutoff_layer:
                truncated_sd[key] = value
        # Keep final norm
        elif key.startswith("model.norm."):
            truncated_sd[key] = value

    # Add classification head as a single CostWiseHead (lm_head.0)
    truncated_sd["lm_head.0.linear_head.weight"] = head_weight

    print(f"Truncated state dict: {len(truncated_sd)} tensors")

    # --- Save state dict as safetensors ---
    save_file(truncated_sd, os.path.join(model_dir, "model.safetensors"))
    print(f"Saved weights to {model_dir}/model.safetensors")

    # --- Save modified config ---
    config_dict = config.to_dict()
    config_dict["num_hidden_layers"] = cutoff_layer
    config_dict["layer_wise"] = True
    config_dict["start_layer"] = cutoff_layer  # Only the cutoff layer has a head
    config_dict["layer_sep"] = cutoff_layer    # Single head
    # Ensure auto_map points to local files
    config_dict["auto_map"] = {
        "AutoConfig": "gemma_config.CostWiseGemmaConfig",
        "AutoModel": "gemma_model.CostWiseGemmaForCausalLM",
        "AutoModelForCausalLM": "gemma_model.CostWiseGemmaForCausalLM",
    }
    config_dict["model_type"] = "cost_wise_gemma"

    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved config (num_hidden_layers={cutoff_layer})")

    # --- Save tokenizer ---
    tokenizer.save_pretrained(model_dir)
    print("Saved tokenizer")

    # --- Copy custom modeling files from source ---
    _copy_custom_code(source_model_name, model_dir)

    print(f"\nTruncated model saved to: {model_dir}")

    # --- Cleanup ---
    del model, full_sd, truncated_sd
    torch.cuda.empty_cache()

    return model_dir


def _copy_custom_code(source_model_name: str, dest_dir: str):
    """Copy gemma_model.py and gemma_config.py from the source model cache."""
    from huggingface_hub import hf_hub_download

    for filename in ["gemma_model.py", "gemma_config.py"]:
        try:
            src_path = hf_hub_download(
                repo_id=source_model_name,
                filename=filename,
            )
            shutil.copy2(src_path, os.path.join(dest_dir, filename))
            print(f"Copied {filename}")
        except Exception as e:
            print(f"Warning: could not copy {filename}: {e}")
            print("  Attempting to find in transformers cache...")
            _try_copy_from_cache(source_model_name, filename, dest_dir)


def _try_copy_from_cache(model_name: str, filename: str, dest_dir: str):
    """Fallback: search the HF cache for the file."""
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if model_name.replace("/", "--") in str(repo.repo_path):
                for revision in repo.revisions:
                    for f in revision.files:
                        if f.file_name == filename:
                            shutil.copy2(str(f.file_path), os.path.join(dest_dir, filename))
                            print(f"  Found and copied {filename} from cache")
                            return
    except Exception:
        pass
    print(f"  Could not find {filename} — you may need to copy it manually")


def validate_truncated_model(
    model_dir: str,
    pairs: list[dict],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> list[float]:
    """Load a truncated model and score pairs to validate it works.

    Args:
        model_dir: Path to truncated model directory.
        pairs: Query-passage pairs.
        device: Device for inference.
        dtype: Model dtype.

    Returns:
        List of scores.
    """
    from torch_inference import TorchReranker

    print(f"\nValidating truncated model at {model_dir}...")
    reranker = TorchReranker(model_name=model_dir, device=device, dtype=dtype)
    scores = reranker.score(pairs, normalize=True)
    reranker.cleanup()
    return scores


def create_multiple_truncated_models(
    source_model_name: str = DEFAULT_MODEL,
    cutoff_layers: list[int] = None,
    output_dir: str = "./truncated_models",
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[int, str]:
    """Create truncated models for multiple cutoff layers.

    Loads the source model only once and extracts all truncated versions.

    Returns:
        Dict mapping cutoff_layer -> model directory path.
    """
    if cutoff_layers is None:
        cutoff_layers = [8, 16, 20, 24, 28]

    print(f"Loading source model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        source_model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )

    config = model.config
    num_layers = config.num_hidden_layers
    layer_wise = getattr(config, "layer_wise", False)

    # Extract head weight
    if layer_wise and isinstance(model.lm_head, torch.nn.ModuleList):
        head_weight = model.lm_head[0].linear_head.weight.data.clone()
    else:
        raise ValueError("Cannot extract CostWiseHead from model")

    full_sd = model.state_dict()
    model_dirs = {}

    for cutoff_layer in sorted(cutoff_layers):
        assert cutoff_layer <= num_layers
        model_dir = os.path.join(output_dir, f"layer_{cutoff_layer}")
        os.makedirs(model_dir, exist_ok=True)

        # Build truncated state dict
        truncated_sd = {}
        for key, value in full_sd.items():
            if key.startswith("model.embed_tokens."):
                truncated_sd[key] = value
            elif key.startswith("model.layers."):
                layer_idx = int(key.split(".")[2])
                if layer_idx < cutoff_layer:
                    truncated_sd[key] = value
            elif key.startswith("model.norm."):
                truncated_sd[key] = value

        truncated_sd["lm_head.0.linear_head.weight"] = head_weight

        # Save weights
        save_file(truncated_sd, os.path.join(model_dir, "model.safetensors"))

        # Save config
        config_dict = config.to_dict()
        config_dict["num_hidden_layers"] = cutoff_layer
        config_dict["layer_wise"] = True
        config_dict["start_layer"] = cutoff_layer
        config_dict["layer_sep"] = cutoff_layer
        config_dict["auto_map"] = {
            "AutoConfig": "gemma_config.CostWiseGemmaConfig",
            "AutoModel": "gemma_model.CostWiseGemmaForCausalLM",
            "AutoModelForCausalLM": "gemma_model.CostWiseGemmaForCausalLM",
        }
        config_dict["model_type"] = "cost_wise_gemma"

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save tokenizer
        tokenizer.save_pretrained(model_dir)

        # Copy custom code
        _copy_custom_code(source_model_name, model_dir)

        print(f"Created truncated model: layer_{cutoff_layer}")
        model_dirs[cutoff_layer] = model_dir

    del model, full_sd
    torch.cuda.empty_cache()
    return model_dirs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create truncated reranker models")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Source model name/path")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[8, 16, 20, 24, 28],
        help="Cutoff layers to create truncated models for",
    )
    parser.add_argument("--output-dir", default="./truncated_models")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_dirs = create_multiple_truncated_models(
        source_model_name=args.model,
        cutoff_layers=args.layers,
        output_dir=args.output_dir,
        device=args.device,
    )
    print("\nCreated models:")
    for layer, path in sorted(model_dirs.items()):
        print(f"  Layer {layer}: {path}")
