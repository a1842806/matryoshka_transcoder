"""Train all Gemma-2-2B layers (except 8, 12, 17) with 10k steps using 2 GPUs in parallel."""

import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import torch
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import (
    TranscoderActivationsStore,
    create_transcoder_config,
)
from src.training.training import train_transcoder
from src.utils.config import get_default_cfg, post_init_cfg


def build_config_for_layer(layer: int, device: str, num_tokens: int = 10_240_000) -> dict:
    """
    Build configuration for a specific layer.
    
    Args:
        layer: Layer number to train
        device: GPU device string (e.g., "cuda:0", "cuda:1")
        num_tokens: Number of tokens for training (default 10k steps = 10_240_000 tokens)
    
    Returns:
        Configuration dictionary
    """
    cfg = get_default_cfg()
    cfg.update(
        {
            "model_name": "gemma-2-2b",
            "dataset_path": "HuggingFaceFW/fineweb-edu",
            "layer": layer,
            "num_tokens": num_tokens,  # 10k steps with batch_size 1024
            "model_batch_size": 4,
            "batch_size": 1024,
            "seq_len": 128,
            "lr": 4e-4,
            "model_dtype": torch.bfloat16,
            "dtype": torch.bfloat16,
            "device": device,
            "scheduler_type": "warmup_decay",
            "warmup_steps": 1024,
            "dict_size": 18432,
            "prefix_sizes": [2304, 4608, 9216, 13824, 18432],
            "top_k": 96,
            "aux_penalty": 1 / 64,
            "n_batches_to_dead": 20,
            "top_k_aux": 256,
            "save_activation_samples": True,
            "sample_collection_freq": 100,
            "max_samples_per_feature": 100,
            "sample_context_size": 20,
            "sample_activation_threshold": 0.1,
            "top_features_to_save": 100,
            "samples_per_feature_to_save": 10,
            "perf_log_freq": 25,
            "checkpoint_freq": 3000,
            "wandb_project": f"gemma-2-2b-layer{layer}-10k-steps",
            "experiment_description": f"10k-steps-layer{layer}",
        }
    )

    cfg["min_lr"] = cfg["lr"] * 0.01

    # Set activation sizes explicitly for Gemma-2-2b MLP layers
    cfg["source_act_size"] = 2304
    cfg["target_act_size"] = 2304

    cfg = create_transcoder_config(
        cfg,
        source_layer=layer,
        target_layer=layer,
        source_site="mlp_in",
        target_site="mlp_out",
    )

    return post_init_cfg(cfg)


def train_single_layer(layer: int, device: str, num_tokens: int = 10_240_000) -> Tuple[int, bool, str]:
    """
    Train a single layer on a specific device.
    
    Args:
        layer: Layer number to train
        device: GPU device string (e.g., "cuda:0", "cuda:1")
        num_tokens: Number of tokens for training
    
    Returns:
        Tuple of (layer, success, message)
    """
    try:
        cfg = build_config_for_layer(layer, device, num_tokens)
        expected_steps = int(cfg["num_tokens"] // cfg["batch_size"])
        run_dir_hint = f"results/{cfg['model_name']}/layer{cfg['layer']}/{expected_steps}"

        print(
            f"[Layer {layer} on {device}] Training Matryoshka transcoder: "
            f"model={cfg['model_name']} layer={cfg['layer']} steps≈{expected_steps}"
        )
        print(f"[Layer {layer} on {device}] Artifacts will be stored under {run_dir_hint}")

        model = HookedTransformer.from_pretrained_no_processing(
            cfg["model_name"], dtype=cfg["model_dtype"]
        )
        model = model.to(device)

        activation_store = TranscoderActivationsStore(model, cfg)
        transcoder = MatryoshkaTranscoder(cfg).to(device)

        train_transcoder(transcoder, activation_store, model, cfg)
        
        print(f"[Layer {layer} on {device}] Training complete. Inspect {run_dir_hint} for checkpoints, metrics, and samples.")
        return (layer, True, f"Training completed successfully for layer {layer}")
    
    except Exception as exc:
        error_msg = f"Training failed for layer {layer} on {device}: {exc}"
        print(f"[Layer {layer} on {device}] ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return (layer, False, error_msg)


def is_layer_already_trained(layer: int, model_name: str, num_tokens: int, batch_size: int, results_dir: str = "results") -> bool:
    """
    Check if a layer has already been trained by looking for a final checkpoint.
    
    Args:
        layer: Layer number to check
        model_name: Model name (e.g., "gemma-2-2b")
        num_tokens: Number of tokens used for training
        batch_size: Batch size used for training
        results_dir: Base results directory
    
    Returns:
        True if final checkpoint exists, False otherwise
    """
    expected_steps = int(num_tokens // batch_size)
    checkpoint_path = Path(results_dir) / model_name / f"layer{layer}" / str(expected_steps) / "checkpoints" / "final.pt"
    return checkpoint_path.exists()


def get_layers_to_train(
    total_layers: int = 26,
    skip_layers: list = [8, 12, 17],
    num_tokens: int = 10_240_000,
    batch_size: int = 1024,
    model_name: str = "gemma-2-2b",
    check_existing: bool = True,
) -> Tuple[list, list]:
    """
    Get list of layers to train, excluding skipped layers and already trained layers.
    
    Args:
        total_layers: Total number of layers in the model (default 26 for Gemma-2-2B)
        skip_layers: List of layer numbers to skip
        num_tokens: Number of tokens for training
        batch_size: Batch size for training
        model_name: Model name
        check_existing: If True, skip layers that already have final checkpoints
    
    Returns:
        Tuple of (layers_to_train, already_trained_layers)
    """
    all_layers = list(range(total_layers))
    layers_to_train = [layer for layer in all_layers if layer not in skip_layers]
    already_trained = []
    
    if check_existing:
        layers_to_check = layers_to_train.copy()
        layers_to_train = []
        for layer in layers_to_check:
            if is_layer_already_trained(layer, model_name, num_tokens, batch_size):
                already_trained.append(layer)
            else:
                layers_to_train.append(layer)
    
    return layers_to_train, already_trained


def worker_train_layer(args: Tuple[int, str, int]) -> Tuple[int, bool, str]:
    """
    Worker function for multiprocessing.
    Must be defined at module level for pickling with spawn method.
    
    Args:
        args: Tuple of (layer, device, num_tokens)
    
    Returns:
        Tuple of (layer, success, message)
    """
    layer, device, num_tokens = args
    return train_single_layer(layer, device, num_tokens)


def main():
    """Main function to train all layers using multiprocessing."""
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("WARNING: No CUDA GPUs found. Training will use CPU (very slow).")
        devices = ["cpu"]
        max_workers = 1
    elif num_gpus == 1:
        print("WARNING: Only 1 GPU found. Training will be sequential.")
        devices = ["cuda:0"]
        max_workers = 1
    else:
        print(f"Found {num_gpus} GPUs. Training will use 2 GPUs in parallel.")
        devices = ["cuda:0", "cuda:1"]
        max_workers = 2
    
    # Number of tokens for 10k steps (batch_size 1024)
    num_tokens = 10_240_000
    batch_size = 1024
    model_name = "gemma-2-2b"
    
    # Get layers to train (0-25, excluding 8, 12, 17, and already trained layers)
    layers_to_train, already_trained = get_layers_to_train(
        total_layers=26,
        skip_layers=[8, 12, 17],
        num_tokens=num_tokens,
        batch_size=batch_size,
        model_name=model_name,
        check_existing=True,
    )
    
    print(f"Total candidate layers: {26 - 3}")  # 26 layers minus 3 skipped
    if already_trained:
        print(f"Already trained layers (skipping): {sorted(already_trained)}")
    print(f"Layers to train: {layers_to_train}")
    print(f"Remaining layers to train: {len(layers_to_train)}")
    
    if not layers_to_train:
        print("\nAll layers have already been trained! Nothing to do.")
        return
    
    if max_workers == 1:
        # Sequential training on single GPU
        print(f"\nTraining {len(layers_to_train)} layers sequentially on {devices[0]}...")
        results = []
        for layer in layers_to_train:
            result = train_single_layer(layer, devices[0], num_tokens)
            results.append(result)
            layer_num, success, message = result
            if success:
                print(f"✓ Layer {layer_num}: {message}")
            else:
                print(f"✗ Layer {layer_num}: {message}")
        
        # Print summary for sequential training
        print("\n" + "="*60)
        print("Training Summary:")
        print("="*60)
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        if already_trained:
            print(f"\nSkipped (already trained): {len(already_trained)} layers")
            print(f"  Layers: {sorted(already_trained)}")
        
        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        for layer, success, message in successful:
            print(f"  ✓ Layer {layer}: {message}")
        
        if failed:
            print(f"\nFailed: {len(failed)}/{len(results)}")
            for layer, success, message in failed:
                print(f"  ✗ Layer {layer}: {message}")
        
        total_completed = len(already_trained) + len(successful)
        total_expected = 26 - 3  # 26 layers minus 3 skipped (8, 12, 17)
        print(f"\nTotal progress: {total_completed}/{total_expected} layers completed")
        print("="*60)
    else:
        # Parallel training on multiple GPUs
        print(f"\nTraining {len(layers_to_train)} layers in parallel on {max_workers} GPUs...")
        
        # Prepare arguments: (layer, device, num_tokens)
        # Round-robin device assignment
        layer_args = [
            (layer, devices[i % len(devices)], num_tokens)
            for i, layer in enumerate(layers_to_train)
        ]
        
        # Print device assignment
        print("\nDevice assignment:")
        for layer, device, _ in layer_args:
            print(f"  Layer {layer:2d} -> {device}")
        print()
        
        # Use multiprocessing to train layers in parallel
        with mp.Pool(processes=max_workers) as pool:
            results = pool.map(worker_train_layer, layer_args)
        
        # Print summary
        print("\n" + "="*60)
        print("Training Summary:")
        print("="*60)
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        if already_trained:
            print(f"\nSkipped (already trained): {len(already_trained)} layers")
            print(f"  Layers: {sorted(already_trained)}")
        
        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        for layer, success, message in successful:
            print(f"  ✓ Layer {layer}: {message}")
        
        if failed:
            print(f"\nFailed: {len(failed)}/{len(results)}")
            for layer, success, message in failed:
                print(f"  ✗ Layer {layer}: {message}")
        
        total_completed = len(already_trained) + len(successful)
        total_expected = 26 - 3  # 26 layers minus 3 skipped (8, 12, 17)
        print(f"\nTotal progress: {total_completed}/{total_expected} layers completed")
        print("="*60)


if __name__ == "__main__":
    # Set multiprocessing start method (required for CUDA)
    # Only set if not already set
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Start method already set, which is fine
        pass
    main()

