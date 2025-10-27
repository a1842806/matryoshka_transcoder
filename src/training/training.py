import torch
import tqdm
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
from utils.activation_samples import ActivationSampleCollector
# Anti-duplication imports removed - reverted to clean state
import multiprocessing as mp
from queue import Empty
import wandb
import json
import os
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, LinearLR


def linear_warmup(step, warmup_steps, base_lr):
    """Linear warmup function for learning rate scheduling."""
    return min(1.0, step / warmup_steps)


def warmup_cosine_decay(step, warmup_steps, total_steps, base_lr):
    """Combined warmup and cosine decay function.""" 
    if step < warmup_steps:
        # Warmup phase: linear increase
        return step / warmup_steps
    else:
        # Decay phase: cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


def create_scheduler(optimizer, cfg, num_batches):
    """Create learning rate scheduler based on configuration."""
    scheduler_type = cfg.get("scheduler_type", "none")
    
    if scheduler_type == "warmup_only":
        warmup_steps = cfg.get("warmup_steps", 1000)
        return LambdaLR(optimizer, lr_lambda=lambda step: linear_warmup(step, warmup_steps, cfg["lr"]))
    
    elif scheduler_type == "decay_only":
        min_lr = cfg.get("min_lr", cfg["lr"] * 0.01)
        return CosineAnnealingLR(optimizer, T_max=num_batches, eta_min=min_lr)
    
    elif scheduler_type == "warmup_decay":
        warmup_steps = cfg.get("warmup_steps", 1000)
        min_lr = cfg.get("min_lr", cfg["lr"] * 0.01)
        return LambdaLR(optimizer, lr_lambda=lambda step: warmup_cosine_decay(step, warmup_steps, num_batches, cfg["lr"]))
    
    else:
        # No scheduler
        return None


def save_checkpoint_mp(sae, cfg, step):
    """
    Save checkpoint without requiring a wandb run object.
    Creates an artifact but doesn't log it to wandb directly.
    Uses organized structure: checkpoints/{model_type}/{base_model}/{name}_{step}/
    """
    # Determine model type (sae or transcoder)
    sae_type = cfg.get("sae_type", "topk")
    if "transcoder" in sae_type or "clt" in sae_type:
        model_type = "transcoder"
    else:
        model_type = "sae"
    
    # Get base model name (e.g., "gpt2-small", "gemma-2-2b")
    base_model = cfg["model_name"]
    
    # Create organized directory structure
    save_dir = f"checkpoints/{model_type}/{base_model}/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    print(f"Model and config saved at step {step} in {save_dir}")
    return save_dir, sae_path, config_path


def train_transcoder(transcoder, activation_store, model, cfg):
    """
    Train a single transcoder (cross-layer feature transformer).
    
    Args:
        transcoder: MatryoshkaTranscoder or similar model
        activation_store: TranscoderActivationsStore providing paired activations
        model: The language model
        cfg: Configuration dictionary
    """
    num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    
    # Create learning rate scheduler
    scheduler = create_scheduler(optimizer, cfg, num_batches)
    
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)
    
    # Initialize activation sample collector if enabled
    sample_collector = None
    if cfg.get("save_activation_samples", False):
        sample_collector = ActivationSampleCollector(
            max_samples_per_feature=cfg.get("max_samples_per_feature", 100),
            context_size=cfg.get("sample_context_size", 20),
            storage_threshold=cfg.get("sample_activation_threshold", 0.1)
        )
        print(f"Activation sample collection enabled (collecting every {cfg.get('sample_collection_freq', 1000)} steps)")
    
    for i in pbar:
        # Get paired activations (source, target)
        source_batch, target_batch = activation_store.next_batch()
        
        # Forward pass - transcoder learns source -> target mapping
        transcoder_output = transcoder(source_batch, target_batch)
        
        # Log metrics
        log_wandb(transcoder_output, i, wandb_run)
        
        # Log dead-feature metrics (count and percentage) with proper visualization
        n_to_dead = cfg.get("n_batches_to_dead", 20)
        dead_mask_log = transcoder.num_batches_not_active >= n_to_dead
        dead_count_log = int(dead_mask_log.sum().item())
        total_features_log = int(transcoder.num_batches_not_active.numel())
        percent_dead_log = (dead_count_log / max(1, total_features_log)) * 100.0
        
        # Log both count and percentage for comprehensive tracking
        # This creates separate plots in W&B: one for absolute counts, one for percentages
        wandb_run.log({
            "dead_features_count": dead_count_log,        # Total number of dead features
            "dead_features_percentage": percent_dead_log,  # Percentage of dead features
            "dead_features_alive": total_features_log - dead_count_log,  # Active features
        }, step=i)
        
        # Log learning rate if scheduler is used
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            wandb_run.log({"learning_rate": current_lr}, step=i)
        
        if i % cfg["perf_log_freq"] == 0:
            log_transcoder_performance(wandb_run, i, model, activation_store, transcoder)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, transcoder, cfg, i)
        
        # Collect activation samples if enabled
        if sample_collector and i % cfg.get("sample_collection_freq", 1000) == 0:
            # Get a batch with tokens for sample collection
            source_batch_with_tokens, batch_tokens = activation_store.get_batch_with_tokens()
            
            # Encode to get sparse features
            with torch.no_grad():
                sparse_features = transcoder.encode(source_batch_with_tokens)
            
            # Collect samples (using source batch tokens as context)
            sample_collector.collect_batch_samples(
                sparse_features,
                batch_tokens,
                activation_store.tokenizer
            )
        
        # Update progress bar with key metrics
        # Convert scalars to Python floats to avoid autograd references in formatting
        loss_val = float(transcoder_output['loss'].detach().cpu())
        l0_val = float(transcoder_output['l0_norm'].detach().cpu())
        l2_val = float(transcoder_output['l2_loss'].detach().cpu())
        fvu_val = float(transcoder_output['fvu'].detach().cpu())
        dead_val = float(transcoder.num_batches_not_active.sum().item())
        pbar.set_description(
            f"Training: {i/num_batches*100:.0f}%, "
            f"Loss: {loss_val:.3f}, "
            f"L0: {l0_val:.0f}, "
            f"L2: {l2_val:.3f}, "
            f"FVU: {fvu_val:.3f}, "
            f"Dead: {dead_val:.0f}"
        )
        
        # Backward pass
        # Backward pass; retain graph to avoid edge cases with reused tensors
        transcoder_output["loss"].backward(retain_graph=True)
        
        # Update decoder weights to maintain unit norm
        transcoder.make_decoder_weights_and_grad_unit_norm()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Save final checkpoint
    save_checkpoint(wandb_run, transcoder, cfg, num_batches)
    
    # Save activation samples if collected
    if sample_collector:
        sample_dir = f"checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}_{num_batches}_activation_samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save samples
        sample_collector.save_samples(
            sample_dir, 
            top_k_features=cfg.get("top_features_to_save", 100),
            samples_per_feature=cfg.get("samples_per_feature_to_save", 10)
        )
        print(f"\nSaved activation samples to {sample_dir}")
    
    wandb_run.finish()


def log_transcoder_performance(wandb_run, step, model, activation_store, transcoder):
    """
    Log transcoder performance metrics (reconstruction quality, etc.).
    """
    # Get a batch for evaluation
    source_batch, target_batch = activation_store.next_batch()
    
    with torch.no_grad():
        # Get transcoder reconstruction
        transcoder_output = transcoder(source_batch, target_batch)
        reconstruction = transcoder_output["transcoder_out"]
        
        # Compute reconstruction quality metrics
        mse = (reconstruction - target_batch).pow(2).mean()
        mae = (reconstruction - target_batch).abs().mean()
        
        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            reconstruction.flatten(), 
            target_batch.flatten(), 
            dim=0
        )
        
        # Log metrics
        wandb_run.log({
            "performance/mse": mse.item(),
            "performance/mae": mae.item(),
            "performance/cosine_similarity": cos_sim.item(),
            "performance/fvu": transcoder_output["fvu"].item(),
        }, step=step)

def train_transcoder_group_seperate_wandb(transcoders, activation_stores, model, cfgs):
    def new_wandb_process(config, log_queue, entity, project):
        run = wandb.init(
            entity=entity, 
            project=project, 
            config=config, 
            name=config["name"]
        )
        
        # Train the transcoder
        train_transcoder(
            transcoders[config["index"]], 
            activation_stores[config["index"]], 
            model, 
            config
        )
        
        # Send completion signal
        log_queue.put(("complete", config["index"]))
        run.finish()

    # Create processes for each transcoder
    processes = []
    log_queue = mp.Queue()
    
    for i, (transcoder, activation_store, cfg) in enumerate(zip(transcoders, activation_stores, cfgs)):
        cfg["index"] = i
        process = mp.Process(
            target=new_wandb_process, 
            args=(cfg, log_queue, cfg.get("entity", None), cfg["wandb_project"])
        )
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    completed = 0
    while completed < len(processes):
        try:
            message, index = log_queue.get(timeout=1)
            if message == "complete":
                completed += 1
                print(f"Transcoder {index} training completed")
        except Empty:
            continue
    
    # Clean up processes
    for process in processes:
        process.join()
    
    print("All transcoder training completed!")