import torch
import tqdm
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
from utils.activation_samples import ActivationSampleCollector
from utils.diversity_regularization import create_diversity_regularizer
from utils.position_stratified_sampling import create_position_sampler
from utils.feature_correlation_monitor import create_correlation_monitor
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


def train_sae(sae, activation_store, model, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
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
        batch = activation_store.next_batch()
        sae_output = sae(batch)
        log_wandb(sae_output, i, wandb_run)
        
        if i % cfg["perf_log_freq"]  == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)
        
        # Collect activation samples periodically
        if sample_collector is not None and i % cfg.get("sample_collection_freq", 1000) == 0:
            with torch.no_grad():
                sample_batch, sample_tokens = activation_store.get_batch_with_tokens()
                sample_acts = sae.encode(sample_batch)
                sample_collector.collect_batch_samples(
                    sample_acts,
                    sample_tokens,
                    model.tokenizer
                )

        loss = sae_output["loss"]
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "L0": f"{sae_output['l0_norm']:.4f}", 
            "L2": f"{sae_output['l2_loss']:.4f}", 
            "FVU": f"{sae_output['fvu']:.4f}"
        })
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(wandb_run, sae, cfg, i)
    
    # Save collected samples at end of training
    if sample_collector is not None:
        sample_dir = f"checkpoints/{cfg.get('model_type', 'sae')}/{cfg['model_name']}/{cfg['name']}_activation_samples"
        sample_collector.save_samples(
            sample_dir,
            top_k_features=cfg.get("top_features_to_save", 100),
            samples_per_feature=cfg.get("samples_per_feature_to_save", 10)
        )
        print(f"\nSaved activation samples to {sample_dir}")
    

def train_sae_group(saes, activation_store, model, cfgs):
    num_batches = cfgs[0]["num_tokens"] // cfgs[0]["batch_size"]
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfgs[0])

    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        batch = activation_store.next_batch()
        counter = 0
        for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_wandb(sae_output, i, wandb_run, index=counter)
            if i % cfg["perf_log_freq"]  == 0:
                log_model_performance(wandb_run, i, model, activation_store, sae, index=counter, batch_tokens=batch_tokens)

            if i % cfg["checkpoint_freq"] == 0:
                save_checkpoint(wandb_run, sae, cfg, i)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "L0": f"{sae_output['l0_norm']:.4f}", 
                "L2": f"{sae_output['l2_loss']:.4f}", 
                "FVU": f"{sae_output['fvu']:.4f}"
            })
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
   
    for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
        save_checkpoint(wandb_run, sae, cfg, i)


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

def train_sae_group_seperate_wandb(saes, activation_store, model, cfgs):
    def new_wandb_process(config, log_queue, entity, project):
        run = wandb.init(
            entity=entity, 
            project=project, 
            config=config, 
            name=config["name"]
        )
        
        while True:
            try:
                # Wait up to 1 second for new data
                log = log_queue.get(timeout=1)
                
                # Check for termination signal
                if log == "DONE":
                    break
                
                # Check if this is a checkpoint signal
                if isinstance(log, dict) and log.get("checkpoint"):
                    # Create and log artifact
                    artifact = wandb.Artifact(
                        name=f"{config['name']}_{log['step']}",
                        type="model",
                        description=f"Model checkpoint at step {log['step']}",
                    )
                    save_dir = log["save_dir"]
                    artifact.add_file(os.path.join(save_dir, "sae.pt"))
                    artifact.add_file(os.path.join(save_dir, "config.json"))
                    run.log_artifact(artifact)
                else:
                    # Log regular metrics
                    wandb.log(log)
            except Empty:
                continue
                
        wandb.finish()
    
    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    print(f"Number of batches: {num_batches}")
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) 
                 for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    # Initialize wandb processes and queues
    wandb_processes = []
    log_queues = []
    
    for i, cfg in enumerate(cfgs):
        log_queue = mp.Queue()
        log_queues.append(log_queue)
        wandb_config = cfg
        wandb_process = mp.Process(
            target=new_wandb_process,
            args=(wandb_config, log_queue, cfg.get("wandb_entity", ""), cfg["wandb_project"]),
        )
        wandb_process.start()
        wandb_processes.append(wandb_process)


    for i in pbar:
        batch = activation_store.next_batch()
        
        for idx, (sae, cfg, optimizer) in enumerate(zip(saes, cfgs, optimizers)):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            
            # Log metrics to appropriate wandb process
            log_dict = {
                k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
                for k, v in sae_output.items() if isinstance(v, (int, float)) or 
                (isinstance(v, torch.Tensor) and v.dim() == 0)
            }
            log_queues[idx].put(log_dict)

            if i % cfg["checkpoint_freq"] == 0:
                # Save checkpoint and send artifact info to wandb process
                save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
                log_queues[idx].put({
                    "checkpoint": True,
                    "step": i,
                    "save_dir": save_dir
                })

            pbar.set_postfix({
                f"Loss_{idx}": f"{loss.item():.4f}", 
                f"L0_{idx}": f"{sae_output['l0_norm']:.4f}",
                f"L2_{idx}": f"{sae_output['l2_loss']:.4f}", 
                f"FVU_{idx}": f"{sae_output['fvu']:.4f}",
            })
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

    # Final checkpoints
    for idx, (sae, cfg) in enumerate(zip(saes, cfgs)):
        save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
        log_queues[idx].put({
            "checkpoint": True,
            "step": i,
            "save_dir": save_dir
        })

    # Clean up wandb processes
    for queue in log_queues:
        queue.put("DONE")
    for process in wandb_processes:
        process.join()


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
        
        # Log learning rate if scheduler is used
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            wandb_run.log({"learning_rate": current_lr}, step=i)
        
        if i % cfg["perf_log_freq"] == 0:
            log_transcoder_performance(wandb_run, i, model, activation_store, transcoder)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, transcoder, cfg, i)
        
        # Collect activation samples periodically for transcoders
        if sample_collector is not None and i % cfg.get("sample_collection_freq", 1000) == 0:
            with torch.no_grad():
                # Get batch with tokens from activation store
                sample_batch, sample_tokens = activation_store.get_batch_with_tokens()
                sample_acts = transcoder.encode(sample_batch)
                sample_collector.collect_batch_samples(
                    sample_acts,
                    sample_tokens,
                    model.tokenizer
                )

        loss = transcoder_output["loss"]
        
        # Log current learning rate if scheduler is used
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "L0": f"{transcoder_output['l0_norm']:.4f}", 
            "L2": f"{transcoder_output['l2_loss']:.4f}", 
            "FVU": f"{transcoder_output['fvu']:.4f}",
            "Dead": f"{transcoder_output['num_dead_features']}",
            "LR": f"{current_lr:.2e}"
        })
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transcoder.parameters(), cfg["max_grad_norm"])
        transcoder.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        
        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
            
        optimizer.zero_grad()

    save_checkpoint(wandb_run, transcoder, cfg, i)
    
    # Save collected samples at end of training
    if sample_collector is not None:
        sample_dir = f"checkpoints/transcoder/{cfg['model_name']}/{cfg['name']}_activation_samples"
        sample_collector.save_samples(
            sample_dir,
            top_k_features=cfg.get("top_features_to_save", 100),
            samples_per_feature=cfg.get("samples_per_feature_to_save", 10)
        )
        print(f"\nSaved activation samples to {sample_dir}")


def log_transcoder_performance(wandb_run, step, model, activation_store, transcoder):
    """
    Log transcoder performance metrics.
    
    Evaluates how well the transcoder transforms features from source to target layer.
    """
    with torch.no_grad():
        # Get a batch of paired activations
        source_batch, target_batch = activation_store.next_batch()
        source_batch = source_batch[:transcoder.config["batch_size"] // 4]
        target_batch = target_batch[:transcoder.config["batch_size"] // 4]
        
        # Get transcoder reconstruction
        transcoder_output = transcoder(source_batch, target_batch)
        target_reconstruct = transcoder_output["sae_out"]
        
        # Compute reconstruction quality
        l2_error = (target_reconstruct.float() - target_batch.float()).pow(2).mean().item()
        relative_error = l2_error / target_batch.float().pow(2).mean().item()
        
        # Sparsity metrics
        l0 = transcoder_output["l0_norm"]
        
        log_dict = {
            "performance/transcoder_l2_error": l2_error,
            "performance/transcoder_relative_error": relative_error,
            "performance/transcoder_sparsity": l0,
        }
        
        wandb_run.log(log_dict, step=step)


def train_transcoder_group_seperate_wandb(transcoders, activation_stores, model, cfgs):
    """
    Train multiple transcoders with separate W&B runs using multiprocessing.
    
    Args:
        transcoders: List of transcoder models
        activation_stores: List of TranscoderActivationsStore (one per transcoder)
        model: The language model
        cfgs: List of configuration dictionaries (one per transcoder)
    """
    def new_wandb_process(config, log_queue, entity, project):
        run = wandb.init(
            entity=entity, 
            project=project, 
            config=config, 
            name=config["name"]
        )
        
        while True:
            try:
                log = log_queue.get(timeout=1)
                
                if log == "DONE":
                    break
                
                if isinstance(log, dict) and log.get("checkpoint"):
                    artifact = wandb.Artifact(
                        name=f"{config['name']}_{log['step']}",
                        type="model",
                        description=f"Model checkpoint at step {log['step']}",
                    )
                    save_dir = log["save_dir"]
                    artifact.add_file(os.path.join(save_dir, "sae.pt"))
                    artifact.add_file(os.path.join(save_dir, "config.json"))
                    run.log_artifact(artifact)
                else:
                    wandb.log(log)
            except Empty:
                continue
                
        wandb.finish()
    
    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    print(f"Number of batches: {num_batches}")
    
    optimizers = [
        torch.optim.Adam(transcoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) 
        for transcoder, cfg in zip(transcoders, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    # Initialize wandb processes and queues
    wandb_processes = []
    log_queues = []
    
    for i, cfg in enumerate(cfgs):
        log_queue = mp.Queue()
        log_queues.append(log_queue)
        wandb_config = cfg
        wandb_process = mp.Process(
            target=new_wandb_process,
            args=(wandb_config, log_queue, cfg.get("wandb_entity", ""), cfg["wandb_project"]),
        )
        wandb_process.start()
        wandb_processes.append(wandb_process)

    for i in pbar:
        for idx, (transcoder, activation_store, cfg, optimizer) in enumerate(
            zip(transcoders, activation_stores, cfgs, optimizers)
        ):
            # Get paired activations
            source_batch, target_batch = activation_store.next_batch()
            
            # Forward pass
            transcoder_output = transcoder(source_batch, target_batch)
            loss = transcoder_output["loss"]
            
            # Log metrics to appropriate wandb process
            log_dict = {
                k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
                for k, v in transcoder_output.items() if isinstance(v, (int, float)) or 
                (isinstance(v, torch.Tensor) and v.dim() == 0)
            }
            log_queues[idx].put(log_dict)

            if i % cfg["checkpoint_freq"] == 0:
                save_dir, _, _ = save_checkpoint_mp(transcoder, cfg, i)
                log_queues[idx].put({
                    "checkpoint": True,
                    "step": i,
                    "save_dir": save_dir
                })

            pbar.set_postfix({
                f"Loss_{idx}": f"{loss.item():.4f}", 
                f"L0_{idx}": f"{transcoder_output['l0_norm']:.4f}",
                f"L2_{idx}": f"{transcoder_output['l2_loss']:.4f}",
                f"FVU_{idx}": f"{transcoder_output['fvu']:.4f}",
            })
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transcoder.parameters(), cfg["max_grad_norm"])
            transcoder.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()

    # Final checkpoints
    for idx, (transcoder, cfg) in enumerate(zip(transcoders, cfgs)):
        save_dir, _, _ = save_checkpoint_mp(transcoder, cfg, i)
        log_queues[idx].put({
            "checkpoint": True,
            "step": i,
            "save_dir": save_dir
        })

    # Clean up wandb processes
    for queue in log_queues:
        queue.put("DONE")
    for process in wandb_processes:
        process.join()