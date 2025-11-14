import torch
import tqdm
import math

from utils.logs import init_wandb, log_wandb, log_model_performance
from utils.activation_samples import ActivationSampleCollector
from utils.clean_results import CleanResultsManager
import wandb
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, LinearLR


def linear_warmup(step, warmup_steps, base_lr):
    return min(1.0, step / warmup_steps)


def warmup_cosine_decay(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


def create_scheduler(optimizer, cfg, num_batches):
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
        return None

def train_transcoder(transcoder, activation_store, model, cfg):
    num_batches = int(cfg["num_tokens"] // cfg["batch_size"])
    optimizer = torch.optim.Adam(transcoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = create_scheduler(optimizer, cfg, num_batches)
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)
    
    results_manager = CleanResultsManager()
    experiment_dir = results_manager.create_experiment_dir(
        model_name=cfg["model_name"],
        layer=cfg["layer"],
        steps=num_batches,
        description=cfg.get("experiment_description", "")
    )
    
    sample_collector = None
    if cfg.get("save_activation_samples", False):
        sample_collector = ActivationSampleCollector(
            max_samples_per_feature=cfg.get("max_samples_per_feature", 100),
            context_size=cfg.get("sample_context_size", 20),
            storage_threshold=cfg.get("sample_activation_threshold", 0.1)
        )
        print(f"Activation sample collection enabled (collecting every {cfg.get('sample_collection_freq', 1000)} steps)")
    
    for i in pbar:
        source_batch, target_batch = activation_store.next_batch()
        transcoder_output = transcoder(source_batch, target_batch)
        
        log_wandb(transcoder_output, i, wandb_run)
        
        n_to_dead = cfg.get("n_batches_to_dead", 20)
        dead_mask_log = transcoder.num_batches_not_active >= n_to_dead
        dead_count_log = int(dead_mask_log.sum().item())
        total_features_log = int(transcoder.num_batches_not_active.numel())
        percent_dead_log = (dead_count_log / max(1, total_features_log)) * 100.0
        
        wandb_run.log({
            "dead_features_count": dead_count_log,
            "dead_features_percentage": percent_dead_log,
            "dead_features_alive": total_features_log - dead_count_log,
        }, step=i)
        
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            wandb_run.log({"learning_rate": current_lr}, step=i)
        
        if i % cfg["perf_log_freq"] == 0:
            log_transcoder_performance(wandb_run, i, model, activation_store, transcoder, cfg)

        if i % cfg["checkpoint_freq"] == 0:
            intermediate_metrics = {
                "loss": float(transcoder_output['loss'].detach().cpu()),
                "l2_loss": float(transcoder_output['l2_loss'].detach().cpu()),
                "l0_norm": float(transcoder_output['l0_norm'].detach().cpu()),
                "fvu": float(transcoder_output['fvu'].detach().cpu()),
                "dead_features": float(transcoder.num_batches_not_active.sum().item())
            }
            results_manager.save_checkpoint(
                experiment_dir=experiment_dir,
                model=transcoder,
                config=cfg,
                step=i,
                metrics=intermediate_metrics,
                is_final=False,
            )
        
        if sample_collector and i % cfg.get("sample_collection_freq", 1000) == 0:
            source_batch_with_tokens, batch_tokens = activation_store.get_batch_with_tokens()
            with torch.no_grad():
                sparse_features = transcoder.encode(source_batch_with_tokens)
            sample_collector.collect_batch_samples(
                sparse_features,
                batch_tokens,
                activation_store.tokenizer
            )
        
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
        
        transcoder_output["loss"].backward(retain_graph=True)
        transcoder.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
    
    final_metrics = {
        "loss": float(transcoder_output['loss'].detach().cpu()),
        "l2_loss": float(transcoder_output['l2_loss'].detach().cpu()),
        "l0_norm": float(transcoder_output['l0_norm'].detach().cpu()),
        "fvu": float(transcoder_output['fvu'].detach().cpu()),
        "dead_features": float(transcoder.num_batches_not_active.sum().item())
    }
    
    final_config = dict(cfg)
    final_config["completed_steps"] = num_batches

    results_manager.save_checkpoint(
        experiment_dir=experiment_dir,
        model=transcoder,
        config=final_config,
        step=num_batches,
        metrics=final_metrics,
        is_final=True,
    )
    
    if sample_collector:
        results_manager.save_activation_samples(
            experiment_dir=experiment_dir,
            sample_collector=sample_collector,
            top_k_features=cfg.get("top_features_to_save", 100),
            samples_per_feature=cfg.get("samples_per_feature_to_save", 10)
        )
    
    wandb_run.finish()


def log_transcoder_performance(wandb_run, step, model, activation_store, transcoder, cfg):
    source_batch, target_batch = activation_store.next_batch()
    
    with torch.no_grad():
        transcoder_output = transcoder(source_batch, target_batch)
        reconstruction = transcoder_output["transcoder_out"]
        
        mse = (reconstruction - target_batch).pow(2).mean()
        mae = (reconstruction - target_batch).abs().mean()
        cos_sim = torch.nn.functional.cosine_similarity(
            reconstruction.flatten(), 
            target_batch.flatten(), 
            dim=0
        )
        
        metrics_to_log = {
            "performance/mse": mse.item(),
            "performance/mae": mae.item(),
            "performance/cosine_similarity": cos_sim.item(),
            "performance/fvu": transcoder_output["fvu"].item(),
        }
        
        wandb_run.log(metrics_to_log, step=step)


    