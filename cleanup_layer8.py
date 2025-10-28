"""Clean up layer 8 training results into organized structure."""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.clean_results import CleanResultsManager

def cleanup_layer8_results():
    """Clean up layer 8 results into organized structure."""
    print("🧹 CLEANING UP LAYER 8 TRAINING RESULTS")
    print("=" * 60)

    manager = CleanResultsManager()
    
    # Find all layer 8 checkpoints
    checkpoint_base = Path("checkpoints/sae/gemma-2-2b")
    layer8_checkpoints = []
    
    for checkpoint_dir in checkpoint_base.iterdir():
        if checkpoint_dir.is_dir() and "blocks.8" in checkpoint_dir.name:
            layer8_checkpoints.append(checkpoint_dir)
    
    if not layer8_checkpoints:
        print("No layer 8 checkpoints found.")
        return
    
    # Sort by step number
    layer8_checkpoints.sort(key=lambda x: int(x.name.split('_')[-1]))
    
    print(f"Found {len(layer8_checkpoints)} layer 8 checkpoints:")
    for cp in layer8_checkpoints:
        step = cp.name.split('_')[-1]
        print(f"  - Step {step}")
    
    # Find the final checkpoint (highest step)
    final_checkpoint = layer8_checkpoints[-1]
    final_step = int(final_checkpoint.name.split('_')[-1])
    
    print(f"\n📊 Final checkpoint: Step {final_step}")
    
    # Create organized experiment directory
    exp_dir = manager.create_experiment_dir(
        model_name="gemma-2-2b",
        layer=8,
        steps=final_step,
        description=f"{final_step}k-steps-layer8"
    )
    
    print(f"📁 Created experiment directory: {exp_dir}")
    
    # Move final checkpoint
    final_checkpoint_path = final_checkpoint / "sae.pt"
    if final_checkpoint_path.exists():
        shutil.move(str(final_checkpoint_path), str(exp_dir / "checkpoint.pt"))
        print(f"✅ Moved final checkpoint: {final_checkpoint_path} → {exp_dir / 'checkpoint.pt'}")
    
    # Move config
    config_path = final_checkpoint / "config.json"
    if config_path.exists():
        shutil.move(str(config_path), str(exp_dir / "config.json"))
        print(f"✅ Moved config: {config_path} → {exp_dir / 'config.json'}")
    
    # Create training log with step information
    log_data = {
        "step": final_step,
        "total_checkpoints": len(layer8_checkpoints),
        "checkpoint_steps": [int(cp.name.split('_')[-1]) for cp in layer8_checkpoints],
        "timestamp": datetime.now().isoformat(),
        "note": "Migrated from old checkpoint structure"
    }
    
    log_path = exp_dir / "training_log.json"
    import json
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"✅ Created training log: {log_path}")
    
    # Move activation samples if they exist
    samples_source = Path("analysis_results/gemma-2-2b_blocks.8.hook_resid_pre_4608_matryoshka-transcoder_8_0.0003_activation_samples")
    if samples_source.exists():
        samples_dest = exp_dir / "activation_samples"
        shutil.move(str(samples_source), str(samples_dest))
        print(f"✅ Moved activation samples: {samples_source} → {samples_dest}")
    
    # Clean up old checkpoint directories
    print(f"\n🗑️  Cleaning up old checkpoint directories...")
    removed_count = 0
    for checkpoint_dir in layer8_checkpoints:
        try:
            shutil.rmtree(checkpoint_dir)
            removed_count += 1
            print(f"  ✅ Removed: {checkpoint_dir.name}")
        except Exception as e:
            print(f"  ⚠️  Could not remove {checkpoint_dir.name}: {e}")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print("=" * 60)
    print(f"✅ Organized layer 8 results:")
    print(f"   📁 Experiment: {exp_dir}")
    print(f"   🔢 Final step: {final_step}")
    print(f"   📊 Checkpoints processed: {len(layer8_checkpoints)}")
    print(f"   🗑️  Old directories removed: {removed_count}")
    print("=" * 60)
    print("You can now view results with:")
    print("  python src/utils/view_activation_samples.py --browse")

if __name__ == "__main__":
    cleanup_layer8_results()
