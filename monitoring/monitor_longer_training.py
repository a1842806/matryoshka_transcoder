#!/usr/bin/env python3
"""
Monitor the progress of the longer training with enhanced activation sample collection.
"""

import os
import time
import json
from pathlib import Path

def check_training_progress():
    """Check the current training progress."""
    
    print("=" * 80)
    print("🔍 MONITORING LONGER TRAINING PROGRESS")
    print("=" * 80)
    
    # Check for recent checkpoint directories
    checkpoint_dir = Path("checkpoints/transcoder/gemma-2-2b/")
    if checkpoint_dir.exists():
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and "matryoshka-transcoder" in d.name]
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if checkpoints:
            latest_checkpoint = checkpoints[0]
            print(f"📁 Latest checkpoint: {latest_checkpoint.name}")
            print(f"🕒 Last modified: {time.ctime(latest_checkpoint.stat().st_mtime)}")
            
            # Check if activation samples exist
            activation_samples_dir = latest_checkpoint / "activation_samples"
            if activation_samples_dir.exists():
                sample_files = list(activation_samples_dir.glob("feature_*_samples.json"))
                print(f"📊 Activation samples: {len(sample_files)} feature files")
                
                # Check collection summary
                summary_file = activation_samples_dir / "collection_summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                    print(f"📈 Total samples collected: {summary.get('total_samples_stored', 0):,}")
                    print(f"🎯 Features with samples: {len(summary.get('features_with_samples', []))}")
                    
                    # Show top features
                    top_features = summary.get('top_features_by_activation', [])[:5]
                    if top_features:
                        print(f"\n🏆 Top 5 Features by Activation:")
                        for i, feat in enumerate(top_features, 1):
                            print(f"  {i}. Feature {feat['feature_idx']}: max={feat['max_activation']:.3f}, samples={feat['num_samples']}")
            else:
                print("⏳ Activation samples not yet collected...")
        else:
            print("⏳ No checkpoints found yet...")
    else:
        print("⏳ Checkpoint directory not found...")
    
    # Check for W&B logs
    wandb_dir = Path("wandb/")
    if wandb_dir.exists():
        runs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
        if runs:
            latest_run = max(runs, key=lambda x: x.stat().st_mtime)
            print(f"\n📊 Latest W&B run: {latest_run.name}")
            print(f"🕒 Started: {time.ctime(latest_run.stat().st_mtime)}")
    
    print("=" * 80)

def estimate_progress():
    """Estimate training progress based on expected steps."""
    
    # Configuration from the training script
    num_tokens = 1_000_000  # 1M tokens
    batch_size = 1024
    seq_len = 64
    tokens_per_step = batch_size * seq_len  # 65,536 tokens per step
    
    expected_steps = num_tokens // tokens_per_step  # ~15 steps
    print(f"📊 Expected training configuration:")
    print(f"   - Total tokens: {num_tokens:,}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Tokens per step: {tokens_per_step:,}")
    print(f"   - Expected steps: ~{expected_steps}")
    print(f"   - Sample collection: every 100 steps")
    print(f"   - Checkpoint saving: every 500 steps")

def main():
    """Main monitoring function."""
    
    estimate_progress()
    print()
    check_training_progress()
    
    print(f"\n💡 Tips:")
    print(f"   - Training should complete in ~{1_000_000 // (1024 * 64)} steps")
    print(f"   - Activation samples will be collected every 100 steps")
    print(f"   - Checkpoints will be saved every 500 steps")
    print(f"   - Run this script periodically to monitor progress")

if __name__ == "__main__":
    main()


