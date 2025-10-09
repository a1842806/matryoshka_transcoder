#!/usr/bin/env python3
"""
Monitor the anti-duplication training progress.

This script monitors the training progress and shows the effectiveness
of the anti-duplication features in real-time.
"""

import os
import time
import json
import glob
from pathlib import Path

def monitor_training_progress():
    """Monitor the training progress and anti-duplication features."""
    
    print("🔍 Monitoring Anti-Duplication Training Progress...")
    print("=" * 80)
    
    # Track various metrics
    last_checkpoint_time = 0
    last_log_time = 0
    correlation_reports = 0
    activation_samples = 0
    
    while True:
        current_time = time.time()
        
        # Check for new checkpoints
        checkpoint_dirs = glob.glob("checkpoints/transcoder/gemma-2-2b/*")
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getctime)
            checkpoint_time = os.path.getctime(latest_checkpoint)
            if checkpoint_time > last_checkpoint_time:
                print(f"✅ New checkpoint created: {os.path.basename(latest_checkpoint)}")
                last_checkpoint_time = checkpoint_time
        
        # Check for correlation logs
        if os.path.exists("anti_duplication_logs"):
            log_files = glob.glob("anti_duplication_logs/*.json")
            if len(log_files) > correlation_reports:
                new_reports = len(log_files) - correlation_reports
                print(f"📊 New correlation reports: {new_reports} files")
                correlation_reports = len(log_files)
                
                # Show latest correlation summary
                if log_files:
                    latest_log = max(log_files, key=os.path.getctime)
                    try:
                        with open(latest_log, 'r') as f:
                            data = json.load(f)
                            print(f"   Latest report: Step {data.get('step', 'unknown')}")
                            if 'redundant_features' in data:
                                redundant_count = len(data['redundant_features'])
                                print(f"   Redundant features detected: {redundant_count}")
                    except Exception as e:
                        print(f"   Error reading log: {e}")
        
        # Check for activation samples
        if os.path.exists("analysis_results"):
            sample_dirs = [d for d in os.listdir("analysis_results") if "activation_samples" in d]
            if len(sample_dirs) > activation_samples:
                new_samples = len(sample_dirs) - activation_samples
                print(f"📁 New activation sample directories: {new_samples}")
                activation_samples = len(sample_dirs)
                
                # Show latest sample directory
                if sample_dirs:
                    latest_samples = max(sample_dirs, key=lambda x: os.path.getctime(os.path.join("analysis_results", x)))
                    print(f"   Latest samples: {latest_samples}")
        
        # Check wandb logs
        wandb_dirs = glob.glob("wandb/run-*")
        if wandb_dirs:
            latest_wandb = max(wandb_dirs, key=os.path.getctime)
            log_files = glob.glob(f"{latest_wandb}/logs/*.log")
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                log_time = os.path.getctime(latest_log)
                if log_time > last_log_time:
                    print(f"📈 W&B logging active: {os.path.basename(latest_wandb)}")
                    last_log_time = log_time
        
        # Show status
        if current_time % 30 < 5:  # Every 30 seconds
            print(f"⏰ {time.strftime('%H:%M:%S')} - Monitoring active...")
            print(f"   Checkpoints: {len(checkpoint_dirs)}")
            print(f"   Correlation reports: {correlation_reports}")
            print(f"   Activation samples: {activation_samples}")
            print(f"   W&B runs: {len(wandb_dirs)}")
        
        time.sleep(5)  # Check every 5 seconds

def main():
    """Main monitoring function."""
    
    print("🛡️  ANTI-DUPLICATION TRAINING MONITOR")
    print("=" * 80)
    print("Monitoring the training progress with anti-duplication features:")
    print("   - Diversity regularization")
    print("   - Position-stratified sampling")
    print("   - Correlation monitoring")
    print("   - Enhanced activation sample collection")
    print()
    
    try:
        monitor_training_progress()
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    main()
