#!/usr/bin/env python3
"""
Inspect checkpoint structure to understand how weights are stored.
"""

import torch
import json
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect the structure of a checkpoint file."""
    
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Successfully loaded checkpoint")
        
        print(f"\n📊 Checkpoint keys: {list(checkpoint.keys())}")
        
        # Recursively explore the structure
        def explore_dict(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, dict):
                        print(f"{prefix}{key}: dict with keys {list(value.keys())}")
                        explore_dict(value, prefix + "  ")
                    elif isinstance(value, torch.Tensor):
                        print(f"{prefix}{key}: Tensor {value.shape} {value.dtype}")
                    elif isinstance(value, list):
                        print(f"{prefix}{key}: List with {len(value)} items")
                        if len(value) > 0:
                            print(f"{prefix}  First item type: {type(value[0])}")
                    else:
                        print(f"{prefix}{key}: {type(value)} = {value}")
        
        explore_dict(checkpoint)
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def main():
    """Main function to inspect checkpoint structure."""
    
    # Find latest checkpoint
    checkpoint_dir = Path("checkpoints/transcoder/gemma-2-2b")
    checkpoint_files = list(checkpoint_dir.glob("**/sae.pt"))
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 Using latest checkpoint: {latest_checkpoint}")
        
        checkpoint = inspect_checkpoint(str(latest_checkpoint))
        
        if checkpoint is not None:
            print(f"\n🎯 Checkpoint inspection complete!")
    else:
        print(f"❌ No checkpoint files found")

if __name__ == "__main__":
    main()
