"""Clean up old messy training results into organized structure."""

import os
import sys
import shutil
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.clean_results import CleanResultsManager

def cleanup_results():
    """Clean up messy results into organized structure."""
    print("🧹 CLEANING UP TRAINING RESULTS")
    print("=" * 50)

    manager = CleanResultsManager()
    
    # Find old checkpoints
    old_checkpoints = Path("checkpoints")
    if not old_checkpoints.exists():
        print("No old checkpoints directory found.")
        return
    
    moved_count = 0
    
    for old_dir in old_checkpoints.rglob("*"):
        if not old_dir.is_dir():
            continue
        
        # Skip if it's not a checkpoint directory
        if not any(f.name.endswith('.pt') for f in old_dir.iterdir()):
            continue
        
        # Parse directory name to extract info
        dir_name = old_dir.name
        if "gemma-2-2b" in dir_name and "blocks." in dir_name:
            # Extract layer number
            try:
                layer_start = dir_name.find("blocks.") + 7
                layer_end = dir_name.find(".", layer_start)
                layer = int(dir_name[layer_start:layer_end])
            except:
                continue
            
            # Extract step count if available
            step_start = dir_name.rfind("_") + 1
            try:
                steps = int(dir_name[step_start:])
            except:
                steps = 0
            
            # Create new experiment directory
            exp_dir = manager.create_experiment_dir(
                model_name="gemma-2-2b",
                layer=layer,
                steps=steps,
                description=f"{steps}k-steps" if steps > 0 else "unknown-steps"
            )
            
            # Move checkpoint files
            for file in old_dir.iterdir():
                if file.name == "sae.pt":
                    shutil.move(str(file), str(exp_dir / "checkpoint.pt"))
                elif file.name == "config.json":
                    shutil.move(str(file), str(exp_dir / "config.json"))
            
            # Remove old directory if empty
            try:
                old_dir.rmdir()
                moved_count += 1
                print(f"✅ Moved: {dir_name}")
            except OSError:
                print(f"⚠️  Partially moved: {dir_name}")
    
    print(f"\n✅ Moved {moved_count} experiments to clean structure")
    print("🎉 Cleanup complete!")

if __name__ == "__main__":
    cleanup_results()