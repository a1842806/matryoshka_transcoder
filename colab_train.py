"""
Google Colab Training Script for Matryoshka Transcoder (Nested Groups)

Copy this entire file into a Colab cell and run it!
Or run: !python colab_train.py --layer 17
"""

import subprocess
import sys
import os

def setup_environment():
    """Setup Google Colab environment."""
    print("="*80)
    print("🚀 MATRYOSHKA TRANSCODER - NESTED GROUPS TRAINING")
    print("="*80)
    
    # Check if already cloned
    if not os.path.exists('matryoshka_transcoder'):
        print("\n📥 Cloning repository...")
        subprocess.run([
            'git', 'clone', 
            'https://github.com/a1842806/matryoshka_transcoder.git'
        ], check=True)
        print("✓ Repository cloned")
    else:
        print("\n✓ Repository already exists")
    
    # Change to repository directory
    os.chdir('matryoshka_transcoder')
    
    # Checkout the nested groups branch
    print("\n🔄 Switching to nested groups branch...")
    subprocess.run(['git', 'checkout', 'interpretability-evaluation'], check=True)
    subprocess.run(['git', 'pull'], check=True)
    print("✓ On interpretability-evaluation branch")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        'torch', 'transformers', 'transformer-lens', 
        'wandb', 'datasets', 'einops', 'jaxtyping', '-q'
    ], check=True)
    print("✓ Dependencies installed")
    
    # Check GPU
    print("\n🖥️  GPU Check...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  No GPU detected! Training will be very slow.")
            print("   Go to: Runtime → Change runtime type → T4 GPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed correctly")
        return False
    
    return True

def train_model(layer: int = 17):
    """Train the model on specified layer."""
    print("\n" + "="*80)
    print(f"🎯 TRAINING LAYER {layer}")
    print("="*80)
    
    # Map layer to training script
    script_map = {
        8: 'src/scripts/train_gemma_layer8_with_warmup_decay_samples.py',
        12: 'src/scripts/train_gemma_layer12_with_warmup_decay_samples.py',
        17: 'src/scripts/train_gemma_layer17_with_warmup_decay_samples.py',
    }
    
    if layer not in script_map:
        print(f"❌ Invalid layer {layer}. Choose from: {list(script_map.keys())}")
        return False
    
    script = script_map[layer]
    
    # Print training info
    info = {
        8: "Quick test (1k steps, ~30 min on T4)",
        12: "Full training (20k steps, ~8 hours on T4)",
        17: "Optimized training (15k steps, ~6 hours on T4) - RECOMMENDED",
    }
    
    print(f"\n📊 Configuration: {info[layer]}")
    print(f"📝 Script: {script}")
    print(f"💾 Checkpoints will be saved to: checkpoints/transcoder/gemma-2-2b/")
    print(f"📈 Monitor progress at W&B dashboard (link will appear below)")
    print("\n" + "-"*80)
    print("Training starting... This will take a while. ☕")
    print("-"*80 + "\n")
    
    # Run training
    try:
        subprocess.run([sys.executable, script], check=True)
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False

def save_to_drive():
    """Save checkpoints to Google Drive."""
    print("\n💾 Saving to Google Drive...")
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        import shutil
        src = 'checkpoints/transcoder/gemma-2-2b/'
        dst = '/content/drive/MyDrive/matryoshka_checkpoints/'
        
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"✓ Checkpoints saved to: {dst}")
            return True
        else:
            print("⚠️  No checkpoints found to save")
            return False
            
    except ImportError:
        print("⚠️  Not running on Google Colab, skipping Drive save")
        return False
    except Exception as e:
        print(f"❌ Failed to save to Drive: {e}")
        return False

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Matryoshka Transcoder on Google Colab')
    parser.add_argument('--layer', type=int, default=17, choices=[8, 12, 17],
                        help='Layer to train (8=quick test, 17=recommended, 12=full)')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving to Google Drive after training')
    
    args = parser.parse_args()
    
    # Setup
    if not setup_environment():
        print("\n❌ Setup failed. Please check errors above.")
        sys.exit(1)
    
    # Optional: Login to W&B
    print("\n🔐 Weights & Biases Login (optional)")
    print("   Press Enter to skip, or login to track metrics")
    try:
        import wandb
        wandb.login()
    except Exception:
        print("   Skipping W&B login...")
    
    # Train
    success = train_model(args.layer)
    
    if not success:
        sys.exit(1)
    
    # Save to Drive
    if not args.no_save:
        save_to_drive()
    
    # Final instructions
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)
    print("\n📁 Your files:")
    print(f"   - Local: checkpoints/transcoder/gemma-2-2b/")
    if not args.no_save:
        print(f"   - Drive: /content/drive/MyDrive/matryoshka_checkpoints/")
    
    print("\n🔍 Next steps:")
    print("   1. Evaluate interpretability:")
    print("      !python src/eval/evaluate_interpretability_standalone.py \\")
    print("          --checkpoint checkpoints/transcoder/gemma-2-2b/YOUR_MODEL \\")
    print("          --layer", args.layer)
    print("\n   2. Analyze activation samples:")
    print("      !python src/utils/analyze_activation_samples.py \\")
    print("          checkpoints/transcoder/gemma-2-2b/YOUR_MODEL_activation_samples/")
    
    print("\n📚 Documentation:")
    print("   - QUICK_START_COLAB.md - Quick reference")
    print("   - COLAB_TRAINING.md - Detailed guide")
    print("   - NESTED_GROUPS_SUMMARY.md - Architecture explanation")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

