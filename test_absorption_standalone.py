#!/usr/bin/env python3
"""Standalone test for absorption evaluation - Layer 8"""

import sys
import os

# Set up paths FIRST before any other imports
PROJECT_ROOT = "/home/datngo/User/matryoshka_sae"
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Clear any conflicting paths and set up clean imports
sys.path = [p for p in sys.path if 'utils' not in p.lower() or PROJECT_ROOT in p]
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, PROJECT_ROOT)

print("="*80)
print("Absorption Evaluation - Layer 8 (Standalone)")
print("="*80)

# Now import everything
print("\n[1/7] Importing PyTorch...")
import torch
print(f"  ✓ PyTorch {torch.__version__}")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"  ✓ Device: {device}")

print("\n[2/7] Importing TransformerLens...")
from transformer_lens import HookedTransformer
print("  ✓ TransformerLens loaded")

print("\n[3/7] Importing model components...")
# Import from src directly with correct path
from models.sae import MatryoshkaTranscoder
print("  ✓ Matryoshka Transcoder")

# Import absorption eval components one by one to isolate issues
os.chdir(PROJECT_ROOT)  # Ensure we're in project root

print("\n[4/7] Loading transcoder checkpoint...")
checkpoint_path = "/home/datngo/User/matryoshka_sae/results/gemma-2-2b/layer8/16000/checkpoints/final.pt"
config_path = "/home/datngo/User/matryoshka_sae/results/gemma-2-2b/layer8/16000/config.json"

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
print(f"  ✓ Checkpoint loaded (keys: {list(checkpoint.keys())})")

# Load config from separate JSON file
import json
with open(config_path, 'r') as f:
    cfg = json.load(f)
print("  ✓ Config loaded")

# Fix dtype and device in config (they may be stored as strings)
if isinstance(cfg.get('dtype'), str):
    cfg['dtype'] = getattr(torch, cfg['dtype'].replace('torch.', ''))
if isinstance(cfg.get('device'), str):
    cfg['device'] = device  # Use our device

# Create transcoder
transcoder = MatryoshkaTranscoder(cfg)

# Load weights
if 'model' in checkpoint:
    transcoder.load_state_dict(checkpoint['model'])
elif 'model_state_dict' in checkpoint:
    transcoder.load_state_dict(checkpoint['model_state_dict'])
else:
    transcoder.load_state_dict(checkpoint)

transcoder = transcoder.to(device)
transcoder.eval()
print(f"  ✓ Transcoder initialized: {cfg['dict_size']} features")

print("\n[5/7] Loading language model...")
model = HookedTransformer.from_pretrained_no_processing(
    "gemma-2-2b",
    dtype=torch.bfloat16,
).to(device)
print("  ✓ Model loaded")

print("\n[6/7] Creating first-letter dataset...")
# Simplified first-letter task
tokenizer = model.tokenizer
vocab_size = tokenizer.vocab_size

# Collect tokens for letter 's'
s_tokens = []
s_token_strings = []

for token_id in range(min(10000, vocab_size)):  # Sample first 10k tokens
    token_str = tokenizer.decode([token_id])
    token_clean = token_str.strip().lstrip("▁").lstrip("Ġ").lstrip()
    
    if len(token_clean) > 0 and token_clean[0].lower() == 's':
        s_tokens.append(token_id)
        s_token_strings.append(token_str)
        if len(s_tokens) >= 100:  # Get 100 's' tokens
            break

print(f"  ✓ Found {len(s_tokens)} tokens starting with 's'")
print(f"  Examples: {s_token_strings[:5]}")

print("\n[7/7] Testing transcoder on 's' tokens...")
# Get activations for these tokens
s_tokens_tensor = torch.tensor(s_tokens, device=device).unsqueeze(1)  # (n, 1)

with torch.no_grad():
    # Get model activations
    _, cache = model.run_with_cache(s_tokens_tensor[:10], stop_at_layer=9)  # Just first 10 for speed
    hook_name = f"blocks.8.hook_resid_pre"
    activations = cache[hook_name][:, 0, :]  # (10, d_model)
    
    # Encode with transcoder
    features, features_topk = transcoder.compute_activations(activations)
    
    # Find most active features
    feature_activity = (features_topk > 0).float().sum(dim=0)  # (n_features,)
    top_features = torch.topk(feature_activity, k=5)
    
    print(f"\n  Top 5 active features for 's' tokens:")
    for i, (feat_idx, count) in enumerate(zip(top_features.indices, top_features.values)):
        print(f"    {i+1}. Feature {feat_idx.item()}: active on {count.item()}/10 tokens")
    
    # Get feature sparsity
    avg_active = (features_topk > 0).float().sum(dim=1).mean().item()
    print(f"\n  Average active features per token: {avg_active:.1f}")

print("\n" + "="*80)
print("✓ STANDALONE TEST COMPLETE!")
print("="*80)
print("\nThis confirms your transcoder can be loaded and used.")
print("The full absorption evaluation would:")
print("  - Create a balanced dataset for all 26 letters")
print("  - Train linear probes as ground truth classifiers")
print("  - Identify which transcoder features track each letter")
print("  - Measure precision, recall, F1 for classification")
print("  - Detect absorption (when features fail to fire)")
print("  - Run ablation experiments to verify causality")
print("\nTo run the full evaluation, the import issues need to be resolved.")
print("Consider running the evaluation as a standalone script like this one.")

