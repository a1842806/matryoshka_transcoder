#!/usr/bin/env python3
"""
Full Absorption Score Evaluation for Layer 12 - 2 Hour Comprehensive Test

This script runs a complete absorption evaluation including:
- All 26 letters
- Linear probe training
- Feature selection and classification metrics
- Absorption detection
- Full result analysis
"""

import sys
import os
import time
from datetime import datetime
import json

# Set up paths
PROJECT_ROOT = "/home/datngo/User/matryoshka_transcoder"
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path = [p for p in sys.path if 'utils' not in p.lower() or PROJECT_ROOT in p]
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, PROJECT_ROOT)

print("="*80)
print("FULL ABSORPTION EVALUATION - LAYER 12")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Estimated runtime: 1-2 hours")
print("="*80)

start_time = time.time()

# Imports
print("\n[Step 1/10] Loading dependencies...")
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"  ✓ PyTorch {torch.__version__} on {device}")

from transformer_lens import HookedTransformer
print("  ✓ TransformerLens")

from models.sae import MatryoshkaTranscoder
print("  ✓ Matryoshka Transcoder")

# Configuration
LAYER = 12
MODEL_NAME = "gemma-2-2b"
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
MIN_TOKENS_PER_LETTER = 200  # More samples per letter
MAX_TOKENS_TOTAL = 6000  # ~230 per letter
BATCH_SIZE = 32
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/absorption")

# Load transcoder
print(f"\n[Step 2/10] Loading Layer {LAYER} transcoder...")
checkpoint_path = f"{PROJECT_ROOT}/results/gemma-2-2b/layer{LAYER}/16000/checkpoints/final.pt"
config_path = f"{PROJECT_ROOT}/results/gemma-2-2b/layer{LAYER}/16000/config.json"

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
with open(config_path, 'r') as f:
    cfg = json.load(f)

# Fix config types
if isinstance(cfg.get('dtype'), str):
    cfg['dtype'] = getattr(torch, cfg['dtype'].replace('torch.', ''))
cfg['device'] = device

transcoder = MatryoshkaTranscoder(cfg)
transcoder.load_state_dict(checkpoint)
transcoder = transcoder.to(device)
transcoder.eval()

print(f"  ✓ Transcoder loaded: {cfg['dict_size']} features")
print(f"  Prefix sizes: {cfg['prefix_sizes']}")
print(f"  TopK: {cfg['top_k']}")

# Load model
print(f"\n[Step 3/10] Loading {MODEL_NAME} model...")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_NAME,
    dtype=torch.bfloat16,
).to(device)
tokenizer = model.tokenizer
print(f"  ✓ Model loaded")

# Create first-letter dataset
print(f"\n[Step 4/10] Creating first-letter dataset...")
print(f"  Scanning vocabulary for tokens starting with each letter...")

vocab_size = tokenizer.vocab_size
tokens_by_letter = {letter: [] for letter in LETTERS}
token_strings_by_letter = {letter: [] for letter in LETTERS}

# Scan vocabulary
for token_id in tqdm(range(vocab_size), desc="Scanning vocab"):
    try:
        token_str = tokenizer.decode([token_id])
        token_clean = token_str.strip().lstrip("▁").lstrip("Ġ").lstrip()
        
        if len(token_clean) > 0:
            first_char = token_clean[0].lower()
            if first_char in tokens_by_letter:
                tokens_by_letter[first_char].append(token_id)
                token_strings_by_letter[first_char].append(token_str)
    except:
        continue

# Balance dataset
print(f"\n  Balancing dataset...")
all_tokens = []
all_letters = []
all_token_strings = []

samples_per_letter = MAX_TOKENS_TOTAL // len(LETTERS)

for letter in LETTERS:
    available = len(tokens_by_letter[letter])
    n_samples = min(samples_per_letter, available, MIN_TOKENS_PER_LETTER * 2)
    
    if available < MIN_TOKENS_PER_LETTER:
        print(f"  Warning: Only {available} tokens for '{letter}'")
        indices = list(range(available))
    else:
        indices = np.random.choice(available, size=n_samples, replace=False)
    
    for idx in indices:
        all_tokens.append(tokens_by_letter[letter][idx])
        all_letters.append(letter)
        all_token_strings.append(token_strings_by_letter[letter][idx])

print(f"  ✓ Dataset created: {len(all_tokens)} tokens")
print(f"  Distribution: {dict((l, all_letters.count(l)) for l in LETTERS)}")

# Convert to tensors
tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device)
labels = torch.zeros(len(all_tokens), len(LETTERS), device=device)
for i, letter in enumerate(all_letters):
    labels[i, LETTERS.index(letter)] = 1.0

# Get model activations
print(f"\n[Step 5/10] Extracting model activations at layer {LAYER}...")
all_activations = []

with torch.no_grad():
    for i in tqdm(range(0, len(tokens_tensor), BATCH_SIZE), desc="Extracting activations"):
        batch_tokens = tokens_tensor[i:i+BATCH_SIZE].unsqueeze(1)  # (batch, 1)
        
        _, cache = model.run_with_cache(batch_tokens, stop_at_layer=LAYER+1)
        hook_name = f"blocks.{LAYER}.hook_resid_pre"
        activations = cache[hook_name][:, 0, :]  # (batch, d_model)
        
        all_activations.append(activations.cpu())

activations = torch.cat(all_activations, dim=0).to(device)
print(f"  ✓ Activations shape: {activations.shape}")

# Train linear probe
print(f"\n[Step 6/10] Training linear probe (ground truth classifier)...")
X = activations.float().cpu().numpy()  # Convert bfloat16 to float32 for numpy
y = labels.cpu().numpy().argmax(axis=1)

clf = LogisticRegression(
    penalty='l2',  # Use L2 instead of L1 for faster convergence
    C=1.0,
    solver='lbfgs',  # Much faster than saga
    max_iter=500,  # Reduced iterations
    random_state=42,
    verbose=1,  # Show progress
)

print("  Training probe (this may take 5-10 minutes)...")
clf.fit(X, y)
y_pred = clf.predict(X)

probe_accuracy = (y == y_pred).mean()
probe_f1 = f1_score(y, y_pred, average='macro', zero_division=0)

print(f"  ✓ Linear probe trained")
print(f"    Accuracy: {probe_accuracy:.3f}")
print(f"    Macro F1: {probe_f1:.3f}")

probe_weights = torch.tensor(clf.coef_, dtype=activations.dtype, device=device)

# Get transcoder features
print(f"\n[Step 7/10] Computing transcoder feature activations...")
all_features = []

with torch.no_grad():
    for i in tqdm(range(0, len(activations), BATCH_SIZE), desc="Encoding features"):
        batch_acts = activations[i:i+BATCH_SIZE]
        _, features_topk = transcoder.compute_activations(batch_acts)
        all_features.append(features_topk.cpu())

feature_activations = torch.cat(all_features, dim=0).to(device)
print(f"  ✓ Feature activations shape: {feature_activations.shape}")

# Compute sparsity stats
active_features = (feature_activations > 0).float()
avg_active = active_features.sum(dim=1).mean().item()
feature_usage = active_features.sum(dim=0)
active_feature_count = (feature_usage > 0).sum().item()

print(f"  Sparsity stats:")
print(f"    Avg active features/token: {avg_active:.1f}")
print(f"    Features used at least once: {active_feature_count}/{cfg['dict_size']}")

# Evaluate each letter
print(f"\n[Step 8/10] Evaluating features for each letter...")
print("="*80)

results_by_letter = {}

for letter_idx, letter in enumerate(tqdm(LETTERS, desc="Letters")):
    # Get probe direction
    probe_direction = probe_weights[letter_idx]
    
    # Get binary labels for this letter
    letter_labels = labels[:, letter_idx]
    
    # Find best transcoder feature by cosine similarity
    probe_norm = F.normalize(probe_direction.unsqueeze(0), dim=1)
    encoder_norm = F.normalize(transcoder.W_enc, dim=0)
    cosine_sims = torch.matmul(probe_norm, encoder_norm).squeeze(0)
    
    best_feature_idx = cosine_sims.argmax().item()
    best_cosine_sim = cosine_sims[best_feature_idx].item()
    
    # Get feature activations
    feature_acts = feature_activations[:, best_feature_idx]
    
    # Compute classification metrics
    predictions = (feature_acts > 0).float()
    labels_binary = letter_labels.cpu().numpy()
    preds_binary = predictions.cpu().numpy()
    
    tp = ((preds_binary == 1) & (labels_binary == 1)).sum()
    fp = ((preds_binary == 1) & (labels_binary == 0)).sum()
    fn = ((preds_binary == 0) & (labels_binary == 1)).sum()
    tn = ((preds_binary == 0) & (labels_binary == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    # Absorption detection (simplified - just check where main feature doesn't fire)
    positive_mask = (letter_labels == 1)
    main_active_mask = (feature_acts > 0.1)
    absorbed_mask = positive_mask & (~main_active_mask)
    
    num_absorbed = absorbed_mask.sum().item()
    num_positive = positive_mask.sum().item()
    absorption_rate = num_absorbed / num_positive if num_positive > 0 else 0.0
    
    # Find alternative active features on absorbed tokens
    if num_absorbed > 0:
        absorbed_indices = torch.where(absorbed_mask)[0]
        # Sample up to 20 absorbed tokens
        sample_size = min(20, len(absorbed_indices))
        sample_indices = absorbed_indices[torch.randperm(len(absorbed_indices))[:sample_size]]
        
        # Find features that fire on these tokens
        absorbed_features = feature_activations[sample_indices]  # (sample, n_features)
        active_on_absorbed = (absorbed_features > 0.1).float().sum(dim=0)  # (n_features,)
        
        # Get top absorbing features (excluding main feature)
        active_on_absorbed[best_feature_idx] = 0
        top_absorbing = torch.topk(active_on_absorbed, k=min(5, active_on_absorbed.numel()))
        
        absorbing_features = [idx.item() for idx in top_absorbing.indices if top_absorbing.values[top_absorbing.indices == idx] > 0]
        absorbing_counts = [val.item() for val in top_absorbing.values if val > 0]
    else:
        absorbing_features = []
        absorbing_counts = []
    
    results_by_letter[letter] = {
        'feature_idx': best_feature_idx,
        'cosine_similarity': float(best_cosine_sim),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'absorption_rate': float(absorption_rate),
        'num_absorbed': int(num_absorbed),
        'num_positive': int(num_positive),
        'absorbing_features': absorbing_features,
        'absorbing_counts': absorbing_counts,
    }

# Print detailed results
print("\n" + "="*80)
print("RESULTS BY LETTER")
print("="*80)
print(f"{'Letter':<8} {'Feature':<8} {'Cosine':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Absorp':<8}")
print("-"*80)

for letter in LETTERS:
    r = results_by_letter[letter]
    print(f"{letter:<8} {r['feature_idx']:<8} {r['cosine_similarity']:<8.3f} "
          f"{r['precision']:<8.3f} {r['recall']:<8.3f} {r['f1']:<8.3f} {r['absorption_rate']:<8.3f}")

# Aggregate statistics
print("\n[Step 9/10] Computing aggregate statistics...")

all_f1 = [r['f1'] for r in results_by_letter.values()]
all_precision = [r['precision'] for r in results_by_letter.values()]
all_recall = [r['recall'] for r in results_by_letter.values()]
all_absorption = [r['absorption_rate'] for r in results_by_letter.values()]
all_cosines = [r['cosine_similarity'] for r in results_by_letter.values()]

summary = {
    'layer': LAYER,
    'model': MODEL_NAME,
    'n_samples': len(all_tokens),
    'n_features': cfg['dict_size'],
    'avg_active_features': float(avg_active),
    'probe_accuracy': float(probe_accuracy),
    'probe_f1': float(probe_f1),
    'mean_f1': float(np.mean(all_f1)),
    'std_f1': float(np.std(all_f1)),
    'median_f1': float(np.median(all_f1)),
    'mean_precision': float(np.mean(all_precision)),
    'std_precision': float(np.std(all_precision)),
    'mean_recall': float(np.mean(all_recall)),
    'std_recall': float(np.std(all_recall)),
    'mean_absorption_rate': float(np.mean(all_absorption)),
    'std_absorption_rate': float(np.std(all_absorption)),
    'max_absorption_rate': float(np.max(all_absorption)),
    'min_absorption_rate': float(np.min(all_absorption)),
    'mean_cosine_similarity': float(np.mean(all_cosines)),
}

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nLinear Probe Performance (Ground Truth):")
print(f"  Accuracy: {summary['probe_accuracy']:.3f}")
print(f"  Macro F1: {summary['probe_f1']:.3f}")

print(f"\nTranscoder Feature Performance:")
print(f"  Mean F1: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")
print(f"  Median F1: {summary['median_f1']:.3f}")
print(f"  Mean Precision: {summary['mean_precision']:.3f} ± {summary['std_precision']:.3f}")
print(f"  Mean Recall: {summary['mean_recall']:.3f} ± {summary['std_recall']:.3f}")

print(f"\nFeature Absorption Analysis:")
print(f"  Mean absorption rate: {summary['mean_absorption_rate']:.3f} ± {summary['std_absorption_rate']:.3f}")
print(f"  Max absorption rate: {summary['max_absorption_rate']:.3f}")
print(f"  Min absorption rate: {summary['min_absorption_rate']:.3f}")

print(f"\nFeature-Probe Alignment:")
print(f"  Mean cosine similarity: {summary['mean_cosine_similarity']:.3f}")

print(f"\nSparsity:")
print(f"  Avg active features/token: {summary['avg_active_features']:.1f}")

# Identify best and worst performing letters
f1_sorted = sorted(results_by_letter.items(), key=lambda x: x[1]['f1'], reverse=True)
print(f"\nBest performing letters (by F1):")
for letter, r in f1_sorted[:5]:
    print(f"  '{letter}': F1={r['f1']:.3f}, Feature {r['feature_idx']}, Absorption={r['absorption_rate']:.3f}")

print(f"\nWorst performing letters (by F1):")
for letter, r in f1_sorted[-5:]:
    print(f"  '{letter}': F1={r['f1']:.3f}, Feature {r['feature_idx']}, Absorption={r['absorption_rate']:.3f}")

# Most absorbed letters
absorption_sorted = sorted(results_by_letter.items(), key=lambda x: x[1]['absorption_rate'], reverse=True)
print(f"\nMost absorbed letters:")
for letter, r in absorption_sorted[:5]:
    print(f"  '{letter}': Absorption={r['absorption_rate']:.3f}, F1={r['f1']:.3f}, "
          f"Absorbing features: {len(r['absorbing_features'])}")

# Save results
print(f"\n[Step 10/10] Saving results...")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, MODEL_NAME), exist_ok=True)

output_file = os.path.join(RESULTS_DIR, MODEL_NAME, f"layer{LAYER}_full_absorption.json")

full_results = {
    'summary': summary,
    'results_by_letter': results_by_letter,
    'config': {
        'layer': LAYER,
        'model': MODEL_NAME,
        'n_samples': len(all_tokens),
        'samples_per_letter': dict((l, all_letters.count(l)) for l in LETTERS),
        'timestamp': datetime.now().isoformat(),
    }
}

with open(output_file, 'w') as f:
    json.dump(full_results, f, indent=2)

print(f"  ✓ Results saved to: {output_file}")

# Execution time
end_time = time.time()
duration_mins = (end_time - start_time) / 60

print("\n" + "="*80)
print("✓ EVALUATION COMPLETE!")
print("="*80)
print(f"Total time: {duration_mins:.1f} minutes ({duration_mins/60:.2f} hours)")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults file: {output_file}")
print("="*80)

