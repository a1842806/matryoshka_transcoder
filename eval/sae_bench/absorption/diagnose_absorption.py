"""
Diagnostic script to understand why absorption fraction is low.
"""

import torch
import json
import numpy as np
from pathlib import Path

# Load results
results_file = Path("results/absorption/gemma-2-2b/layer12/absorption_results.json")
data = json.load(open(results_file))

print("="*80)
print("Absorption Diagnosis")
print("="*80)

# Check a few letters
for letter in ['a', 's', 't']:
    if letter not in data['results_by_letter']:
        continue
    
    letter_data = data['results_by_letter'][letter]
    print(f"\nLetter '{letter.upper()}':")
    print(f"  Words evaluated: {letter_data['n_words_evaluated']}")
    print(f"  Non-zero absorption: {letter_data['n_non_zero_absorption']}")
    print(f"  Positive probe proj: {letter_data['n_positive_probe_proj']}")
    print(f"  Full absorption rate: {letter_data['full_absorption_rate']:.4f}")
    print(f"  Cosine similarity: {letter_data['cosine_similarity']:.4f}")
    
    # Check absorption fractions
    abs_fracs = letter_data['absorption_fractions']
    print(f"  Absorption fractions:")
    print(f"    Min: {min(abs_fracs):.6f}")
    print(f"    Max: {max(abs_fracs):.6f}")
    print(f"    Mean: {np.mean(abs_fracs):.6f}")
    print(f"    Non-zero count: {sum(1 for f in abs_fracs if f > 0)}")

print("\n" + "="*80)
print("Analysis:")
print("="*80)
print("""
The absorption_fraction metric measures:
  - How much probe signal comes from 'absorbing' features vs main features
  - Requires: main_feats_probe_proj < act_probe_proj AND
              absorbers contribute >= 40% of probe projection

If absorption_fraction is 0 for all words, it likely means:
  1. Main features are working correctly (main_feats_probe_proj >= act_probe_proj)
     → This is GOOD! Features are accounting for the signal.
  
  2. OR absorbers don't contribute enough (< 40% threshold)
     → Other features aren't significantly absorbing the signal.

The full_absorption_rate (0.1%) shows cases where:
  - Main feature doesn't fire
  - But other aligned features do fire
  - This is the actual "absorption" cases

Low absorption could indicate:
  ✓ Good feature quality (features work correctly)
  ✓ Well-aligned features (not being absorbed)
  
To see more absorption, you might need to:
  - Evaluate on harder/more diverse words
  - Look at cases where probe fires but main feature doesn't
  - Check if thresholds (0.4, 0.1) are appropriate
  - Examine individual word results to see what's happening
""")

print("\n" + "="*80)
print("Recommendations:")
print("="*80)
print("""
1. The near-zero absorption fraction might actually indicate GOOD feature quality
2. Check full_absorption_rate - this shows actual absorption cases (currently 0.1%)
3. To see more absorption, you could:
   a. Lower the threshold (0.4 → 0.2) to be less strict
   b. Evaluate on more diverse/harder words
   c. Look at words where main feature activation is low but probe projection is high
   d. Check if the probe is working correctly (high cosine similarity)
""")

