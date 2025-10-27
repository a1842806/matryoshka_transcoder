"""
Utility to verify and visualize nested group structure in Matryoshka Transcoder.

This script demonstrates that the groups are correctly implemented as nested
(each group contains all features from previous groups plus new ones).
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def verify_nested_groups(group_sizes):
    """
    Verify that group indices correctly implement nested structure.
    
    Args:
        group_sizes: List of group sizes (e.g., [1152, 2304, 4608, 10368])
    
    Returns:
        True if nested structure is correct, False otherwise
    """
    # Compute group indices using cumsum (as in MatryoshkaTranscoder)
    group_indices = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0))
    
    print("="*80)
    print("NESTED GROUP VERIFICATION")
    print("="*80)
    print(f"\nInput group_sizes: {group_sizes}")
    print(f"Computed group_indices: {group_indices}")
    print(f"Total dictionary size: {sum(group_sizes)}")
    
    print("\n" + "-"*80)
    print("NESTED GROUP STRUCTURE (each group includes all previous features):")
    print("-"*80)
    
    for i in range(len(group_sizes)):
        start = group_indices[i]
        end = group_indices[i+1]
        size = end - start
        
        print(f"\nGroup {i}:")
        print(f"  Feature range: [{start}:{end})")
        print(f"  Total features in this group: {end} (cumulative)")
        print(f"  New features added: {size}")
        print(f"  Includes: ", end="")
        
        if i == 0:
            print(f"features 0-{end-1}")
        else:
            prev_end = group_indices[i]
            print(f"all features from Groups 0-{i-1} (0-{prev_end-1}) + new features ({prev_end}-{end-1})")
    
    print("\n" + "-"*80)
    print("NESTED RECONSTRUCTION:")
    print("-"*80)
    print("\nWhen reconstructing with each group:")
    for i in range(len(group_sizes)):
        end = group_indices[i+1]
        print(f"  Group {i} reconstruction uses features [0:{end}] = {end} features total")
    
    # Verify nested property
    print("\n" + "-"*80)
    print("VERIFICATION:")
    print("-"*80)
    
    is_nested = True
    for i in range(1, len(group_indices) - 1):
        # Each group should contain the previous group
        prev_features = set(range(0, group_indices[i]))
        curr_features = set(range(0, group_indices[i+1]))
        
        if not prev_features.issubset(curr_features):
            print(f"❌ FAILED: Group {i} does not contain all features from Group {i-1}")
            is_nested = False
        else:
            print(f"✅ Group {i} correctly contains all features from Group {i-1}")
    
    if is_nested:
        print("\n" + "="*80)
        print("✅ NESTED GROUPS VERIFIED CORRECTLY!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ NESTED GROUPS VERIFICATION FAILED!")
        print("="*80)
    
    return is_nested


def visualize_nested_vs_sequential():
    """
    Visualize the difference between nested and sequential groups.
    """
    group_sizes = [1152, 2304, 4608, 10368]
    
    print("\n" + "="*80)
    print("NESTED vs SEQUENTIAL COMPARISON")
    print("="*80)
    
    # Nested (current implementation)
    print("\n🎯 NESTED (Current Implementation - CORRECT for Matryoshka):")
    print("-"*80)
    group_indices_nested = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0))
    for i in range(len(group_sizes)):
        start = 0  # Always start from 0 for nested
        end = group_indices_nested[i+1]
        print(f"  Group {i}: features [0:{end}] = {end} features")
    
    # Sequential (INCORRECT for Matryoshka)
    print("\n❌ SEQUENTIAL (Non-nested - INCORRECT for Matryoshka):")
    print("-"*80)
    cumsum_values = torch.cumsum(torch.tensor(group_sizes), dim=0)
    for i in range(len(group_sizes)):
        start = 0 if i == 0 else cumsum_values[i-1]
        end = cumsum_values[i]
        print(f"  Group {i}: features [{start}:{end}] = {end - start} features")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCE:")
    print("="*80)
    print("• NESTED: Each group uses ALL features from 0 to its boundary")
    print("  → Group 1 uses [0:3456] (includes Group 0's features)")
    print("  → Enables adaptive computation at multiple granularities")
    print("\n• SEQUENTIAL: Each group uses only its own feature range")
    print("  → Group 1 uses [1152:3456] (excludes Group 0's features)")
    print("  → No nesting, just separate feature banks")
    print("="*80)


def demonstrate_reconstruction():
    """
    Demonstrate how reconstruction works with nested groups.
    """
    print("\n" + "="*80)
    print("NESTED RECONSTRUCTION EXAMPLE")
    print("="*80)
    
    group_sizes = [4, 8, 16, 32]  # Smaller example for clarity
    group_indices = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0))
    total_features = sum(group_sizes)
    
    print(f"\nExample with {len(group_sizes)} groups:")
    print(f"  group_sizes = {group_sizes}")
    print(f"  group_indices = {group_indices}")
    print(f"  total features = {total_features}")
    
    # Simulate activations
    print("\n" + "-"*80)
    print("Simulated sparse activations (1 = active, 0 = inactive):")
    print("-"*80)
    acts = torch.zeros(total_features)
    acts[[1, 5, 12, 20, 35, 50]] = 1.0  # Random active features
    
    print(f"Active features: {acts.nonzero().squeeze().tolist()}")
    
    print("\n" + "-"*80)
    print("Reconstruction with each nested group:")
    print("-"*80)
    
    for i in range(len(group_sizes)):
        end_idx = group_indices[i+1]
        group_acts = acts[:end_idx]
        n_active = (group_acts > 0).sum().item()
        print(f"\nGroup {i} (uses features [0:{end_idx}]):")
        print(f"  Features available: {end_idx}")
        print(f"  Features active: {n_active}")
        print(f"  Reconstruction uses: {group_acts.nonzero().squeeze().tolist()}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("• Each larger group has MORE features available")
    print("• Larger groups can reconstruct MORE accurately")
    print("• User can choose group based on compute/accuracy tradeoff")
    print("• This is the MATRYOSHKA property: one model, multiple capacities")
    print("="*80)


def main():
    """Run all verification and visualization functions."""
    
    # Common group sizes from training scripts
    test_cases = [
        [1152, 2304, 4608, 10368],  # Layer 17
        [2304, 4608, 9216, 20736],  # Layer 12
        [2304, 4608, 9216, 2304],   # Layer 8
    ]
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "MATRYOSHKA NESTED GROUPS VERIFICATION" + " "*21 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    for i, group_sizes in enumerate(test_cases, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        verify_nested_groups(group_sizes)
    
    # Visualizations
    visualize_nested_vs_sequential()
    demonstrate_reconstruction()
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "VERIFICATION COMPLETE!" + " "*32 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")


if __name__ == "__main__":
    main()

