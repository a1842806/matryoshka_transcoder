"""
End-to-end integration test for activation sample collection.

This test verifies that the entire pipeline works:
1. Sample collection during training
2. Saving samples to disk
3. Loading and analyzing samples
4. Generating interpretability reports
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.activation_samples import ActivationSampleCollector
from utils.analyze_activation_samples import ActivationSampleAnalyzer
import tempfile
import shutil
from unittest.mock import Mock


def test_end_to_end_workflow():
    """Test complete workflow from collection to analysis."""
    
    print("\n" + "="*80)
    print("END-TO-END ACTIVATION SAMPLE COLLECTION TEST")
    print("="*80)
    
    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    
    try:
        # Step 1: Initialize collector
        print("\n[Step 1] Initializing ActivationSampleCollector...")
        collector = ActivationSampleCollector(
            max_samples_per_feature=20,
            context_size=15,
            storage_threshold=0.2
        )
        
        # Create mock tokenizer
        vocab = {i: f"word{i}" for i in range(100)}
        mock_tokenizer = Mock()
        mock_tokenizer.decode = lambda tokens: " ".join([vocab.get(t, f"<{t}>") for t in tokens])
        
        # Step 2: Simulate collecting samples during training
        print("[Step 2] Simulating sample collection during training...")
        
        num_training_steps = 5
        batch_size = 4
        seq_len = 20
        n_features = 10
        
        for step in range(num_training_steps):
            # Create realistic activation patterns
            activations = torch.zeros(batch_size, seq_len, n_features)
            
            # Feature 0: Activates on positions 5-7
            activations[:, 5:8, 0] = torch.rand(batch_size, 3) * 0.5 + 0.3
            
            # Feature 3: Activates on positions 10-12
            activations[:, 10:13, 3] = torch.rand(batch_size, 3) * 0.6 + 0.4
            
            # Feature 7: Rare but strong activations
            if step % 2 == 0:
                activations[0, 15, 7] = 0.9
            
            # Generate tokens
            tokens = torch.randint(0, 100, (batch_size, seq_len))
            
            # Collect samples
            collector.collect_batch_samples(
                activations,
                tokens,
                mock_tokenizer,
                feature_indices=[0, 3, 7]
            )
            
            print(f"  Step {step+1}/{num_training_steps}: Collected samples")
        
        # Step 3: Verify collection
        print("\n[Step 3] Verifying collected samples...")
        stats = collector.get_all_statistics()
        print(f"  Total samples seen: {stats['total_samples_seen']}")
        print(f"  Total samples stored: {stats['total_samples_stored']}")
        print(f"  Features tracked: {stats['num_features_tracked']}")
        
        assert stats['total_samples_stored'] > 0, "No samples were stored!"
        assert len(collector.feature_samples) > 0, "No features have samples!"
        
        # Step 4: Examine feature samples
        print("\n[Step 4] Examining feature samples...")
        for feature_idx in [0, 3, 7]:
            samples = collector.get_top_samples(feature_idx, k=3)
            if samples:
                print(f"\n  Feature {feature_idx}: {len(samples)} samples")
                print(f"    Max activation: {samples[0].activation_value:.3f}")
                print(f"    Example text: {samples[0].text}")
        
        # Step 5: Save samples to disk
        print("\n[Step 5] Saving samples to disk...")
        collector.save_samples(
            tmpdir,
            top_k_features=10,
            samples_per_feature=5
        )
        
        # Verify files were created
        files = os.listdir(tmpdir)
        print(f"  Files created: {len(files)}")
        assert "collection_summary.json" in files, "Summary file not created!"
        
        feature_files = [f for f in files if f.startswith("feature_")]
        assert len(feature_files) > 0, "No feature files created!"
        print(f"  Feature files: {len(feature_files)}")
        
        # Step 6: Load and analyze samples
        print("\n[Step 6] Loading and analyzing samples...")
        analyzer = ActivationSampleAnalyzer(tmpdir)
        
        print(f"  Features loaded: {len(analyzer.features)}")
        assert len(analyzer.features) > 0, "Failed to load features!"
        
        # Step 7: Generate report
        print("\n[Step 7] Generating interpretability report...")
        report_path = os.path.join(tmpdir, "test_report.md")
        analyzer.generate_interpretability_report(
            report_path,
            top_k_features=5,
            examples_per_feature=3
        )
        
        assert os.path.exists(report_path), "Report not generated!"
        
        # Verify report content
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        assert "Feature" in report_content, "Report missing feature information!"
        assert "Statistics" in report_content, "Report missing statistics!"
        
        report_size = len(report_content)
        print(f"  Report generated: {report_size} characters")
        
        # Step 8: Test feature report printing
        print("\n[Step 8] Testing feature report printing...")
        if analyzer.features:
            first_feature = list(analyzer.features.keys())[0]
            print(f"  Printing report for Feature {first_feature}...")
            analyzer.print_feature_report(first_feature, num_examples=2)
        
        # Step 9: Test analysis functions
        print("\n[Step 9] Testing analysis functions...")
        analyzer.analyze_feature_clustering()
        
        # Success!
        print("\n" + "="*80)
        print("✅ END-TO-END TEST PASSED!")
        print("="*80)
        print("\nAll pipeline components working correctly:")
        print("  ✓ Sample collection during training")
        print("  ✓ Storage and statistics tracking")
        print("  ✓ Saving to disk")
        print("  ✓ Loading from disk")
        print("  ✓ Analysis and reporting")
        print("  ✓ Interpretability features")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
            print(f"\n[Cleanup] Removed temporary directory: {tmpdir}")


if __name__ == "__main__":
    success = test_end_to_end_workflow()
    sys.exit(0 if success else 1)

