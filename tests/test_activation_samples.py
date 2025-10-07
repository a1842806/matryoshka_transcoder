"""
Comprehensive tests for activation sample collection functionality.

Tests both functionality and interpretability following Anthropic's approach.
"""

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.activation_samples import ActivationSampleCollector, ActivationSample
import tempfile
import json
import unittest
from unittest.mock import Mock


class TestActivationSampleCollector(unittest.TestCase):
    """Test basic functionality of ActivationSampleCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = ActivationSampleCollector(
            max_samples_per_feature=10,
            context_size=20,
            storage_threshold=0.1
        )
        
        # Create mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode = lambda tokens: " ".join([f"tok{t}" for t in tokens])
    
    def test_initialization(self):
        """Test collector initializes with correct parameters."""
        self.assertEqual(self.collector.max_samples, 10)
        self.assertEqual(self.collector.context_size, 20)
        self.assertEqual(self.collector.storage_threshold, 0.1)
        self.assertEqual(len(self.collector.feature_samples), 0)
    
    def test_collect_batch_samples_basic(self):
        """Test basic sample collection from a batch."""
        # Create simple test data
        batch_size = 2
        seq_len = 10
        n_features = 5
        
        # Create activations with some above threshold
        activations = torch.zeros(batch_size, seq_len, n_features)
        activations[0, 5, 2] = 0.8  # Feature 2, position 5
        activations[1, 3, 4] = 0.6  # Feature 4, position 3
        
        # Create tokens
        tokens = torch.randint(0, 100, (batch_size, seq_len))
        
        # Collect samples
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer,
            feature_indices=[2, 4]
        )
        
        # Check samples were collected
        self.assertIn(2, self.collector.feature_samples)
        self.assertIn(4, self.collector.feature_samples)
        self.assertEqual(len(self.collector.feature_samples[2]), 1)
        self.assertEqual(len(self.collector.feature_samples[4]), 1)
        
        # Check activation values
        samples_f2 = self.collector.get_top_samples(2)
        self.assertAlmostEqual(samples_f2[0].activation_value, 0.8, places=5)
    
    def test_max_samples_limit(self):
        """Test that collector respects max_samples limit."""
        batch_size = 2
        seq_len = 10
        n_features = 3
        
        # Create many high activations for feature 0
        for _ in range(15):  # More than max_samples=10
            activations = torch.zeros(batch_size, seq_len, n_features)
            activations[0, 5, 0] = torch.rand(1).item() + 0.5  # Random value > 0.5
            tokens = torch.randint(0, 100, (batch_size, seq_len))
            
            self.collector.collect_batch_samples(
                activations,
                tokens,
                self.mock_tokenizer,
                feature_indices=[0]
            )
        
        # Should not exceed max_samples
        self.assertLessEqual(len(self.collector.feature_samples[0]), 10)
    
    def test_top_k_samples_sorted(self):
        """Test that get_top_samples returns samples sorted by activation."""
        batch_size = 2
        seq_len = 10
        n_features = 3
        
        # Create samples with different activations
        activation_values = [0.3, 0.8, 0.5, 0.9, 0.2]
        for val in activation_values:
            activations = torch.zeros(batch_size, seq_len, n_features)
            activations[0, 5, 1] = val
            tokens = torch.randint(0, 100, (batch_size, seq_len))
            
            self.collector.collect_batch_samples(
                activations,
                tokens,
                self.mock_tokenizer,
                feature_indices=[1]
            )
        
        # Get top samples
        samples = self.collector.get_top_samples(1)
        
        # Check they are sorted descending
        activations = [s.activation_value for s in samples]
        self.assertEqual(activations, sorted(activations, reverse=True))
        self.assertAlmostEqual(samples[0].activation_value, 0.9, places=5)
    
    def test_storage_threshold(self):
        """Test that samples below threshold are not stored."""
        batch_size = 2
        seq_len = 10
        n_features = 3
        
        # Create activations below threshold
        activations = torch.zeros(batch_size, seq_len, n_features)
        activations[0, 5, 0] = 0.05  # Below threshold of 0.1
        tokens = torch.randint(0, 100, (batch_size, seq_len))
        
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer,
            feature_indices=[0]
        )
        
        # Should not have any samples
        self.assertNotIn(0, self.collector.feature_samples)
    
    def test_get_feature_statistics(self):
        """Test feature statistics calculation."""
        batch_size = 2
        seq_len = 10
        n_features = 3
        
        # Add samples for feature 0
        activation_values = [0.5, 0.7, 0.3, 0.9]
        for val in activation_values:
            activations = torch.zeros(batch_size, seq_len, n_features)
            activations[0, 5, 0] = val
            tokens = torch.randint(0, 100, (batch_size, seq_len))
            
            self.collector.collect_batch_samples(
                activations,
                tokens,
                self.mock_tokenizer,
                feature_indices=[0]
            )
        
        stats = self.collector.get_feature_statistics(0)
        
        self.assertEqual(stats['num_samples'], 4)
        self.assertAlmostEqual(stats['max_activation'], 0.9, places=5)
        self.assertGreater(stats['mean_activation'], 0.3)
        self.assertLess(stats['mean_activation'], 0.9)
    
    def test_save_and_load_samples(self):
        """Test saving and loading samples to/from disk."""
        batch_size = 2
        seq_len = 10
        n_features = 3
        
        # Collect some samples
        activations = torch.zeros(batch_size, seq_len, n_features)
        activations[0, 5, 0] = 0.8
        activations[1, 3, 1] = 0.6
        tokens = torch.randint(0, 100, (batch_size, seq_len))
        
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer
        )
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            self.collector.save_samples(tmpdir, top_k_features=10, samples_per_feature=5)
            
            # Check files were created
            files = os.listdir(tmpdir)
            self.assertIn("collection_summary.json", files)
            
            # Check feature files exist
            feature_files = [f for f in files if f.startswith("feature_")]
            self.assertGreater(len(feature_files), 0)
            
            # Load summary and verify
            with open(os.path.join(tmpdir, "collection_summary.json"), 'r') as f:
                summary = json.load(f)
            
            self.assertGreater(summary['total_samples_seen'], 0)
            self.assertGreater(summary['total_samples_stored'], 0)
            
            # Test loading
            new_collector = ActivationSampleCollector()
            new_collector.load_samples(tmpdir)
            
            # Verify loaded samples match
            self.assertEqual(len(new_collector.feature_samples), len(self.collector.feature_samples))
    
    def test_context_extraction(self):
        """Test that context tokens are properly extracted."""
        batch_size = 1
        seq_len = 30
        n_features = 2
        
        # Create activation at position 15 (middle of sequence)
        activations = torch.zeros(batch_size, seq_len, n_features)
        activations[0, 15, 0] = 0.8
        
        # Create tokens with identifiable pattern
        tokens = torch.arange(seq_len).unsqueeze(0)
        
        self.collector = ActivationSampleCollector(
            max_samples_per_feature=10,
            context_size=10,  # 5 tokens on each side
            storage_threshold=0.1
        )
        
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer,
            feature_indices=[0]
        )
        
        samples = self.collector.get_top_samples(0)
        self.assertEqual(len(samples), 1)
        
        # Context should be around position 15
        context_tokens = samples[0].context_tokens
        self.assertGreater(len(context_tokens), 5)
        self.assertIn(15, context_tokens)  # Should include the activating position


class TestActivationSampleInterpretability(unittest.TestCase):
    """Test interpretability aspects using real-like data."""
    
    def setUp(self):
        """Set up with realistic tokenizer."""
        self.collector = ActivationSampleCollector(
            max_samples_per_feature=20,
            context_size=30,
            storage_threshold=0.2
        )
        
        # Create more realistic mock tokenizer
        self.vocab = {
            0: "the", 1: "cat", 2: "dog", 3: "sat", 4: "on", 
            5: "mat", 6: "ran", 7: "quickly", 8: "jumped", 9: "high",
            10: "ate", 11: "food", 12: "happy", 13: "sad", 14: "."
        }
        
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode = lambda tokens: " ".join([self.vocab.get(t, f"<unk{t}>") for t in tokens])
    
    def test_interpretable_patterns(self):
        """Test that high-activation samples show interpretable patterns."""
        # Simulate a feature that activates for "cat" related content
        test_sequences = [
            ([0, 1, 3, 4, 0, 5, 14], 1, 0.9),  # "the cat sat on the mat ." - high activation at "cat"
            ([0, 1, 6, 7, 14], 1, 0.85),        # "the cat ran quickly ." - high activation at "cat"
            ([0, 2, 3, 4, 0, 5, 14], 1, 0.1),   # "the dog sat on the mat ." - low activation (dog, not cat)
            ([0, 1, 10, 11, 14], 1, 0.8),       # "the cat ate food ." - high activation at "cat"
        ]
        
        feature_idx = 42  # Arbitrary feature that responds to "cat"
        
        for tokens, position, activation in test_sequences:
            batch_size = 1
            seq_len = len(tokens)
            n_features = 100
            
            activations = torch.zeros(batch_size, seq_len, n_features)
            activations[0, position, feature_idx] = activation
            
            token_tensor = torch.tensor([tokens])
            
            self.collector.collect_batch_samples(
                activations,
                token_tensor,
                self.mock_tokenizer,
                feature_indices=[feature_idx]
            )
        
        # Get top samples for the "cat" feature
        samples = self.collector.get_top_samples(feature_idx, k=5)
        
        # All high-activation samples should contain "cat"
        high_act_samples = [s for s in samples if s.activation_value > 0.7]
        self.assertGreater(len(high_act_samples), 0)
        
        for sample in high_act_samples:
            # Text should be interpretable
            self.assertIsInstance(sample.text, str)
            self.assertGreater(len(sample.text), 0)
            
            # Context should provide meaning
            self.assertIn("cat", sample.context_text.lower())
    
    def test_feature_specialization(self):
        """Test that different features capture different patterns."""
        # Feature 10: "cat" related
        # Feature 20: "dog" related
        
        test_data = [
            ([0, 1, 3, 4, 0, 5], 1, 10, 0.9),   # cat
            ([0, 1, 6, 7, 14], 1, 10, 0.85),    # cat
            ([0, 2, 3, 4, 0, 5], 1, 20, 0.9),   # dog
            ([0, 2, 6, 7, 14], 1, 20, 0.85),    # dog
        ]
        
        for tokens, position, feature_idx, activation in test_data:
            batch_size = 1
            seq_len = len(tokens)
            n_features = 50
            
            activations = torch.zeros(batch_size, seq_len, n_features)
            activations[0, position, feature_idx] = activation
            
            token_tensor = torch.tensor([tokens])
            
            self.collector.collect_batch_samples(
                activations,
                token_tensor,
                self.mock_tokenizer,
                feature_indices=[feature_idx]
            )
        
        # Get samples for both features
        cat_samples = self.collector.get_top_samples(10)
        dog_samples = self.collector.get_top_samples(20)
        
        # Cat feature should only have cat examples
        for sample in cat_samples:
            self.assertIn("cat", sample.context_text.lower())
            self.assertNotIn("dog", sample.context_text.lower())
        
        # Dog feature should only have dog examples
        for sample in dog_samples:
            self.assertIn("dog", sample.context_text.lower())
            self.assertNotIn("cat", sample.context_text.lower())
    
    def test_activation_distribution(self):
        """Test that activation values follow expected distribution."""
        # Create samples with varying activations
        feature_idx = 5
        activations_list = []
        
        for i in range(50):
            batch_size = 1
            seq_len = 10
            n_features = 20
            
            # Random activations above threshold
            activation = 0.2 + torch.rand(1).item() * 0.8
            activations_list.append(activation)
            
            activations = torch.zeros(batch_size, seq_len, n_features)
            activations[0, 5, feature_idx] = activation
            
            tokens = torch.randint(0, len(self.vocab), (batch_size, seq_len))
            
            self.collector.collect_batch_samples(
                activations,
                tokens,
                self.mock_tokenizer,
                feature_indices=[feature_idx]
            )
        
        # Get statistics
        stats = self.collector.get_feature_statistics(feature_idx)
        
        # Check reasonable statistics
        self.assertGreater(stats['max_activation'], 0.2)  # Should have some high activations
        self.assertLess(stats['min_activation'], 1.0)     # Min should be less than max possible
        self.assertGreater(stats['std_activation'], 0.0)  # Should have variance
        self.assertLess(stats['max_activation'], 1.1)     # Reasonable upper bound
        
        # Top samples should have highest activations
        samples = self.collector.get_top_samples(feature_idx, k=5)
        top_activations = [s.activation_value for s in samples]
        
        # All top-5 should be in the upper half of all activations
        median_activation = sorted(activations_list)[len(activations_list) // 2]
        for act in top_activations:
            self.assertGreater(act, median_activation)


class TestActivationSampleEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up collector and mock tokenizer."""
        self.collector = ActivationSampleCollector()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.decode = lambda tokens: " ".join([f"tok{t}" for t in tokens])
    
    def test_empty_activations(self):
        """Test handling of batch with no activations above threshold."""
        activations = torch.zeros(2, 10, 5)
        tokens = torch.randint(0, 100, (2, 10))
        
        # Should not crash
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer
        )
        
        # Should have no samples
        self.assertEqual(len(self.collector.feature_samples), 0)
    
    def test_single_token_sequence(self):
        """Test handling of very short sequences."""
        activations = torch.tensor([[[0.5, 0.8, 0.3]]])  # [1, 1, 3]
        tokens = torch.tensor([[42]])  # [1, 1]
        
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer
        )
        
        # Should still collect the sample
        samples = self.collector.get_top_samples(1)
        self.assertEqual(len(samples), 1)
    
    def test_feature_not_found(self):
        """Test getting samples for feature with no data."""
        samples = self.collector.get_top_samples(999)
        self.assertEqual(len(samples), 0)
        
        stats = self.collector.get_feature_statistics(999)
        self.assertEqual(stats['num_samples'], 0)
    
    def test_clear_functionality(self):
        """Test clearing all collected samples."""
        # Collect some samples
        activations = torch.ones(2, 10, 5) * 0.5
        tokens = torch.randint(0, 100, (2, 10))
        
        self.collector.collect_batch_samples(
            activations,
            tokens,
            self.mock_tokenizer
        )
        
        self.assertGreater(len(self.collector.feature_samples), 0)
        
        # Clear
        self.collector.clear()
        
        self.assertEqual(len(self.collector.feature_samples), 0)
        self.assertEqual(self.collector.total_samples_seen, 0)


def run_tests():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestActivationSampleCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestActivationSampleInterpretability))
    suite.addTests(loader.loadTestsFromTestCase(TestActivationSampleEdgeCases))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. See details above.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

