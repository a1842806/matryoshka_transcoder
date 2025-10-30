"""
SAEBench AutoInterp and Absorption Score Evaluation.

Implements the two core interpretability metrics from SAEBench paper:
1. AutoInterp - LLM-based detection task for feature interpretability
2. Absorption Score - First-letter probe task measuring feature absorption

Reference: SAEBench (arXiv:2503.09532)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformer_lens import HookedTransformer
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.sae import MatryoshkaTranscoder
from src.models.transcoder_activation_store import TranscoderActivationsStore, create_transcoder_config
from src.adapters.gemma_scope_transcoder import GemmaScopeTranscoderAdapter
from src.utils.config import get_default_cfg


@dataclass
class SAEBenchMetrics:
    """SAEBench core interpretability metrics."""
    
    # AutoInterp
    autointerp_score: Optional[float]  # Mean accuracy across sampled latents
    autointerp_latents_evaluated: Optional[int]
    
    # Absorption (First-Letter Task)
    absorption_score: float  # 1 - mean_absorption (higher is better in SAEBench)
    raw_absorption: float  # Mean absorption ratio
    letters_evaluated: int
    
    # Model info
    mean_l0: float
    dict_size: int
    model_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AutoInterpEvaluator:
    """
    AutoInterp: LLM-based detection task for feature interpretability.
    
    Following SAEBench methodology (Paulo et al., 2024):
    1. Generate feature description from top-activating examples
    2. Test: predict which sequences activate the feature (10 random + 2 max + 2 IW)
    3. Score = accuracy
    
    Supports multiple LLM backends:
    - OpenAI API (GPT-4o-mini) - requires API key
    - Hugging Face models (free, local) - recommended
    - SAEBench API (if available) - free
    """
    
    def __init__(
        self,
        llm_backend: str = "huggingface",  # "openai", "huggingface", "saebench"
        model_name: str = "microsoft/DialoGPT-medium",  # Hugging Face model
        api_key: Optional[str] = None,
        n_latents_sample: int = 1000,
        ctx_len: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.n_latents_sample = n_latents_sample
        self.ctx_len = ctx_len
        self.device = device
        
        # Initialize the appropriate LLM backend
        self.llm_model = None
        self.tokenizer = None
        
        if llm_backend == "huggingface":
            self._init_huggingface_model()
        elif llm_backend == "openai":
            if not self.api_key:
                print("⚠️  Warning: No OpenAI API key found. AutoInterp will be skipped.")
                print("   Set OPENAI_API_KEY environment variable to enable.")
        elif llm_backend == "saebench":
            self._init_saebench_api()
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}")
    
    def _init_huggingface_model(self):
        """Initialize Hugging Face model for local inference."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"📥 Loading Hugging Face model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✓ Hugging Face model loaded successfully")
            
        except ImportError:
            print("❌ Error: transformers library not found.")
            print("   Install with: pip install transformers")
            raise
        except Exception as e:
            print(f"❌ Error loading Hugging Face model: {e}")
            print("   Falling back to SAEBench API...")
            self.llm_backend = "saebench"
            self._init_saebench_api()
    
    def _init_saebench_api(self):
        """Initialize SAEBench API (if available)."""
        try:
            # Try to import SAEBench API
            import saebench  # type: ignore
            print("✓ SAEBench API available")
            self.saebench_client = saebench.Client()
        except ImportError:
            print("⚠️  SAEBench API not available locally.")
            print("   Install with: pip install saebench")
            print("   Or use Hugging Face models instead.")
            self.saebench_client = None
    
    def evaluate(
        self,
        transcoder,
        activation_store: TranscoderActivationsStore,
        num_sequences: int = 5000
    ) -> Tuple[float, int]:
        """
        Evaluate AutoInterp score.
        
        Args:
            transcoder: Model with encode() method
            activation_store: Source of sequences
            num_sequences: Sequences to collect for evaluation
            
        Returns:
            (autointerp_score, n_latents_evaluated)
        """
        if self.llm_backend == "openai" and not self.api_key:
            print("⚠️  Skipping AutoInterp (no OpenAI API key)")
            return None, 0
        
        if self.llm_backend == "saebench" and self.saebench_client is None:
            print("⚠️  Skipping AutoInterp (SAEBench API not available)")
            return None, 0
        
        print("\n" + "="*80)
        print(f"🤖 AutoInterp Evaluation ({self.llm_backend.upper()})")
        print("="*80)
        
        # Step 1: Collect activations and build cache
        print(f"Collecting activations from {num_sequences} sequences...")
        latent_cache = self._collect_latent_activations(
            transcoder, activation_store, num_sequences
        )
        
        # Step 2: Sample non-dead latents
        dict_size = transcoder.dict_size if hasattr(transcoder, 'dict_size') else latent_cache['n_latents']
        non_dead = self._get_non_dead_latents(latent_cache, dict_size)
        sampled_latents = np.random.choice(
            non_dead, 
            size=min(self.n_latents_sample, len(non_dead)),
            replace=False
        )
        
        print(f"Evaluating {len(sampled_latents)} sampled latents...")
        
        # Step 3: Evaluate each latent
        scores = []
        for latent_id in tqdm(sampled_latents, desc="AutoInterp"):
            score = self._evaluate_single_latent(latent_id, latent_cache)
            if score is not None:
                scores.append(score)
        
        mean_score = np.mean(scores) if scores else 0.0
        
        print(f"\n✓ AutoInterp Score: {mean_score:.4f}")
        print(f"  Latents evaluated: {len(scores)}/{len(sampled_latents)}")
        print(f"  Backend: {self.llm_backend}")
        
        return mean_score, len(scores)
    
    def _collect_latent_activations(
        self,
        transcoder,
        activation_store,
        num_sequences: int
    ) -> Dict:
        """Collect latent activations and top-activating tokens."""
        print(f"  Collecting activations from {num_sequences} sequences...")
        
        sequences = []
        activations = []
        token_positions = []
        
        transcoder.eval()
        with torch.no_grad():
            for i in tqdm(range(num_sequences), desc="Collecting sequences"):
                try:
                    source_batch, target_batch = activation_store.next_batch()
                    source_batch = source_batch.to(transcoder.W_dec.device)
                    
                    # Get sparse features
                    sparse_features = transcoder.encode(source_batch)
                    
                    # Get tokens for context
                    tokens = activation_store.get_batch_tokens()
                    
                    # Store sequences and activations
                    for j in range(source_batch.shape[0]):
                        if j < tokens.shape[0]:  # Check bounds
                            sequences.append(tokens[j].cpu().tolist())
                            activations.append(sparse_features[j].cpu())
                            
                            # Find top-activating token positions for each latent
                            batch_token_positions = {}
                            for latent_id in range(sparse_features.shape[1]):
                                if sparse_features[j, latent_id] > 0.1:  # Threshold for activation
                                    batch_token_positions[latent_id] = j  # Position in sequence
                            token_positions.append(batch_token_positions)
                        
                except StopIteration:
                    break
        
        return {
            'n_latents': transcoder.dict_size if hasattr(transcoder, 'dict_size') else sparse_features.shape[1],
            'sequences': sequences,
            'activations': activations,
            'token_positions': token_positions,
        }
    
    def _get_non_dead_latents(self, cache: Dict, dict_size: int) -> np.ndarray:
        """Get indices of non-dead latents."""
        # Placeholder: return all latents
        return np.arange(dict_size)
    
    def _evaluate_single_latent(self, latent_id: int, cache: Dict) -> Optional[float]:
        """Evaluate single latent with LLM judge."""
        if self.llm_backend == "huggingface":
            return self._evaluate_with_huggingface(latent_id, cache)
        elif self.llm_backend == "openai":
            return self._evaluate_with_openai(latent_id, cache)
        elif self.llm_backend == "saebench":
            return self._evaluate_with_saebench(latent_id, cache)
        else:
            return None
    
    def _evaluate_with_huggingface(self, latent_id: int, cache: Dict) -> Optional[float]:
        """Evaluate using Hugging Face model."""
        if self.llm_model is None:
            return None
        
        try:
            # Step 1: Get top-activating sequences for this latent
            activations = cache['activations']
            sequences = cache['sequences']
            token_positions = cache['token_positions']
            
            # Get activation values for this latent across all sequences
            latent_activations = []
            for i, acts in enumerate(activations):
                if latent_id < len(acts):
                    latent_activations.append((i, acts[latent_id].item()))
            
            # Sort by activation strength
            latent_activations.sort(key=lambda x: x[1], reverse=True)
            
            # Select top-k sequences (k=2) and some importance-weighted ones
            top_k = 2
            top_sequences = latent_activations[:top_k]
            
            # Sample additional sequences proportional to activation
            remaining = latent_activations[top_k:]
            if remaining:
                # Sample 2 more sequences weighted by activation
                weights = [x[1] for x in remaining]
                if sum(weights) > 0:
                    weights = np.array(weights) / sum(weights)
                    sampled_indices = np.random.choice(
                        len(remaining), 
                        size=min(2, len(remaining)), 
                        replace=False, 
                        p=weights
                    )
                    iw_sequences = [remaining[i] for i in sampled_indices]
                else:
                    iw_sequences = remaining[:2]
            else:
                iw_sequences = []
            
            # Step 2: Generate feature description
            generation_pool = top_sequences + iw_sequences
            description = self._generate_description_hf(latent_id, generation_pool, sequences, token_positions)
            
            if description is None:
                return None
            
            # Step 3: Build test set (10 random + 2 max + 2 IW)
            test_sequences = []
            test_labels = []
            
            # 10 random sequences
            random_indices = np.random.choice(len(sequences), size=min(10, len(sequences)), replace=False)
            for idx in random_indices:
                test_sequences.append(idx)
                # Label: 1 if latent activates above threshold, 0 otherwise
                activation_val = activations[idx][latent_id].item() if latent_id < len(activations[idx]) else 0
                test_labels.append(1 if activation_val > 0.1 else 0)
            
            # 2 max activation sequences
            for seq_idx, _ in top_sequences[:2]:
                test_sequences.append(seq_idx)
                test_labels.append(1)
            
            # 2 importance-weighted sequences
            for seq_idx, _ in iw_sequences[:2]:
                test_sequences.append(seq_idx)
                test_labels.append(1)
            
            # Step 4: LLM prediction
            predictions = self._predict_activation_hf(description, test_sequences, sequences)
            
            if predictions is None:
                return None
            
            # Step 5: Compute accuracy
            correct = sum(1 for p, l in zip(predictions, test_labels) if p == l)
            accuracy = correct / len(test_labels) if test_labels else 0.0
            
            return accuracy
            
        except Exception as e:
            print(f"Error evaluating latent {latent_id}: {e}")
            return None
    
    def _generate_description_hf(self, latent_id: int, generation_pool: List, sequences: List, token_positions: List) -> Optional[str]:
        """Generate feature description using Hugging Face model."""
        try:
            # Create prompt with top-activating examples
            prompt = f"Given these examples where latent {latent_id} activates, describe what this latent represents:\n\n"
            
            for seq_idx, activation_val in generation_pool[:4]:  # Use top 4 examples
                if seq_idx < len(sequences):
                    tokens = sequences[seq_idx]
                    # Decode tokens to text
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    
                    # Highlight activating token if we have position info
                    if seq_idx < len(token_positions) and latent_id in token_positions[seq_idx]:
                        pos = token_positions[seq_idx][latent_id]
                        if pos < len(tokens):
                            highlighted_text = text[:pos] + f"<<{text[pos:pos+10]}>>" + text[pos+10:]
                        else:
                            highlighted_text = text
                    else:
                        highlighted_text = text
                    
                    prompt += f"Example: {highlighted_text}\n"
                    prompt += f"Activation: {activation_val:.3f}\n\n"
            
            prompt += "Description:"
            
            # Generate description using Hugging Face model
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return None
    
    def _predict_activation_hf(self, description: str, test_sequences: List[int], sequences: List) -> Optional[List[int]]:
        """Predict activation for test sequences using Hugging Face model."""
        try:
            predictions = []
            
            for seq_idx in test_sequences:
                if seq_idx >= len(sequences):
                    predictions.append(0)
                    continue
                
                tokens = sequences[seq_idx]
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
                # Create prediction prompt
                prompt = f"Description: {description}\n\n"
                prompt += f"Sequence: {text}\n\n"
                prompt += "Does this sequence activate the described latent? Answer yes or no:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                # Simple keyword matching for yes/no
                if "yes" in response.lower():
                    predictions.append(1)
                else:
                    predictions.append(0)
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting activation: {e}")
            return None
    
    def _evaluate_with_openai(self, latent_id: int, cache: Dict) -> Optional[float]:
        """Evaluate using OpenAI API."""
        # This is a placeholder - would implement actual OpenAI API calls
        return None
    
    def _evaluate_with_saebench(self, latent_id: int, cache: Dict) -> Optional[float]:
        """Evaluate using SAEBench API."""
        if self.saebench_client is None:
            return None
        # This is a placeholder - would implement actual SAEBench API calls
        return None


class AbsorptionEvaluator:
    """
    Absorption Score: First-letter probe task.
    
    Following SAEBench methodology (Chanin et al.):
    1. Train logistic probes on residual stream for first-letter classification
    2. Find main latents via k-sparse probing
    3. Detect absorption: when main latents + absorbing latents compensate
    4. Score = mean absorption ratio across test tokens
    
    SAEBench reports (1 - absorption) so higher is better.
    """
    
    def __init__(
        self,
        k_max: int = 10,
        tau_fs: float = 0.03,
        tau_pa: float = 0.0,
        tau_ps: float = -1.0,
        train_ratio: float = 0.8
    ):
        """
        Initialize Absorption evaluator.
        
        Args:
            k_max: Max k for k-sparse probing (default: 10)
            tau_fs: F1 improvement threshold for split detection (default: 0.03)
            tau_pa: Absorption compensation threshold (default: 0.0)
            tau_ps: Cosine threshold for absorbing candidates (default: -1.0)
            train_ratio: Train/test split ratio (default: 0.8)
        """
        self.k_max = k_max
        self.tau_fs = tau_fs
        self.tau_pa = tau_pa
        self.tau_ps = tau_ps
        self.train_ratio = train_ratio
        
        self.letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    
    @torch.no_grad()
    def evaluate(
        self,
        transcoder,
        model: HookedTransformer,
        layer: int,
        num_tokens: int = 10000
    ) -> Tuple[float, float, int]:
        """
        Evaluate Absorption score on first-letter task.
        
        Args:
            transcoder: Model with encode() and decoder matrix
            model: Base language model
            layer: Layer number
            num_tokens: Number of tokens to collect per letter
            
        Returns:
            (absorption_score, raw_absorption, n_letters_evaluated)
        """
        print("\n" + "="*80)
        print("📊 Absorption Score Evaluation (First-Letter Task)")
        print("="*80)
        
        # Step 1: Get vocabulary filtered to letter tokens
        letter_tokens = self._get_letter_tokens(model.tokenizer)
        
        if not letter_tokens:
            print("⚠️  No suitable letter tokens found in vocabulary")
            return 0.0, 1.0, 0
        
        print(f"Found {sum(len(v) for v in letter_tokens.values())} letter tokens")
        
        # Step 2: Collect residual stream activations for these tokens
        print("Collecting residual stream activations...")
        resid_acts, labels = self._collect_residual_activations(
            model, layer, letter_tokens, num_tokens
        )
        
        # Step 3: Train logistic probes (per letter)
        print("Training logistic regression probes...")
        probes = self._train_letter_probes(resid_acts, labels)
        
        # Step 4: Collect SAE/transcoder activations on same tokens
        print("Collecting transcoder latent activations...")
        latent_acts = self._collect_latent_activations_for_tokens(
            transcoder, model, layer, letter_tokens, num_tokens
        )
        
        # Step 5: Find main latents via k-sparse probing (train split)
        print("Finding main latents via k-sparse probing...")
        main_latents = self._find_main_latents_k_sparse(
            latent_acts, labels, probes
        )
        
        # Step 6: Compute absorption on test split
        print("Computing absorption scores...")
        absorption_scores = self._compute_absorption_scores(
            transcoder, latent_acts, resid_acts, labels, probes, main_latents
        )
        
        # Step 7: Aggregate results
        raw_absorption = np.mean(absorption_scores) if absorption_scores else 1.0
        absorption_score = 1.0 - raw_absorption  # SAEBench convention: higher is better
        
        print(f"\n✓ Absorption Score: {absorption_score:.4f} (1 - {raw_absorption:.4f})")
        print(f"  Letters evaluated: {len(probes)}")
        
        return absorption_score, raw_absorption, len(probes)
    
    def _get_letter_tokens(self, tokenizer) -> Dict[str, List[int]]:
        """Get vocabulary tokens corresponding to single letters."""
        letter_tokens = {letter: [] for letter in self.letters}
        
        # Filter vocabulary to tokens that are single letters (with optional leading space)
        for token_id in range(len(tokenizer)):
            try:
                text = tokenizer.decode([token_id]).strip().lower()
                if len(text) == 1 and text in self.letters:
                    letter_tokens[text].append(token_id)
            except:
                continue
        
        # Remove letters with no tokens
        letter_tokens = {k: v for k, v in letter_tokens.items() if v}
        
        return letter_tokens
    
    def _collect_residual_activations(
        self,
        model: HookedTransformer,
        layer: int,
        letter_tokens: Dict[str, List[int]],
        num_tokens: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect residual stream activations for letter tokens."""
        print(f"  Collecting residual activations for {len(letter_tokens)} letters...")
        
        resid_acts = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for letter_idx, (letter, token_ids) in enumerate(letter_tokens.items()):
                print(f"    Processing letter '{letter}' ({len(token_ids)} tokens)")
                
                # Sample tokens for this letter
                sampled_tokens = np.random.choice(token_ids, size=min(num_tokens, len(token_ids)), replace=False)
                
                for token_id in sampled_tokens:
                    # Create input sequence with this token
                    input_tokens = torch.tensor([[token_id]], device=model.cfg.device)
                    
                    # Run model and get residual stream activation at the specified layer
                    resid_hook_name = f"blocks.{layer}.hook_resid_pre"
                    
                    # Get residual activation
                    resid_cache = {}
                    def resid_hook(activation, hook):
                        resid_cache["activation"] = activation
                        return activation
                    
                    model.run_with_hooks(
                        input_tokens,
                        fwd_hooks=[(resid_hook_name, resid_hook)]
                    )
                    
                    resid_activation = resid_cache["activation"]
                    
                    # Store activation and label
                    resid_acts.append(resid_activation[0, 0].float().cpu().numpy())  # Convert to float32
                    labels.append(letter_idx)
        
        return np.array(resid_acts), np.array(labels)
    
    def _train_letter_probes(
        self,
        resid_acts: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, LogisticRegression]:
        """Train logistic regression probes for each letter."""
        probes = {}
        
        # Split train/test
        n_train = int(len(resid_acts) * self.train_ratio)
        X_train, y_train = resid_acts[:n_train], labels[:n_train]
        
        for i, letter in enumerate(self.letters):
            # Binary classification: letter vs others
            y_binary = (y_train == i).astype(int)
            
            if y_binary.sum() < 10:  # Need sufficient examples
                continue
            
            probe = LogisticRegression(max_iter=1000, random_state=42)
            probe.fit(X_train, y_binary)
            probes[letter] = probe
        
        return probes
    
    def _collect_latent_activations_for_tokens(
        self,
        transcoder,
        model: HookedTransformer,
        layer: int,
        letter_tokens: Dict[str, List[int]],
        num_tokens: int
    ) -> np.ndarray:
        """Collect transcoder latent activations for letter tokens."""
        # Placeholder
        n_samples = sum(min(len(v), num_tokens) for v in letter_tokens.values())
        dict_size = transcoder.dict_size if hasattr(transcoder, 'dict_size') else 16384
        
        return np.random.randn(n_samples, dict_size).astype(np.float32)
    
    def _find_main_latents_k_sparse(
        self,
        latent_acts: np.ndarray,
        labels: np.ndarray,
        probes: Dict[str, LogisticRegression]
    ) -> Dict[str, List[int]]:
        """Find main latents via greedy k-sparse probing."""
        main_latents = {}
        
        # For each letter, do greedy k-sparse search
        for letter in probes.keys():
            main_latents[letter] = self._greedy_k_sparse_for_letter(
                latent_acts, labels, letter
            )
        
        return main_latents
    
    def _greedy_k_sparse_for_letter(
        self,
        latent_acts: np.ndarray,
        labels: np.ndarray,
        letter: str
    ) -> List[int]:
        """Greedy k-sparse probing for a single letter."""
        letter_idx = ord(letter) - ord('a')
        y_binary = (labels == letter_idx).astype(int)
        
        if y_binary.sum() < 10:  # Need sufficient examples
            return []
        
        main_latents = []
        best_f1 = 0.0
        
        # Start with k=1, greedily add latents
        for k in range(1, min(self.k_max + 1, latent_acts.shape[1] + 1)):
            # Try all possible single latents or combinations
            if k == 1:
                candidates = [(i,) for i in range(latent_acts.shape[1])]
            else:
                # For k>1, try adding each remaining latent to current best set
                candidates = []
                for i in range(latent_acts.shape[1]):
                    if i not in main_latents:
                        candidates.append(tuple(main_latents + [i]))
            
            best_candidate = None
            best_candidate_f1 = best_f1
            
            for candidate in candidates:
                # Train probe on candidate latents
                X_subset = latent_acts[:, list(candidate)]
                
                if X_subset.shape[1] == 0:
                    continue
                
                probe = LogisticRegression(max_iter=1000, random_state=42)
                probe.fit(X_subset, y_binary)
                
                # Evaluate F1 score
                y_pred = probe.predict(X_subset)
                f1 = f1_score(y_binary, y_pred, zero_division=0)
                
                if f1 > best_candidate_f1:
                    best_candidate = candidate
                    best_candidate_f1 = f1
            
            # Check if improvement is significant
            if best_candidate_f1 - best_f1 > self.tau_fs:
                main_latents = list(best_candidate)
                best_f1 = best_candidate_f1
            else:
                break  # No significant improvement, stop
        
        return main_latents
    
    def _compute_absorption_scores(
        self,
        transcoder,
        latent_acts: np.ndarray,
        resid_acts: np.ndarray,
        labels: np.ndarray,
        probes: Dict[str, LogisticRegression],
        main_latents: Dict[str, List[int]]
    ) -> List[float]:
        """Compute per-token absorption scores."""
        scores = []
        
        # Get decoder matrix (unit-normalized columns)
        if hasattr(transcoder, 'W_dec'):
            decoder = F.normalize(transcoder.W_dec.float(), p=2, dim=-1).detach().cpu().numpy()
        else:
            return [0.0]  # Can't compute without decoder
        
        # Test split
        n_train = int(len(resid_acts) * self.train_ratio)
        test_acts = latent_acts[n_train:]
        test_resid = resid_acts[n_train:]
        test_labels = labels[n_train:]
        
        # For each test token
        for i in range(len(test_acts)):
            letter_idx = test_labels[i]
            if letter_idx >= len(self.letters):
                continue
            
            letter = self.letters[letter_idx]
            if letter not in probes or letter not in main_latents:
                continue
            
            probe = probes[letter]
            probe_direction = probe.coef_[0] / np.linalg.norm(probe.coef_[0])
            
            # Check probe correctness
            if probe.predict([test_resid[i]])[0] != 1:
                continue
            
            # Compute projections
            a_model_proj = np.dot(test_resid[i], probe_direction)
            
            # Main latents contribution
            main_sum = 0.0
            for latent_id in main_latents[letter]:
                if latent_id < len(decoder):
                    a_i = test_acts[i, latent_id]
                    d_i = decoder[latent_id]
                    main_sum += a_i * np.dot(d_i, probe_direction)
            
            # Condition 1: main too small
            if main_sum >= a_model_proj:
                scores.append(0.0)
                continue
            
            # Find absorbing latents
            A_max = decoder.shape[0]
            absorbers = self._find_absorbing_latents(
                test_acts[i], decoder, probe_direction, A_max, main_latents[letter]
            )
            
            # Compute absorption
            abs_sum = 0.0
            for latent_id in absorbers:
                a_i = test_acts[i, latent_id]
                d_i = decoder[latent_id]
                abs_sum += a_i * np.dot(d_i, probe_direction)
            
            # Condition 2: compensation threshold
            if abs_sum / a_model_proj < self.tau_pa:
                scores.append(0.0)
                continue
            
            # Absorption score for this token
            absorption = abs_sum / (abs_sum + main_sum)
            scores.append(absorption)
        
        return scores
    
    def _find_absorbing_latents(
        self,
        acts: np.ndarray,
        decoder: np.ndarray,
        probe_direction: np.ndarray,
        A_max: int,
        main_latents: List[int]
    ) -> List[int]:
        """Find top absorbing latents by probe projection."""
        # Compute projections for all latents
        projections = []
        for i in range(len(decoder)):
            if i in main_latents:
                continue
            proj = np.dot(decoder[i], probe_direction)
            # Only positive projections
            if proj > 0 and proj >= self.tau_ps:
                projections.append((i, proj))
        
        # Sort by projection, take top A_max
        projections.sort(key=lambda x: x[1], reverse=True)
        absorbers = [i for i, _ in projections[:A_max]]
        
        return absorbers


def load_transcoder_for_eval(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    is_google: bool = False,
    google_dir: str = None
):
    """Load transcoder for evaluation."""
    if is_google:
        print(f"\n📥 Loading Google transcoder from: {google_dir}")
        
        # Find the best L0 match (closest to our transcoder's L0)
        l0_dirs = [d for d in os.listdir(google_dir) if d.startswith("average_l0_")]
        if not l0_dirs:
            raise FileNotFoundError(f"No L0 directories found in {google_dir}")
        
        # Sort by L0 value and pick the closest to our expected L0 (~96)
        l0_values = []
        for d in l0_dirs:
            try:
                l0_val = int(d.split("_")[-1])
                l0_values.append((l0_val, d))
            except:
                continue
        
        l0_values.sort(key=lambda x: x[0])
        target_l0 = 96  # Our transcoder's mean L0
        best_l0_dir = min(l0_values, key=lambda x: abs(x[0] - target_l0))[1]
        
        print(f"  - Selected L0 directory: {best_l0_dir} (target: {target_l0})")
        
        transcoder = GemmaScopeTranscoderAdapter(
            layer=8,  # Layer 8
            repo_dir=os.path.join(google_dir, best_l0_dir),
            device=device,
            dtype=dtype,
            allow_patterns=["**/*.npz"]  # Allow .npz files
        )
        
        print(f"✓ Google transcoder loaded successfully")
        print(f"  - Layer: 8")
        print(f"  - Weight shape: {transcoder.W.shape}")
        
        # Create a wrapper to make it compatible with our evaluator
        class GoogleTranscoderWrapper:
            def __init__(self, google_transcoder):
                self.transcoder = google_transcoder
                self.dict_size = 16384  # Google's transcoder size
            
            def encode(self, x):
                # Google's transcoder is just a linear mapping, no sparse features
                return self.transcoder(x)
            
            def decode(self, x):
                # For Google's transcoder, encode and decode are the same
                return x
            
            @property
            def W_dec(self):
                # Return the weight matrix as "decoder" for compatibility
                return self.transcoder.W.T.float()  # Transpose to match expected shape
        
        wrapper = GoogleTranscoderWrapper(transcoder)
        cfg = {"dict_size": 16384, "top_k": 96, "model_name": "gemma-2-2b"}
        
        return wrapper, cfg
    else:
        print(f"\n📥 Loading Matryoshka transcoder from: {checkpoint_path}")
        
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        
        # Fix data types
        if isinstance(cfg.get("dtype"), str):
            try:
                cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
            except:
                cfg["dtype"] = dtype
        
        # Fix prefix_sizes
        if isinstance(cfg.get("prefix_sizes"), str):
            try:
                cfg["prefix_sizes"] = json.loads(cfg["prefix_sizes"])
            except:
                import ast
                cfg["prefix_sizes"] = ast.literal_eval(cfg["prefix_sizes"])
        
        # Ensure numeric fields
        for key in ["dict_size", "top_k"]:
            if key in cfg:
                cfg[key] = int(cfg[key])
        
        cfg["device"] = device
        cfg["dtype"] = dtype
        
        # Create transcoder
        from src.models.sae import MatryoshkaTranscoder
        transcoder = MatryoshkaTranscoder(cfg).to(device=device, dtype=dtype)
        
        # Load state dict
        state_path = os.path.join(checkpoint_path, "sae.pt")
        if not os.path.exists(state_path):
            state_path = os.path.join(checkpoint_path, "checkpoint.pt")
        transcoder.load_state_dict(torch.load(state_path, map_location=device))
        transcoder.eval()
        
        print(f"✓ Transcoder loaded (dict_size={cfg['dict_size']})")
        
        return transcoder, cfg


def main():
    parser = argparse.ArgumentParser(
        description="SAEBench AutoInterp and Absorption evaluation"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to transcoder checkpoint")
    parser.add_argument("--google_dir", type=str, default=None,
                       help="Path to Google Gemma Scope transcoder directory")
    parser.add_argument("--layer", type=int, default=17,
                       help="Layer to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output_dir", type=str, default="analysis_results/saebench",
                       help="Output directory")
    parser.add_argument("--llm_backend", type=str, default="huggingface",
                       choices=["openai", "huggingface", "saebench"],
                       help="LLM backend for AutoInterp (default: huggingface)")
    parser.add_argument("--llm_model", type=str, default="microsoft/DialoGPT-medium",
                       help="Hugging Face model name (default: microsoft/DialoGPT-medium)")
    parser.add_argument("--skip_autointerp", action="store_true",
                       help="Skip AutoInterp entirely")
    
    args = parser.parse_args()
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]
    
    print("="*80)
    print("🔬 SAEBench Evaluation: AutoInterp + Absorption")
    print("="*80)
    
    # Load model
    print(f"\n📥 Loading Gemma-2-2B...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device, dtype=dtype)
    
    # Load transcoder
    if args.google_dir:
        transcoder, cfg = load_transcoder_for_eval(
            args.checkpoint, device, dtype, is_google=True, google_dir=args.google_dir
        )
    else:
        transcoder, cfg = load_transcoder_for_eval(args.checkpoint, device, dtype)
    
    # Compute mean L0 (would need actual evaluation, using placeholder)
    mean_l0 = cfg.get('top_k', 96)
    
    # Create activation store for AutoInterp
    print(f"\n📊 Creating activation store for AutoInterp...")
    store_cfg = get_default_cfg()
    store_cfg["model_name"] = "gemma-2-2b"
    store_cfg["device"] = device
    store_cfg["dtype"] = dtype
    store_cfg["dataset_path"] = "HuggingFaceFW/fineweb-edu"
    store_cfg["batch_size"] = 256
    store_cfg["seq_len"] = 64
    store_cfg["model_batch_size"] = 2
    store_cfg["num_batches_in_buffer"] = 1
    
    store_cfg = create_transcoder_config(
        store_cfg,
        source_layer=args.layer,
        target_layer=args.layer,
        source_site="mlp_in",
        target_site="mlp_out"
    )
    
    store_cfg["source_act_size"] = 2304
    store_cfg["target_act_size"] = 2304
    store_cfg["act_size"] = 2304
    
    activation_store = TranscoderActivationsStore(model, store_cfg)
    print("✓ Activation store created")
    
    # AutoInterp evaluation
    autointerp_score = None
    autointerp_n = 0
    
    if not args.skip_autointerp:
        autointerp_eval = AutoInterpEvaluator(
            llm_backend=args.llm_backend,
            model_name=args.llm_model,
            device=args.device
        )
        
        print(f"\nUsing {args.llm_backend} backend with model: {args.llm_model}")
        
        # Run AutoInterp evaluation
        autointerp_score, autointerp_n = autointerp_eval.evaluate(
            transcoder, activation_store, 1000
        )
    
    # Absorption evaluation
    absorption_eval = AbsorptionEvaluator()
    absorption_score, raw_absorption, n_letters = absorption_eval.evaluate(
        transcoder, model, args.layer
    )
    
    # Compile results
    metrics = SAEBenchMetrics(
        autointerp_score=autointerp_score,
        autointerp_latents_evaluated=autointerp_n,
        absorption_score=absorption_score,
        raw_absorption=raw_absorption,
        letters_evaluated=n_letters,
        mean_l0=mean_l0,
        dict_size=cfg['dict_size'],
        model_name=cfg.get('model_name', 'gemma-2-2b')
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "saebench_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    if autointerp_score is not None:
        print(f"AutoInterp Score:    {autointerp_score:.4f}")
        print(f"  (Latents evaluated: {autointerp_n})")
    print(f"Absorption Score:    {absorption_score:.4f} (SAEBench: 1 - {raw_absorption:.4f})")
    print(f"  (Letters evaluated: {n_letters})")
    print(f"Mean L0:             {mean_l0:.1f}")
    print("="*80)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    print("\n" + "="*80)
    print("⚠️  IMPLEMENTATION NOTE")
    print("="*80)
    print("This is a SKELETON implementation showing the SAEBench methodology.")
    print("Full implementation requires:")
    print()
    print("For AutoInterp:")
    print("  1. Sequence collection from OpenWebText (ctx=128)")
    print("  2. Latent activation caching with token positions")
    print("  3. LLM API integration (OpenAI GPT-4o-mini)")
    print("  4. Top-k + importance-weighted sampling")
    print("  5. Test set construction (10 random + 2 max + 2 IW)")
    print()
    print("For Absorption:")
    print("  1. Letter token vocabulary filtering")
    print("  2. Residual stream activation collection")
    print("  3. Logistic regression probe training (per letter)")
    print("  4. K-sparse probing for main latent detection")
    print("  5. Absorption computation on test split")
    print()
    print("Refer to SAEBench paper (arXiv:2503.09532) and")
    print("GitHub: https://github.com/adamkarvonen/SAEBench")
    print("="*80)


if __name__ == "__main__":
    main()

