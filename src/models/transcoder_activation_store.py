import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import load_dataset
from utils.config import get_hook_name


class TranscoderActivationsStore:
    """
    Activation store for training transcoders.
    
    Collects paired activations from source and target layers/sites
    to train cross-layer feature transformations.
    """
    
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        # Try to load dataset with fallback options
        try:
            self.dataset = iter(load_dataset(
                cfg["dataset_path"], 
                split=cfg.get("dataset_split", "train"),
                streaming=True, 
                trust_remote_code=False  # Remove trust_remote_code
            ))
        except Exception as e:
            print(f"Warning: Failed to load {cfg['dataset_path']}: {e}")
            print("Falling back to C4 dataset...")
            self.dataset = iter(load_dataset(
                "c4", 
                split="train",
                streaming=True,
                trust_remote_code=False
            ))
        
        # Source and target hook points
        self.source_hook_point = cfg["source_hook_point"]
        self.target_hook_point = cfg["target_hook_point"]
        
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.tokens_column = self._get_tokens_column()
        self.config = cfg
        self.tokenizer = model.tokenizer
        
        # Layer to stop at (target layer + 1)
        self.stop_at_layer = cfg.get("target_layer", cfg["layer"]) + 1

    def _get_tokens_column(self):
        """Determine which column contains the text/tokens"""
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):
        """Generate a batch of tokens from the dataset"""
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], 
                    truncate=True, 
                    move_to_device=True, 
                    prepend_bos=True
                ).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        
        token_tensor = torch.tensor(
            all_tokens,
            dtype=torch.long,
            device=self.device
        )[:self.model_batch_size * self.context_size]
        
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_paired_activations(self, batch_tokens: torch.Tensor):
        """
        Get activations from both source and target hook points.
        
        For Gemma-2 models with ln2_normalized source, applies RMSNorm scaling
        to get the actual MLP input (post-norm, post-scale).
        
        Args:
            batch_tokens: Tokenized input (batch_size, seq_len)
        
        Returns:
            Tuple of (source_acts, target_acts)
        """
        # Prepare names to cache, with optional fallback for mlp_in
        names = [self.target_hook_point, self.source_hook_point]
        fallback_ln2_name = None
        layer_for_fallback = None
        if "hook_mlp_in" in self.source_hook_point:
            try:
                parts = self.source_hook_point.split(".")
                layer_for_fallback = int(parts[1])
                fallback_ln2_name = f"blocks.{layer_for_fallback}.ln2.hook_normalized"
                names.append(fallback_ln2_name)
            except Exception:
                pass

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=names,
                stop_at_layer=self.stop_at_layer,
            )

        # Try direct mlp_in; if not present, reconstruct from ln2 normalized * ln2.w
        try:
            source_acts = cache[self.source_hook_point]
        except KeyError:
            if fallback_ln2_name is None or layer_for_fallback is None:
                raise
            ln2_norm = cache[fallback_ln2_name]
            # Try multiple common attribute names for RMSNorm scale
            scale_param = None
            ln2_mod = self.model.blocks[layer_for_fallback].ln2
            for attr in ["w", "weight", "scale", "g"]:
                if hasattr(ln2_mod, attr):
                    scale_param = getattr(ln2_mod, attr)
                    break
            # If no explicit scale parameter found, fall back to normalized activations
            if scale_param is None:
                source_acts = ln2_norm
            else:
                source_acts = ln2_norm * scale_param

        target_acts = cache[self.target_hook_point]

        # Ensure dtypes match training dtype (e.g., bfloat16) to avoid matmul errors
        desired_dtype = self.config.get("dtype", None)
        if desired_dtype is not None:
            try:
                source_acts = source_acts.to(dtype=desired_dtype)
                target_acts = target_acts.to(dtype=desired_dtype)
            except Exception:
                pass

        return source_acts, target_acts

    def _fill_buffer(self):
        """
        Fill the activation buffer with paired source-target activations.
        
        Returns:
            Tuple of (source_buffer, target_buffer)
        """
        all_source_acts = []
        all_target_acts = []
        
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            source_acts, target_acts = self.get_paired_activations(batch_tokens)
            
            # Flatten (batch_size, seq_len, act_size) -> (batch_size * seq_len, act_size)
            all_source_acts.append(
                source_acts.reshape(-1, self.config.get("source_act_size", self.config["act_size"]))
            )
            all_target_acts.append(
                target_acts.reshape(-1, self.config.get("target_act_size", self.config["act_size"]))
            )
        
        return (
            torch.cat(all_source_acts, dim=0),
            torch.cat(all_target_acts, dim=0)
        )

    def _get_dataloader(self):
        """Create a DataLoader from the current buffer"""
        dataset = TensorDataset(self.source_buffer, self.target_buffer)
        return DataLoader(
            dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True
        )

    def next_batch(self):
        """
        Get the next batch of paired activations.
        
        Returns:
            Tuple of (source_batch, target_batch)
        """
        try:
            return next(self.dataloader_iter)
        except (StopIteration, AttributeError):
            # Refill buffer when exhausted
            self.source_buffer, self.target_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)

    def get_batch_with_tokens(self):
        """Get a fresh batch of activations with corresponding tokens for sample collection."""
        batch_tokens = self.get_batch_tokens()
        source_acts, target_acts = self.get_paired_activations(batch_tokens)
        # Return source activations and tokens for sample collection
        return source_acts, batch_tokens


def create_transcoder_config(base_cfg, source_layer, target_layer, source_site, target_site):
    """
    Helper function to create transcoder configuration.
    
    Automatically handles model-specific hook naming (e.g., Gemma-2 vs GPT-2).
    
    Args:
        base_cfg: Base configuration dictionary
        source_layer: Source layer number
        target_layer: Target layer number  
        source_site: Source site (e.g., "resid_pre", "resid_mid", "resid_post", "mlp_out", "attn_out")
        target_site: Target site
    
    Returns:
        Updated configuration dictionary
    """
    cfg = base_cfg.copy()
    
    # Set layer information
    cfg["source_layer"] = source_layer
    cfg["target_layer"] = target_layer
    cfg["source_site"] = source_site
    cfg["target_site"] = target_site
    
    # Create hook points using model-aware helper
    # This handles Gemma-2 vs GPT-2 naming differences
    cfg["source_hook_point"] = get_hook_name(source_site, source_layer, cfg["model_name"])
    cfg["target_hook_point"] = get_hook_name(target_site, target_layer, cfg["model_name"])
    
    # Set layer for hook_point (for compatibility with existing code)
    cfg["layer"] = source_layer
    cfg["hook_point"] = cfg["source_hook_point"]
    
    # Update name
    cfg["name"] = (
        f"{cfg['model_name']}_{cfg['source_hook_point']}_to_{cfg['target_hook_point']}_"
        f"{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    )
    
    return cfg

