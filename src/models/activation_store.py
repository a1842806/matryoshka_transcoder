import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import Dataset, load_dataset
import tqdm

class ActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        # Try to load dataset with fallback options
        try:
            self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", 
                                             streaming=True, trust_remote_code=False))
        except Exception as e:
            print(f"Warning: Failed to load {cfg['dataset_path']}: {e}")
            print("Falling back to C4 dataset...")
            self.dataset = iter(load_dataset("c4", split="train", 
                                             streaming=True, trust_remote_code=False))
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.tokens_column = self._get_tokens_column()
        self.config = cfg
        self.tokenizer = model.tokenizer
        
        # Store recent tokens for activation sample collection
        self.store_tokens = cfg.get("store_tokens_for_samples", False)
        self.current_tokens = None

    def _get_tokens_column(self):
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
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(batch["text"], 
                                              truncate=True, 
                                              move_to_device=True, 
                                              prepend_bos=True).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(
            all_tokens,
            dtype=torch.long,
            device=self.device
        )[:self.model_batch_size * self.context_size]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.config["layer"] +1,
            )
        return cache[self.hook_point]

    def _fill_buffer(self):
        all_activations = []
        all_tokens = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens).reshape(-1, self.config["act_size"])
            all_activations.append(activations)
            if self.store_tokens:
                all_tokens.append(batch_tokens)
        
        # Store tokens if needed for sample collection
        if self.store_tokens and all_tokens:
            self.current_tokens = torch.cat(all_tokens, dim=0)
        
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        return DataLoader(TensorDataset(self.activation_buffer), 
                          batch_size=self.config["batch_size"], 
                          shuffle=True)

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]
    
    def get_batch_with_tokens(self):
        """Get a fresh batch of activations with corresponding tokens for sample collection."""
        batch_tokens = self.get_batch_tokens()
        activations = self.get_activations(batch_tokens)
        return activations, batch_tokens

