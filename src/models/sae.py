git import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import compute_fvu


class BaseAutoencoder(nn.Module):
    """Base class for autoencoder models."""

    def __init__(self, cfg):
        super().__init__()

        self.config = cfg
        torch.manual_seed(self.config["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.config["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.config["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config["act_size"], self.config["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.config["dict_size"], self.config["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros((self.config["dict_size"],)).to(
            cfg["device"]
        )

        self.to(cfg["dtype"]).to(cfg["device"])

    def preprocess_input(self, x):
        if self.config["input_unit_norm"]:
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        if self.config["input_unit_norm"]:
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    @torch.no_grad()
    def update_inactive_features(self, acts):
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def encode(self, x):
        raise NotImplementedError("Encode method must be implemented by subclasses")


class MatryoshkaTranscoder(BaseAutoencoder):
    """
    Matryoshka Transcoder: Maps activations from source layer to target layer
    using hierarchical feature groups at multiple abstraction levels.
    
    Key differences from SAE:
    - Input: activations from source layer (e.g., resid_mid)
    - Output: reconstruction of target layer (e.g., mlp_out)
    - Loss: measures quality of cross-layer transformation
    """
    
    def __init__(self, cfg):
        # Don't call super().__init__ yet, we need to handle dimensions first
        nn.Module.__init__(self)
        
        self.config = cfg
        torch.manual_seed(self.config["seed"])
        
        # Matryoshka group configuration - NESTED GROUPS
        # Each group includes all features from previous groups
        # Example: group_sizes=[1152, 2304, 4608, 10368]
        #   Group 0: features 0-1151 (1152 features)
        #   Group 1: features 0-3455 (1152+2304 features)
        #   Group 2: features 0-8063 (1152+2304+4608 features)
        #   Group 3: features 0-18431 (all features)
        total_dict_size = sum(cfg["group_sizes"])
        self.group_sizes = cfg["group_sizes"]
        self.group_indices = [0] + list(torch.cumsum(torch.tensor(cfg["group_sizes"]), dim=0))
        self.active_groups = len(cfg["group_sizes"])
        
        # Source and target dimensions (may differ for cross-layer mapping)
        self.source_act_size = cfg.get("source_act_size", cfg["act_size"])
        self.target_act_size = cfg.get("target_act_size", cfg["act_size"])
        
        # Encoder: source layer → latent features
        self.b_enc = nn.Parameter(torch.zeros(total_dict_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.source_act_size, total_dict_size)
            )
        )
        
        # Decoder: latent features → target layer
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(total_dict_size, self.target_act_size)
            )
        )
        self.b_dec = nn.Parameter(torch.zeros(self.target_act_size))
        
        # Normalize decoder weights
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        
        # Dead feature tracking
        self.num_batches_not_active = torch.zeros(total_dict_size, device=cfg["device"])
        self.register_buffer('threshold', torch.tensor(0.0))
        
        self.to(cfg["dtype"]).to(cfg["device"])
    
    def preprocess_input(self, x):
        """Preprocess source activations (optional normalization)"""
        if self.config.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            x_std = x.std(dim=-1, keepdim=True)
            x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        return x, None, None
    
    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        """Postprocess target reconstruction"""
        if self.config.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """Maintain unit norm constraint on decoder weights"""
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed
    
    @torch.no_grad()
    def update_inactive_features(self, acts):
        """Track dead features"""
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0
    
    def compute_activations(self, x_source):
        """Encode source layer activations to sparse latent features"""
        pre_acts = x_source @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            # Global TopK across entire batch
            acts_topk = torch.topk(
                acts.flatten(), 
                self.config["top_k"] * x_source.shape[0], 
                dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
            self.update_threshold(acts_topk)
        else:
            # Use threshold for inference
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk
    
    def forward(self, x_source, x_target):
        """
        Forward pass: encode source → sparse features → decode to target reconstruction
        
        Args:
            x_source: Source layer activations (e.g., resid_mid)
            x_target: Target layer activations (e.g., mlp_out) - for loss computation
        
        Returns:
            Dictionary with reconstruction, loss components, and metrics
        """
        # Preprocess source activations
        x_source, x_mean, x_std = self.preprocess_input(x_source)
        self.x_mean = x_mean
        self.x_std = x_std
        
        # Encode to sparse features
        all_acts, all_acts_topk = self.compute_activations(x_source)
        
        # Decode using hierarchical NESTED groups (each group includes all previous groups)
        # This matches the original Matryoshka paper design
        intermediate_reconstructs = []
        
        for i in range(self.active_groups):
            # NESTED: Always start from 0, end at cumulative size
            start_idx = 0
            end_idx = self.group_indices[i+1]
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            acts_topk_slice = all_acts_topk[:, start_idx:end_idx]
            x_reconstruct = acts_topk_slice @ W_dec_slice + self.b_dec
            intermediate_reconstructs.append(x_reconstruct)
        
        # Postprocess reconstruction
        x_reconstruct = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        # Update dead feature tracking
        self.update_inactive_features(all_acts_topk)
        
        # Compute loss components
        l2_loss = (x_reconstruct - x_target).pow(2).mean()
        l1_loss = all_acts_topk.abs().mean()
        l0_norm = (all_acts_topk > 0).sum(dim=-1).float().mean()
        l1_norm = all_acts_topk.abs().sum(dim=-1).mean()
        
        # Auxiliary loss for dead features
        aux_loss = self.get_auxiliary_loss(x_target, x_reconstruct, all_acts)
        
        # Total loss
        loss = l2_loss + self.config["l1_coeff"] * l1_loss + aux_loss
        
        # Compute FVU for each group
        fvus = []
        for intermediate_reconstruct in intermediate_reconstructs:
            intermediate_reconstruct = self.postprocess_output(intermediate_reconstruct, x_mean, x_std)
            fvu = compute_fvu(x_target, intermediate_reconstruct)
            fvus.append(fvu)
        
        # Compute final FVU
        final_fvu = compute_fvu(x_target, x_reconstruct)
        
        return {
            "transcoder_out": x_reconstruct,
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "fvu": final_fvu,
            "fvu_min": min(fvus) if fvus else final_fvu,
            "fvu_max": max(fvus) if fvus else final_fvu,
            "fvu_mean": sum(fvus) / len(fvus) if fvus else final_fvu,
        }
    
    def get_loss_dict(self, x_target, x_reconstruct, all_acts, all_acts_topk, 
                     x_mean, x_std, intermediate_reconstructs):
        """
        Compute loss dictionary for training.
        
        Args:
            x_target: Target activations
            x_reconstruct: Reconstructed activations
            all_acts: All activations (before TopK)
            all_acts_topk: TopK activations
            x_mean, x_std: Preprocessing statistics
            intermediate_reconstructs: Intermediate reconstructions from each group
        
        Returns:
            Dictionary with loss components and metrics
        """
        l2_loss = (x_reconstruct - x_target).pow(2).mean()
        l1_loss = all_acts_topk.abs().mean()
        l0_norm = (all_acts_topk > 0).sum(dim=-1).float().mean()
        l1_norm = all_acts_topk.abs().sum(dim=-1).mean()
        
        aux_loss = self.get_auxiliary_loss(x_target, x_reconstruct, all_acts)
        loss = l2_loss + self.config["l1_coeff"] * l1_loss + aux_loss
        
        # Compute FVU for each group
        fvus = []
        for intermediate_reconstruct in intermediate_reconstructs:
            intermediate_reconstruct = self.postprocess_output(intermediate_reconstruct, x_mean, x_std)
            fvu = compute_fvu(x_target, intermediate_reconstruct)
            fvus.append(fvu)
        
        # Compute final FVU
        final_fvu = compute_fvu(x_target, x_reconstruct)
        
        return {
            "loss": loss,
            "l2_loss": l2_loss,
            "l1_loss": l1_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "fvu": final_fvu,
            "fvu_min": min(fvus) if fvus else final_fvu,
            "fvu_max": max(fvus) if fvus else final_fvu,
            "fvu_mean": sum(fvus) / len(fvus) if fvus else final_fvu,
        }
    
    def get_auxiliary_loss(self, x_target, x_reconstruct, acts):
        """
        Auxiliary loss to revive dead features.
        
        Args:
            x_target: Target activations
            x_reconstruct: Reconstructed activations
            acts: All activations (before TopK)
        
        Returns:
            Auxiliary loss tensor
        """
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x_target.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.config["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.config["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        return torch.tensor(0, dtype=x_target.dtype, device=x_target.device)
    
    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        """
        Update threshold for inference based on training activations.
        
        Args:
            acts_topk: TopK activations from training
            lr: Learning rate for threshold update
        """
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive
    
    def encode(self, x_source):
        """
        Encode source activations to sparse features.
        
        Args:
            x_source: Source layer activations
        
        Returns:
            Sparse feature activations
        """
        original_shape = x_source.shape
        x_source, x_mean, x_std = self.preprocess_input(x_source)
        self.x_mean = x_mean
        self.x_std = x_std
        
        x_source = x_source.reshape(-1, x_source.shape[-1])
        _, result = self.compute_activations(x_source)
        
        # Zero out features beyond active groups
        max_act_index = self.group_indices[self.active_groups]
        result[:, max_act_index:] = 0
        
        if len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1], -1)
        return result
    
    def decode(self, acts_topk):
        """
        Decode sparse features to target layer reconstruction.
        Uses NESTED groups: only features up to the active group size.
        
        Args:
            acts_topk: Sparse feature activations
        
        Returns:
            Reconstruction of target layer
        """
        # For nested groups, use only features up to the current active group size
        max_feature_idx = self.group_indices[self.active_groups]
        acts_topk_active = acts_topk[:, :max_feature_idx]
        W_dec_active = self.W_dec[:max_feature_idx, :]
        
        reconstruct = acts_topk_active @ W_dec_active + self.b_dec
        return self.postprocess_output(reconstruct, self.x_mean, self.x_std)