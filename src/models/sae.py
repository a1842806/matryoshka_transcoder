import torch
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


class GlobalBatchTopKMatryoshkaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        total_dict_size = sum(cfg["group_sizes"])
        self.group_sizes = cfg["group_sizes"]
        
        self.group_indices = [0] + list(torch.cumsum(torch.tensor(cfg["group_sizes"]), dim=0))
        self.active_groups = len(cfg["group_sizes"])

        self.b_dec = nn.Parameter(torch.zeros(self.config["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(total_dict_size))
        
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], total_dict_size)
            )
        )
        
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(total_dict_size, cfg["act_size"])
            )
        )
        
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.num_batches_not_active = torch.zeros(total_dict_size, device=cfg["device"])
        self.register_buffer('threshold', torch.tensor(0.0))
        self.to(cfg["dtype"]).to(cfg["device"])

    def compute_activations(self, x_cent):
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            acts_topk = torch.topk(
                acts.flatten(), 
                self.config["top_k"] * x_cent.shape[0], 
                dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
            self.update_threshold(acts_topk)
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk

    def encode(self, x):
        original_shape = x.shape
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std

        x = x.reshape(-1, x.shape[-1])
        x_cent = x - self.b_dec
        _, result = self.compute_activations(x_cent)
        max_act_index = self.group_indices[self.active_groups]
        result[:, max_act_index:] = 0
        if len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1], -1)
        return result
    
    def decode(self, acts_topk):
        reconstruct = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(reconstruct, self.x_mean, self.x_std)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        x_reconstruct = self.b_dec

        intermediate_reconstructs = []
        all_acts, all_acts_topk = self.compute_activations(x_cent)

        for i in range(self.active_groups):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i+1]
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            acts_topk = all_acts_topk[:, start_idx:end_idx]
            x_reconstruct = acts_topk @ W_dec_slice + x_reconstruct
            intermediate_reconstructs.append(x_reconstruct)

        self.update_inactive_features(all_acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, all_acts, all_acts_topk, x_mean, 
                                  x_std, intermediate_reconstructs)
        return output

    def get_loss_dict(self, x, x_reconstruct, all_acts, all_acts_topk, x_mean, x_std, intermediate_reconstructs):
        total_l2_loss = (self.b_dec - x.float()).pow(2).mean()
        l2_losses = torch.tensor([]).to(x.device)
        for intermediate_reconstruct in intermediate_reconstructs:
            l2_losses = torch.cat([l2_losses, (intermediate_reconstruct.float() - 
                                             x.float()).pow(2).mean().unsqueeze(0)])
            total_l2_loss += (intermediate_reconstruct.float() - x.float()).pow(2).mean()

        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()
        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)

        l1_norm = all_acts_topk.float().abs().sum(-1).mean()
        l0_norm = (all_acts_topk > 0).float().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, all_acts)
        loss = mean_l2_loss + l1_loss + aux_loss
        
        # Compute FVU (Fraction of Variance Unexplained)
        fvu = compute_fvu(x, x_reconstruct)
        
        # Compute FVU for each intermediate reconstruction
        fvu_per_group = torch.tensor([]).to(x.device)
        for intermediate_reconstruct in intermediate_reconstructs:
            fvu_group = compute_fvu(x, intermediate_reconstruct)
            fvu_per_group = torch.cat([fvu_per_group, fvu_group.unsqueeze(0)])
        
        num_dead_features = (self.num_batches_not_active > self.config["n_batches_to_dead"]).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": all_acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": mean_l2_loss,
            "min_l2_loss": min_l2_loss,
            "max_l2_loss": max_l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
            "fvu": fvu,
            "fvu_min": fvu_per_group.min() if len(fvu_per_group) > 0 else fvu,
            "fvu_max": fvu_per_group.max() if len(fvu_per_group) > 0 else fvu,
            "fvu_mean": fvu_per_group.mean() if len(fvu_per_group) > 0 else fvu,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, all_acts):
        residual = x.float() - x_reconstruct.float()
        aux_reconstruct = torch.zeros_like(residual)
        
        acts = all_acts
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        
        if dead_features.sum() > 0:
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.config["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            aux_reconstruct = aux_reconstruct + x_reconstruct_aux
                
        if aux_reconstruct.abs().sum() > 0:
            aux_loss = self.config["aux_penalty"] * (aux_reconstruct.float() - residual.float()).pow(2).mean()
            return aux_loss
            
        return torch.tensor(0.0, device=x.device)
    
    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive


class BatchTopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer('threshold', torch.tensor(0.0))
        
    def compute_activations(self, x):
        x_cent = x - self.b_dec
        pre_acts = x_cent @ self.W_enc
        acts = F.relu(pre_acts)
        
        if self.training:
            acts_topk = torch.topk(
                acts.flatten(), 
                self.config["top_k"] * x.shape[0], 
                dim=-1
            )
            acts_topk = (
                torch.zeros_like(acts.flatten())
                .scatter(-1, acts_topk.indices, acts_topk.values)
                .reshape(acts.shape)
            )
        else:
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        acts, acts_topk = self.compute_activations(x)
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        self.update_threshold(acts_topk)
        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def encode(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std
        acts, acts_topk = self.compute_activations(x)
        return acts_topk
    
    def decode(self, acts_topk):
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(x_reconstruct, self.x_mean, self.x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        
        # Compute FVU
        fvu = compute_fvu(x, x_reconstruct)
        
        num_dead_features = (
            self.num_batches_not_active > self.config["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
            "fvu": fvu,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
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
        return torch.tensor(0, dtype=x.dtype, device=x.device)
        
    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive


class TopKSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.config["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def encode(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        self.x_mean = x_mean
        self.x_std = x_std
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.config["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        return acts_topk

    def decode(self, acts):
        out = acts @ self.W_dec + self.b_dec
        return self.postprocess_output(out, self.x_mean, self.x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        
        # Compute FVU
        fvu = compute_fvu(x, x_reconstruct)
        
        num_dead_features = (
            self.num_batches_not_active > self.config["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "fvu": fvu,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
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
        return torch.tensor(0, dtype=x.dtype, device=x.device)


class VanillaSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        
        # Compute FVU
        fvu = compute_fvu(x, x_reconstruct)
        
        num_dead_features = (
            self.num_batches_not_active > self.config["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "fvu": fvu,
        }
        return output


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLU(nn.Module):
    def __init__(self, feature_size, bandwidth, device='cpu'):
        super().__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size, device=device))
        self.bandwidth = bandwidth

    def forward(self, x):
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReLUSAE(BaseAutoencoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], 
                                bandwidth=cfg["bandwidth"], device=cfg["device"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, 
                              self.config["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.config["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        
        # Compute FVU
        fvu = compute_fvu(x, x_reconstruct)
        
        num_dead_features = (
            self.num_batches_not_active > self.config["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
            "fvu": fvu,
        }
        return output


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
        
        # Matryoshka group configuration
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
            # Use learned threshold for inference
            acts_topk = torch.where(acts > self.threshold, acts, torch.zeros_like(acts))
        
        return acts, acts_topk
    
    def forward(self, x_source, x_target):
        """
        Forward pass for training.
        
        Args:
            x_source: Activations from source layer (batch_size, source_act_size)
            x_target: Activations from target layer (batch_size, target_act_size)
        
        Returns:
            Dictionary with loss, reconstructions, and metrics
        """
        # Preprocess inputs
        x_source, x_mean_src, x_std_src = self.preprocess_input(x_source)
        x_target, x_mean_tgt, x_std_tgt = self.preprocess_input(x_target)
        
        # Encode source to sparse features
        all_acts, all_acts_topk = self.compute_activations(x_source)
        
        # Incrementally decode using each Matryoshka group
        x_reconstruct = self.b_dec
        intermediate_reconstructs = []
        
        for i in range(self.active_groups):
            start_idx = self.group_indices[i]
            end_idx = self.group_indices[i+1]
            
            # Get decoder weights for this group
            W_dec_slice = self.W_dec[start_idx:end_idx, :]
            acts_topk = all_acts_topk[:, start_idx:end_idx]
            
            # Add this group's contribution to reconstruction
            x_reconstruct = acts_topk @ W_dec_slice + x_reconstruct
            intermediate_reconstructs.append(x_reconstruct)
        
        # Update dead feature tracking
        self.update_inactive_features(all_acts_topk)
        
        # Compute losses (measured against TARGET layer)
        output = self.get_loss_dict(
            x_target, x_reconstruct, all_acts, all_acts_topk, 
            x_mean_tgt, x_std_tgt, intermediate_reconstructs
        )
        return output
    
    def get_loss_dict(self, x_target, x_reconstruct, all_acts, all_acts_topk, 
                     x_mean, x_std, intermediate_reconstructs):
        """
        Compute losses for Matryoshka Transcoder.
        Key: All losses measure reconstruction quality of TARGET layer.
        """
        # Bias-only baseline loss
        total_l2_loss = (self.b_dec - x_target.float()).pow(2).mean()
        l2_losses = torch.tensor([]).to(x_target.device)
        
        # Loss for each group's incremental reconstruction
        for intermediate_reconstruct in intermediate_reconstructs:
            l2_loss = (intermediate_reconstruct.float() - x_target.float()).pow(2).mean()
            l2_losses = torch.cat([l2_losses, l2_loss.unsqueeze(0)])
            total_l2_loss += l2_loss
        
        # Summary statistics across groups
        min_l2_loss = l2_losses.min()
        max_l2_loss = l2_losses.max()
        mean_l2_loss = total_l2_loss / (len(intermediate_reconstructs) + 1)
        
        # Sparsity metrics
        l1_norm = all_acts_topk.float().abs().sum(-1).mean()
        l0_norm = (all_acts_topk > 0).float().sum(-1).mean()
        l1_loss = self.config["l1_coeff"] * l1_norm
        
        # Auxiliary loss for dead feature reactivation
        aux_loss = self.get_auxiliary_loss(x_target, x_reconstruct, all_acts)
        
        # Total loss
        loss = mean_l2_loss + l1_loss + aux_loss
        
        # Compute FVU (Fraction of Variance Unexplained)
        fvu = compute_fvu(x_target, x_reconstruct)
        
        # Compute FVU for each intermediate reconstruction
        fvu_per_group = torch.tensor([]).to(x_target.device)
        for intermediate_reconstruct in intermediate_reconstructs:
            fvu_group = compute_fvu(x_target, intermediate_reconstruct)
            fvu_per_group = torch.cat([fvu_per_group, fvu_group.unsqueeze(0)])
        
        # Count dead features
        num_dead_features = (self.num_batches_not_active > self.config["n_batches_to_dead"]).sum()
        
        # Postprocess output
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        
        return {
            "sae_out": sae_out,
            "feature_acts": all_acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": mean_l2_loss,
            "min_l2_loss": min_l2_loss,
            "max_l2_loss": max_l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
            "threshold": self.threshold,
            "fvu": fvu,
            "fvu_min": fvu_per_group.min() if len(fvu_per_group) > 0 else fvu,
            "fvu_max": fvu_per_group.max() if len(fvu_per_group) > 0 else fvu,
            "fvu_mean": fvu_per_group.mean() if len(fvu_per_group) > 0 else fvu,
        }
    
    def get_auxiliary_loss(self, x_target, x_reconstruct, all_acts):
        """
        Auxiliary loss to reactivate dead features.
        Uses residual between target and current reconstruction.
        """
        residual = x_target.float() - x_reconstruct.float()
        aux_reconstruct = torch.zeros_like(residual)
        
        dead_features = self.num_batches_not_active >= self.config["n_batches_to_dead"]
        
        if dead_features.sum() > 0:
            # Select top-k activations among dead features
            acts_topk_aux = torch.topk(
                all_acts[:, dead_features],
                min(self.config["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(all_acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            # Reconstruct residual using dead features
            aux_reconstruct = acts_aux @ self.W_dec[dead_features]
        
        if aux_reconstruct.abs().sum() > 0:
            aux_loss = self.config["aux_penalty"] * (aux_reconstruct.float() - residual.float()).pow(2).mean()
            return aux_loss
        
        return torch.tensor(0.0, device=x_target.device)
    
    @torch.no_grad()
    def update_threshold(self, acts_topk, lr=0.01):
        """Update activation threshold using exponential moving average"""
        positive_mask = acts_topk > 0
        if positive_mask.any():
            min_positive = acts_topk[positive_mask].min()
            self.threshold = (1 - lr) * self.threshold + lr * min_positive
    
    def encode(self, x_source):
        """
        Encode source layer activations to sparse features.
        
        Args:
            x_source: Source layer activations
        
        Returns:
            Sparse feature activations (respecting active groups)
        """
        original_shape = x_source.shape
        x_source, x_mean, x_std = self.preprocess_input(x_source)
        self.x_mean = x_mean
        self.x_std = x_std
        
        x_source = x_source.reshape(-1, x_source.shape[-1])
        _, result = self.compute_activations(x_source)
        
        # Zero out inactive groups
        max_act_index = self.group_indices[self.active_groups]
        result[:, max_act_index:] = 0
        
        if len(original_shape) == 3:
            result = result.reshape(original_shape[0], original_shape[1], -1)
        return result
    
    def decode(self, acts_topk):
        """
        Decode sparse features to target layer reconstruction.
        
        Args:
            acts_topk: Sparse feature activations
        
        Returns:
            Reconstruction of target layer
        """
        reconstruct = acts_topk @ self.W_dec + self.b_dec
        return self.postprocess_output(reconstruct, self.x_mean, self.x_std)
