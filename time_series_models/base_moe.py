import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self,
                 n_embd: int = 512,
                 dropout_p: float = 0.1):
        super().__init__()

        self.n_embd = n_embd
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(n_embd, n_embd*2)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(n_embd*2, n_embd)
        self.relu_2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu_2(x)
        x = self.dropout(x)
        return x


class MoE(nn.Module):
    """
    This module implements mixture of experts.
    Args: n_embd (int): Number of Embeddings,
          n_experts (int): Number of Experts,
          top_k (int): Top k experts selected,
          dropout_p (int): dropout probability
    """
    def __init__(self,
                 n_embd: int,
                 n_experts: int,
                 top_k: int,
                 balancing_coef: float = 1e-4,
                 dropout_p: float=0.1,
                 device: str= 'cpu'):
        super().__init__()

        self.n_embd = n_embd
        self.n_experts = n_experts
        self.top_k = top_k
        self.dropout_p = dropout_p
        self.balancing_coef = balancing_coef
        self.device = device

        self.router = nn.Sequential(nn.Linear(self.n_embd, self.n_experts),
                                    nn.Softmax(dim=-1))
        self.softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList([FeedForward(self.n_embd, dropout_p) for _ in range(self.n_experts)])
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.return_gate_loss = False

    def forward(self, x):
        """
        Forward method of the MoE layer.

        Args: x (torch.Tensor): Input tensor
        Returns: torch.Tensor: Output tensor
        """
        B, T, _ = x.size() # B, T, n_embd
        x_flat = x.view(B*T, -1) # (B*T,)
        output_flat = torch.zeros_like(x_flat)

        g_x = self.router(x) # B, T, n_experts
        top_weights, top_indicies = g_x.topk(self.top_k, dim=-1)

        g_x_flat = g_x.view(B*T, -1)
        top_weights_flat = top_weights.view(B*T, -1)
        top_indicies_flat = top_indicies.view(B*T, -1)

        for i in range(self.top_k):
            expert_idx = top_indicies_flat[:, i] # (B*T,)
            expert_weight = top_weights_flat[:, i]

            for idx in range(self.n_experts):
                expert_mask = expert_idx == idx
                if expert_mask.any():
                    expert_out = self.experts[idx](x_flat[expert_mask])
                    output_flat[expert_mask] += expert_out * expert_weight[expert_mask].unsqueeze(-1)

        output = output_flat.view(B, T, -1)

        target_sharing = torch.ones([self.n_experts], device=self.device) / self.n_experts
        experts_sharing = g_x_flat.mean(dim=0)
        gating_loss = F.mse_loss(target_sharing, experts_sharing) * self.balancing_coef

        return output, gating_loss

class MoEVectorized(nn.Module):
    """
    A vectorized mixture-of-experts (MoE) layer.

    Args:
        n_embd (int): Dimensionality of the input embeddings.
        n_experts (int): Number of experts.
        top_k (int): Number of experts to select per token.
        balancing_coef (float): Coefficient for the gating loss.
        dropout_p (float): Dropout probability.
        device (str): Device to use.
    """
    def __init__(self,
                 n_embd: int,
                 n_experts: int,
                 top_k: int,
                 balancing_coef: float = 1e-4,
                 dropout_p: float = 0.1):
        super().__init__()
        self.n_embd = n_embd
        self.n_experts = n_experts
        self.top_k = top_k
        self.dropout_p = dropout_p
        self.balancing_coef = balancing_coef

        # The router produces a probability over experts for each token.
        self.router = nn.Sequential(
            nn.Linear(self.n_embd, self.n_experts),
            nn.Softmax(dim=-1)
        )
        # Create a list of expert feed-forward networks.
        self.experts = nn.ModuleList([
            FeedForward(self.n_embd, dropout_p) for _ in range(self.n_experts)
        ])
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        """
        Vectorized forward pass for the MoE layer.
        Args:
            x (torch.Tensor): Input of shape (B, T, n_embd)
        Returns:
            output (torch.Tensor): Output tensor of shape (B, T, n_embd)
            gating_loss (torch.Tensor): A scalar balancing loss.
        """
        B, T, D = x.size()
        x_flat = x.view(B * T, D)

        # Compute routing probabilities for each token.
        router_out = self.router(x)  # shape: (B, T, n_experts)
        router_out_flat = router_out.view(B * T, self.n_experts)

        # Select top-k experts per token.
        top_weights, top_indices = router_out_flat.topk(self.top_k, dim=-1)  # shapes: (B*T, top_k)

        # Compute outputs for all experts in parallel.
        # Each expert is applied to all tokens; output shape: (B*T, n_experts, D)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)

        # Now, gather the outputs of the top-k experts for each token.
        # Expand top_indices to match the feature dimension.
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)  # shape: (B*T, top_k, D)
        top_expert_outputs = torch.gather(expert_outputs, 1, top_indices_expanded)  # shape: (B*T, top_k, D)

        # Weight each expert's output by its corresponding gating weight.
        top_weights = top_weights.unsqueeze(-1)  # shape: (B*T, top_k, 1)
        output_flat = (top_expert_outputs * top_weights).sum(dim=1)  # shape: (B*T, D)
        output = output_flat.view(B, T, D)

        # Compute balancing (gating) loss.
        target_sharing = torch.ones(self.n_experts, device=x.device) / self.n_experts
        experts_sharing = router_out_flat.mean(dim=0)
        gating_loss = F.mse_loss(target_sharing, experts_sharing) * self.balancing_coef

        return output, gating_loss

