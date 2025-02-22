import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MoENNParams, TrainEvalParams, VanillaNNParams


class MexicanHat(nn.Module):
    def __init__(self, alpha_init=1):
        super(MexicanHat, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32), requires_grad=True)
    def forward(self, x):
        x = (1-x**2*self.alpha**2)*torch.exp(-x**2*self.alpha**2)
        return x

class GatingNetwork(nn.Module):
    def __init__(self,
                 n_experts: int = 8,
                 hidden_dim: int = 64,
                 gating_dim: int = 16,
                 gating_noise_std: float = 1e-2,
                 device: str=None):
        super().__init__()

        self.n_experts = n_experts
        self.gating_dim = gating_dim
        self.gating_noise_std = gating_noise_std

        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.gate = nn.Sequential(nn.Linear(hidden_dim, gating_dim),
                                  nn.ReLU(),
                                  nn.Linear(gating_dim, n_experts)
                                  )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, hidden_dim =  x.size()
        if self.gating_noise_std:
            noise = torch.randn([B, self.n_experts], device=self.device) * self.gating_noise_std
        else:
            noise = torch.zeros([B, self.n_experts], device=self.device)
        x = self.softmax(self.gate(x)) + noise
        return x # B, n_experts

class Expert(nn.Module):
    def __init__(self,
                 hidden_dim: int = 64,
                 n_layers: int = 1):
        super().__init__()

        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                              nn.Sigmoid()) for _ in range(n_layers)])

    def forward(self, x):
        # x.size() = B, hidden_dim
        for layer in self.layers:
            x = layer(x) + x
        return x

class MoE(nn.Module):
    def __init__(self,
                 hidden_dim: int = 64,
                 n_experts: int = 8,
                 n_layers_per_expert: int = 1,
                 gating_dim: int= 8,
                 top_k: int = 2,
                 balancing_coef: float = 1e-2,
                 device: str = None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.n_layers_per_expert = n_layers_per_expert
        self.gating_dim = gating_dim
        self.top_k = top_k
        self.balancing_coef = balancing_coef

        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.shared_expert = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                           nn.Sigmoid())
        self.shared_expert_weight = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)

        self.routing_experts = nn.ModuleList([Expert(self.hidden_dim, self.n_layers_per_expert) for _ in range(self.n_experts)])
        self.router_experts_weight = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)

        self.gate = GatingNetwork(self.n_experts, self.hidden_dim, self.gating_dim)

    def forward(self,
                x: torch.Tensor,
                return_gate_loss: bool = True):
        B, hidden_dim =  x.size()
        assert hidden_dim == self.hidden_dim

        gating_weights = self.gate(x) # B, n_experts

        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1) # B, k
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        router_experts_output = torch.zeros([B, self.hidden_dim], device=self.device)

        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_weights[:, i]

            # output tensor to collect each expert's output

            for expert_idx in range(self.n_experts):
                expert_mask = expert_indices == expert_idx

                if expert_mask.any():
                    expert_out = self.routing_experts[expert_idx](x[expert_mask])
                    router_experts_output[expert_mask] += expert_out * expert_weights[expert_mask].unsqueeze(-1)

        final_out = x + self.shared_expert_weight*self.shared_expert(x) + self.router_experts_weight*router_experts_output

        gate_loss = torch.tensor(0)
        if return_gate_loss:
            experts_prob = gating_weights.mean(dim=0)
            target_prob = torch.ones([self.n_experts], device=self.device) / self.n_experts
            gate_loss = F.mse_loss(target_prob, experts_prob) * self.balancing_coef

        return final_out, gate_loss

class MoENeuralNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 moe_nn_params):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = moe_nn_params.hidden_dim
        self.gating_dim = moe_nn_params.gating_dim
        self.n_experts = moe_nn_params.n_experts
        self.top_k = moe_nn_params.top_k
        self.balancing_coef = moe_nn_params.balancing_coef
        self.n_layers_per_expert = moe_nn_params.n_layers_per_expert

        self.fc1 = nn.Linear(self.in_features, self.hidden_dim)
        self.activation = nn.Sigmoid()
        self.sparse = MoE(self.hidden_dim, self.n_experts, self.n_layers_per_expert, self.gating_dim, self.top_k, self.balancing_coef)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_features)

    def forward(self,
                x: torch.Tensor,
                return_gate_loss: bool=False):

        x = self.fc1(x)
        x = self.activation(x)
        x, gate_loss = self.sparse(x, return_gate_loss)
        x = self.fc2(x)

        return x, gate_loss

    def get_active_parameters(self):

        linear_layers_params = list(self.fc1.parameters()) + list(self.fc2.parameters())
        shared_expert_params = list(self.sparse.shared_expert.parameters())
        routing_expert_params = list(self.sparse.routing_experts.parameters())

        total_active_params = (sum([p.numel() for p in linear_layers_params]) +
                               sum([p.numel() for p in shared_expert_params]) +
                               sum([p.numel() for p in routing_expert_params]) // self.n_experts * self.top_k)
        return total_active_params
if __name__ == "__main__":
    moe_nn_params = MoENNParams(); nn_params = VanillaNNParams()
    train_eval_params = TrainEvalParams()
    device = train_eval_params.device

    batch_size = 32
    in_features = 15
    out_features = 1
    x = torch.rand([batch_size, in_features]).to(device)

    test_network = MoENeuralNetwork(in_features, out_features, moe_nn_params).to(device)
    active_params = test_network.get_active_parameters()
    total_params = sum([p.numel() for p in test_network.parameters()])
    print(f"Total Params: {total_params}\nActive Params: {active_params}")
    y, gate_loss = test_network(x, return_gate_loss=True)
    print(f"Input Shape: {x.size()}",
          f"\nOutput Shape: {y.size()}")

    print(f"Gate Loss: {gate_loss:.3f}")

