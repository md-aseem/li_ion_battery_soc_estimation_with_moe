import torch.nn as nn
import torch
from config import VanillaNNParams


class VanillaNeuralNetwork(nn.Module):
    """Neural network to predict terminal voltage from SOC, current, and temperature."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 nn_params
                 ):
        super().__init__()

        self.n_hidden_layers = nn_params.n_hidden_layers
        self.hidden_dim = nn_params.hidden_dim

        self.fc1 = nn.Linear(in_features, self.hidden_dim)
        self.act = nn.Sigmoid()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                   nn.Sigmoid()) for _ in range(self.n_hidden_layers)])
        self.fc2 = nn.Linear(self.hidden_dim, out_features)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.act(self.fc1(x))
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        return torch.clip(x, 0, 1)


if __name__ == "__main__":

    in_features = 15
    out_features = 1
    batch_size = 100
    nn_params = VanillaNNParams()
    test_nn_model = VanillaNeuralNetwork(in_features, out_features, nn_params)

    x = torch.rand([batch_size, in_features])
    y = test_nn_model(x)

    print(f"Total Params: {sum([p.numel() for p in test_nn_model.parameters()])}")