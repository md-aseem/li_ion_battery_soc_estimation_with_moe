import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from config import TrainEvalParams, MoENNParams, CalceDataParams, HistoryColsParams, TransformerParams
from utils.preprocess_data import PreprocessCalceA123
from time_series_models.base_moe import MoEVectorized
from utils.trainer import Trainer

"""
The file implements a transformer architecture with mixture of experts
"""


class Block(nn.Module):
    def __init__(self,
                 transformer_params: TransformerParams,
                 moe_params: MoENNParams,
                 device: str = "cpu"):
        super().__init__()

        self.transformer_params = transformer_params
        self.moe_params = moe_params
        self.device = device

        self.to_k = nn.Linear(self.transformer_params.n_embd, self.transformer_params.n_embd)
        self.to_q = nn.Linear(self.transformer_params.n_embd, self.transformer_params.n_embd)
        self.to_v = nn.Linear(self.transformer_params.n_embd, self.transformer_params.n_embd)

        self.attn = nn.MultiheadAttention(self.transformer_params.n_embd,
                                          self.transformer_params.n_heads,
                                          self.transformer_params.dropout)

        self.ln1 = nn.LayerNorm(self.transformer_params.n_embd)
        self.ln2 = nn.LayerNorm(self.transformer_params.n_embd)

        self.shared_expert = nn.Sequential(nn.Linear(self.transformer_params.n_embd, self.transformer_params.n_embd),
                                           nn.LeakyReLU())
        self.moe = MoEVectorized(self.transformer_params.n_embd, self.moe_params.n_experts,
                       self.moe_params.top_k)

        self.moe_fake = nn.Sequential(nn.Linear(self.transformer_params.n_embd, self.transformer_params.n_embd),
                                 nn.LeakyReLU())

    def forward(self, x):

        q = self.to_q(x); k = self.to_k(x); v = self.to_v(x)
        attn_out, _ = self.attn(q, k, v)
        x = self.ln1(x+attn_out)
        moe_out, gating_loss = self.moe(x.contiguous())
        x = self.ln2(x + moe_out + self.shared_expert(x))
        # x = self.moe_fake(x); gating_loss = torch.tensor(0, dtype=torch.float32)
        return x, gating_loss


class Transformer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 transformer_params: TransformerParams,
                 moe_params: MoENNParams,
                 device: str ="cuda"):
        super().__init__()

        self.transformer_params = transformer_params
        self.moe_params = moe_params

        self.fc1 = nn.LSTM(in_features, self.transformer_params.n_embd, batch_first=True)
        self.blocks = nn.ModuleList([Block(self.transformer_params, self.moe_params, device=device) for _ in range(self.transformer_params.n_blocks)])
        self.fc2 = nn.Linear(self.transformer_params.n_embd, out_features)

    def forward(self, x):
        x, _ = self.fc1(x)
        gating_losses = []
        for block in self.blocks:
            block_out, gating_loss  = block(x)
            x = x + block_out
            gating_losses.append(gating_loss)

        x = self.fc2(x)
        gating_loss = torch.tensor(gating_losses).mean()
        return x, gating_loss


if __name__ == "__main__":

    transformer_params = TransformerParams()
    moe_params = MoENNParams()

    data_params = CalceDataParams()
    history_params = HistoryColsParams()
    preprocess_calce = PreprocessCalceA123(data_params, history_params)
    drive_cycle_path = r"..\data\calce_lfp\drive_cycles"
    df = preprocess_calce.load_dfs(drive_cycle_path, create_sequence_cols=False)
    seq_cols = ['Voltage(V)', 'Current(A)', 'Temperature (C)_1']
    target_col = 'soc'

    train_filter = {"testpart": ["US06"]}
    test_filter = {"testpart": ['FUD']}

    train_df = preprocess_calce.filter(df, filtering_conditions=train_filter)
    test_df = preprocess_calce.filter(df, filtering_conditions=test_filter).iloc[:10000, :]

    train_df_scaler, scaler = preprocess_calce.standardize_data(train_df, seq_cols)
    test_df_scaler, _ = preprocess_calce.standardize_data(df, feature_cols=seq_cols, scaler=scaler)

    train_feature_data, train_target_data = preprocess_calce.add_3d_sequence_data_per_col(train_df_scaler,
                                                                              seq_cols=seq_cols,
                                                                              target_col='soc',
                                                                              history_length=data_params.history_length)

    test_feature_data, test_target_data = preprocess_calce.add_3d_sequence_data_per_col(test_df_scaler,
                                                                              seq_cols=seq_cols,
                                                                              target_col='soc',
                                                                              history_length=data_params.history_length)

    model = Transformer(len(seq_cols), 1, transformer_params, moe_params, device="cuda")
    print(f"Total Params: {sum([p.numel() for p in model.parameters()])}")

    train_params = TrainEvalParams()
    trainer = Trainer(model, train_params, device='cuda')

    train_dataset = SequenceBatteryDataSet(train_feature_data, train_target_data, device=train_params.device)
    train_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, shuffle=True)

    test_dataset = SequenceBatteryDataSet(test_feature_data, test_target_data, device=train_params.device)
    test_loader = DataLoader(test_dataset, batch_size=train_params.batch_size, shuffle=True)

    model, losses = trainer.train(train_loader)

    plt.figure()
    plt.semilogy(losses)
    plt.title("Loss Curve")
    plt.show()

    y_train, y_pred_train = trainer.evaluate(train_loader)

    plt.figure()
    plt.plot(y_train.cpu().numpy()[::1000][:, -1], y_pred_train.cpu().detach().numpy()[::1000][:, -1], 'o')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Training Predictions")
    plt.show()

    y_test, y_pred_test = trainer.evaluate(test_loader)

    plt.figure()
    plt.plot(y_test.cpu().numpy()[::1000][:, -1], y_pred_test.cpu().detach().numpy()[::1000][:, -1], 'o')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Training Predictions")
    plt.show()