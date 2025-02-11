import time
from typing import List
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils.preprocess_data import PreprocessCalceA123
from torch.utils.data import Dataset, DataLoader
from config import TrainEvalParams, VanillaNNParams, MoENNParams, CalceDataParams
from model.mixture_of_experts_nn import MoENeuralNetwork
import numpy as np
from utils.helpers import parameter_count

class BatteryDataset(Dataset):
    def __init__(self, df, feature_cols: List[str], target_col: str):

        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.target = torch.tensor(df[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return x, y

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 training_params,
                 device: str):

        self.model = model
        self.training_params = training_params
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.training_params.learning_rate)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader):

        model = self.model
        model.train()

        optimizer = self.optimizer
        loss_fn = self.loss_fn
        losses = []; min_loss = float('inf'); loss = float('inf')

        for epoch in range(self.training_params.n_epochs):
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):

                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                model_out = self.model(x)
                if isinstance(model_out, tuple): # moe model outputs gate_loss too
                    y_pred, gate_loss = model_out
                else:
                    gate_loss = 0 # no gate loss for regular NNs
                    y_pred = model_out

                loss = loss_fn(y_pred.squeeze(-1), y) + gate_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss/len(train_loader)
            losses.append(avg_loss)

            if epoch % (self.training_params.n_epochs//10) == 0:
                print(f"Epoch: {epoch}, Loss: {avg_loss:.7f}")

        model.eval()

        return model, losses

    def evaluate(self, val_loader):
        self.model.eval()
        all_y, all_y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                model_out = self.model(x)
                if isinstance(model_out, tuple): # moe model outputs gate_loss too
                    y_pred, gate_loss = model_out
                else:
                    gate_loss = 0 # no gate loss for regular NNs
                    y_pred = model_out

                all_y.append(y)
                all_y_pred.append(y_pred)

        all_y = torch.concatenate(all_y)
        all_y_pred = torch.concatenate(all_y_pred)

        return all_y, all_y_pred

if __name__ == "__main__":
    data_loading_start_time = time.time()
    data_params = CalceDataParams()
    preprocess_calce = PreprocessCalceA123()
    drive_cycle_path = r"..\data\calce_lfp\drive_cycles"
    df = preprocess_calce.load_dfs(drive_cycle_path)

    feature_cols = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'amb_temp'] + [f"Voltage(V)-{i + 1}" for i in range(data_params.history_length)] + [f"Current(A)-{i + 1}" for i in range(data_params.history_length)]
    target_col = 'soc'

    training_conditions = {"testpart": ["DST", "FUD"]}
    train_df = preprocess_calce.filter(df, training_conditions)
    preprocess_calce.plot(train_df, x_axis="Test_Time(s)", y_axes=["Voltage(V)"]) # plotting the data before training
    train_df = preprocess_calce.add_sequence_data(train_df, ['Current(A)', 'Voltage(V)'], history_length=data_params.history_length)

    test_conditions = {"testpart": ["US06"]}
    test_df = preprocess_calce.filter(df, test_conditions)
    test_df = preprocess_calce.add_sequence_data(test_df, ['Current(A)', 'Voltage(V)'], history_length=data_params.history_length)

    train_df, scaler = preprocess_calce.standardize_data(train_df, feature_cols=feature_cols)
    test_df, _ = preprocess_calce.standardize_data(test_df, feature_cols=feature_cols, scaler=scaler)

    data_loading_end_time = time.time()

    train_eval_params = TrainEvalParams()
    moe_nn_params = MoENNParams()
    moe_model = MoENeuralNetwork(in_features=len(feature_cols), out_features=1, moe_nn_params=moe_nn_params).to(device=train_eval_params.device)
    print(f"Parameter Count(Moe): {parameter_count(moe_model)}")

    training_dataset = BatteryDataset(train_df, feature_cols, target_col)
    test_dataset = BatteryDataset(test_df, feature_cols, target_col)

    train_loader = DataLoader(training_dataset, batch_size=train_eval_params.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=train_eval_params.batch_size, shuffle=False)

    training_start_time = time.time()
    trainer = Trainer(moe_model, train_eval_params, train_eval_params.device)
    model, losses = trainer.train(train_loader)
    training_end_time = time.time()
    print(f"Training Time: {(training_end_time - training_start_time):.2f} secs")
    trainer.evaluate(val_loader)

    plt.figure()
    plt.semilogy(losses)
    plt.show()