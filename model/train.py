import time
from typing import List
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils.preprocess_data import PreprocessData
from torch.utils.data import Dataset, DataLoader
from config import DataParams, NeuralNetworkParams
from model.neural_network_model import VoltageNN
import numpy as np

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
                y_pred = self.model(x)
                loss = loss_fn(y_pred.squeeze(-1), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss/len(train_loader)
            losses.append(avg_loss)

            if epoch % (self.training_params.n_epochs//10) == 0:
                print(f"Epoch: {epoch}, Loss: {avg_loss:.7f}")

            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(model.state_dict(), "saved/best_model.pt")

        model.eval()

        return model, losses

    def evaluate(self, val_loader):
        self.model.eval()
        all_y, all_y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                all_y.append(y.cpu().numpy())
                all_y_pred.append(y_pred.cpu().numpy())

        all_y = np.concatenate(all_y)
        all_y_pred = np.concatenate(all_y_pred)

        plt.plot(all_y, label="True")
        plt.plot(all_y_pred, label="Predicted")
        plt.legend()
        plt.show()
        pass

if __name__ == "__main__":
    data_path = '../data/Stage_1'
    data_params = DataParams()
    read_process_data = PreprocessData(data_params)
    df = read_process_data.load_dfs(data_path)
    filtering_conditions = {"testpart": "hppc"}
    df = read_process_data.fitler_data(df, filtering_conditions)
    read_process_data.plot(df, y_axes=["c_vol", "soc"], conditions={"testpoint": 1}) # plotting the data before training
    df = read_process_data.add_sequence_data(df, 'c_cur', num_points=data_params.num_points)
    feature_cols = ['c_cur', 'soc', 'ocv', 'dva', 'ocv_ch', 'ocv_dch'] + [f"c_cur-{i + 1}" for i in range(data_params.num_points)]
    target_col = 'c_vol'
    df, _ = read_process_data.standardize_data(df, feature_cols=feature_cols)

    model = VoltageNN(in_features=len(feature_cols),
                      hidden_feature=NeuralNetworkParams.hidden_dim,
                      out_features=1).to(device=NeuralNetworkParams.device)

    nn_params = NeuralNetworkParams()
    training_dataset = BatteryDataset(df[df['testpoint'] <= 3], feature_cols, target_col)
    val_dataset = BatteryDataset(df[df['testpoint'] == 4], feature_cols, target_col)

    train_loader = DataLoader(training_dataset, batch_size=nn_params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=nn_params.batch_size, shuffle=False)

    start_time = time.time()
    trainer = Trainer(model, nn_params, nn_params.device)
    model, losses = trainer.train(train_loader)
    end_time = time.time()
    print(f"Training Time: {end_time - start_time}:2f")
    trainer.evaluate(val_loader)

    plt.figure()
    plt.semilogy(losses)
    plt.show()