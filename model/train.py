from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BatteryDataset(Dataset):
    def __init__(self, df, feature_cols: List[str], target_col: str):

        self.features = torch.tensor(df[feature_cols].values.astype('float32'))
        self.target = torch.tensor(df[target_col].values.astype('float32'))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return x, y

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 training_params):

        self.model = model
        self.training_params = training_params
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.training_params.learning_rate)

    def train(self, train_loader):

        model = self.model
        model.train()

        optimizer = self.optimizer
        loss_fn = self.loss_fn
        losses = []; min_loss = float('inf'); loss = float('inf')

        for epoch in self.training_params.n_epochs:
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            avg_loss = epoch_loss/len(train_loader)
            losses.append(avg_loss)

            if epoch % (self.training_params.n_epochs//10) == 0:
                print(f"Epoch: {epoch}, Loss: {avg_loss.item():.7f}")

            if avg_loss < min_loss:
                min_loss = avg_loss
                torch.save(model.state_dict(), "saved/best_model.pt")

        model.eval()

        return model, losses

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x, y
                y_pred = self.model(x)
                val_loss += self.loss_fn(y_pred, y).item()
        return val_loss / len(val_loader)