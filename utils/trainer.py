import torch.nn as nn
import torch
import torch.nn.functional as F
import time

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

    def train(self, train_loader, val_loader=None):

        model = self.model
        model.to(device=self.device)
        model.train()

        optimizer = self.optimizer
        loss_fn = self.loss_fn
        losses = []; min_loss = float('inf'); loss = float('inf')

        for epoch in range(self.training_params.n_epochs):
            model.train()
            start_time = time.time()
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(train_loader):

                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                model_out = self.model(x)
                if isinstance(model_out, tuple): # moe flat_feature_models outputs gate_loss too
                    y_pred, gate_loss = model_out
                else:
                    gate_loss = 0 # no gate loss for regular NNs
                    y_pred = model_out

                loss = F.mse_loss(y_pred.squeeze(-1), y).mean() + gate_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            end_time = time.time()
            avg_loss = epoch_loss/len(train_loader)
            losses.append(avg_loss)

            if epoch % (self.training_params.n_epochs//10) == 0:
                print(f"Epoch: {epoch}, Loss: {avg_loss:.6f}, Time/epoch: {(end_time-start_time):.3f} secs")
                if val_loader:
                    y, y_pred = self.evaluate(val_loader)
                    val_loss = torch.abs(y-y_pred.squeeze(-1)).mean()
                    print(f"Validation Loss: {val_loss:.6f}")

        model.eval()

        return model, losses

    def evaluate(self, val_loader):
        self.model.eval()
        all_y, all_y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                model_out = self.model(x)
                if isinstance(model_out, tuple): # moe flat_feature_models outputs gate_loss too
                    y_pred, gate_loss = model_out
                else:
                    gate_loss = 0 # no gate loss for regular NNs
                    y_pred = model_out

                all_y.append(y)
                all_y_pred.append(y_pred.squeeze(-1))

        all_y = torch.concatenate(all_y)
        all_y_pred = torch.concatenate(all_y_pred)

        return all_y, all_y_pred