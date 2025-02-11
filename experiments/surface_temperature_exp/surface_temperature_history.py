import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from config import ExperimentalDesignParams, VanillaNNParams, CalceDataParams, TrainEvalParams
from model.neural_network_model import VanillaNeuralNetwork
from utils.preprocess_data import PreprocessCalceA123
from model.train import BatteryDataset, Trainer
import time
from tqdm import tqdm
"""
This file will perform an experiment to quantify the length of temperature data on the accuracy of the model.
"""

experimental_design_params = ExperimentalDesignParams()

history_length_list = []
mse_list = []
mae_list = []
training_time_list = []

### Data Loading ###
train_params = TrainEvalParams()
data_params = CalceDataParams()
preprocess_calce = PreprocessCalceA123(data_params)
drive_cycle_path = r"../../data/calce_lfp/drive_cycles"
df = preprocess_calce.load_dfs(drive_cycle_path)

HISTORY_LENGTHS = torch.arange(50, 0, -10, dtype=torch.int8)

print(f"Device: {train_params.device}")
for history_length in tqdm(HISTORY_LENGTHS):
    print(f"History Length: {history_length}\n")
    for run in tqdm(range(experimental_design_params.n_runs)):
        ### Data Preprocessing ###
        feature_cols = (['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'amb_temp'] +
                        [f"Voltage(V)-{i + 1}" for i in range(history_length)] +
                        [f"Current(A)-{i + 1}" for i in range(history_length)] +
                        [f"Temperature (C)_1-{i + 1}" for i in range(history_length)] )

        target_col = 'soc'

        training_conditions = {"testpart": ["DST", "FUD"]}
        train_df = preprocess_calce.filter(df, training_conditions)
        train_df = preprocess_calce.add_sequence_data_per_col(train_df, seq_cols=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                                                                        history_length=[data_params.history_length,
                                                                                        data_params.history_length,
                                                                                        history_length])

        test_conditions = {"testpart": ["US06"]}
        test_df = preprocess_calce.filter(df, test_conditions)
        test_df = preprocess_calce.add_sequence_data_per_col(test_df, seq_cols=['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                                                                      history_length=[data_params.history_length,
                                                                                      data_params.history_length,
                                                                                      history_length])

        train_df, scaler = preprocess_calce.standardize_data(train_df, feature_cols=feature_cols)
        test_df, _ = preprocess_calce.standardize_data(test_df, feature_cols=feature_cols, scaler=scaler)

        training_dataset = BatteryDataset(train_df, feature_cols, target_col)
        test_dataset = BatteryDataset(test_df, feature_cols, target_col)

        ### Neural Network Creation ###
        nn_params = VanillaNNParams()
        nn_model = VanillaNeuralNetwork(len(feature_cols), out_features=1, nn_params=nn_params).to(train_params.device)

        ### Training ###
        train_loader = DataLoader(training_dataset, batch_size=train_params.batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=train_params.batch_size, shuffle=False)

        training_start_time = time.time()
        trainer = Trainer(nn_model, train_params, train_params.device)
        nn_model, losses = trainer.train(train_loader, val_loader)
        training_end_time = time.time()
        training_time = (training_end_time - training_start_time)
        print(f"Training Time: {training_time:.2f} secs")

        ### Plotting ###
        plt.figure()
        plt.semilogy(losses)
        plt.title("Loss Curve")
        plt.savefig(f"results/loss_curve_{history_length}.png")
        plt.show()

        y, y_pred = trainer.evaluate(val_loader)
        plt.figure()
        plt.plot(y.cpu().numpy()[:10000], 'o', label="True")
        plt.plot(y_pred.cpu().numpy()[:10000], 'o', label="Predicted")
        plt.legend()
        plt.savefig(f"results/prediction_{history_length}.png")
        plt.show()

        mse_loss = F.mse_loss(y, y_pred)
        mae_loss = torch.abs(y-y_pred).mean()

        print(f"MAE: {mse_loss:.2f}\nMSE: {mse_loss:.2f}")
        mse_list.append(mse_loss); mae_list.append(mae_loss); history_length_list.append(history_length); training_time_list.append(training_time)


df = pd.DataFrame({"mae": mae_list,
                   "mse": mse_list,
                   "history_length": history_length_list})
df.to_csv("results/training_results.csv")