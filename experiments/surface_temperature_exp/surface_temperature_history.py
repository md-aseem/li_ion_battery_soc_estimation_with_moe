import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from config import ExperimentalDesignParams, VanillaNNParams, CalceDataParams, TrainEvalParams, HistoryColsParams
from flat_feature_models.vanilla_nn import VanillaNeuralNetwork
from utils.preprocess_data import PreprocessCalceA123
from flat_feature_models.train import BatteryDataset, Trainer
import time
from tqdm import tqdm
"""
This file will perform an experiment to quantify the length of temperature data on the accuracy of the flat_feature_models.
"""

experimental_design_params = ExperimentalDesignParams()

history_length_list = []
mse_list = []
mae_list = []
training_time_list = []

### Data Loading ###
drive_cycle_path = r"../../data/calce_lfp/drive_cycles"
train_params = TrainEvalParams()
data_params = CalceDataParams()

HISTORY_LENGTHS = torch.arange(5, 0, -1, dtype=torch.int8)

print(f"Device: {train_params.device}")
for history_length in tqdm(HISTORY_LENGTHS):
    print(f"History Length: {history_length}\n")

    history_col_params = HistoryColsParams(['Voltage(V)', 'Current(A)', 'Temperature (C)_1'],
                                           [data_params.history_length, data_params.history_length, history_length])
    preprocess_calce = PreprocessCalceA123(data_params, history_col_params)
    df = preprocess_calce.load_dfs(drive_cycle_path)

    for run in tqdm(range(experimental_design_params.n_runs)):
        ### Data Preprocessing ###
        feature_cols = (['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'amb_temp'] +
                        [f"Voltage(V)-{i + 1}" for i in range(data_params.history_length)] +
                        [f"Current(A)-{i + 1}" for i in range(data_params.history_length)] +
                        [f"Temperature (C)_1-{i + 1}" for i in range(history_length)] )

        target_col = 'soc'

        training_conditions = {"testpart": ["DST", "FUD"]}
        train_df = preprocess_calce.filter(df, training_conditions)

        test_conditions = {"testpart": ["US06"]}
        test_df = preprocess_calce.filter(df, test_conditions)

        train_df_scaler, scaler = preprocess_calce.standardize_data(train_df, feature_cols=feature_cols)
        test_df_scaler, _ = preprocess_calce.standardize_data(test_df, feature_cols=feature_cols, scaler=scaler)

        training_dataset = BatteryDataset(train_df_scaler, feature_cols, target_col)
        test_dataset = BatteryDataset(test_df_scaler, feature_cols, target_col)

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