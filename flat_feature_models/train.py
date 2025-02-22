import time
from typing import List
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils.preprocess_data import PreprocessCalceA123
from torch.utils.data import Dataset, DataLoader
from config import TrainEvalParams, VanillaNNParams, MoENNParams, CalceDataParams, HistoryColsParams
from flat_feature_models.moe_nn import MoENeuralNetwork
import numpy as np
from utils.helpers import parameter_count

if __name__ == "__main__":
    data_loading_start_time = time.time()
    data_params = CalceDataParams()
    history_params = HistoryColsParams()
    preprocess_calce = PreprocessCalceA123(data_params, history_params)
    drive_cycle_path = r"../data/calce_lfp/drive_cycles"
    df = preprocess_calce.load_dfs(drive_cycle_path, create_sequence_cols=True)

    feature_cols = (['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'amb_temp'] +
                    [f"Voltage(V)-{i + 1}" for i in range(data_params.history_length)] +
                    [f"Current(A)-{i + 1}" for i in range(data_params.history_length)] +
                    [f"Temperature (C)_1-{i + 1}" for i in range(data_params.history_length)])

    target_col = 'soc'

    training_conditions = {"testpart": ["DST", "US06"]}
    train_df = preprocess_calce.filter(df, training_conditions)
    preprocess_calce.plot(train_df, x_axis="Test_Time(s)", y_axes=["Voltage(V)"]) # plotting the data before training

    test_conditions = {"testpart": ["FUD"]}
    test_df = preprocess_calce.filter(df, test_conditions)

    train_df = train_df.iloc[:10024, :]
    test_df = test_df.iloc[:10024, :]
    train_df_scaler, scaler = train_df, 1 #preprocess_calce.standardize_data(train_df, feature_cols=feature_cols)
    test_df_scaler, _ = test_df, 1 #preprocess_calce.standardize_data(test_df, feature_cols=feature_cols, scaler=scaler)

    data_loading_end_time = time.time()

    train_eval_params = TrainEvalParams()
    moe_nn_params = MoENNParams()
    moe_model = MoENeuralNetwork(in_features=len(feature_cols), out_features=1, moe_nn_params=moe_nn_params).to(device=train_eval_params.device)
    print(f"Parameter Count(Moe): {parameter_count(moe_model)}")

    training_dataset = BatteryDataset(train_df_scaler, feature_cols, target_col)
    test_dataset = BatteryDataset(test_df_scaler, feature_cols, target_col)

    train_loader = DataLoader(training_dataset, batch_size=train_eval_params.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=train_eval_params.batch_size, shuffle=False)

    training_start_time = time.time()
    trainer = Trainer(moe_model, train_eval_params, train_eval_params.device)
    moe_model, losses = trainer.train(train_loader)
    training_end_time = time.time()
    print(f"Training Time: {(training_end_time - training_start_time):.2f} secs")

    ### Training Data Prediction ###
    y_pred = moe_model(torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device='cuda'))[0]
    y = torch.tensor(train_df[target_col].values, dtype=torch.float32, device='cuda')

    plt.plot(y.cpu().numpy(), y_pred.cpu().detach().numpy())
    plt.show()
    trainer.evaluate(train_loader)

    ### Plotting ###
    plt.figure()
    plt.semilogy(losses)
    plt.title("Loss Curve")
    plt.show()

    y, y_pred = trainer.evaluate(val_loader)
    plt.figure()
    plt.plot(y.cpu().numpy()[:10000], 'o', label="True")
    plt.plot(y_pred.cpu().numpy()[:10000], 'o', label="Predicted")
    plt.legend()
    plt.show()