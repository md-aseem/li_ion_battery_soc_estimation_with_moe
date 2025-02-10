import torch
import torch.nn as nn
from config import ExperimentalDesignParams, VanillaNNParams, CalceDataParams
from model.neural_network_model import VanillaNeuralNetwork
from utils.preprocess_data import PreprocessCalceA123

"""
This file will perform an experiment to quantify the length of temperature data on the accuracy of the model.
"""

experimental_design_params = ExperimentalDesignParams()

history_length_list = []
mse_list = []
mae_list = []

for run in range(experimental_design_params.n_runs):

    ### Data Loading ###
    data_params = CalceDataParams()
    preprocess_calce = PreprocessCalceA123()
    drive_cycle_path = r"..\data\calce_lfp\drive_cycles"
    df = preprocess_calce.load_dfs(drive_cycle_path)

    history_lengths = torch.arange(0, 100, 10, dtype=torch.int8)

    for history_length in history_lengths:
        feature_cols = (['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'amb_temp'] +
                        [f"Voltage(V)-{i + 1}" for i in range(history_length)] +
                        [f"Current(A)-{i + 1}" for i in range(history_length)] +
                        [f"Temperature (C)_1-{i + 1}" for i in range(history_length)] )

        target_col = 'soc'

        training_conditions = {"testpart": ["DST", "FUD"]}
        train_df = preprocess_calce.filter(df, training_conditions)
        preprocess_calce.plot(train_df, x_axis="Test_Time(s)", y_axes=["Voltage(V)"])  # plotting the data before training
        train_df = preprocess_calce.add_sequence_data_per_col(train_df, ['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                                                      history_length=[data_params.history_length,
                                                                      data_params.history_length,
                                                                      history_length])

        test_conditions = {"testpart": ["US06"]}
        test_df = preprocess_calce.filter(df, test_conditions)
        test_df = preprocess_calce.add_sequence_data_per_col(test_df, ['Current(A)', 'Voltage(V)', 'Temperature (C)_1'],
                                                      history_length=[data_params.history_length,
                                                                      data_params.history_length,
                                                                      history_length])

        train_df, scaler = preprocess_calce.standardize_data(train_df, feature_cols=feature_cols)
        test_df, _ = preprocess_calce.standardize_data(test_df, feature_cols=feature_cols, scaler=scaler)

        # Neural Network Creation
        nn_params = VanillaNNParams()
        nn_model = VanillaNeuralNetwork(len(feature_cols), out_features=1, nn_params=nn_params)