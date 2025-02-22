from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils.trainer import Trainer
from config import TrainEvalParams, MoENNParams, CalceDataParams, HistoryColsParams, TransformerParams
from utils.preprocess_data import PreprocessCalceA123
from time_series_models.Transformer_MoE import Transformer
from time_series_models.custom_dataset import SequenceBatteryDataSet

moe_params = MoENNParams(n_experts=16, top_k=1)
vanilla_nn_params = MoENNParams(n_experts=1, top_k=1)
nn_params = [moe_params, vanilla_nn_params]

transformer_params = TransformerParams()
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

train_params = TrainEvalParams()
train_dataset = SequenceBatteryDataSet(train_feature_data, train_target_data, device=train_params.device)
train_loader = DataLoader(train_dataset, batch_size=train_params.batch_size, shuffle=True)

test_dataset = SequenceBatteryDataSet(test_feature_data, test_target_data, device=train_params.device)
test_loader = DataLoader(test_dataset, batch_size=train_params.batch_size, shuffle=True)

model_names = ["moe", "vanilla_nn"]
models = []
model_losses = []
trainers = []

for params in nn_params:

    model = Transformer(len(seq_cols), 1, transformer_params, params, device="cuda")
    print(f"Total Params: {sum([p.numel() for p in model.parameters()])}")

    trainer = Trainer(model, train_params, device='cuda')

    model, losses = trainer.train(train_loader)
    models.append(model); model_losses.append(losses); trainers.append(trainer)

plt.figure()
for index, losses in enumerate(model_losses):
    plt.semilogy(losses[5:], label=model_names[index])
plt.title("Loss Curve")
plt.legend()
plt.savefig("results/loss_curve.png")
plt.show()

plt.figure()
for index, trainer in enumerate(trainers):
    y_train, y_pred_train = trainer.evaluate(train_loader)
    plt.plot(y_train.cpu().numpy()[::1000][:, -1], y_pred_train.cpu().detach().numpy()[::1000][:, -1], 'o', label=model_names[index])
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Training Data Predictions")
plt.legend()
plt.savefig("results/training_data_pred.png")
plt.show()

plt.figure()
for index, trainer in enumerate(trainers):
    y_train, y_pred_train = trainer.evaluate(test_loader)
    plt.plot(y_train.cpu().numpy()[::1000][:, -1], y_pred_train.cpu().detach().numpy()[::1000][:, -1], 'o', label=model_names[index])
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Testing Data Predictions")
plt.legend()
plt.savefig("results/testing_data_pred.png")
plt.show()