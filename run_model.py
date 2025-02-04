import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from kalman_filter.kf_variants import SimpleEKF
from model.neural_network_model import VoltageNN

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==============================================
# Step 1: Generate Synthetic Demo Data
# ==============================================
def generate_demo_data(num_samples=5000):
    """Generate synthetic battery data (current, voltage, temperature, SOC)."""
    time = np.arange(num_samples)
    dt = 1  # Time step (1 second)

    # Simulate current (discharge/charge cycles)
    current = 10 * np.sin(2 * np.pi * time / 1000) + np.random.normal(0, 0.1, num_samples)

    # Simulate true SOC using Coulomb counting with noise
    Q_max = 5000  # Total capacity (Ah)
    soc_true = np.zeros(num_samples)
    for t in range(1, num_samples):
        soc_true[t] = soc_true[t - 1] + current[t - 1] * dt / Q_max
    soc_true = np.clip(soc_true + np.random.normal(0, 0.001, num_samples), 0, 1)  # Add noise

    # Simulate temperature (sinusoidal variation)
    temperature = 25 + 10 * np.sin(2 * np.pi * time / 2000)

    # Simulate voltage using a nonlinear function of SOC, current, and temperature
    voltage = (
            3.7 * soc_true
            - 0.5 * current
            + 0.02 * (temperature - 25)
            + np.random.normal(0, 0.01, num_samples)  # Measurement noise
    )

    return current, voltage, temperature, soc_true


current, voltage, temperature, soc_true = generate_demo_data()


# Prepare data
X = np.column_stack([soc_true, current, temperature])
y = voltage.reshape(-1, 1)

# Split into train/validation
split = int(0.5 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Convert to PyTorch tensors
train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Initialize model, loss, optimizer
model = VoltageNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}")


# ==============================================
# Step 4: Run EKF with Synthetic Data
# ==============================================
Q_max = 5000  # Must match data generation
ekf = SimpleEKF(Q_max)
dt = 1  # Time step (1 second)

# Test on validation data
soc_ekf = []
soc_coulomb = []
for i in range(split, len(current)):
    # Ground truth (for plotting only)
    true_soc = soc_true[i]

    # Coulomb counting (for comparison)
    if i == split:
        soc_coulomb.append(soc_true[split])
    else:
        soc_coulomb.append(soc_coulomb[-1] + current[i] * dt / Q_max)

    # EKF update
    estimated_soc = ekf.update(
        current[i], voltage[i], temperature[i], dt, model
    )
    soc_ekf.append(estimated_soc)

# ==============================================
# Step 5: Plot Results
# ==============================================
plt.figure(figsize=(12, 6))
plt.plot(soc_true[split:], label="True SOC", linestyle="--")
plt.plot(soc_ekf, label="EKF Estimate")
plt.plot(soc_coulomb, label="Coulomb Counting", alpha=0.6)
plt.xlabel("Time Step")
plt.ylabel("SOC")
plt.title("SOC Estimation using NN + EKF")
plt.legend()
plt.grid(True)
plt.show()