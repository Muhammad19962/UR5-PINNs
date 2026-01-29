import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RobotPowerNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def run_power_model(
    csv_path: str = "robot_data.csv",
    sample_every: int = 1,          
    use_acc: bool = False,          
    power_abs_current: bool = True, 
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
):
    set_seed(seed)

    print("--- Loading UR5 Dataset ---")
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if sample_every < 1:
        sample_every = 1
    if sample_every > 1:
        df = df.iloc[::sample_every, :].copy()

    q_cols = [f"actual_q_{i}" for i in range(6)]
    qd_cols = [f"actual_qd_{i}" for i in range(6)]

    features = q_cols + qd_cols

    if use_acc:
        qdd_cols = [f"actual_qdd_{i}" for i in range(6)]
        features += qdd_cols

    if power_abs_current:
        df["power"] = df["actual_robot_voltage"] * df["actual_robot_current"].abs()
    else:
        df["power"] = df["actual_robot_voltage"] * df["actual_robot_current"]

    target = ["power"]

    df = df[features + target].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 100:
        raise ValueError("Not enough valid rows after cleaning. Check column names and data quality.")

    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    split = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train_raw)
    X_test_s = scaler_X.transform(X_test_raw)

    y_train_s = scaler_y.fit_transform(y_train_raw)
    y_test_s = scaler_y.transform(y_test_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_s, dtype=torch.float32, device=device)

    model = RobotPowerNN(input_dim=X_train_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_train = X_train_t.shape[0]
    steps_per_epoch = max(1, int(np.ceil(n_train / batch_size)))

    train_losses = []
    print(f"Training on {n_train} samples (device={device})...")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for s in range(steps_per_epoch):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= steps_per_epoch
        train_losses.append(epoch_loss)

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4d}/{epochs} | Train Loss (scaled MSE): {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_test_s = model(X_test_t).cpu().numpy()
        y_true_test_s = y_test_t.cpu().numpy()

    y_pred_test = scaler_y.inverse_transform(y_pred_test_s)
    y_true_test = scaler_y.inverse_transform(y_true_test_s)

    r2 = r2_score(y_true_test, y_pred_test)
    mse = mean_squared_error(y_true_test, y_pred_test)
    rmse = float(np.sqrt(mse))

    print("\n" + "=" * 40)
    print("Test Metrics (real units)")
    print(f"R2:    {r2:.6f}")
    print(f"MSE:   {mse:.6f} (W^2)")
    print(f"RMSE:  {rmse:.6f} (W)")
    print("=" * 40)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss (scaled MSE)")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("power_nn_training_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_test[:200], label="True Power")
    plt.plot(y_pred_test[:200], label="Predicted Power", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Power (W)")
    plt.title("Power Predictionn")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("power_nn_test_preview.png", dpi=300)
    plt.close()

    print("Saved: power_nn_training_loss.png, power_nn_test_preview.png")


if __name__ == "__main__":
    run_power_model(
        csv_path="robot_data.csv",
        sample_every=1,        
        use_acc=False,
        power_abs_current=True,
        epochs=300,
        batch_size=512,
        lr=1e-3,
        seed=42,
    )
