import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def run_high_performance_pinn(
    csv_path: str = "robot_data.csv",
    downsample_step: int = 2,
    epochs: int = 1000,
    batch_size: int = 2048,
    lr: float = 0.005,
    physics_weight: float = 0.05,
    use_abs_current: bool = False,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("--- Loading MyUR5 Dataset ---")
    df = pd.read_csv(csv_path, sep=None, engine="python")

    if downsample_step < 1:
        downsample_step = 1
    df = df.iloc[::downsample_step, :].copy()

    q_cols = [f"actual_q_{i}" for i in range(6)]
    qd_cols = [f"actual_qd_{i}" for i in range(6)]
    qdd_cols = [f"target_qdd_{i}" for i in range(6)]
    features = q_cols + qd_cols + qdd_cols

    if use_abs_current:
        df["power"] = df["actual_robot_voltage"] * df["actual_robot_current"].abs()
    else:
        df["power"] = df["actual_robot_voltage"] * df["actual_robot_current"]

    df = df[features + ["power"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 200:
        raise ValueError("Not enough valid rows after cleaning. Check column names and file content.")

    X = df[features].values.astype(np.float32)
    y = df[["power"]].values.astype(np.float32)

    split = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train_raw).astype(np.float32)
    X_test_s = scaler_X.transform(X_test_raw).astype(np.float32)

    y_train_s = scaler_y.fit_transform(y_train_raw).astype(np.float32)
    y_test_s = scaler_y.transform(y_test_raw).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_s_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_s_t = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    X_test_s_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    y_test_s_t = torch.tensor(y_test_s, dtype=torch.float32, device=device)

    X_train_raw_t = torch.tensor(X_train_raw, dtype=torch.float32, device=device)
    X_test_raw_t = torch.tensor(X_test_raw, dtype=torch.float32, device=device)

    def l2_mag(x):
        return torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + 1e-12)

    class RobotPINN(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.friction = nn.Parameter(torch.tensor(0.5, dtype=torch.float32, device=device))
            self.inertia = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, device=device))

        def forward(self, x):
            return self.net(x)

    model = RobotPINN(len(features)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_train = X_train_s_t.shape[0]
    steps_per_epoch = max(1, int(np.ceil(n_train / batch_size)))
    train_loss_hist = []

    print(f"Training on {n_train} samples (device={device})...")
    for epoch in range(1, epochs + 1):
        model.train()

        perm = torch.randperm(n_train, device=device)
        total_epoch_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0

        for s in range(steps_per_epoch):
            idx = perm[s * batch_size : (s + 1) * batch_size]

            xb_s = X_train_s_t[idx]
            yb_s = y_train_s_t[idx]
            xb_raw = X_train_raw_t[idx]

            optimizer.zero_grad()

            pred_s = model(xb_s)
            loss_data = criterion(pred_s, yb_s)

            vel_raw = xb_raw[:, 6:12]
            acc_raw = xb_raw[:, 12:18]

            v_mag = l2_mag(vel_raw)
            a_mag = l2_mag(acc_raw)

            p_phys_raw = (model.friction * (v_mag ** 2)) + (model.inertia * v_mag * a_mag)

            p_phys_s = torch.tensor(
                scaler_y.transform(p_phys_raw.detach().cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )

            loss_phys = torch.mean((pred_s - p_phys_s) ** 2)
            loss = loss_data + (physics_weight * loss_phys)

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()

        total_epoch_loss /= steps_per_epoch
        total_data_loss /= steps_per_epoch
        total_phys_loss /= steps_per_epoch
        train_loss_hist.append(total_epoch_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs} | Total: {total_epoch_loss:.6f} | "
                f"Data: {total_data_loss:.6f} | Phys: {total_phys_loss:.6f} | "
                f"Kf: {model.friction.item():.4f} | Ki: {model.inertia.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        y_pred_test_s = model(X_test_s_t).detach().cpu().numpy()
        y_true_test_s = y_test_s_t.detach().cpu().numpy()

    pred_watts = scaler_y.inverse_transform(y_pred_test_s)
    actual_watts = scaler_y.inverse_transform(y_true_test_s)

    r2 = r2_score(actual_watts, pred_watts)
    mse = mean_squared_error(actual_watts, pred_watts)
    rmse = float(np.sqrt(mse))

    print("\n" + "=" * 45)
    print("Test Metrics (Watts)")
    print(f"R2:   {r2:.6f}")
    print(f"MSE:  {mse:.6f} (W^2)")
    print(f"RMSE: {rmse:.6f} (W)")
    print("=" * 45)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss (scaled)")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pinn_training_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    n_scatter = min(2000, len(actual_watts))
    plt.scatter(actual_watts[:n_scatter], pred_watts[:n_scatter], alpha=0.6, s=10, label="Predicted vs Actual")

    lims = [
        float(min(actual_watts.min(), pred_watts.min())),
        float(max(actual_watts.max(), pred_watts.max())),
    ]
    plt.plot(lims, lims, "r--", linewidth=2, label="Ideal Fit ($y=x$)")
    plt.title(f"UR5 Power Consumption Prediction = {r2:.4f}")
    plt.xlabel("Measured Power (W)")
    plt.ylabel("Predicted Power (W)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("pinn_power_scatter.png", dpi=300)
    plt.close()

    save_payload = {
        "model_state_dict": model.state_dict(),
        "scaler_X_mean": scaler_X.mean_,
        "scaler_X_scale": scaler_X.scale_,
        "scaler_y_mean": scaler_y.mean_,
        "scaler_y_scale": scaler_y.scale_,
        "features": features,
        "downsample_step": downsample_step,
        "use_abs_current": use_abs_current,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "physics_weight": physics_weight,
        "seed": seed,
    }
    torch.save(save_payload, "pinn_power_model.pth")

    print("Saved: pinn_power_model.pth, pinn_training_loss.png, pinn_power_scatter.png")


if __name__ == "__main__":
    run_high_performance_pinn(
        csv_path="robot_data.csv",
        downsample_step=2,
        epochs=1000,
        batch_size=2048,
        lr=0.005,
        physics_weight=0.05,
        use_abs_current=False,
        seed=42,
    )
