import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

FILE_PATH = "robot_data.csv"
START_ROW = 2000
END_ROW = 7000

EPOCHS = 1500
LR = 1e-3

HZ_FALLBACK = 125
DT_FALLBACK = 1.0 / HZ_FALLBACK

WINDOW_DEFAULT = 101
WINDOW_J5 = 301
POLY_ORDER = 3

TRAIN_RATIO = 0.8

SAVE_MODEL = "trajectory_nn.pth"
SAVE_FIG = "joint_trajectory_fit_nn.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_time(df: pd.DataFrame) -> np.ndarray:
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").values.astype(np.float32)
        t = t - t[0]
        if len(t) > 2:
            dt = float(np.nanmedian(np.diff(t)))
            if (not np.isfinite(dt)) or dt <= 0:
                dt = DT_FALLBACK
        else:
            dt = DT_FALLBACK
        if not np.all(np.isfinite(t)):
            t = np.arange(len(df), dtype=np.float32) * dt
        return t
    return np.arange(len(df), dtype=np.float32) * DT_FALLBACK


def prepare_data(file_path: str):
    df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    df = df.iloc[START_ROW:END_ROW, :].copy()

    t_sec = _infer_time(df)
    q = np.zeros((len(df), 6), dtype=np.float32)
    qd = np.zeros((len(df), 6), dtype=np.float32)

    for i in range(6):
        q_col = f"actual_q_{i}"
        qd_col = f"actual_qd_{i}"
        if (q_col not in df.columns) or (qd_col not in df.columns):
            raise ValueError(f"Missing column(s): {q_col} or {qd_col}")

        window = WINDOW_J5 if i == 4 else WINDOW_DEFAULT

        q_i = pd.to_numeric(df[q_col], errors="coerce").values.astype(np.float32)
        qd_i = pd.to_numeric(df[qd_col], errors="coerce").values.astype(np.float32)

        if np.any(~np.isfinite(q_i)) or np.any(~np.isfinite(qd_i)):
            mask = np.isfinite(q_i) & np.isfinite(qd_i)
            t_sec = t_sec[mask]
            q_i = q_i[mask]
            qd_i = qd_i[mask]

        if len(q_i) < window + 5:
            raise ValueError("Segment too short for the selected Savitzky-Golay window.")

        q[:, i] = savgol_filter(q_i, window, POLY_ORDER).astype(np.float32)
        qd[:, i] = savgol_filter(qd_i, window, POLY_ORDER).astype(np.float32)

    n = min(len(t_sec), len(q), len(qd))
    t_sec = t_sec[:n]
    q = q[:n]
    qd = qd[:n]

    t0 = float(t_sec[0])
    t1 = float(t_sec[-1])
    t_span = max(1e-6, t1 - t0)
    t_norm = (t_sec - t0) / t_span

    scaler_q = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler_qd = MinMaxScaler(feature_range=(0.0, 1.0))

    q_s = scaler_q.fit_transform(q).astype(np.float32)
    qd_s = scaler_qd.fit_transform(qd).astype(np.float32)

    return t_norm.astype(np.float32), t_span, q_s, qd_s, scaler_q, scaler_qd


class SmoothTrajectoryNet(nn.Module):
    def __init__(self, fourier_dim=128, hidden=512):
        super().__init__()
        B = torch.randn(1, fourier_dim) * 3.5
        self.register_buffer("B", B)

        self.net = nn.Sequential(
            nn.Linear(2 * fourier_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 256),
            nn.Tanh(),
            nn.Linear(256, 6),
        )

    def forward(self, t_norm):
        proj = t_norm @ self.B
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(feat)


def train_and_plot():
    torch.manual_seed(42)
    np.random.seed(42)

    t_norm_np, t_span, q_np, qd_np, scaler_q, scaler_qd = prepare_data(FILE_PATH)

    n = len(t_norm_np)
    split = int(TRAIN_RATIO * n)

    t_train = torch.tensor(t_norm_np[:split, None], dtype=torch.float32, device=DEVICE)
    q_train = torch.tensor(q_np[:split], dtype=torch.float32, device=DEVICE)

    t_test = torch.tensor(t_norm_np[split:, None], dtype=torch.float32, device=DEVICE)
    q_test = torch.tensor(q_np[split:], dtype=torch.float32, device=DEVICE)

    model = SmoothTrajectoryNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 1.0], dtype=torch.float32, device=DEVICE)

    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        q_pred = model(t_train)
        loss_train = torch.mean(loss_weights * (q_pred - q_train) ** 2)

        loss_train.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            q_pred_test = model(t_test)
            loss_test = torch.mean(loss_weights * (q_pred_test - q_test) ** 2)

        train_losses.append(float(loss_train.item()))
        test_losses.append(float(loss_test.item()))

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {loss_train.item():.6f} | Test Loss: {loss_test.item():.6f}")

    model.eval()
    with torch.no_grad():
        q_pred_all = model(torch.tensor(t_norm_np[:, None], dtype=torch.float32, device=DEVICE)).cpu().numpy()

    q_true_all = q_np
    t_plot = t_norm_np

    mse_per_joint = []
    rmse_per_joint = []
    r2_per_joint = []

    split_idx = split
    q_true_test_np = q_np[split_idx:]
    q_pred_test_np = q_pred_all[split_idx:]

    for j in range(6):
        mse_j = mean_squared_error(q_true_test_np[:, j], q_pred_test_np[:, j])
        rmse_j = float(np.sqrt(mse_j))
        r2_j = r2_score(q_true_test_np[:, j], q_pred_test_np[:, j])

        mse_per_joint.append(float(mse_j))
        rmse_per_joint.append(float(rmse_j))
        r2_per_joint.append(float(r2_j))

    print("\n" + "=" * 60)
    print("Test Metrics (scaled 0..1) per joint")
    for j in range(6):
        print(f"Joint {j+1}: R2={r2_per_joint[j]:.6f} | MSE={mse_per_joint[j]:.6f} | RMSE={rmse_per_joint[j]:.6f}")
    print("=" * 60)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(t_plot, q_true_all[:, i], label="Actual (filtered)", linewidth=3)
        axes[i].plot(t_plot, q_pred_all[:, i], linestyle="--", label="NN predicted", linewidth=2.5)
        axes[i].set_title(f"Joint {i+1} Trajectory", fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel("Scaled Position")

    plt.figlegend(["Actual (filtered)", "NN predicted"], loc="lower center", ncol=2, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(SAVE_FIG, dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted MSE")
    plt.title("Training Curve (NN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nn_training_curve.png", dpi=300)
    plt.close()

    payload = {
        "model_state_dict": model.state_dict(),
        "t_span": float(t_span),
        "scaler_q_min": scaler_q.data_min_.astype(np.float32),
        "scaler_q_max": scaler_q.data_max_.astype(np.float32),
        "scaler_qd_min": scaler_qd.data_min_.astype(np.float32),
        "scaler_qd_max": scaler_qd.data_max_.astype(np.float32),
        "metrics_test_scaled": {
            "mse_per_joint": np.array(mse_per_joint, dtype=np.float32),
            "rmse_per_joint": np.array(rmse_per_joint, dtype=np.float32),
            "r2_per_joint": np.array(r2_per_joint, dtype=np.float32),
        },
        "config": {
            "FILE_PATH": FILE_PATH,
            "START_ROW": START_ROW,
            "END_ROW": END_ROW,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "SMOOTHING_WINDOW_DEFAULT": WINDOW_DEFAULT,
            "SMOOTHING_WINDOW_J5": WINDOW_J5,
            "POLY_ORDER": POLY_ORDER,
            "TRAIN_RATIO": TRAIN_RATIO,
        },
    }
    torch.save(payload, SAVE_MODEL)

    print(f"\nSaved: {SAVE_MODEL}")
    print(f"Saved: {SAVE_FIG}")
    print("Saved: nn_training_curve.png")


if __name__ == "__main__":
    train_and_plot()
