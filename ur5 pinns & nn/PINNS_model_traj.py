import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")


FILE_PATH = "robot_data.csv"
START_ROW = 2000
END_ROW = 7000

EPOCHS = 500
LR = 1e-3

HZ_FALLBACK = 125
DT_FALLBACK = 1.0 / HZ_FALLBACK

WINDOW_DEFAULT = 101
WINDOW_J5 = 301
POLY_ORDER = 3

TRAIN_RATIO = 0.8

SAVE_MODEL = "trajectory_pinn.pth"
SAVE_FIG = "joint_trajectory_fit.png"

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


def dq_dt_from_qpred(q_pred, t_norm, t_span):
    grads = []
    for j in range(6):
        g = torch.autograd.grad(
            outputs=q_pred[:, j].sum(),
            inputs=t_norm,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(g)
    dq_dt_norm = torch.cat(grads, dim=1)
    dq_dt_sec = dq_dt_norm / t_span
    return dq_dt_sec


def train_and_plot():
    t_norm_np, t_span, q_np, qd_np, scaler_q, scaler_qd = prepare_data(FILE_PATH)

    n = len(t_norm_np)
    split = int(TRAIN_RATIO * n)

    t_train = torch.tensor(t_norm_np[:split, None], dtype=torch.float32, device=DEVICE, requires_grad=True)
    q_train = torch.tensor(q_np[:split], dtype=torch.float32, device=DEVICE)
    qd_train = torch.tensor(qd_np[:split], dtype=torch.float32, device=DEVICE)

    t_test = torch.tensor(t_norm_np[split:, None], dtype=torch.float32, device=DEVICE, requires_grad=True)
    q_test = torch.tensor(q_np[split:], dtype=torch.float32, device=DEVICE)
    qd_test = torch.tensor(qd_np[split:], dtype=torch.float32, device=DEVICE)

    model = SmoothTrajectoryNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 1.0], dtype=torch.float32, device=DEVICE)

    for epoch in range(EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        q_pred = model(t_train)
        loss_data = torch.mean(loss_weights * (q_pred - q_train) ** 2)

        qd_pred = dq_dt_from_qpred(q_pred, t_train, t_span)
        loss_phys = torch.mean((qd_pred - qd_train) ** 2)

        total_loss = 40.0 * loss_data + 0.5 * loss_phys
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        q_pred_all = model(torch.tensor(t_norm_np[:, None], dtype=torch.float32, device=DEVICE)).cpu().numpy()

    q_true_all = q_np
    t_plot = t_norm_np

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(t_plot, q_true_all[:, i], label="Actual (filtered)", linewidth=3)
        axes[i].plot(t_plot, q_pred_all[:, i], linestyle="--", label="Model predicted", linewidth=2.5)
        axes[i].set_title(f"Joint {i+1} Trajectory", fontweight="bold")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel("Scaled Position")

    plt.figlegend(["Actual (filtered)", "Model predicted"], loc="lower center", ncol=2, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(SAVE_FIG, dpi=300)
    plt.show()

    payload = {
        "model_state_dict": model.state_dict(),
        "t_span": float(t_span),
        "scaler_q_min": scaler_q.data_min_.astype(np.float32),
        "scaler_q_max": scaler_q.data_max_.astype(np.float32),
        "scaler_qd_min": scaler_qd.data_min_.astype(np.float32),
        "scaler_qd_max": scaler_qd.data_max_.astype(np.float32),
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

    print(f"Saved: {SAVE_MODEL}")
    print(f"Saved: {SAVE_FIG}")


if __name__ == "__main__":
    train_and_plot()
