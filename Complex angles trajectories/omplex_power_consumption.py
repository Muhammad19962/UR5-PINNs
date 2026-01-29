import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

FILE_SIMPLE = "robot_data.csv"
FILE_COMPLEX = "complex_data.csv"

EPOCHS = 500
LEARNING_RATE = 0.002
BATCH_SIZE = 1024

HZ = 125
DT_DEFAULT = 1.0 / HZ

SMOOTHING_WINDOW = 20
PHYSICS_WEIGHT = 0.1

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "lines.linewidth": 2.5, "font.family": "sans-serif"})


def rolling_smooth(df, window):
    cols_to_smooth = ["actual_robot_voltage", "actual_robot_current"]
    for c in df.columns:
        if ("actual_qd" in c) or (c in cols_to_smooth):
            df[c] = df[c].rolling(window=window, min_periods=1).mean()
    return df


def load_and_prep(filename):
    df = pd.read_csv(filename, sep=r"\s+", engine="python")
    df = rolling_smooth(df, SMOOTHING_WINDOW)

    df["power_w"] = df["actual_robot_voltage"] * df["actual_robot_current"].abs()

    vel_cols = [c for c in df.columns if "actual_qd" in c]
    if len(vel_cols) < 6:
        raise ValueError(f"Expected 6 velocity columns, found {len(vel_cols)} in {filename}")

    X_vel = df[vel_cols].values.astype(np.float32)

    if "timestamp" in df.columns:
        t = df["timestamp"].values.astype(np.float32)
        t = t - t[0]
        dt = float(np.median(np.diff(t))) if len(t) > 2 else DT_DEFAULT
        if (not np.isfinite(dt)) or dt <= 0:
            dt = DT_DEFAULT
    else:
        t = np.arange(len(df), dtype=np.float32) * DT_DEFAULT
        dt = DT_DEFAULT

    X_acc = (np.gradient(X_vel, axis=0) / dt).astype(np.float32)

    X = np.hstack([X_vel, X_acc]).astype(np.float32)  # [v(6), a(6)]
    y = df[["power_w"]].values.astype(np.float32)

    return X, y, t


def time_split(X, y, t, train_ratio=0.8):
    n = len(X)
    if n < 100:
        raise ValueError("Not enough samples for a meaningful split.")
    split = int(train_ratio * n)
    return X[:split], y[:split], t[:split], X[split:], y[split:], t[split:]


class RobotPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.friction = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.inertia = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x):
        return self.net(x)


def safe_limit_index(t, limit_val):
    idxs = np.where(t <= limit_val)[0]
    if len(idxs) == 0:
        return len(t) - 1
    return int(idxs[-1])


def train_and_eval():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading datasets...")
    X_s, y_s, t_s = load_and_prep(FILE_SIMPLE)
    X_c, y_c, t_c = load_and_prep(FILE_COMPLEX)

    Xs_tr, ys_tr, ts_tr, Xs_te, ys_te, ts_te = time_split(X_s, y_s, t_s, 0.8)
    Xc_tr, yc_tr, tc_tr, Xc_te, yc_te, tc_te = time_split(X_c, y_c, t_c, 0.8)

    X_train_raw = np.vstack([Xs_tr, Xc_tr]).astype(np.float32)
    y_train_raw = np.vstack([ys_tr, yc_tr]).astype(np.float32)

    X_test_raw_s = Xs_te.astype(np.float32)
    y_test_raw_s = ys_te.astype(np.float32)

    X_test_raw_c = Xc_te.astype(np.float32)
    y_test_raw_c = yc_te.astype(np.float32)

    scaler_x = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train_raw).astype(np.float32)
    X_test_sc_s = scaler_x.transform(X_test_raw_s).astype(np.float32)
    X_test_sc_c = scaler_x.transform(X_test_raw_c).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_sc_t = torch.tensor(X_train_sc, dtype=torch.float32, device=device)
    X_train_raw_t = torch.tensor(X_train_raw, dtype=torch.float32, device=device)
    y_train_raw_t = torch.tensor(y_train_raw, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(X_train_sc_t, X_train_raw_t, y_train_raw_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = RobotPINN(input_dim=X_train_sc.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    loss_fn_data = nn.MSELoss()

    train_hist_total, train_hist_data, train_hist_phys = [], [], []

    print("\nTraining (data loss + physics loss)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_L = 0.0
        total_Ld = 0.0
        total_Lp = 0.0
        n_batches = 0

        for xb_sc, xb_raw, yb_raw in loader:
            p_pred = model(xb_sc)
            loss_data = loss_fn_data(p_pred, yb_raw)

            # PINNs Implementation 
            X_tensor = xb_raw
            vel = X_tensor[:, :6]
            acc = X_tensor[:, 6:]
            v_mag = torch.sum(torch.abs(vel), dim=1, keepdim=True)
            a_mag = torch.sum(torch.abs(acc), dim=1, keepdim=True)
            p_phys = (model.friction * v_mag**2) + (model.inertia * v_mag * a_mag)
            loss_phys = torch.mean((p_pred - p_phys)**2)

            loss = loss_data + (0.1 * loss_phys)
            loss.backward()
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            total_L += float(loss.item())
            total_Ld += float(loss_data.item())
            total_Lp += float(loss_phys.item())
            n_batches += 1

        total_L /= max(1, n_batches)
        total_Ld /= max(1, n_batches)
        total_Lp /= max(1, n_batches)

        train_hist_total.append(total_L)
        train_hist_data.append(total_Ld)
        train_hist_phys.append(total_Lp)

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | Total {total_L:.6f} | Data {total_Ld:.6f} | Phys {total_Lp:.6f} "
                f"| Kf {model.friction.item():.4f} | Ki {model.inertia.item():.4f}"
            )

    def predict_watts(X_sc_np):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_sc_np, dtype=torch.float32, device=device)).cpu().numpy()
        return pred.flatten()

    pred_s = predict_watts(X_test_sc_s)
    pred_c = predict_watts(X_test_sc_c)

    true_s = y_test_raw_s.flatten()
    true_c = y_test_raw_c.flatten()

    pred_all = np.concatenate([pred_s, pred_c])
    true_all = np.concatenate([true_s, true_c])

    r2 = r2_score(true_all, pred_all)
    mse = mean_squared_error(true_all, pred_all)
    rmse = float(np.sqrt(mse))

    print("\n" + "=" * 50)
    print("Test Metrics (Watts)")
    print(f"R2:   {r2:.6f}")
    print(f"MSE:  {mse:.6f} (W^2)")
    print(f"RMSE: {rmse:.6f} (W)")
    print("=" * 50)

    plt.figure(figsize=(10, 6))
    plt.plot(train_hist_total, label="Total Loss")
    plt.plot(train_hist_data, label="Data Loss")
    plt.plot(train_hist_phys, label="Physics Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pinn_training_losses.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    n_scatter = min(2000, len(true_all))
    plt.scatter(true_all[:n_scatter], pred_all[:n_scatter], alpha=0.6, s=10, label="Predicted vs Actual")
    lims = [float(min(true_all.min(), pred_all.min())), float(max(true_all.max(), pred_all.max()))]
    plt.plot(lims, lims, "r--", linewidth=2, label="Ideal ($y=x$)")
    plt.title(f"Power Prediction \n$R^2$={r2:.4f}")
    plt.xlabel("Measured Power (W)")
    plt.ylabel("Predicted Power (W)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("pinn_power_scatter.png", dpi=300)
    plt.close()

    limit_val = 30.0
    idx_s = safe_limit_index(ts_te, limit_val)
    idx_c = safe_limit_index(tc_te, limit_val)

    t_plot_s = ts_te[: idx_s + 1]
    t_plot_c = tc_te[: idx_c + 1]

    plt.figure(figsize=(14, 8))
    plt.plot(t_plot_s, pred_s[: idx_s + 1], linestyle="--", label="Simple (Predicted)")
    plt.plot(t_plot_c, pred_c[: idx_c + 1], label="Complex (Predicted)")

    def label_peak(t, p):
        if len(p) == 0:
            return
        i = int(np.argmax(p))
        plt.text(t[i], p[i] + 1.5, f"{p[i]:.1f}W", fontweight="bold", ha="center")

    label_peak(t_plot_s, pred_s[: idx_s + 1])
    label_peak(t_plot_c, pred_c[: idx_c + 1])

    plt.xlim(0, limit_val)
    plt.title("Instantaneous Power Comparison)")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig("pinn_power_simple_vs_complex.png", dpi=300)
    plt.close()

    save_payload = {
        "model_state_dict": model.state_dict(),
        "scaler_x_mean": scaler_x.mean_.astype(np.float32),
        "scaler_x_scale": scaler_x.scale_.astype(np.float32),
        "config": {
            "FILE_SIMPLE": FILE_SIMPLE,
            "FILE_COMPLEX": FILE_COMPLEX,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "BATCH_SIZE": BATCH_SIZE,
            "HZ": HZ,
            "SMOOTHING_WINDOW": SMOOTHING_WINDOW,
            "PHYSICS_WEIGHT": PHYSICS_WEIGHT,
        },
        "learned_constants": {"Kf": float(model.friction.item()), "Ki": float(model.inertia.item())},
    }
    torch.save(save_payload, "pinn_power_model.pth")

    print("\nSaved files:")
    print(" - pinn_power_model.pth")
    print(" - pinn_training_losses.png")
    print(" - pinn_power_scatter.png")
    print(" - pinn_power_simple_vs_complex.png")


if __name__ == "__main__":
    train_and_eval()
