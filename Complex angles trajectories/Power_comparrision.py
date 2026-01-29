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

HZ = 125
DT_DEFAULT = 1.0 / HZ

EPOCHS = 500
LR = 0.002
BATCH_SIZE = 1024

SMOOTHING_WINDOW = 20
PHYSICS_WEIGHT = 0.1

PLOT_LIMIT_S = None  
SAVE_PLOT = "pinn_power_simple_vs_complex.png"
SAVE_MODEL = "pinn_power_model.pth"

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "lines.linewidth": 2.5, "font.family": "sans-serif"})

def rolling_smooth(df, window):
    cols_to_smooth = ["actual_robot_voltage", "actual_robot_current"]
    for c in df.columns:
        if ("actual_qd" in c) or (c in cols_to_smooth):
            df[c] = df[c].rolling(window=window, min_periods=1).mean()
    return df


def infer_time_and_dt(df):
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").values.astype(np.float32)
        t = t - t[0]
        if len(t) > 2:
            dt = float(np.median(np.diff(t)))
            if (not np.isfinite(dt)) or dt <= 0:
                dt = DT_DEFAULT
        else:
            dt = DT_DEFAULT
        return t, dt
    else:
        t = np.arange(len(df), dtype=np.float32) * DT_DEFAULT
        return t, DT_DEFAULT


def load_and_prep(filename):
    df = pd.read_csv(filename, sep=r"\s+", engine="python")
    df = rolling_smooth(df, SMOOTHING_WINDOW)

    # Power in Watts
    df["power_w"] = pd.to_numeric(df["actual_robot_voltage"], errors="coerce") * \
                    pd.to_numeric(df["actual_robot_current"], errors="coerce").abs()

    vel_cols = [c for c in df.columns if "actual_qd" in c]
    if len(vel_cols) < 6:
        raise ValueError(f"{filename}: expected 6 velocity columns like actual_qd_0..5")

    X_vel = df[vel_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    t, dt = infer_time_and_dt(df)
    X_acc = (np.gradient(X_vel, axis=0) / dt).astype(np.float32)

    X = np.hstack([X_vel, X_acc]).astype(np.float32)   # [v(6), a(6)]
    y = df[["power_w"]].values.astype(np.float32)
    return X, y, t


def time_split(X, y, t, train_ratio=0.8):
    n = len(X)
    split = int(train_ratio * n)
    return X[:split], y[:split], t[:split], X[split:], y[split:], t[split:]


#Model Arc
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
        
        self.friction = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))  # Kf
        self.inertia = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))   # Ki

    def forward(self, x):
        return self.net(x)


def safe_limit_index(t, limit_val):
    if limit_val is None:
        return len(t) - 1
    idxs = np.where(t <= float(limit_val))[0]
    if len(idxs) == 0:
        return len(t) - 1
    return int(idxs[-1])

def train_pinn_and_plot():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading datasets...")
    X_s, y_s, t_s = load_and_prep(FILE_SIMPLE)
    X_c, y_c, t_c = load_and_prep(FILE_COMPLEX)

    Xs_tr, ys_tr, ts_tr, Xs_te, ys_te, ts_te = time_split(X_s, y_s, t_s, 0.8)
    Xc_tr, yc_tr, tc_tr, Xc_te, yc_te, tc_te = time_split(X_c, y_c, t_c, 0.8)

    X_train_raw = np.vstack([Xs_tr, Xc_tr]).astype(np.float32)
    y_train_raw = np.vstack([ys_tr, yc_tr]).astype(np.float32)

    X_test_raw = np.vstack([Xs_te, Xc_te]).astype(np.float32)
    y_test_raw = np.vstack([ys_te, yc_te]).astype(np.float32)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_x.fit_transform(X_train_raw).astype(np.float32)
    y_train_sc = scaler_y.fit_transform(y_train_raw).astype(np.float32)
    X_test_sc = scaler_x.transform(X_test_raw).astype(np.float32)
    y_test_sc = scaler_y.transform(y_test_raw).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_sc_t = torch.tensor(X_train_sc, dtype=torch.float32, device=device)
    y_train_sc_t = torch.tensor(y_train_sc, dtype=torch.float32, device=device)
    X_train_raw_t = torch.tensor(X_train_raw, dtype=torch.float32, device=device)

    dataset = torch.utils.data.TensorDataset(X_train_sc_t, y_train_sc_t, X_train_raw_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = RobotPINN(input_dim=X_train_sc.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\nTraining model...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        L_total = 0.0
        batches = 0

        for xb_sc, yb_sc, xb_raw in loader:
            optimizer.zero_grad()

            p_pred_sc = model(xb_sc)

            loss_data = loss_fn(p_pred_sc, yb_sc)

            X_tensor = xb_raw
            vel = X_tensor[:, :6]
            acc = X_tensor[:, 6:]
            v_mag = torch.sum(torch.abs(vel), dim=1, keepdim=True)
            a_mag = torch.sum(torch.abs(acc), dim=1, keepdim=True)
            p_phys_raw = (model.friction * v_mag**2) + (model.inertia * v_mag * a_mag)
            p_phys_sc = torch.tensor(
                scaler_y.transform(p_phys_raw.detach().cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )

            loss_phys = torch.mean((p_pred_sc - p_phys_sc) ** 2)
            loss = loss_data + (PHYSICS_WEIGHT * loss_phys)
            loss.backward()
            optimizer.step()

            L_total += float(loss.item())
            batches += 1

        if epoch % 50 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{EPOCHS} | Loss {L_total/max(1,batches):.6f} "
                f"| Kf {model.friction.item():.4f} | Ki {model.inertia.item():.4f}"
            )

#eval
    model.eval()
    with torch.no_grad():
        pred_test_sc = model(torch.tensor(X_test_sc, dtype=torch.float32, device=device)).cpu().numpy()
    pred_test_w = scaler_y.inverse_transform(pred_test_sc).flatten()
    true_test_w = y_test_raw.flatten()

    r2 = r2_score(true_test_w, pred_test_w)
    mse = mean_squared_error(true_test_w, pred_test_w)
    rmse = float(np.sqrt(mse))

    print("\n" + "=" * 50)
    print("Test Metrics (Watts)")
    print(f"R2:   {r2:.6f}")
    print(f"MSE:  {mse:.6f} (W^2)")
    print(f"RMSE: {rmse:.6f} (W)")
    print("=" * 50)

    def predict_full(X_raw):
        X_sc = scaler_x.transform(X_raw).astype(np.float32)
        with torch.no_grad():
            p_sc = model(torch.tensor(X_sc, dtype=torch.float32, device=device)).cpu().numpy()
        return scaler_y.inverse_transform(p_sc).flatten()

    pred_simple = predict_full(X_s)
    pred_complex = predict_full(X_c)

    idx_s = safe_limit_index(t_s, PLOT_LIMIT_S)
    idx_c = safe_limit_index(t_c, PLOT_LIMIT_S)

    t_plot_s = t_s[: idx_s + 1]
    t_plot_c = t_c[: idx_c + 1]
    p_plot_s = pred_simple[: idx_s + 1]
    p_plot_c = pred_complex[: idx_c + 1]

    plt.figure(figsize=(14, 7))
    plt.plot(t_plot_s, p_plot_s, linestyle="--", label="Simple Trajectory (PINN)")
    plt.plot(t_plot_c, p_plot_c, label="Complex Trajectory (PINN)")

    def label_peak(t, p, dy=1.5):
        if len(p) == 0:
            return
        i = int(np.argmax(p))
        plt.text(t[i], p[i] + dy, f"{p[i]:.1f}W", fontweight="bold", ha="center")

    label_peak(t_plot_s, p_plot_s)
    label_peak(t_plot_c, p_plot_c)

    if PLOT_LIMIT_S is not None:
        plt.xlim(0, float(PLOT_LIMIT_S))

    plt.title("Instantaneous Power Comparison (Simple vs Complex)")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT, dpi=300)
    plt.show()

    print(f"\nSaved plot: {SAVE_PLOT}")

    payload = {
        "model_state_dict": model.state_dict(),
        "scaler_x_mean": scaler_x.mean_.astype(np.float32),
        "scaler_x_scale": scaler_x.scale_.astype(np.float32),
        "scaler_y_mean": scaler_y.mean_.astype(np.float32),
        "scaler_y_scale": scaler_y.scale_.astype(np.float32),
        "learned_constants": {"Kf": float(model.friction.item()), "Ki": float(model.inertia.item())},
        "config": {
            "EPOCHS": EPOCHS,
            "LR": LR,
            "BATCH_SIZE": BATCH_SIZE,
            "SMOOTHING_WINDOW": SMOOTHING_WINDOW,
            "PHYSICS_WEIGHT": PHYSICS_WEIGHT,
            "HZ": HZ,
        },
    }
    torch.save(payload, SAVE_MODEL)
    print(f"Saved model: {SAVE_MODEL}")


if __name__ == "__main__":
    train_pinn_and_plot()
