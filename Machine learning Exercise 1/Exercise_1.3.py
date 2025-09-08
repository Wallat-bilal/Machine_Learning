# Exercise_1.3.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path(r"C:\Users\walat\PycharmProjects\PythonProject\Machine learning Exercise 1\Data\ex01_data.npy")
SEED = 100
N_TRAIN = 100
DEGREE = 9

# -----------------------------
# Utilities
# -----------------------------
def load_xy(npy_path: Path):
    """
    Load Nx2 (or Nx3) array.
    Column 0 -> x (x1), Column 1 -> y (x2 noisy).
    If Column 2 exists, treat it as optional ground-truth y (noise-free).
    """
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected Nx2 or Nx3 array; got shape {arr.shape}")
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    y_true = arr[:, 2].astype(float) if arr.shape[1] >= 3 else None
    return x, y, y_true

def design_matrix(x, degree: int):
    """Vandermonde-style matrix with columns [x^0, x^1, ..., x^degree]."""
    x = np.asarray(x).reshape(-1)
    return np.vander(x, N=degree + 1, increasing=True)

def fit_ols(Phi, y):
    """OLS closed form: w = (Phi^T Phi)^(-1) Phi^T y."""
    y = np.asarray(y).reshape(-1, 1)
    A = Phi.T @ Phi
    w = np.linalg.solve(A, Phi.T @ y)
    return w.ravel()

def predict(Phi, w):
    return Phi @ w

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# Main
# -----------------------------
def main():
    # Load data
    x_all, y_all, y_true_all = load_xy(DATA_PATH)
    N = x_all.shape[0]

    # Randomly choose 100 training indices with the required seed
    np.random.seed(SEED)
    train_idx = np.random.choice(N, size=N_TRAIN, replace=False)
    test_idx = np.setdiff1d(np.arange(N), train_idx)

    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_te, y_te = x_all[test_idx], y_all[test_idx]

    # Fit degree-9 polynomial (no regularization for this exercise)
    Phi_tr = design_matrix(x_tr, DEGREE)
    w = fit_ols(Phi_tr, y_tr)

    # Errors
    tr_mse = mse(y_tr, predict(Phi_tr, w))
    te_mse = mse(y_te, predict(design_matrix(x_te, DEGREE), w))

    # Smooth curve for plotting
    x_plot = np.linspace(x_all.min(), x_all.max(), 500)
    y_plot = predict(design_matrix(x_plot, DEGREE), w)

    # Prepare output dir
    out_dir = Path("sample_size_deg9")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_all, y_all, s=18, alpha=0.25, label="All data")
    plt.scatter(x_tr, y_tr, s=40, label=f"Train ({N_TRAIN} pts)")
    plt.plot(x_plot, y_plot, linewidth=2.2, label=f"Fit (deg {DEGREE})")

    # Optional ground truth overlay if available in column 3
    if y_true_all is not None:
        order = np.argsort(x_all)
        plt.plot(x_all[order], y_true_all[order], linewidth=2, alpha=0.9, label="Ground truth")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Degree {DEGREE} polynomial — n_train={N_TRAIN}, seed={SEED}\n"
              f"Train MSE={tr_mse:.4f} | Test MSE={te_mse:.4f}")
    plt.legend(loc="best")
    plt.tight_layout()
    save_path = out_dir / f"deg{DEGREE}_n{N_TRAIN}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

    # Report
    print(f"\nDegree {DEGREE} with n_train={N_TRAIN} (seed={SEED})")
    print(f"Train MSE: {tr_mse:.6f}")
    print(f"Test  MSE: {te_mse:.6f}")
    print(f"Saved plot to: {save_path.resolve()}")

if __name__ == "__main__":
    main()
