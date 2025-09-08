# exercise_1.2.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path(r"C:\Users\walat\PycharmProjects\PythonProject\Machine learning Exercise 1\Data\ex01_data.npy")
DEGREE = 9
LAMBDAS = [10.0, 1.0, 0.1, 0.01]
SEED = 100
N_TRAIN = 10

# -----------------------------
# Utilities
# -----------------------------
def load_xy(npy_path: Path):
    """Load Nx2 array, use first column as x (x1) and second as y (x2)."""
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected Nx2 array; got shape {arr.shape}")
    x = arr[:, 0].astype(float)  # x1 -> x
    y = arr[:, 1].astype(float)  # x2 -> y
    return x, y

def design_matrix(x, degree: int):
    """Vandermonde-style design matrix with columns [x^0, x^1, ..., x^degree]."""
    x = np.asarray(x).reshape(-1)
    return np.vander(x, N=degree + 1, increasing=True)

def fit_ridge(Phi, y, lam: float):
    """
    Closed-form ridge solution:
        w = (Phi^T Phi + lam * I)^(-1) Phi^T y
    """
    y = np.asarray(y).reshape(-1, 1)
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b = Phi.T @ y
    w = np.linalg.solve(A, b)
    return w.reshape(-1)

def predict(Phi, w):
    return Phi @ w

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# Main (Part 2: Ridge @ degree 9)
# -----------------------------
def main():
    # Load data
    x_all, y_all = load_xy(DATA_PATH)
    N = x_all.shape[0]

    # Use the *same* 10 training points as specified (np.random.seed(100))
    np.random.seed(SEED)
    train_idx = np.random.choice(N, size=N_TRAIN, replace=False)
    test_idx = np.setdiff1d(np.arange(N), train_idx)

    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_te, y_te = x_all[test_idx], y_all[test_idx]

    # Dense grid for smooth curves
    x_plot = np.linspace(x_all.min(), x_all.max(), 400)

    # Output directory
    out_dir = Path("ridge_degree9")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for lam in LAMBDAS:
        # Fit ridge model at degree 9
        Phi_tr = design_matrix(x_tr, DEGREE)
        w = fit_ridge(Phi_tr, y_tr, lam=lam)

        # Errors
        yhat_tr = predict(Phi_tr, w)
        tr_mse = mse(y_tr, yhat_tr)

        Phi_te = design_matrix(x_te, DEGREE)
        yhat_te = predict(Phi_te, w)
        te_mse = mse(y_te, yhat_te)

        results.append((lam, tr_mse, te_mse, w))

        # Plot curve
        Phi_plot = design_matrix(x_plot, DEGREE)
        y_plot = predict(Phi_plot, w)

        plt.figure(figsize=(8, 5))
        plt.scatter(x_all, y_all, s=18, alpha=0.25, label="All data")
        plt.scatter(x_tr, y_tr, s=45, label="Train (10 pts)")
        plt.plot(x_plot, y_plot, linewidth=2.2, label=f"Degree {DEGREE}, λ={lam:g}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Ridge regression (degree {DEGREE}) — λ={lam:g}\n"
                  f"Train MSE={tr_mse:.4f} | Test MSE={te_mse:.4f}")
        plt.legend(loc="best")
        plt.tight_layout()

        save_path = out_dir / f"ridge_deg{DEGREE}_lambda_{str(lam).replace('.', '_')}.png"
        plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()

    # Print results table
    print(f"\nMSE for degree {DEGREE} with L2 regularization (train=10 pts, seed={SEED})")
    print(f"{'lambda':>8} | {'Train MSE':>12} | {'Test MSE':>12} | {'||w||^2':>10}")
    print("-" * 50)
    for lam, tr, te, w in results:
        w_norm_sq = float(np.sum(w**2))
        print(f"{lam:8.2g} | {tr:12.6f} | {te:12.6f} | {w_norm_sq:10.2f}")

    print(f"\nSaved plots to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
