# exercise_1.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path(r"C:\Users\walat\PycharmProjects\PythonProject\Machine learning Exercise 1\Data\ex01_data.npy")
DEGREES = [0, 1, 2, 3, 4, 9]
SEED = 100
N_TRAIN = 10
LAMBDA = 0.0  # no regularization for Part 1

# -----------------------------
# Utilities
# -----------------------------
def load_xy(npy_path: Path):
    """Load Nx2 array, use first column as x (x1) and second as y (x2)."""
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected Nx2 array; got shape {arr.shape}")
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    return x, y

def design_matrix(x, degree: int):
    """Vandermonde-style design matrix with columns [x^0, x^1, ..., x^degree]."""
    x = np.asarray(x).reshape(-1)
    return np.vander(x, N=degree + 1, increasing=True)

def fit_ridge(Phi, y, lam: float = 0.0):
    """Closed-form ridge/OLS solution: w = (Phi^T Phi + lam*I)^(-1) Phi^T y"""
    y = y.reshape(-1, 1)
    A = Phi.T @ Phi
    if lam > 0:
        A = A + lam * np.eye(A.shape[0])
    w = np.linalg.solve(A, Phi.T @ y)
    return w.reshape(-1)

def predict(Phi, w):
    return Phi @ w

def mse(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# Main experiment (Part 1)
# -----------------------------
# --- replace your main() with this version ---
def main():
    # Load data
    x_all, y_all = load_xy(DATA_PATH)
    N = x_all.shape[0]

    # Select 10 random training points (as required)
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(N, size=N_TRAIN, replace=False)
    test_idx = np.setdiff1d(np.arange(N), train_idx)

    x_tr, y_tr = x_all[train_idx], y_all[train_idx]
    x_te, y_te = x_all[test_idx], y_all[test_idx]

    # Smooth x-grid for plotting curves
    x_plot = np.linspace(x_all.min(), x_all.max(), 400)

    # For saving individual figures
    out_dir = Path("degree_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep results for table
    results = []

    for deg in DEGREES:
        # Fit
        Phi_tr = design_matrix(x_tr, deg)
        w = fit_ridge(Phi_tr, y_tr, lam=0.0)

        # Errors
        tr_mse = mse(y_tr, predict(Phi_tr, w))
        te_mse = mse(y_te, predict(design_matrix(x_te, deg), w))
        results.append((deg, tr_mse, te_mse))

        # Curve on dense grid
        y_plot = predict(design_matrix(x_plot, deg), w)

        # ---- individual figure for this degree ----
        plt.figure(figsize=(8, 5))
        plt.scatter(x_all, y_all, s=18, alpha=0.25, label="All data")
        plt.scatter(x_tr, y_tr, s=45, label="Train (10 pts)")
        plt.plot(x_plot, y_plot, linewidth=2.2, label=f"Fit (deg {deg})")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"Polynomial fit — degree {deg} (seed={SEED})\n"
                  f"Train MSE={tr_mse:.4f} | Test MSE={te_mse:.4f}")
        plt.legend(loc="best")
        plt.tight_layout()

        save_path = out_dir / f"degree_{deg}.png"
        plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()

    # Print results table
    print("\nMSE by degree (train on 10 points, test on the rest)")
    print(f"{'Degree':>6} | {'Train MSE':>12} | {'Test MSE':>12}")
    print("-" * 37)
    for deg, tr, te in results:
        print(f"{deg:>6} | {tr:12.6f} | {te:12.6f}")

    best_test = min(results, key=lambda t: t[2])
    print(f"\nLowest test MSE achieved by degree {best_test[0]}.")

    print(f"\nSaved individual plots to: {out_dir.resolve()}")


# -----------------------------
# Optional: hooks for Parts 2 & 3 (regularization & data size)
# -----------------------------
def fit_and_eval(x, y, degrees, n_train, lam=0.0, seed=SEED):
    """Reusable experiment helper to vary degree, lambda, and number of points."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_train, replace=False)
    tr_idx = idx
    te_idx = np.setdiff1d(np.arange(x.shape[0]), tr_idx)
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_te, y_te = x[te_idx], y[te_idx]

    out = []
    for d in degrees:
        w = fit_ridge(design_matrix(x_tr, d), y_tr, lam=lam)
        tr_mse = mse(y_tr, predict(design_matrix(x_tr, d), w))
        te_mse = mse(y_te, predict(design_matrix(x_te, d), w))
        out.append({"degree": d, "lambda": lam, "n_train": n_train,
                    "train_mse": tr_mse, "test_mse": te_mse, "w": w})
    return out

if __name__ == "__main__":
    main()
