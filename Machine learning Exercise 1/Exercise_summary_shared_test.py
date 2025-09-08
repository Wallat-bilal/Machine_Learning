# exercise_summary_shared_test.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path(r"C:\Users\walat\PycharmProjects\PythonProject\Machine learning Exercise 1\Data\ex01_data.npy")

# Fixed test set for *all* experiments
TEST_SEED = 999
TEST_FRACTION = 0.20   # try to keep 20% for test (adjusted below if needed)

# Experiment settings (match your assignment)
DEGREES_SWEEP = [0, 1, 2, 3, 4, 9]          # Part 1
RIDGE_DEGREE = 9
RIDGE_LAMBDAS = [10.0, 1.0, 0.1, 0.01]      # Part 2
SEED_TRAIN = 100                            # selection seed for training subsets
N_TRAIN_DEGREE = 10                         # Part 1
N_TRAIN_RIDGE = 10                          # Part 2
N_TRAIN_SAMPLESIZE = 100                    # Part 3

OUT_DIR = Path("shared_test_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def fit_ridge(Phi, y, lam: float = 0.0):
    """
    Ridge/OLS closed-form:
        w = (Phi^T Phi + lam * I)^(-1) Phi^T y
    Set lam=0 for OLS.
    """
    y = np.asarray(y).reshape(-1, 1)
    A = Phi.T @ Phi
    if lam > 0:
        A = A + lam * np.eye(Phi.shape[1])
    w = np.linalg.solve(A, Phi.T @ y)
    return w.ravel()

def predict(Phi, w):
    return Phi @ w

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def choose_shared_test_indices(N, want_test_fraction=0.20, need_train_at_least=100, seed=TEST_SEED):
    """
    Pick a *fixed* test set once, with size adjusted so that the training pool
    has at least `need_train_at_least` points for Part 3.
    """
    rng = np.random.default_rng(seed)
    max_test = N - need_train_at_least
    if max_test < 1:
        raise ValueError(f"Dataset too small: need at least {need_train_at_least + 1} points.")
    test_size = int(round(want_test_fraction * N))
    test_size = max(1, min(test_size, max_test))
    test_idx = rng.choice(N, size=test_size, replace=False)
    return np.sort(test_idx)

def choose_train_indices(pool_idx, n_train, seed=SEED_TRAIN):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool_idx, size=n_train, replace=False))

def save_summary_csv(rows, path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# -----------------------------
# Main
# -----------------------------
def main():
    # Load data
    x_all, y_all, y_true_all = load_xy(DATA_PATH)
    N = x_all.shape[0]

    # Shared test set for *everything*
    test_idx = choose_shared_test_indices(
        N, want_test_fraction=TEST_FRACTION, need_train_at_least=N_TRAIN_SAMPLESIZE, seed=TEST_SEED
    )
    train_pool_idx = np.setdiff1d(np.arange(N), test_idx)

    x_test, y_test = x_all[test_idx], y_all[test_idx]

    # Dense x-grid for smooth curves (for plotting)
    x_plot = np.linspace(x_all.min(), x_all.max(), 600)

    summary_rows = []

    # ---------- Part 1: Degree sweep with 10 training points ----------
    deg10_idx = choose_train_indices(train_pool_idx, N_TRAIN_DEGREE, seed=SEED_TRAIN)
    x_tr_deg, y_tr_deg = x_all[deg10_idx], y_all[deg10_idx]

    deg_results = []
    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_all, y_all, s=14, alpha=0.25, label="All data")
    plt.scatter(x_tr_deg, y_tr_deg, s=40, edgecolor="k", label=f"Train (10 pts)")
    for d in DEGREES_SWEEP:
        Phi_tr = design_matrix(x_tr_deg, d)
        w = fit_ridge(Phi_tr, y_tr_deg, lam=0.0)  # OLS
        tr_mse = mse(y_tr_deg, predict(Phi_tr, w))
        te_mse = mse(y_test, predict(design_matrix(x_test, d), w))
        deg_results.append((d, tr_mse, te_mse))

        y_plot = predict(design_matrix(x_plot, d), w)
        plt.plot(x_plot, y_plot, linewidth=2, label=f"deg {d}")

        summary_rows.append({
            "experiment": "degree_sweep",
            "degree": d,
            "lambda": 0.0,
            "n_train": N_TRAIN_DEGREE,
            "train_mse": f"{tr_mse:.6f}",
            "test_mse": f"{te_mse:.6f}",
            "w_norm_sq": ""
        })

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Part 1 — Degree sweep (shared test set)")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part1_degree_sweep_overlay.png", dpi=150)
    plt.show()

    # (optional) Degree vs MSE chart
    degs = [d for d, _, _ in deg_results]
    tr = [tr for _, tr, _ in deg_results]
    te = [te for _, _, te in deg_results]
    plt.figure(figsize=(7, 4.6))
    order = np.argsort(degs)
    plt.plot(np.array(degs)[order], np.array(tr)[order], marker="o", label="Train MSE")
    plt.plot(np.array(degs)[order], np.array(te)[order], marker="o", label="Test MSE")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("Part 1 — MSE vs degree")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part1_mse_vs_degree.png", dpi=150)
    plt.show()

    # ---------- Part 2: Ridge at degree 9 with same 10 training points ----------
    ridge_results = []
    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_all, y_all, s=14, alpha=0.25, label="All data")
    plt.scatter(x_tr_deg, y_tr_deg, s=40, edgecolor="k", label=f"Train (10 pts)")
    for lam in RIDGE_LAMBDAS:
        Phi_tr = design_matrix(x_tr_deg, RIDGE_DEGREE)
        w = fit_ridge(Phi_tr, y_tr_deg, lam=lam)
        tr_mse = mse(y_tr_deg, predict(Phi_tr, w))
        te_mse = mse(y_test, predict(design_matrix(x_test, RIDGE_DEGREE), w))
        ridge_results.append((lam, tr_mse, te_mse, float(np.sum(w**2))))

        y_plot = predict(design_matrix(x_plot, RIDGE_DEGREE), w)
        plt.plot(x_plot, y_plot, linewidth=2, label=f"λ={lam:g}")

        summary_rows.append({
            "experiment": "ridge_degree9",
            "degree": RIDGE_DEGREE,
            "lambda": lam,
            "n_train": N_TRAIN_RIDGE,
            "train_mse": f"{tr_mse:.6f}",
            "test_mse": f"{te_mse:.6f}",
            "w_norm_sq": f"{np.sum(w**2):.6f}"
        })

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Part 2 — Degree 9 with L2 (shared test set)")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part2_ridge_overlay.png", dpi=150)
    plt.show()

    # (optional) Lambda vs MSE chart (log-x)
    lams = [lam for lam, _, _, _ in ridge_results]
    tr = [tr for _, tr, _, _ in ridge_results]
    te = [te for _, _, te, _ in ridge_results]
    order = np.argsort(lams)
    plt.figure(figsize=(7, 4.6))
    plt.semilogx(np.array(lams)[order], np.array(tr)[order], marker="o", label="Train MSE")
    plt.semilogx(np.array(lams)[order], np.array(te)[order], marker="o", label="Test MSE")
    plt.xlabel("λ (log scale)")
    plt.ylabel("MSE")
    plt.title("Part 2 — MSE vs λ (degree 9)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part2_mse_vs_lambda.png", dpi=150)
    plt.show()

    # ---------- Part 3: Degree 9, 100 training points ----------
    deg100_idx = choose_train_indices(train_pool_idx, N_TRAIN_SAMPLESIZE, seed=SEED_TRAIN)
    x_tr_100, y_tr_100 = x_all[deg100_idx], y_all[deg100_idx]

    Phi_tr_100 = design_matrix(x_tr_100, RIDGE_DEGREE)
    w_100 = fit_ridge(Phi_tr_100, y_tr_100, lam=0.0)  # OLS at degree 9
    tr_mse_100 = mse(y_tr_100, predict(Phi_tr_100, w_100))
    te_mse_100 = mse(y_test, predict(design_matrix(x_test, RIDGE_DEGREE), w_100))

    summary_rows.append({
        "experiment": "sample_size",
        "degree": RIDGE_DEGREE,
        "lambda": 0.0,
        "n_train": N_TRAIN_SAMPLESIZE,
        "train_mse": f"{tr_mse_100:.6f}",
        "test_mse": f"{te_mse_100:.6f}",
        "w_norm_sq": f"{np.sum(w_100**2):.6f}"
    })

    # Overlay plot comparing deg-9 with 10 vs 100 points
    plt.figure(figsize=(9, 5.5))
    plt.scatter(x_all, y_all, s=14, alpha=0.25, label="All data")
    plt.scatter(x_tr_deg, y_tr_deg, s=35, edgecolor="k", label="Train (10)")
    plt.scatter(x_tr_100, y_tr_100, s=20, alpha=0.7, label="Train (100)")

    # fit for 10-point (OLS degree 9) to visualize overfit vs 100
    Phi_tr_10 = design_matrix(x_tr_deg, RIDGE_DEGREE)
    w_10 = fit_ridge(Phi_tr_10, y_tr_deg, lam=0.0)

    y_plot_10 = predict(design_matrix(x_plot, RIDGE_DEGREE), w_10)
    y_plot_100 = predict(design_matrix(x_plot, RIDGE_DEGREE), w_100)

    plt.plot(x_plot, y_plot_10, linewidth=2, label="deg9 (10 pts)")
    plt.plot(x_plot, y_plot_100, linewidth=2, label="deg9 (100 pts)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Part 3 — Effect of sample size (degree 9)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "part3_sample_size_overlay.png", dpi=150)
    plt.show()

    # ---------- Optional: ground truth overlay (if column 3 exists) ----------
    if y_true_all is not None:
        order_all = np.argsort(x_all)
        plt.figure(figsize=(9, 5.5))
        plt.scatter(x_all, y_all, s=10, alpha=0.25, label="Noisy data")
        plt.plot(x_all[order_all], y_true_all[order_all], linewidth=2, label="Ground truth")
        plt.plot(x_plot, y_plot_10, linewidth=2, label="deg9 (10 pts)")
        plt.plot(x_plot, y_plot_100, linewidth=2, label="deg9 (100 pts)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Ground truth vs estimates (if available)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "optional_ground_truth_overlay.png", dpi=150)
        plt.show()

    # ---------- Save summary ----------
    csv_path = OUT_DIR / "summary_results.csv"
    save_summary_csv(summary_rows, csv_path)

    # Pretty print
    print("\n=== Summary (shared test set) ===")
    print(f"{'experiment':<16} {'deg':>3} {'λ':>8} {'n_train':>8} {'train_mse':>12} {'test_mse':>12} {'||w||^2':>12}")
    for r in summary_rows:
        print(f"{r['experiment']:<16} {int(r['degree']):>3} {float(r['lambda']):>8.3g} {int(r['n_train']):>8} "
              f"{r['train_mse']:>12} {r['test_mse']:>12} {r['w_norm_sq']:>12}")
    print(f"\nSaved figures & CSV to: {OUT_DIR.resolve()}")
    print(f"CSV: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
