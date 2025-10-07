# exercise_lda_pca_compare.py
# MNIST (mnist_all.mat) → LDA(2/9) vs PCA(2/9) dimensionality reduction
# Classification with Gaussian Bayes (full covariance, i.e., QDA) in reduced spaces

import os
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# -------------------
# CONFIG
# -------------------
DATA_DIR = r"C:\Users\walat\PycharmProjects\PythonProject\Machine Learning Exercise 3\Data\mnist_all"
MAT_FILE = "mnist_all.mat"
SEED = 42
SAVE_FIGS = True

rng = np.random.default_rng(SEED)

# -------------------
# DATA LOADING
# -------------------
def create_complete_datasets(mat):
    """Concatenate train0..train9 and test0..test9 into full arrays."""
    trainset, traintargets, testset, testtargets = [], [], [], []
    for d in range(10):
        Xtr = mat[f"train{d}"]
        Xte = mat[f"test{d}"]
        ytr = np.full(len(Xtr), d, dtype=int)
        yte = np.full(len(Xte), d, dtype=int)
        trainset.append(Xtr); traintargets.append(ytr)
        testset.append(Xte);  testtargets.append(yte)
    trainset = np.concatenate(trainset).astype(np.float32) / 255.0
    traintargets = np.concatenate(traintargets).astype(int)
    testset = np.concatenate(testset).astype(np.float32) / 255.0
    testtargets = np.concatenate(testtargets).astype(int)
    return trainset, traintargets, testset, testtargets

print("Loading MNIST (mnist_all.mat)...")
mat = loadmat(os.path.join(DATA_DIR, MAT_FILE))
Xtr, ytr, Xte, yte = create_complete_datasets(mat)
print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

# -------------------
# STANDARDIZE (fit on train only)
# -------------------
scaler = StandardScaler(with_mean=True, with_std=True)
Xtr_std = scaler.fit_transform(Xtr)
Xte_std = scaler.transform(Xte)

# -------------------
# REDUCERS
# -------------------
def pca_reduce(Xtr_s, Xte_s, n):
    pca = PCA(n_components=n, svd_solver="full", random_state=SEED)
    Ztr = pca.fit_transform(Xtr_s)
    Zte = pca.transform(Xte_s)
    return Ztr, Zte, pca

def lda_reduce(Xtr_s, ytr, Xte_s, n):
    # LDA can return up to C-1=9 components for 10 classes
    lda = LDA(solver="svd", n_components=n)
    Ztr = lda.fit_transform(Xtr_s, ytr)
    Zte = lda.transform(Xte_s)
    return Ztr, Zte, lda

# -------------------
# GAUSSIAN BAYES (full covariance, QDA) in reduced space
# -------------------
def fit_gaussians_full(X, y, eps=1e-6):
    """Per-class mean, full covariance (regularized), and MLE priors."""
    classes = np.unique(y)
    params = {}
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(axis=0)
        # rowvar=False → (dxd) covariance
        Sigma = np.cov(Xc, rowvar=False) + eps * np.eye(X.shape[1])
        params[c] = (mu, Sigma, Xc.shape[0])
    N = X.shape[0]
    priors = {c: params[c][2] / N for c in params}
    return params, priors

def log_gaussian_pdf(X, mu, Sigma):
    """Vectorized log N(x|mu,Sigma)"""
    X = np.atleast_2d(X)
    d = X.shape[1]
    # Use eigh for stability on symmetric matrices
    vals, vecs = np.linalg.eigh(Sigma)
    # regularize small eigenvalues
    vals = np.clip(vals, 1e-12, None)
    iSigma = (vecs @ np.diag(1.0/vals) @ vecs.T)
    logdet = np.log(vals).sum()
    diff = X - mu
    m = np.einsum('ni,ij,nj->n', diff, iSigma, diff)
    return -0.5 * (d*np.log(2*np.pi) + logdet + m)

def predict_gaussian_qda(X, params, priors):
    classes = sorted(params.keys())
    scores = []
    for c in classes:
        mu, Sigma, _ = params[c]
        scores.append(log_gaussian_pdf(X, mu, Sigma) + np.log(priors[c]))
    scores = np.column_stack(scores)  # (n_samples, n_classes)
    preds = np.argmax(scores, axis=1)
    return np.array([classes[i] for i in preds])

def evaluate_block(name, Ztr, Zte, ytr, yte, plot_cm=True):
    params, priors = fit_gaussians_full(Ztr, ytr)
    yhat = predict_gaussian_qda(Zte, params, priors)
    acc = accuracy_score(yte, yhat)
    print(f"{name} → accuracy: {acc*100:.2f}%")
    # Confusion matrix
    cm = confusion_matrix(yte, yhat, labels=np.arange(10))
    if SAVE_FIGS and plot_cm:
        plt.figure(figsize=(6,5))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        fname = f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(fname, dpi=140)
        print(f"Saved: {fname}")
    return acc, cm

# -------------------
# RUN: PCA(2), PCA(9), LDA(2), LDA(9)
# -------------------
print("\n=== Dimensionality reduction + QDA classification ===")

# PCA(2)
Ztr_pca2, Zte_pca2, pca2 = pca_reduce(Xtr_std, Xte_std, n=2)
if SAVE_FIGS:
    idx = rng.choice(len(ytr), size=4000, replace=False)
    plt.figure(figsize=(6,5))
    plt.scatter(Ztr_pca2[idx,0], Ztr_pca2[idx,1], c=ytr[idx], s=5, alpha=0.6)
    plt.title("PCA(2) — train projection")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig("pca2_scatter.png", dpi=140)
    print("Saved: pca2_scatter.png")
acc_pca2, cm_pca2 = evaluate_block("PCA(2) + QDA", Ztr_pca2, Zte_pca2, ytr, yte)

# PCA(9)
Ztr_pca9, Zte_pca9, pca9 = pca_reduce(Xtr_std, Xte_std, n=9)
acc_pca9, cm_pca9 = evaluate_block("PCA(9) + QDA", Ztr_pca9, Zte_pca9, ytr, yte)

# LDA(2)
Ztr_lda2, Zte_lda2, lda2 = lda_reduce(Xtr_std, ytr, Xte_std, n=2)
if SAVE_FIGS:
    idx = rng.choice(len(ytr), size=4000, replace=False)
    plt.figure(figsize=(6,5))
    plt.scatter(Ztr_lda2[idx,0], Ztr_lda2[idx,1], c=ytr[idx], s=5, alpha=0.6)
    plt.title("LDA(2) — train projection")
    plt.xlabel("LD1"); plt.ylabel("LD2")
    plt.tight_layout(); plt.savefig("lda2_scatter.png", dpi=140)
    print("Saved: lda2_scatter.png")
acc_lda2, cm_lda2 = evaluate_block("LDA(2) + QDA", Ztr_lda2, Zte_lda2, ytr, yte)

# LDA(9)
Ztr_lda9, Zte_lda9, lda9 = lda_reduce(Xtr_std, ytr, Xte_std, n=9)
acc_lda9, cm_lda9 = evaluate_block("LDA(9) + QDA", Ztr_lda9, Zte_lda9, ytr, yte)

# -------------------
# SUMMARY
# -------------------
print("\n================ SUMMARY ================")
print(f"PCA(2) + QDA : {acc_pca2*100:.2f}%")
print(f"PCA(9) + QDA : {acc_pca9*100:.2f}%")
print(f"LDA(2) + QDA : {acc_lda2*100:.2f}%")
print(f"LDA(9) + QDA : {acc_lda9*100:.2f}%")
print("Saved figures: pca2_scatter.png, lda2_scatter.png and confusion matrices for each setting.")
