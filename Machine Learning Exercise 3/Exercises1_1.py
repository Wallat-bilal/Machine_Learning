# Exercises.py
# End-to-end MNIST dimensionality-reduction + classification
# Works with the classic UFLDL "mnist_all.mat" that contains train0..train9 and test0..test9

import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# =========================
# 0) CONFIG
# =========================
DATA_DIR = r"C:\Users\walat\PycharmProjects\PythonProject\Machine Learning Exercise 3\Data\mnist_all"
MAT_FILE = "mnist_all.mat"  # expected inside DATA_DIR
SEED = 42
PCA_VAR_TARGETS = [0.90, 0.95, 0.99]  # report how many PCs needed for each
PCA_FOR_VIS = 2                       # for quick scatter visualization
THREE_CLASSES = {5, 6, 8}             # for the LDA vs PCA(2) task

rng = np.random.default_rng(SEED)

# =========================
# 1) LOAD DATA
# =========================
def load_mnist_all_mat(path_dir, mat_file="mnist_all.mat"):
    mat = loadmat(os.path.join(path_dir, mat_file))
    # mat keys: train0..train9, test0..test9 (each: N x 784 uint8)
    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
    for d in range(10):
        Xtr = mat[f"train{d}"]
        Xte = mat[f"test{d}"]
        ytr = np.full((Xtr.shape[0],), d, dtype=int)
        yte = np.full((Xte.shape[0],), d, dtype=int)
        Xtr_list.append(Xtr); ytr_list.append(ytr)
        Xte_list.append(Xte); yte_list.append(yte)
    Xtr = np.vstack(Xtr_list).astype(np.float32) / 255.0
    ytr = np.hstack(ytr_list).astype(int)
    Xte = np.vstack(Xte_list).astype(np.float32) / 255.0
    yte = np.hstack(yte_list).astype(int)
    return Xtr, ytr, Xte, yte

print("Loading MNIST (mnist_all.mat)...")
Xtr, ytr, Xte, yte = load_mnist_all_mat(DATA_DIR, MAT_FILE)
n_tr, d = Xtr.shape
n_te = Xte.shape[0]
print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

# =========================
# 2) STANDARDIZE FEATURES (fit on train)
# =========================
scaler = StandardScaler(with_mean=True, with_std=True)
Xtr_std = scaler.fit_transform(Xtr)
Xte_std = scaler.transform(Xte)

# =========================
# 3) PCA FIT (on TRAIN only)
# =========================
print("\nFitting PCA on standardized train data...")
pca_full = PCA(svd_solver="full", random_state=SEED)
pca_full.fit(Xtr_std)

explained = pca_full.explained_variance_ratio_
cum_explained = np.cumsum(explained)

def n_components_for_variance(target):
    return int(np.searchsorted(cum_explained, target) + 1)

print("Explained variance targets and component counts:")
for v in PCA_VAR_TARGETS:
    print(f"  {int(v*100)}% → {n_components_for_variance(v)} components")

# Save scree/cumulative plots
plt.figure()
plt.plot(np.arange(1, len(explained)+1), explained)
plt.xlabel("Component index")
plt.ylabel("Explained variance ratio")
plt.title("PCA Scree Plot (MNIST)")
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=120)

plt.figure()
plt.plot(np.arange(1, len(cum_explained)+1), cum_explained)
plt.axhline(0.90, ls="--"); plt.axhline(0.95, ls="--"); plt.axhline(0.99, ls="--")
plt.xlabel("Components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Cumulative Explained Variance")
plt.tight_layout()
plt.savefig("pca_cumulative.png", dpi=120)
print("Saved plots: pca_scree.png, pca_cumulative.png")

# =========================
# 4) PROJECT DATA TO PCA SPACES
#    (a) 2D for visualization
#    (b) K components to hit 95% variance
# =========================
pca_vis = PCA(n_components=PCA_FOR_VIS, svd_solver="full", random_state=SEED)
Xtr_vis = pca_vis.fit_transform(Xtr_std)
Xte_vis = pca_vis.transform(Xte_std)

target_var = 0.95
K = n_components_for_variance(target_var)
pca_k = PCA(n_components=K, svd_solver="full", random_state=SEED)
Xtr_pca = pca_k.fit_transform(Xtr_std)
Xte_pca = pca_k.transform(Xte_std)
print(f"\nUsing K={K} PCs to retain ~{int(target_var*100)}% variance")

# Optional: tiny scatter of a subset (2D PCA) for sanity
idx = rng.choice(len(ytr), size=4000, replace=False)
plt.figure(figsize=(6,5))
plt.scatter(Xtr_vis[idx, 0], Xtr_vis[idx, 1], c=ytr[idx], s=5, alpha=0.6)
plt.title("Train set projected to first 2 PCs")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_train_scatter_2d.png", dpi=140)
print("Saved plot: pca_train_scatter_2d.png")

# =========================
# 5) CLASSIFIERS IN REDUCED SPACE (10 classes)
# =========================
def run_knn(train_X, train_y, test_X, test_y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(train_X, train_y)
    yhat = knn.predict(test_X)
    acc = accuracy_score(test_y, yhat)
    return acc, yhat

def run_logreg(train_X, train_y, test_X, test_y, C=1.0):
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-3,
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(train_X, train_y)
    yhat = clf.predict(test_X)
    acc = accuracy_score(test_y, yhat)
    return acc, yhat

print("\n=== Classification in PCA(K) space (10 classes) ===")
acc_knn_pca, yhat_knn_pca = run_knn(Xtr_pca, ytr, Xte_pca, yte, k=3)
print(f"k-NN (k=3) on PCA({K}) → accuracy: {acc_knn_pca*100:.2f}%")

acc_lr_pca, yhat_lr_pca = run_logreg(Xtr_pca, ytr, Xte_pca, yte, C=1.0)
print(f"Logistic Regression on PCA({K}) → accuracy: {acc_lr_pca*100:.2f}%")

print("\n=== Baseline (original standardized space, 10 classes) ===")
acc_knn_base, _ = run_knn(Xtr_std, ytr, Xte_std, yte, k=3)
print(f"k-NN (k=3) on original → accuracy: {acc_knn_base*100:.2f}%")
acc_lr_base, _ = run_logreg(Xtr_std, ytr, Xte_std, yte, C=1.0)
print(f"Logistic Regression on original → accuracy: {acc_lr_base*100:.2f}%")

# Confusion matrix and report for the best PCA model (10 classes)
best_name, best_yhat = ("kNN-PCA", yhat_knn_pca) if acc_knn_pca >= acc_lr_pca else ("LogReg-PCA", yhat_lr_pca)
cm = confusion_matrix(yte, best_yhat)
print(f"\nBest PCA-space model (10 classes): {best_name}")
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification report:")
print(classification_report(yte, best_yhat, digits=4))

# =========================
# 6) THREE-CLASS TASK (5, 6, 8): LDA(2) vs PCA(2)
#     (1) Reduce to 2D using LDA and classify
#     (2) Redo with PCA(2) and compare
#     IMPORTANT: fit scaler on the 5/6/8 subset only
# =========================
print("\n=== Three-class experiment: digits {5,6,8} ===")

# 1) Subset RAW data first (not the already-standardized arrays)
mask_tr_raw = np.isin(ytr, [5, 6, 8])
mask_te_raw = np.isin(yte, [5, 6, 8])

Xtr_568_raw, ytr_568 = Xtr[mask_tr_raw], ytr[mask_tr_raw]
Xte_568_raw, yte_568 = Xte[mask_te_raw], yte[mask_te_raw]

# 2) Fit a NEW scaler on the subset train only
scaler_568 = StandardScaler(with_mean=True, with_std=True)
Xtr_568 = scaler_568.fit_transform(Xtr_568_raw)
Xte_568 = scaler_568.transform(Xte_568_raw)

# --- LDA as a CLASSIFIER (this is the one that should hit ~94%) ---
lda_clf = LinearDiscriminantAnalysis(solver="svd")  # classifier; no n_components needed
lda_clf.fit(Xtr_568, ytr_568)
acc_lda_clf = lda_clf.score(Xte_568, yte_568)
print(f"LDA classifier (5/6/8) → accuracy: {acc_lda_clf*100:.2f}%")

# --- LDA as 2D PROJECTION (for plotting + kNN/LogReg on 2D) ---
lda2 = LinearDiscriminantAnalysis(solver="svd", n_components=2)
Xtr_lda2 = lda2.fit_transform(Xtr_568, ytr_568)
Xte_lda2 = lda2.transform(Xte_568)

plt.figure(figsize=(6,5))
plt.scatter(Xtr_lda2[:,0], Xtr_lda2[:,1], c=ytr_568, s=6, alpha=0.7)
plt.title("Digits {5,6,8} projected to LDA(2)")
plt.xlabel("LD1"); plt.ylabel("LD2")
plt.tight_layout()
plt.savefig("lda_568_scatter_2d.png", dpi=140)
print("Saved plot: lda_568_scatter_2d.png")

# Classify in LDA(2) space
acc_knn_lda2, _ = run_knn(Xtr_lda2, ytr_568, Xte_lda2, yte_568, k=3)
acc_lr_lda2,  _ = run_logreg(Xtr_lda2, ytr_568, Xte_lda2, yte_568, C=1.0)
print(f"k-NN on LDA(2) → accuracy: {acc_knn_lda2*100:.2f}%")
print(f"LogReg on LDA(2) → accuracy: {acc_lr_lda2*100:.2f}%")

# --- PCA(2) comparator on the SAME subset + scaler ---
pca2 = PCA(n_components=2, svd_solver="full", random_state=SEED)
Xtr_pca2 = pca2.fit_transform(Xtr_568)
Xte_pca2 = pca2.transform(Xte_568)

plt.figure(figsize=(6,5))
plt.scatter(Xtr_pca2[:,0], Xtr_pca2[:,1], c=ytr_568, s=6, alpha=0.7)
plt.title("Digits {5,6,8} projected to PCA(2)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_568_scatter_2d.png", dpi=140)
print("Saved plot: pca_568_scatter_2d.png")

acc_knn_pca2, _ = run_knn(Xtr_pca2, ytr_568, Xte_pca2, yte_568, k=3)
acc_lr_pca2,  _ = run_logreg(Xtr_pca2, ytr_568, Xte_pca2, yte_568, C=1.0)
print(f"k-NN on PCA(2) → accuracy: {acc_knn_pca2*100:.2f}%")
print(f"LogReg on PCA(2) → accuracy: {acc_lr_pca2*100:.2f}%")

# Summary for the 3-class task
print("\n============ SUMMARY (three classes: 5,6,8) ============ ")
print(f"LDA classifier (no dim reduction step in pipeline): {acc_lda_clf*100:.2f}%")
print(f"k-NN  on LDA(2): {acc_knn_lda2*100:.2f}%")
print(f"LogReg on LDA(2): {acc_lr_lda2*100:.2f}%")
print(f"k-NN  on PCA(2): {acc_knn_pca2*100:.2f}%")
print(f"LogReg on PCA(2): {acc_lr_pca2*100:.2f}%")
print("Saved figures: lda_568_scatter_2d.png, pca_568_scatter_2d.png")


# =========================
# 7) QUICK SUMMARY
# =========================
print("\n================ SUMMARY (10 classes) ================ ")
for v in PCA_VAR_TARGETS:
    print(f"PCs to reach {int(v*100)}% variance: {n_components_for_variance(v)}")
print(f"Using K={K} PCs for evaluation")
print(f"k-NN  (orig) : {acc_knn_base*100:.2f}%")
print(f"LogReg (orig): {acc_lr_base*100:.2f}%")
print(f"k-NN  PCA({K}): {acc_knn_pca*100:.2f}%")
print(f"LogReg PCA({K}): {acc_lr_pca*100:.2f}%")
print("Saved figures: pca_scree.png, pca_cumulative.png, pca_train_scatter_2d.png")

print("\n============ SUMMARY (three classes: 5,6,8) ============ ")
print(f"k-NN  on LDA(2): {acc_knn_lda2*100:.2f}%")
print(f"LogReg on LDA(2): {acc_lr_lda2*100:.2f}%")
print(f"k-NN  on PCA(2): {acc_knn_pca2*100:.2f}%")
print(f"LogReg on PCA(2): {acc_lr_pca2*100:.2f}%")
print("Saved figures: lda_568_scatter_2d.png, pca_568_scatter_2d.png")
