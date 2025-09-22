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
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# =========================
# I could also use Gaussian classifier (QDA/LDA) on PCA features
# =========================

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
#qda = QuadraticDiscriminantAnalysis(store_covariance=False)
#qda.fit(Xtr_pca, ytr); acc_qda = qda.score(Xte_pca, yte)

#lda_clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")  # robust in high-dim
#lda_clf.fit(Xtr_pca, ytr); acc_lda = lda_clf.score(Xte_pca, yte)


# =========================
# 0) CONFIG
# =========================
DATA_DIR = r"C:\Users\walat\PycharmProjects\PythonProject\Machine Learning Exercise 3\Data\mnist_all"
MAT_FILE = "mnist_all.mat"  # expected inside DATA_DIR
SEED = 42
PCA_VAR_TARGETS = [0.90, 0.95, 0.99]  # report how many PCs needed for each
PCA_FOR_VIS = 2                       # for quick scatter visualization

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
# 2) STANDARDIZE FEATURES
#    (center + scale using train stats)
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
#    (a) small 2D for visualization
#    (b) K components to hit 95% variance as default
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
scatter = plt.scatter(Xtr_vis[idx, 0], Xtr_vis[idx, 1], c=ytr[idx], s=5, alpha=0.6)
plt.title("Train set projected to first 2 PCs")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_train_scatter_2d.png", dpi=140)
print("Saved plot: pca_train_scatter_2d.png")

# =========================
# 5) CLASSIFIERS IN REDUCED SPACE
#    (a) k-NN in PCA(K) space
#    (b) Logistic Regression in PCA(K) space
#    (c) (optional) baselines in original space for comparison
# =========================
def run_knn(train_X, train_y, test_X, test_y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(train_X, train_y)
    yhat = knn.predict(test_X)
    acc = accuracy_score(test_y, yhat)
    return acc, yhat

def run_logreg(train_X, train_y, test_X, test_y, C=1.0):
    # multinomial softmax
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,  # try 0.5 or 0.2 if still slow
        solver="lbfgs",  # or "saga"
        max_iter=1000,  # was 200
        tol=1e-3,  # slightly looser tolerance
        random_state=SEED,
        n_jobs=-1  # keep if your sklearn version supports it
    )
    clf.fit(train_X, train_y)
    yhat = clf.predict(test_X)
    acc = accuracy_score(test_y, yhat)
    return acc, yhat

print("\n=== Classification in PCA(K) space ===")
acc_knn_pca, yhat_knn_pca = run_knn(Xtr_pca, ytr, Xte_pca, yte, k=3)
print(f"k-NN (k=3) on PCA({K}) → accuracy: {acc_knn_pca*100:.2f}%")

acc_lr_pca, yhat_lr_pca = run_logreg(Xtr_pca, ytr, Xte_pca, yte, C=1.0)
print(f"Logistic Regression on PCA({K}) → accuracy: {acc_lr_pca*100:.2f}%")

print("\n=== Baseline (original standardized space) ===")
acc_knn_base, _ = run_knn(Xtr_std, ytr, Xte_std, yte, k=3)
print(f"k-NN (k=3) on original → accuracy: {acc_knn_base*100:.2f}%")
acc_lr_base, _ = run_logreg(Xtr_std, ytr, Xte_std, yte, C=1.0)
print(f"Logistic Regression on original → accuracy: {acc_lr_base*100:.2f}%")

# Confusion matrix and report for the best PCA model
best_name, best_yhat = ("kNN-PCA", yhat_knn_pca) if acc_knn_pca >= acc_lr_pca else ("LogReg-PCA", yhat_lr_pca)
cm = confusion_matrix(yte, best_yhat)
print(f"\nBest PCA-space model: {best_name}")
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification report:")
print(classification_report(yte, best_yhat, digits=4))

# =========================
# 6) QUICK SUMMARY
# =========================
print("\n================ SUMMARY ================ ")
for v in PCA_VAR_TARGETS:
    print(f"PCs to reach {int(v*100)}% variance: {n_components_for_variance(v)}")
print(f"Using K={K} PCs for evaluation")
print(f"k-NN  (orig) : {acc_knn_base*100:.2f}%")
print(f"LogReg (orig): {acc_lr_base*100:.2f}%")
print(f"k-NN  PCA({K}): {acc_knn_pca*100:.2f}%")
print(f"LogReg PCA({K}): {acc_lr_pca*100:.2f}%")
print("Saved figures: pca_scree.png, pca_cumulative.png, pca_train_scatter_2d.png")
