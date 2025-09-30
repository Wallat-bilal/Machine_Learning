import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture

# -------------------
# CONFIG
# -------------------
DATA_PATH = r"C:\Users\walat\PycharmProjects\PythonProject\Machine Learning Exercise 4\Data\2D568class.mat"
RANDOM_STATE = 42
GRID_STEPS = 300     # resolution for contour grid
SCALE_BY_255 = True  # set True if your 2D features were saved as 0–255 (hints divided by 255)

# -------------------
# LOAD DATA (train only, 2D for classes 5,6,8)
# -------------------
mat = loadmat(DATA_PATH)
# Keys per the provided data/hints: trn5_2dim, trn6_2dim, trn8_2dim  :contentReference[oaicite:1]{index=1}
trn5 = mat["trn5_2dim"].astype(np.float64)
trn6 = mat["trn6_2dim"].astype(np.float64)
trn8 = mat["trn8_2dim"].astype(np.float64)

if SCALE_BY_255:
    trn5 /= 255.0
    trn6 /= 255.0
    trn8 /= 255.0

X5, X6, X8 = trn5, trn6, trn8
X_all = np.vstack([X5, X6, X8])  # mixed training data (no labels)

# -------------------
# SUPERVISED GAUSSIANS (per class): means & covariances
# -------------------
def mean_cov(X, eps=1e-8):
    mu = X.mean(axis=0)
    # rowvar=False for (n,d)
    Sigma = np.cov(X, rowvar=False)
    # tiny regularization for stability
    Sigma = Sigma + eps * np.eye(Sigma.shape[0])
    return mu, Sigma

mu5, S5 = mean_cov(X5)
mu6, S6 = mean_cov(X6)
mu8, S8 = mean_cov(X8)

# -------------------
# UNSUPERVISED GMM on mixed data (3 components)
# -------------------
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=RANDOM_STATE,
    n_init=10
).fit(X_all)

weights = gmm.weights_           # (3,)
means_gmm = gmm.means_           # (3, 2)
covs_gmm = gmm.covariances_      # (3, 2, 2)

# -------------------
# PRINT COMPARISON: means & covariances
# -------------------
np.set_printoptions(precision=4, suppress=True)
print("\n=== Supervised class Gaussians (means) ===")
print(f"Class 5 mean: {mu5}")
print(f"Class 6 mean: {mu6}")
print(f"Class 8 mean: {mu8}")

print("\n=== Supervised class Gaussians (covariances) ===")
print("S5 =\n", S5)
print("S6 =\n", S6)
print("S8 =\n", S8)

print("\n=== GMM (unsupervised on mixed data) ===")
for k in range(3):
    print(f"\nComponent {k}: weight={weights[k]:.3f}")
    print("mean =", means_gmm[k])
    print("cov =\n", covs_gmm[k])

# -------------------
# CONTOUR HELPERS
# -------------------
def make_grid(X_list, steps=200, padding=0.1):
    Xcat = np.vstack(X_list)
    xmin, ymin = Xcat.min(axis=0)
    xmax, ymax = Xcat.max(axis=0)
    # padding
    xr = xmax - xmin
    yr = ymax - ymin
    xmin -= padding * xr; xmax += padding * xr
    ymin -= padding * yr; ymax += padding * yr
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, steps),
        np.linspace(ymin, ymax, steps)
    )
    XY = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, XY

def biv_gaussian_pdf(X, mu, Sigma):
    # manual stable logpdf -> exp (2D only; fine for viz)
    diff = X - mu
    invS = np.linalg.inv(Sigma)
    logdet = np.log(np.linalg.det(Sigma))
    m = np.einsum('ni,ij,nj->n', diff, invS, diff)
    d = X.shape[1]
    logpdf = -0.5 * (d*np.log(2*np.pi) + logdet + m)
    return np.exp(logpdf)

# -------------------
# BUILD GRID & DENSITIES
# -------------------
xx, yy, XY = make_grid([X5, X6, X8], steps=GRID_STEPS, padding=0.12)

# GMM total density
logprob = gmm.score_samples(XY)           # log p(x)
Z_gmm = np.exp(logprob).reshape(xx.shape)

# Individual GMM components (for intuition)
Z_gmm_comp = []
for k in range(3):
    Zk = weights[k] * biv_gaussian_pdf(XY, means_gmm[k], covs_gmm[k])
    Z_gmm_comp.append(Zk.reshape(xx.shape))

# Supervised class Gaussians
Z5 = biv_gaussian_pdf(XY, mu5, S5).reshape(xx.shape)
Z6 = biv_gaussian_pdf(XY, mu6, S6).reshape(xx.shape)
Z8 = biv_gaussian_pdf(XY, mu8, S8).reshape(xx.shape)

# -------------------
# PLOTS
# -------------------
# 1) Mixed data + GMM total density
plt.figure(figsize=(6,5))
plt.scatter(X_all[:,0], X_all[:,1], s=5, alpha=0.35, c='k', label='train (mixed)')
cs = plt.contour(xx, yy, Z_gmm, levels=10)
plt.clabel(cs, inline=1, fontsize=8)
plt.title("Unsupervised GMM (3 components) on mixed 2D data")
plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.legend()
plt.tight_layout()
plt.savefig("gmm_total_contour.png", dpi=140)
print("Saved: gmm_total_contour.png")

# 2) Individual GMM components
fig, axes = plt.subplots(1, 3, figsize=(13,4), sharex=True, sharey=True)
for k, ax in enumerate(axes):
    ax.scatter(X_all[:,0], X_all[:,1], s=5, alpha=0.2, c='k')
    cs = ax.contour(xx, yy, Z_gmm_comp[k], levels=10)
    ax.clabel(cs, inline=1, fontsize=8)
    ax.set_title(f"GMM component {k}\n(w={weights[k]:.2f})")
    ax.set_xlabel("dim 1")
axes[0].set_ylabel("dim 2")
plt.tight_layout()
plt.savefig("gmm_components_contours.png", dpi=140)
print("Saved: gmm_components_contours.png")

# 3) Supervised class Gaussians (5,6,8) contours
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(X5[:,0], X5[:,1], s=5, alpha=0.25, label='class 5')
ax.scatter(X6[:,0], X6[:,1], s=5, alpha=0.25, label='class 6')
ax.scatter(X8[:,0], X8[:,1], s=5, alpha=0.25, label='class 8')
c5 = ax.contour(xx, yy, Z5, colors='C0', levels=8)
c6 = ax.contour(xx, yy, Z6, colors='C1', levels=8)
c8 = ax.contour(xx, yy, Z8, colors='C2', levels=8)
ax.clabel(c5, inline=1, fontsize=8); ax.clabel(c6, inline=1, fontsize=8); ax.clabel(c8, inline=1, fontsize=8)
ax.set_title("Supervised Gaussian models per class (5,6,8)")
ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.legend()
plt.tight_layout()
plt.savefig("supervised_class_gaussians_contours.png", dpi=140)
print("Saved: supervised_class_gaussians_contours.png")

print("\nDone.")
