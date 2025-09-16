import os
import numpy as np
from numpy.linalg import slogdet, inv
import matplotlib.pyplot as plt

# ------------ config ------------
# Point this to your local folder:
DATA_DIR = r"C:\Users\walat\PycharmProjects\PythonProject\Machine Learning Exercis 2\Data\dataset1_G_Noisy_ASCII"
EPS = 1e-6  # covariance regularization
# --------------------------------

def load_txt(name):
    return np.loadtxt(os.path.join(DATA_DIR, name))

# ---- load data ----
# Train
trn_x = load_txt("trn_x.txt")              # class 1 features
trn_x_label = load_txt("trn_x_class.txt")  # should be all 1s
trn_y = load_txt("trn_y.txt")              # class 2 features
trn_y_label = load_txt("trn_y_class.txt")  # should be all 2s

# Tests
tst_x = load_txt("tst_x.txt")
tst_x_label = load_txt("tst_x_class.txt")
tst_y = load_txt("tst_y.txt")
tst_y_label = load_txt("tst_y_class.txt")
tst_xy = load_txt("tst_xy.txt")
tst_xy_label = load_txt("tst_xy_class.txt")
tst_xy_126 = load_txt("tst_xy_126.txt")
tst_xy_126_label = load_txt("tst_xy_126_class.txt")

assert trn_x.shape[1] == trn_y.shape[1], "feature dims must match"
d = trn_x.shape[1]

# ---- estimate Gaussian params ----
def mean_cov(X):
    mu = np.mean(X, axis=0)
    # row-vectors are samples => rowvar=False so we get (d x d)
    Sigma = np.cov(X, rowvar=False)
    # regularize
    Sigma = Sigma + EPS * np.eye(Sigma.shape[0])
    return mu, Sigma

mu_x, Sigma_x = mean_cov(trn_x)
mu_y, Sigma_y = mean_cov(trn_y)

# ---- priors from training frequencies ----
Nx, Ny = trn_x.shape[0], trn_y.shape[0]
prior_x_mle = Nx / (Nx + Ny)
prior_y_mle = Ny / (Nx + Ny)

# ---- multivariate normal log-pdf (stable) ----
def log_gaussian_pdf(X, mu, Sigma):
    """
    X: (n, d), mu: (d,), Sigma: (d, d)
    returns: (n,)
    """
    X = np.atleast_2d(X)
    d = X.shape[1]
    # precompute inverse and log|Sigma|
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Covariance not positive definite")
    iSigma = inv(Sigma)
    diff = X - mu
    # Mahalanobis terms
    m = np.einsum('ni,ij,nj->n', diff, iSigma, diff)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + m)

def classify(X, pri_x, pri_y):
    log_like_x = log_gaussian_pdf(X, mu_x, Sigma_x)
    log_like_y = log_gaussian_pdf(X, mu_y, Sigma_y)
    # add log priors for posteriors up to normalization
    log_post_x = log_like_x + np.log(pri_x)
    log_post_y = log_like_y + np.log(pri_y)
    # class 1 if posterior_x > posterior_y, else class 2
    yhat = np.where(log_post_x > log_post_y, 1, 2)
    return yhat, log_post_x, log_post_y

def accuracy(yhat, ytrue):
    ytrue = ytrue.astype(int)
    return np.mean(yhat == ytrue)

# ---------- (a) classify tst_xy with data-driven priors ----------
yhat_xy, _, _ = classify(tst_xy, prior_x_mle, prior_y_mle)
acc_xy = accuracy(yhat_xy, tst_xy_label)
print(f"(a) Accuracy on tst_xy with MLE priors: {acc_xy*100:.2f}%")

# ---------- (b) classify tst_xy_126 with uniform prior ----------
prior_x_uniform = 0.5
prior_y_uniform = 0.5
yhat_xy126_uniform, _, _ = classify(tst_xy_126, prior_x_uniform, prior_y_uniform)
acc_xy126_uniform = accuracy(yhat_xy126_uniform, tst_xy_126_label)
print(f"(b) Accuracy on tst_xy_126 with uniform priors: {acc_xy126_uniform*100:.2f}%")

# ---------- (c) classify tst_xy_126 with non-uniform prior (0.9 / 0.1) ----------
prior_x_nonuni = 0.9
prior_y_nonuni = 0.1
yhat_xy126_nonuni, _, _ = classify(tst_xy_126, prior_x_nonuni, prior_y_nonuni)
acc_xy126_nonuni = accuracy(yhat_xy126_nonuni, tst_xy_126_label)
print(f"(c) Accuracy on tst_xy_126 with priors (0.9, 0.1): {acc_xy126_nonuni*100:.2f}%")

# absolute improvement vs (b)
if acc_xy126_uniform > 0:
    improvement = (acc_xy126_nonuni / acc_xy126_uniform) - 1.0
else:
    improvement = np.nan
print(f"Absolute improvement vs uniform: {improvement*100:.2f}%")

# ---------- (optional) visualize train data + Gaussian ellipses ----------
def plot_confidence_ellipse(ax, mu, Sigma, n_std=2.0, **kwargs):
    """
    Draws an ellipse corresponding to n_std standard deviations (≈confidence).
    """
    from matplotlib.patches import Ellipse
    import numpy.linalg as la

    # eigen-decomposition
    vals, vecs = la.eigh(Sigma)
    # sort eigenvalues/vectors
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # angle
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # width/height: 2 * n_std * sqrt(eigvals)
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mu, width=width, height=height, angle=theta, fill=False, **kwargs)
    ax.add_patch(ell)
    return ell

fig, ax = plt.subplots()
ax.scatter(trn_x[:,0], trn_x[:,1], s=12, label='class x (1)')
ax.scatter(trn_y[:,0], trn_y[:,1], s=12, label='class y (2)')
plot_confidence_ellipse(ax, mu_x, Sigma_x, n_std=2.0, linewidth=2)
plot_confidence_ellipse(ax, mu_y, Sigma_y, n_std=2.0, linewidth=2)
ax.set_xlabel('feature 1'); ax.set_ylabel('feature 2'); ax.legend()
ax.set_title('Training data with Gaussian 2-std ellipses')
plt.show()
