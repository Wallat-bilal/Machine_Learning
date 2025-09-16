import os
import numpy as np
from numpy.linalg import slogdet, inv

# ------------ config ------------
DATA_DIR = r"C:\Users\walat\PycharmProjects\PythonProject\Machine Learning Exercis 2\Data\dataset1_G_Noisy_ASCII"
EPS = 1e-6  # covariance regularization for stability
# --------------------------------

def load_txt(name):
    return np.loadtxt(os.path.join(DATA_DIR, name))

# ---- load data ----
trn_x = load_txt("trn_x.txt")              # class 1 (x)
trn_y = load_txt("trn_y.txt")              # class 2 (y)
tst_xy_126 = load_txt("tst_xy_126.txt")
tst_xy_126_label = load_txt("tst_xy_126_class.txt").astype(int)

# ---- fit Gaussians ----
def mean_cov(X):
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False) + EPS*np.eye(X.shape[1])
    return mu, Sigma

mu_x, Sigma_x = mean_cov(trn_x)
mu_y, Sigma_y = mean_cov(trn_y)

# ---- log Gaussian pdf ----
def log_gaussian_pdf(X, mu, Sigma):
    X = np.atleast_2d(X)
    d = X.shape[1]
    sign, logdet = slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Covariance not positive definite")
    iSigma = inv(Sigma)
    diff = X - mu
    m = np.einsum('ni,ij,nj->n', diff, iSigma, diff)  # Mahalanobis terms
    return -0.5 * (d*np.log(2*np.pi) + logdet + m)

def classify(X, prior_x, prior_y):
    log_like_x = log_gaussian_pdf(X, mu_x, Sigma_x)
    log_like_y = log_gaussian_pdf(X, mu_y, Sigma_y)
    log_post_x = log_like_x + np.log(prior_x)
    log_post_y = log_like_y + np.log(prior_y)
    return np.where(log_post_x > log_post_y, 1, 2)

def accuracy(yhat, ytrue):
    return np.mean(yhat == ytrue)

# ---- (b) uniform priors for comparison ----
yhat_uniform = classify(tst_xy_126, prior_x=0.5, prior_y=0.5)
acc_uniform = accuracy(yhat_uniform, tst_xy_126_label)

# ---- (c) non-uniform priors 0.9 / 0.1 ----
yhat_nonuni = classify(tst_xy_126, prior_x=0.9, prior_y=0.1)
acc_nonuni = accuracy(yhat_nonuni, tst_xy_126_label)

# ---- compare ----
improvement = (acc_nonuni / acc_uniform - 1.0) if acc_uniform > 0 else np.nan

print(f"(b) Accuracy on tst_xy_126 with uniform priors: {acc_uniform*100:.2f}%")
print(f"(c) Accuracy on tst_xy_126 with priors (0.9, 0.1): {acc_nonuni*100:.2f}%")
print(f"Absolute improvement vs uniform: {improvement*100:.2f}%")
