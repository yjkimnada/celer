"""
===========================================================
Lasso/Logreg dual objectives with and without extrapolation
===========================================================

The example runs cyclic coordinate descent on a small dataset, and plot
dual objectives obtained with and without extrapolation.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.linalg import norm
from numba import njit

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression

print(__doc__)


@njit
def xlogx(x):
    if x <= 1e-10:
        return 0
    else:
        return x * np.log(x)


@njit
def sigmoid(x):
    """Vectorwise sigmoid."""
    return 1. / (1. + np.exp(- x))


@njit
def primal_logreg(Xw, y, w, alpha):
    """Logreg primal objective for labels in {-1, 1}."""
    p_obj = alpha * np.sum(np.abs(w))
    p_obj += np.log(1. + np.exp(- y * Xw)).sum()
    return p_obj


@njit
def dual_logreg(theta, y, alpha):
    """Logreg dual objective for labels in {-1, 1} and feasible theta.

    Dual constraint is: norm(X.T.dot(theta), ord=np.inf) <= 1."""
    y_theta = y * theta * alpha

    d_obj = 0.
    for y_theta_i in y_theta:
        d_obj -= xlogx(y_theta_i) + xlogx(1. - y_theta_i)
    return d_obj


@njit
def ST(x, u):
    """Soft-thresholding of scalar x at level u."""
    if x > u:
        return x - u
    elif x < - u:
        return x + u
    else:
        return 0.


@njit
def cd_one_epoch(w, X, y, Xw, alpha, lc):
    for j in range(X.shape[1]):
        old_w_j = w[j]
        grad_j = np.dot(X[:, j], - y * sigmoid(- y * Xw))
        w[j] = ST(w[j] - grad_j / lc[j], alpha / lc[j])

        if old_w_j != w[j]:
            Xw += (w[j] - old_w_j) * X[:, j]


def solver_logreg(X, y, alpha, max_iter=10000, tol=1e-4, f_gap=10,
                  K=6, true_sign=None):
    """
    Solve logreg with CD. Labels in {-1, 1}. Dense X.
    """
    n_samples, n_features = X.shape
    last_K_res = np.zeros((K, n_samples))
    onesKm1 = np.ones(K - 1)  # ones K minus 1
    U = np.zeros((K - 1, n_samples))

    E, gaps, d_objs, d_objs_acc = [], [], [], []

    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)

    lc = norm(X, axis=0) ** 2 / 4

    identified = []
    for it in range(max_iter):
        if true_sign is not None:
                identified.append(((w != 0) == true_sign).all())
        if it % f_gap == 0:
            p_obj = primal_logreg(Xw, y, w, alpha)
            E.append(p_obj)

            if it // f_gap < K:
                last_K_res[it // f_gap] = Xw
                Xw_acc = Xw
            else:
                for k in range(K - 1):
                    last_K_res[k] = last_K_res[k + 1]
                last_K_res[K - 1] = Xw

                for k in range(K - 1):
                    U[k] = last_K_res[k + 1] - last_K_res[k]
                C = np.dot(U, U.T)

                try:
                    z = np.linalg.solve(C, onesKm1)
                    c = z / z.sum()
                    Xw_acc = np.sum(last_K_res[:-1] *
                                       np.expand_dims(c, axis=1), axis=0)
                except np.linalg.LinAlgError as e:
                    print("Linalg solving failed, falling back on theta_res")
                    Xw_acc = Xw

            theta_acc = y * sigmoid(-y * Xw_acc) / alpha
            norm_Xtheta_acc = np.max(np.abs(np.dot(X.T, theta_acc)))
            theta_acc /= norm_Xtheta_acc
            d_objs_acc.append(dual_logreg(theta_acc, y, alpha))

            theta = y * sigmoid(-y * Xw) / alpha
            norm_Xtheta = np.max(np.abs(np.dot(X.T, theta)))
            theta /= norm_Xtheta

            d_obj = dual_logreg(theta, y, alpha)
            d_objs.append(d_obj)
            gap = p_obj - d_obj
            gaps.append(gap)
            print("Iteration %d, p_obj::%.5f, d_obj::%.5f, gap::%.2e" %
                  (it, p_obj, d_obj, gap))
            if gap < tol:
                print("Early exit.")
                break

        cd_one_epoch(w, X, y, Xw, alpha, lc)

    return (w, np.array(E), np.array(gaps), np.array(d_objs),
            np.array(d_objs_acc), np.array(identified))

# TODO make this compatible with modern sklearn
dataset = "leukemia"
data = fetch_mldata(dataset)
X = np.asfortranarray(data.data.astype(float))
y = data.target.astype(float)

alpha_max = np.max(np.abs(X.T.dot(y))) / 2.

alpha_div = 10
alpha = alpha_max / alpha_div

# Get very precise solution with sklearn:
clf = LogisticRegression(
    C=1.0 / alpha, penalty="l1", fit_intercept=False, tol=1e-10, max_iter=1000)

clf.fit(X, y)
p_star = primal_logreg(X.dot(clf.coef_[0]), y, clf.coef_[0], alpha)
true_supp = (clf.coef_[0] != 0)


f_gap = 10
w, E, gaps, d_objs, d_objs_acc, identified = solver_logreg(
    X, y, alpha, true_sign=true_supp, f_gap=f_gap)

assert identified[-1]
ided = np.where(np.diff(identified))[0][-1] + 1
# ugly: really check that it did not move after
assert identified[ided:].all()

c_list = sns.color_palette("colorblind", 8)
matplotlib.rcParams["text.usetex"] = True
label1 = (r"$\mathcal{D}(\hat \theta) - \mathcal{D}"
          r"(\theta^{(t)}_{\mathrm{res}})$")
label2 = (r"$\mathcal{D}(\hat \theta) - \mathcal{D}"
          r"(\theta^{(t)}_{\mathrm{accel}})$")
fig = plt.figure(figsize=[6., 2.5])

plt.semilogy(
    f_gap * np.arange(len(d_objs)), p_star - d_objs, c=c_list[1], label=label1)

plt.semilogy(
    f_gap * np.arange(len(d_objs)), p_star - d_objs_acc, c=c_list[0],
    label=label2)

plt.xlabel(r"CD epoch $t$")

plt.axvline(ided, linestyle="--", color="black")
plt.tight_layout()
plt.legend()
plt.show(block=False)
