# flake8: noqa F401
import inspect
import numbers

import numpy as np

from scipy import sparse
from abc import ABCMeta, abstractmethod
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_array
from sklearn.utils.validation import column_or_1d, check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.model_selection import check_cv
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.linear_model import ElasticNetCV, lasso_path
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sklearn
from sklearn.linear_model import MultiTaskLassoCV as _MultiTaskLassoCV
from sklearn.linear_model import (Lasso as Lasso_sklearn,
                                  LassoCV as _LassoCV,
                                  LogisticRegression as LogReg_sklearn)
from sklearn.linear_model._coordinate_descent import (LinearModelCV as
                                                      _LinearModelCV)
from sklearn.linear_model._coordinate_descent import (_alpha_grid,
                                                      _path_residuals)
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

from .homotopy import celer_path, mtl_path

# Hack because `model = Lasso()` is hardcoded in _LinearModelCV definition
lines = inspect.getsource(_LinearModelCV)
exec(lines)  # when this is executed Lasso is our class, not sklearn's
lines = inspect.getsource(_LassoCV)
lines = lines.replace('LassoCV', 'LassoCV_sklearn')
exec(lines)

lines = inspect.getsource(_MultiTaskLassoCV)
lines = lines.replace('MultiTaskLassoCV', 'MultiTaskLassoCV_sklearn')
exec(lines)


class Lasso(Lasso_sklearn):
    """
    Lasso scikit-learn estimator based on Celer solver

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``Lasso`` object is not advised.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    gap_freq : int
        Number of coordinate descent epochs between each duality gap
        computations.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import Lasso
    >>> clf = Lasso(alpha=0.1)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1, gap_freq=10, max_epochs=50000, max_iter=100,
    p0=10, prune=0, tol=1e-06, verbose=0)
    >>> print(clf.coef_)
    [0.85 0.  ]
    >>> print(clf.intercept_)
    0.15

    See also
    --------
    celer_path
    LassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
       "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html
    """

    def __init__(self, alpha=1., max_iter=100, gap_freq=10,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-4, prune=0,
                 fit_intercept=True, normalize=False, warm_start=False,
                 positive=False):
        super(Lasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.positive = positive

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer."""
        results = celer_path(
            X, y, "lasso", alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune,
            positive=self.positive,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))

        return results


class LassoCV(LassoCV_sklearn):
    """
    LassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * ||w||_1

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    gap_freq : int, optional (default=10)
        In the inner loop, the duality gap is computed every `gap_freq`
        coordinate descent epochs.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=False)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape ()
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    Lasso
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0, gap_freq=10,
                 max_epochs=50000, p0=10, prune=0, precompute='auto',
                 positive=False, n_jobs=1):
        super(LassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose, n_jobs=n_jobs)
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.positive = positive

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer."""
        alphas, coefs, dual_gaps = celer_path(
            X, y, "lasso", alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, gap_freq=self.gap_freq,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, prune=self.prune, positive=self.positive,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return alphas, coefs, dual_gaps


class MultiTaskLasso(MultiTaskLasso_sklearn):
    """
    MultiTaskLasso scikit-learn estimator based on Celer solver

    The optimization objective for MultiTaskLasso is::

    (1 / (2 * n_samples)) * ||y - X W||^2_2 + alpha * ||W||_{21}

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``Lasso`` object is not advised.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    gap_freq : int
        Number of coordinate descent epochs between each duality gap
        computations.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import Lasso
    >>> clf = Lasso(alpha=0.1)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1, gap_freq=10, max_epochs=50000, max_iter=100,
    p0=10, prune=0, tol=1e-06, verbose=0)
    >>> print(clf.coef_)
    [0.85 0.  ]
    >>> print(clf.intercept_)
    0.15

    See also
    --------
    celer_path
    LassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
       "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html
    """

    def __init__(self, alpha=1., max_iter=100, gap_freq=10,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-4, prune=0,
                 fit_intercept=True, normalize=False, warm_start=False):
        super().__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer."""
        results = mtl_path(
            X, y, alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            gap_freq=self.gap_freq, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune)

        return results


class MultiTaskLassoCV(MultiTaskLassoCV_sklearn):
    """
    MultiTaskLassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    The optimization objective for Multi-task Lasso is::

    (1 / (2 * n_samples)) * ||y - X W||^2_2 + alpha * ||W||_{21}

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    gap_freq : int, optional (default=10)
        In the inner loop, the duality gap is computed every `gap_freq`
        coordinate descent epochs.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=False)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    n_jobs : int
        to run CV in parallel.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features, n_outputs)
        parameter vector (w in the cost function formula)

    intercept_ : array, shape (n_outputs,)
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape (n_alphas,)
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    Lasso
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0, gap_freq=10,
                 max_epochs=50000, p0=10, prune=0, precompute='auto',
                 n_jobs=1):
        super().__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose, n_jobs=n_jobs)
        self.gap_freq = gap_freq
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer."""

        # Works !
        # alphas, coefs, dual_gaps = lasso_path(
        #     X, y, alphas=alphas, coef_init=coef_init,
        #     max_iter=self.max_iter, tol=self.tol)

        alphas, coefs, dual_gaps = mtl_path(
            X, y, alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, gap_freq=self.gap_freq,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, prune=self.prune)

        return alphas, coefs, dual_gaps


class LogisticRegression(LogReg_sklearn):
    """
    Sparse Logistic regression scikit-learn estimator based on Celer solver.

    The optimization objective for sparse Logistic regression is::

    \sum_1^n_samples log(1 + e^{-y_i x_i^T w}) + 1. / C * ||w||_1

    The solvers use a working set strategy. To solve problems restricted to a
    subset of features, Celer uses coordinate descent while PN-Celer uses
    a Prox-Newton strategy (detailed in [1], Sec 5.2).

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.

    penalty : 'l1'.
        Other penalties are not supported.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    fit_intercept : bool, optional (default=False)
        Whether or not to fit an intercept. Currently True is not supported.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Only False is supported so far.


    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ :  ndarray of shape (1,) or (n_classes,)
        constant term in decision function. Not handled yet.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import LogisticRegression
    >>> clf = LogisticRegression(C=1.)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 1])
    LogisticRegression(C=1.0, penalty='l1', tol=0.0001, fit_intercept=False,
    max_iter=50, verbose=False, max_epochs=50000, p0=10, warm_start=False)

    >>> print(clf.coef_)
    [[0.4001237  0.01949392]]

    See also
    --------
    celer_path

    References
    ----------
    .. [1] M. Massias, S. Vaiter, A. Gramfort, J. Salmon
       "Dual Extrapolation for Sparse Generalized Linear Models",
       preprint, https://arxiv.org/abs/1907.05830
    """

    def __init__(self, C=1., penalty='l1', tol=1e-4, fit_intercept=False,
                 max_iter=50, verbose=False, max_epochs=50000,
                 p0=10, warm_start=False):
        super(LogisticRegression, self).__init__(
            tol=tol, C=C)

        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.max_iter = max_iter
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        # TODO handle normalization, centering
        # TODO intercept
        if self.fit_intercept:
            raise NotImplementedError(
                "Fitting an intercept is not implement yet")
        # TODO support warm start
        if self.penalty != 'l1':
            raise NotImplementedError(
                'Only L1 penalty is supported, got %s' % self.penalty)

        if not isinstance(self.C, numbers.Number) or self.C <= 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        # below are copy pasted excerpts from sklearn.linear_model._logistic
        X, y = check_X_y(X, y, accept_sparse='csr', order="C")
        check_classification_targets(y)
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        self.classes_ = enc.classes_
        n_classes = len(enc.classes_)

        if n_classes <= 2:
            coefs = self.path(
                X, 2 * y_ind - 1, np.array([self.C]))[0]
            self.coef_ = coefs.T  # must be [1, n_features]
            self.intercept_ = 0
        else:
            self.coef_ = np.empty([n_classes, X.shape[1]])
            self.intercept_ = 0.
            multiclass = OneVsRestClassifier(self).fit(X, y)
            self.coef_ = multiclass.coef_

        return self

    def path(self, X, y, Cs, coef_init=None, **kwargs):
        """
        Compute sparse Logistic Regression path with Celer-PN.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Cs : ndarray
            Values of regularization strenghts for which solutions are
            computed

        coef_init : array, shape (n_features,), optional
            Initial value of the coefficients.

        Returns
        -------
        coefs_ : array, shape (len(Cs), n_features)
            Computed coefficients for each value in Cs.

        dual_gaps : array, shape (len(Cs),)
            Corresponding duality gaps at the end of optimization.
        """
        _, coefs, dual_gaps = celer_path(
            X, y, "logreg", alphas=1. / Cs, coef_init=coef_init,
            max_iter=self.max_iter, max_epochs=self.max_epochs,
            p0=self.p0, verbose=self.verbose, tol=self.tol,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return coefs, dual_gaps
