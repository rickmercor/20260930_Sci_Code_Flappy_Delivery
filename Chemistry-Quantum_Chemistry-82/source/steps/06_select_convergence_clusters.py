"""
Follow the source's Gaussian-mixture clustering with BIC model selection. For each candidate count q from one through max_clusters, fit a full-covariance mixture by standard EM and use BIC=-2 log L+p log n with p=(q-1)+qd+qd(d+1)/2. The benchmark starts with the point farthest from the grand mean and repeatedly adds the point maximizing its minimum squared distance to selected means; all components start with the biased empirical covariance plus regularization times the identity and equal weights. Stop after 250 iterations or when the log-likelihood change is at most 1e-10 times one plus the previous magnitude, choose the smallest-BIC candidate, and relabel components by lexicographically sorted means. Return the selected count, numeric labels, BIC vector, mean posterior entropy, and the second-smallest minus smallest BIC.



Retain the likelihood and responsibilities from the last E-step, before its M-step, for BIC and entropy; do not run an additional E-step. Add 1e-15 to the component responsibility sums in each M-step. Farthest-point ties use the lowest row index, and BIC ties use the smallest component count.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
tuple: Selected count, labels, BIC vector, posterior entropy, and BIC margin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_convergence_clusters(features: np.ndarray, max_clusters: int = 4, regularization: float = 1e-6) -> tuple:
    """Select a full-covariance Gaussian mixture by BIC.

    Parameters
    ----------
    features : np.ndarray
        Finite nonconstant feature matrix with at least eight rows and two columns.
    max_clusters : int
        Largest candidate count, from two through min(4,n-1).
    regularization : float
        Positive covariance diagonal regularizer.
    Returns
    -------
    tuple
        Selected count, numeric labels, BIC vector, mean posterior entropy, and BIC margin.
    Raises
    ------
    ValueError
        If shape, variability, candidate-count, or regularization contracts fail.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def _fit_full_gmm(features: np.ndarray, clusters: int, regularization: float) -> tuple:
    n_samples, dimension = features.shape
    center = np.mean(features, axis=0)
    selected = [int(np.argmax(np.sum((features - center) ** 2, axis=1)))]
    while len(selected) < clusters:
        distances = np.min(
            np.stack(
                [np.sum((features - features[index]) ** 2, axis=1) for index in selected],
                axis=1,
            ),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    means = features[selected].copy()
    global_covariance = np.cov(features, rowvar=False, bias=True)
    if dimension == 1:
        global_covariance = np.array([[float(global_covariance)]], dtype=float)
    global_covariance = np.asarray(global_covariance, dtype=float) + regularization * np.eye(dimension)
    covariances = np.repeat(global_covariance[None, :, :], clusters, axis=0)
    mixing = np.full(clusters, 1.0 / clusters, dtype=float)
    previous = -np.inf
    responsibilities = np.full((n_samples, clusters), 1.0 / clusters, dtype=float)
    for _ in range(250):
        log_probability = np.empty((n_samples, clusters), dtype=float)
        for cluster in range(clusters):
            sign, logdet = np.linalg.slogdet(covariances[cluster])
            if sign <= 0:
                raise ValueError("covariance is not positive definite")
            delta = features - means[cluster]
            solved = np.linalg.solve(covariances[cluster], delta.T).T
            quadratic = np.sum(delta * solved, axis=1)
            log_probability[:, cluster] = (
                np.log(mixing[cluster])
                - 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)
            )
        normalizer = _logsumexp(log_probability, axis=1)
        log_likelihood = float(np.sum(normalizer))
        responsibilities = np.exp(log_probability - normalizer[:, None])
        effective = np.sum(responsibilities, axis=0) + 1e-15
        mixing = effective / n_samples
        means = (responsibilities.T @ features) / effective[:, None]
        for cluster in range(clusters):
            delta = features - means[cluster]
            covariances[cluster] = (
                (responsibilities[:, cluster, None] * delta).T @ delta / effective[cluster]
                + regularization * np.eye(dimension)
            )
        if np.isfinite(previous) and abs(log_likelihood - previous) <= 1e-10 * (1.0 + abs(previous)):
            break
        previous = log_likelihood
    parameters = (clusters - 1) + clusters * dimension + clusters * dimension * (dimension + 1) / 2
    bic = float(-2.0 * log_likelihood + parameters * np.log(n_samples))
    return bic, log_likelihood, responsibilities, means


def _oracle_select_convergence_clusters(
    features: np.ndarray, max_clusters: int = 4, regularization: float = 1e-6
) -> tuple:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[0] < 8 or features.shape[1] < 2:
        raise ValueError("features must have shape (n,d) with n at least 8 and d at least 2")
    if not np.all(np.isfinite(features)) or np.any(np.std(features, axis=0) <= 1e-12):
        raise ValueError("feature columns must be finite and nonconstant")
    if not isinstance(max_clusters, (int, np.integer)) or not 2 <= int(max_clusters) <= min(4, features.shape[0] - 1):
        raise ValueError("max_clusters must be an integer from 2 through min(4,n-1)")
    if not np.isscalar(regularization) or not np.isfinite(regularization) or float(regularization) <= 0.0:
        raise ValueError("regularization must be a positive finite scalar")
    bics = []
    fits = []
    for clusters in range(1, int(max_clusters) + 1):
        fit = _fit_full_gmm(features, clusters, float(regularization))
        bics.append(fit[0])
        fits.append(fit)
    bics_array = np.asarray(bics, dtype=float)
    best_index = int(np.argmin(bics_array))
    best_clusters = best_index + 1
    responsibilities = fits[best_index][2]
    means = fits[best_index][3]
    order = np.lexsort(means[:, ::-1].T)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(best_clusters)
    labels = inverse[np.argmax(responsibilities, axis=1)]
    clipped = np.clip(responsibilities, 1e-300, 1.0)
    entropy = float(-np.mean(np.sum(clipped * np.log(clipped), axis=1)))
    sorted_bics = np.sort(bics_array)
    bic_gap = float(sorted_bics[1] - sorted_bics[0]) if sorted_bics.size > 1 else 0.0
    return int(best_clusters), labels.astype(float), bics_array, entropy, bic_gap

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common="""import numpy as np
def _x(n,d):
 i=np.arange(n,dtype=float)[:,None];j=np.arange(d,dtype=float)[None,:];return .3*np.sin(.31*(i+1)*(j+1))+((i[:,0]%3)-1)[:,None]*np.linspace(.8,.2,d)[None,:]
def _flat(v):
 p=[]
 for x in v if isinstance(v,tuple) else (v,):p.extend(np.asarray(x,dtype=float).ravel().tolist())
 return np.asarray(p)
def _e(g,*a):
 try:g(*a)
 except ValueError:return 1.0
 return 0.0"""
    return [
        {"setup":common+chr(10)+"x=_x(18,4)","call":"_flat(select_convergence_clusters(x,4,1e-6))","gold_call":"_flat(_oracle_select_convergence_clusters(x,4,1e-6))","tol":1e-8},
        {"setup":common+chr(10)+"x=_x(8,2)","call":"_flat(select_convergence_clusters(x,2,1e-5))","gold_call":"_flat(_oracle_select_convergence_clusters(x,2,1e-5))","tol":1e-8},
        {"setup":common+chr(10)+"x=_x(31,7)","call":"_flat(select_convergence_clusters(x,3,1e-7))","gold_call":"_flat(_oracle_select_convergence_clusters(x,3,1e-7))","tol":1e-7},
        {"setup":common+chr(10)+"x=np.ones((8,3))","call":"_e(select_convergence_clusters,x,4,1e-6)","gold_call":"_e(_oracle_select_convergence_clusters,x,4,1e-6)","tol":1e-12}
    ]
