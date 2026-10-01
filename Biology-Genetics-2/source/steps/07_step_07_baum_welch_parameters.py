"""
Re-estimate the archaic emission parameters and the two transition probabilities

by expectation maximisation, holding the modern human emission and the initial

state distribution fixed, and return the fitted parameters with the log-likelihood

reached.

The modern human emission is pinned before training because the genome-wide



observation distribution, once its upper tail has been trimmed, already is an



estimate of that state under the assumption that introgressed haplotypes are a



small minority. What remains unknown is the shape of the archaic emission and



how often the chain enters and leaves the archaic state, so the free parameter



set is the archaic shape and rate together with p and q, and Baum-Welch is run



over that set alone. Each expectation step evaluates the posteriors of the



current chain, and the maximisation step re-estimates the two transition



probabilities and refits the archaic gamma to the observations weighted by the



posterior of the archaic state. Iteration stops once the log-likelihood of the



observation sequence changes by less than a fixed tolerance, which is checked



before the parameters of that iteration are updated, so the returned



parameters are those that produced the converged log-likelihood. When the



iteration cap binds instead, the loop exits after a maximisation step and the



returned log-likelihood is the one evaluated before that last update.

Returns
-------
dict with native Python float values under the keys shape_archaic, rate_archaic, p, q and loglik and a native Python int under n_iter
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def baum_welch_parameters(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
    p: float = 0.01,
    q: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
) -> dict:
    """Return the expectation maximisation fit of the free model parameters.

    Parameters
    ----------
    observations : np.ndarray
        Strictly positive observations of shape (m,).
    shape_null : float
        Fixed gamma shape of the modern human state.
    rate_null : float
        Fixed gamma rate of the modern human state.
    shape_archaic : float
        Starting gamma shape of the archaic state.
    rate_archaic : float
        Starting gamma rate of the archaic state.
    p : float
        Starting transition probability into the archaic state.
    q : float
        Starting transition probability out of the archaic state.
    pi_archaic : float
        Fixed prior probability of the archaic state at the first tree.
    max_iter : int
        Largest number of expectation maximisation iterations.
    loglik_tol : float
        Convergence tolerance on the log-likelihood.

    Returns
    -------
    fitted : dict
        Dictionary holding shape_archaic, rate_archaic, p, q, n_iter and loglik.
        ``n_iter`` is the index, counting from zero, of the iteration the loop
        stopped on: the number of maximisation steps already applied when the
        convergence test fires, or ``max_iter - 1`` when the iteration cap binds.

    Raises
    ------
    ValueError
        If observations is not one dimensional or holds fewer than two points, if
        any observation is not strictly positive, if any gamma parameter is not
        strictly positive, if p, q or pi_archaic does not lie strictly between 0
        and 1, if max_iter is not positive, or if loglik_tol is not positive.
    """
    return fitted

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import digamma, gammaln


def _log_emissions(observations, a0, b0, a1, b1):
    """Two-row log emission matrix of the gamma densities in shape and rate form."""
    log_x = np.log(observations)
    rows = []
    for a, b in ((a0, b0), (a1, b1)):
        rows.append(a * np.log(b) - gammaln(a) + (a - 1.0) * log_x - b * observations)
    return np.vstack(rows)


def _forward_backward(log_emissions, p, q, pi_archaic):
    """Log-domain forward and backward recursions of the two-state chain."""
    m = log_emissions.shape[1]
    log_t = np.log(np.array([[1.0 - p, p], [q, 1.0 - q]]))
    log_pi = np.log(np.array([1.0 - pi_archaic, pi_archaic]))
    alphas = np.zeros((2, m))
    scalers = np.zeros(m)
    alphas[:, 0] = log_pi + log_emissions[:, 0]
    scalers[0] = np.logaddexp(alphas[0, 0], alphas[1, 0])
    alphas[:, 0] -= scalers[0]
    for i in range(1, m):
        for z in range(2):
            alphas[z, i] = log_emissions[z, i] + np.logaddexp(
                alphas[0, i - 1] + log_t[0, z], alphas[1, i - 1] + log_t[1, z]
            )
        scalers[i] = np.logaddexp(alphas[0, i], alphas[1, i])
        alphas[:, i] -= scalers[i]
    betas = np.zeros((2, m))
    for i in range(m - 2, -1, -1):
        for z in range(2):
            betas[z, i] = np.logaddexp(
                betas[0, i + 1] + log_emissions[0, i + 1] + log_t[z, 0],
                betas[1, i + 1] + log_emissions[1, i + 1] + log_t[z, 1],
            )
        betas[:, i] -= np.logaddexp(betas[0, i], betas[1, i])
    log_gamma = alphas + betas
    log_gamma -= np.logaddexp(log_gamma[0], log_gamma[1])
    return np.exp(log_gamma), alphas, betas, log_t, float(scalers.sum())


def _weighted_gamma_mle(observations, weights, bracket=(1e-6, 1e6), n_bisect=200):
    """Maximum-likelihood gamma shape and rate of a weighted positive sample."""
    total = weights.sum()
    mean = float((weights * observations).sum() / total)
    mean_log = float((weights * np.log(observations)).sum() / total)
    s = np.log(mean) - mean_log
    low, high = bracket
    for _ in range(n_bisect):
        mid = 0.5 * (low + high)
        if np.log(mid) - digamma(mid) - s > 0.0:
            low = mid
        else:
            high = mid
    shape = 0.5 * (low + high)
    return float(shape), float(shape / mean)


def _oracle_baum_welch_parameters(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
    p: float = 0.01,
    q: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
) -> dict:
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1 or observations.size < 2:
        raise ValueError("observations must be one dimensional with at least two points")
    if np.any(observations <= 0.0):
        raise ValueError("observations must be strictly positive")
    for value in (shape_null, rate_null, shape_archaic, rate_archaic):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("gamma shapes and rates must be strictly positive")
    for name, value in (("p", p), ("q", q), ("pi_archaic", pi_archaic)):
        if not np.isfinite(value) or not (0.0 < value < 1.0):
            raise ValueError("%s must lie strictly between 0 and 1" % name)
    if int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(loglik_tol) or loglik_tol <= 0.0:
        raise ValueError("loglik_tol must be strictly positive")

    m = observations.size
    previous = None
    loglik = 0.0
    n_iter = 0
    for n_iter in range(int(max_iter)):
        log_emissions = _log_emissions(observations, shape_null, rate_null, shape_archaic, rate_archaic)
        gammas, alphas, betas, log_t, loglik = _forward_backward(
            log_emissions, p, q, pi_archaic
        )
        if previous is not None and abs(loglik - previous) < loglik_tol:
            break
        previous = loglik
        xi_01 = np.zeros(m - 1)
        xi_10 = np.zeros(m - 1)
        for i in range(m - 1):
            terms = np.array([
                alphas[0, i] + log_t[0, 0] + log_emissions[0, i + 1] + betas[0, i + 1],
                alphas[0, i] + log_t[0, 1] + log_emissions[1, i + 1] + betas[1, i + 1],
                alphas[1, i] + log_t[1, 0] + log_emissions[0, i + 1] + betas[0, i + 1],
                alphas[1, i] + log_t[1, 1] + log_emissions[1, i + 1] + betas[1, i + 1],
            ])
            peak = terms.max()
            norm = peak + np.log(np.exp(terms - peak).sum())
            xi_01[i] = np.exp(terms[1] - norm)
            xi_10[i] = np.exp(terms[2] - norm)
        p = float(xi_01.sum() / gammas[0, :-1].sum())
        q = float(xi_10.sum() / gammas[1, :-1].sum())
        shape_archaic, rate_archaic = _weighted_gamma_mle(observations, gammas[1])

    return {
        "shape_archaic": float(shape_archaic),
        "rate_archaic": float(rate_archaic),
        "p": float(p),
        "q": float(q),
        "n_iter": int(n_iter),
        "loglik": float(loglik),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_OBS = np.array([
    22100.0, 19500.0, 22000.0, 15200.0, 13800.0, 11300.0, 23500.0, 19200.0,
    14100.0, 6000.0, 8500.0, 8100.0, 6200.0, 20500.0, 138300.0, 148500.0,
    134400.0, 142800.0, 46000.0, 18300.0, 11500.0, 11900.0, 18300.0, 8200.0,
])
'''
    return [
        # --- Normal scenario: a run that converges before the iteration cap ---
        {
            "setup": fixture + """observations = _OBS.copy()
kwargs = dict(
    shape_null=7.020653403450456,
    rate_null=0.00048052951410882235,
    shape_archaic=4.904678331714,
    rate_archaic=4.3371334e-05,
    p=0.01,
    q=0.1,
    pi_archaic=0.05,
    max_iter=200,
    loglik_tol=1e-2,
)

_KEYS = ("shape_archaic", "rate_archaic", "p", "q", "n_iter", "loglik")

def _packed(d):
    # Six significant digits: a correct bisection fit drifts in the last digits with the
    # order of the weighted sums, well below this, while a wrong parameter still differs.
    return [float(d[k]) if k == "n_iter" else float("%.6g" % float(d[k])) for k in _KEYS]
""",
            "call": "_packed(baum_welch_parameters(observations, **kwargs))",
            "gold_call": "_packed(_oracle_baum_welch_parameters(observations, **kwargs))",
            "tol": 1e-6,
        },
        # --- Boundary case: a single iteration allowed by the cap ---
        {
            "setup": fixture + """observations = _OBS.copy()
kwargs = dict(
    shape_null=7.020653403450456,
    rate_null=0.00048052951410882235,
    shape_archaic=4.904678331714,
    rate_archaic=4.3371334e-05,
    p=0.01,
    q=0.1,
    pi_archaic=0.05,
    max_iter=1,
    loglik_tol=1e-2,
)

_KEYS = ("shape_archaic", "rate_archaic", "p", "q", "n_iter", "loglik")

def _packed(d):
    # Six significant digits: a correct bisection fit drifts in the last digits with the
    # order of the weighted sums, well below this, while a wrong parameter still differs.
    return [float(d[k]) if k == "n_iter" else float("%.6g" % float(d[k])) for k in _KEYS]
""",
            "call": "_packed(baum_welch_parameters(observations, **kwargs))",
            "gold_call": "_packed(_oracle_baum_welch_parameters(observations, **kwargs))",
            "tol": 1e-6,
        },
        # --- Edge case: a non-positive convergence tolerance ---
        {
            "setup": fixture + """observations = _OBS.copy()
def run_model():
    try:
        baum_welch_parameters(observations, 7.0, 5e-4, 5.0, 4e-5, loglik_tol=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_baum_welch_parameters(observations, 7.0, 5e-4, 5.0, 4e-5, loglik_tol=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
