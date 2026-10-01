"""
Run the forward and backward recursions of the two-state chain in the log domain

and return the posterior state probabilities at every marginal tree together with

the log-likelihood of the observation sequence.

Introgressed material arrives as haplotypes rather than as isolated positions,

so neighbouring marginal trees along a chromosome are far from independent and

the ancestry state is carried across them by a Markov chain. With state 0 the

modern human state and state 1 the archaic state, the transition matrix is



    R = [[1 - p, p], [q, 1 - q]],



where p is the probability of entering the archaic state between adjacent trees

and q the probability of leaving it, so that the expected number of trees spanned

by an archaic tract is 1 / q. The forward and backward recursions are carried out

in the log domain, and their combination at each tree gives the posterior

probability of each state there. The same recursions deliver the log-likelihood

of the full observation sequence, which is the quantity monitored for convergence

when the parameters are re-estimated.

Returns
-------
tuple of an np.ndarray of shape (2, m) holding the posterior state probabilities as a float64 array and a native Python float log-likelihood
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def forward_backward_posteriors(
    log_emissions: np.ndarray,
    p: float,
    q: float,
    pi_archaic: float = 0.05,
) -> tuple:
    """Return posterior state probabilities and the sequence log-likelihood.

    Parameters
    ----------
    log_emissions : np.ndarray
        Log emission densities of shape (2, m).
    p : float
        Transition probability into the archaic state.
    q : float
        Transition probability out of the archaic state.
    pi_archaic : float
        Prior probability of the archaic state at the first tree.

    Returns
    -------
    result : tuple
        The posterior state probabilities of shape (2, m) followed by the
        log-likelihood of the observation sequence.

    Raises
    ------
    ValueError
        If log_emissions does not have shape (2, m) with m at least two, or if p,
        q or pi_archaic does not lie strictly between 0 and 1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_forward_backward_posteriors(
    log_emissions: np.ndarray,
    p: float,
    q: float,
    pi_archaic: float = 0.05,
) -> tuple:
    """Reference implementation."""
    log_emissions = np.asarray(log_emissions, dtype=float)
    if log_emissions.ndim != 2 or log_emissions.shape[0] != 2:
        raise ValueError("log_emissions must have shape (2, m)")
    if log_emissions.shape[1] < 2:
        raise ValueError("at least two marginal trees are required")
    for name, value in (("p", p), ("q", q), ("pi_archaic", pi_archaic)):
        if not np.isfinite(value) or not (0.0 < value < 1.0):
            raise ValueError("%s must lie strictly between 0 and 1" % name)

    m = log_emissions.shape[1]
    log_t = np.log(np.array([[1.0 - p, p], [q, 1.0 - q]]))
    log_pi = np.log(np.array([1.0 - pi_archaic, pi_archaic]))

    alphas = np.zeros((2, m), dtype=float)
    scalers = np.zeros(m, dtype=float)
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

    betas = np.zeros((2, m), dtype=float)
    for i in range(m - 2, -1, -1):
        for z in range(2):
            betas[z, i] = np.logaddexp(
                betas[0, i + 1] + log_emissions[0, i + 1] + log_t[z, 0],
                betas[1, i + 1] + log_emissions[1, i + 1] + log_t[z, 1],
            )
        betas[:, i] -= np.logaddexp(betas[0, i], betas[1, i])

    log_gamma = alphas + betas
    log_gamma -= np.logaddexp(log_gamma[0], log_gamma[1])
    return np.exp(log_gamma), float(scalers.sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_EMISSION_MATRIX = np.array([
    [-11.10352, -10.46046, -10.84378, -14.43700, -11.03712, -10.32724],
    [-13.28459, -12.70932, -12.54344, -12.17722, -12.12295, -12.16929],
])
'''
    return [
        # --- Normal scenario ---
        {
            "setup": fixture + """log_emissions = _EMISSION_MATRIX.copy()
p = 0.07181926246625578
q = 0.41792948477327807
pi_archaic = 0.05

def _pack(fn):
    posteriors, loglik = fn(log_emissions, p, q, pi_archaic=pi_archaic)
    return ([float(posteriors.shape[0]), float(posteriors.shape[1])]
            + [round(float(v), 12) for v in posteriors.ravel()]
            + [round(float(loglik), 9)])

def run_model():
    return _pack(forward_backward_posteriors)

def run_oracle():
    return _pack(_oracle_forward_backward_posteriors)
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Boundary case: the shortest admissible chain ---
        {
            "setup": fixture + """log_emissions = _EMISSION_MATRIX[:, :2].copy()
p = 0.01
q = 0.5
pi_archaic = 0.5

def _pack(fn):
    posteriors, loglik = fn(log_emissions, p, q, pi_archaic=pi_archaic)
    return ([float(posteriors.shape[0]), float(posteriors.shape[1])]
            + [round(float(v), 12) for v in posteriors.ravel()]
            + [round(float(loglik), 9)])

def run_model():
    return _pack(forward_backward_posteriors)

def run_oracle():
    return _pack(_oracle_forward_backward_posteriors)
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Edge case: a transition probability of exactly one ---
        {
            "setup": fixture + """log_emissions = _EMISSION_MATRIX.copy()
def run_model():
    try:
        forward_backward_posteriors(log_emissions, 1.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_forward_backward_posteriors(log_emissions, 1.0, 0.4)
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
