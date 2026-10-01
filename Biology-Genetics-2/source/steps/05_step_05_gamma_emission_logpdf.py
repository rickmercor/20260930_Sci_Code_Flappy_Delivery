"""
Evaluate the log emission density of both hidden states at every marginal tree,

returning a two-row matrix of log densities.

The ancestry model treats each marginal tree as one emission of a two-state

hidden Markov chain, state 0 standing for ordinary variation of the recipient

population and state 1 for a haplotype of archaic origin. Both states emit the

same scalar observation, the product of focal branch length and the number of

coalescences the branch spans, and both do so through a gamma density in the

shape and rate parameterisation, with the archaic state carrying a smaller rate

and therefore a heavier upper tail. Working in the log domain throughout is what

keeps the forward and backward recursions numerically stable, because the

observations span several orders of magnitude and the densities of the two states

differ by many powers of e on the trees that carry the signal.

Returns
-------
np.ndarray of shape (2, m), the log emission densities as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gamma_emission_logpdf(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
) -> np.ndarray:
    """Return the two-state log emission matrix for the observations.

    Parameters
    ----------
    observations : np.ndarray
        Strictly positive observations of shape (m,).
    shape_null : float
        Gamma shape of the modern human state.
    rate_null : float
        Gamma rate of the modern human state.
    shape_archaic : float
        Gamma shape of the archaic state.
    rate_archaic : float
        Gamma rate of the archaic state.

    Returns
    -------
    log_emissions : np.ndarray
        Array of shape (2, m) holding the log emission density of each state.

    Raises
    ------
    ValueError
        If observations is not one dimensional, if any observation is not
        strictly positive, or if any shape or rate is not strictly positive.
    """
    return log_emissions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammaln


def _oracle_gamma_emission_logpdf(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
) -> np.ndarray:
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1:
        raise ValueError("observations must be one dimensional")
    if np.any(observations <= 0.0):
        raise ValueError("observations must be strictly positive")
    parameters = (shape_null, rate_null, shape_archaic, rate_archaic)
    if any((not np.isfinite(v)) or v <= 0.0 for v in parameters):
        raise ValueError("gamma shapes and rates must be strictly positive")

    log_x = np.log(observations)
    log_emissions = np.empty((2, observations.size), dtype=float)
    for row, (a, b) in enumerate(((shape_null, rate_null), (shape_archaic, rate_archaic))):
        log_emissions[row] = a * np.log(b) - gammaln(a) + (a - 1.0) * log_x - b * observations
    return log_emissions

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_OBS = np.array([22100.0, 13800.0, 6000.0, 46000.0, 138300.0, 148500.0])
'''
    return [
        # --- Normal scenario: fitted parameters of the two states ---
        {
            "setup": fixture + """observations = _OBS.copy()
shape_null = 7.020653403450456
rate_null = 0.00048052951410882235
shape_archaic = 4.922380327204293
rate_archaic = 4.317547561158265e-05
""",
            "call": "gamma_emission_logpdf(observations, shape_null, rate_null, shape_archaic, rate_archaic).tolist()",
            "gold_call": "_oracle_gamma_emission_logpdf(observations, shape_null, rate_null, shape_archaic, rate_archaic).tolist()",
        },
        # --- Boundary case: both states given identical parameters ---
        {
            "setup": fixture + """observations = _OBS.copy()
shape_null = 1.0
rate_null = 1e-4
shape_archaic = 1.0
rate_archaic = 1e-4
""",
            "call": "gamma_emission_logpdf(observations, shape_null, rate_null, shape_archaic, rate_archaic).tolist()",
            "gold_call": "_oracle_gamma_emission_logpdf(observations, shape_null, rate_null, shape_archaic, rate_archaic).tolist()",
        },
        # --- Edge case: a non-positive rate must be rejected ---
        {
            "setup": fixture + """observations = _OBS.copy()
def run_model():
    try:
        gamma_emission_logpdf(observations, 5.0, 1e-4, 3.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_gamma_emission_logpdf(observations, 5.0, 1e-4, 3.0, 0.0)
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
