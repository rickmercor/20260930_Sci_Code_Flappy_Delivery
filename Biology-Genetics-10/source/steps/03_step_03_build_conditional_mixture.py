"""
Discretize the factor distribution conditional on a lead allele.

At factor value f, variant j has ALT probability

q_j(f) = Phi((b_j f - tau_j) / sqrt(psi_j)). Conditioning on lead state s

reweights the standard-normal factor density by q_0(f)^s[1-q_0(f)]^(1-s).

A fixed Gauss-Legendre rule on a symmetric finite interval turns this posterior

factor law into normalized mixture weights and partner Bernoulli probabilities.

Inputs

------

loadings : oriented one-factor loadings, lead first

uniqueness : residual variances

thresholds : fixed Gaussian thresholds

lead_state : conditioned lead allele, zero or one

quadrature_order : number of fixed Gauss-Legendre nodes

factor_bound : positive half-width of the integration interval

Returns

-------

factor_nodes : transformed quadrature nodes

mixture_weights : normalized lead-conditioned factor weights

partner_probabilities : ALT probabilities for every node and partner

Returns
-------
tuple of factor_nodes (G,), mixture_weights (G,), and partner_probabilities (G, p - 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_conditional_mixture(
    loadings: np.ndarray,
    uniqueness: np.ndarray,
    thresholds: np.ndarray,
    lead_state: int,
    quadrature_order: int,
    factor_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Build a fixed-quadrature approximation to the lead-conditioned factor law.

    Parameters
    ----------
    loadings : np.ndarray
        One-dimensional oriented loadings, with the lead first.
    uniqueness : np.ndarray
        Positive residual variances in the same order.
    thresholds : np.ndarray
        Fixed Gaussian thresholds in the same order.
    lead_state : int
        Conditioned lead allele, either zero or one.
    quadrature_order : int
        Number of Gauss-Legendre nodes.
    factor_bound : float
        Positive half-width of the symmetric factor interval.

    Returns
    -------
    factor_nodes : np.ndarray
        Transformed quadrature nodes.
    mixture_weights : np.ndarray
        Normalized posterior factor weights.
    partner_probabilities : np.ndarray
        Partner ALT probabilities with shape (quadrature_order, p - 1).

    Raises
    ------
    ValueError
        If `loadings` is not a finite one-dimensional vector with at least two
        entries; if `uniqueness` or `thresholds` has a different shape; if a
        uniqueness is nonpositive or nonfinite; if a threshold is nonfinite;
        if `lead_state` is not zero or one; if `quadrature_order` is not an
        integer of at least eight; if `factor_bound` is not numeric, positive,
        and finite; or if the lead-conditioned quadrature has no positive
        finite mass.
    '''
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math  # noqa: E402

import numpy as np  # noqa: E402, F811


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard-normal CDF without an external dependency."""
    flat = np.asarray(values, dtype=float).ravel()
    result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in flat),
        dtype=float,
        count=flat.size,
    )
    return result.reshape(np.asarray(values).shape)


def _oracle_build_conditional_mixture(
    loadings: np.ndarray,
    uniqueness: np.ndarray,
    thresholds: np.ndarray,
    lead_state: int,
    quadrature_order: int,
    factor_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of fixed lead-conditioned quadrature."""
    b = np.asarray(loadings, dtype=float)
    psi = np.asarray(uniqueness, dtype=float)
    tau = np.asarray(thresholds, dtype=float)
    if b.ndim != 1 or b.size < 2 or not np.isfinite(b).all():
        raise ValueError("loadings must be a finite vector with at least two entries")
    if psi.shape != b.shape or tau.shape != b.shape:
        raise ValueError("uniqueness and thresholds must match loadings")
    if not np.isfinite(psi).all() or np.any(psi <= 0.0) or not np.isfinite(tau).all():
        raise ValueError("uniqueness must be positive and all parameters finite")
    if lead_state not in (0, 1):
        raise ValueError("lead_state must be zero or one")
    if not isinstance(quadrature_order, (int, np.integer)) or int(quadrature_order) < 8:
        raise ValueError("quadrature_order must be an integer of at least eight")
    if not isinstance(factor_bound, (int, float, np.integer, np.floating)):
        raise ValueError("factor_bound must be numeric")
    if not np.isfinite(float(factor_bound)) or float(factor_bound) <= 0.0:
        raise ValueError("factor_bound must be positive and finite")

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    factor_nodes = float(factor_bound) * nodes
    base = (
        float(factor_bound)
        * weights
        * np.exp(-0.5 * factor_nodes * factor_nodes)
        / math.sqrt(2.0 * math.pi)
    )
    lead_eta = (b[0] * factor_nodes - tau[0]) / math.sqrt(psi[0])
    lead_probability = _normal_cdf(lead_eta)
    base *= lead_probability if lead_state == 1 else (1.0 - lead_probability)
    normalizer = float(np.sum(base))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("lead-conditioned quadrature has zero mass")
    mixture_weights = base / normalizer
    partner_eta = (
        factor_nodes[:, None] * b[None, 1:] - tau[None, 1:]
    ) / np.sqrt(psi)[None, 1:]
    partner_probabilities = _normal_cdf(partner_eta)
    return factor_nodes, mixture_weights, partner_probabilities

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811

_LOADINGS = np.array(
    [0.9103664774626048, 0.8619342151577695, -0.8849182223819824,
     0.7808688094430303, -0.8402966482242443, 0.6476484200955405,
     -0.7546055221635047, 0.8232127859153063]
)
_UNIQUENESS = np.array(
    [0.17123287671232873, 0.25706940874035994, 0.21691973969631234,
     0.3902439024390244, 0.293901542952276, 0.5805515239477504,
     0.4305705059203445, 0.322320709105568]
)
_THRESHOLDS = np.array(
    [1.000490545619381, 0.7332357810248108, -0.5813930077385457,
     0.5104216363344174, -0.7332357810248108, 1.2278262639421003,
     -0.18445243845167293, 0.31060942561200097]
)


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """loadings = _LOADINGS.copy()
uniqueness = _UNIQUENESS.copy()
thresholds = _THRESHOLDS.copy()
lead_state = 1
quadrature_order = 64
factor_bound = 8.0
def pack(result):
    return np.concatenate([np.asarray(value, dtype=float).ravel() for value in result])
""",
            "call": "pack(build_conditional_mixture(loadings, uniqueness, thresholds, lead_state, quadrature_order, factor_bound))",
            "gold_call": "pack(_oracle_build_conditional_mixture(loadings, uniqueness, thresholds, lead_state, quadrature_order, factor_bound))",
        },
        {
            "setup": """loadings = np.array([0.5, -0.5])
uniqueness = np.array([0.75, 0.75])
thresholds = np.array([0.0, 0.0])
lead_state = 0
quadrature_order = 8
factor_bound = 4.0
def pack(result):
    return np.concatenate([np.asarray(value, dtype=float).ravel() for value in result])
""",
            "call": "pack(build_conditional_mixture(loadings, uniqueness, thresholds, lead_state, quadrature_order, factor_bound))",
            "gold_call": "pack(_oracle_build_conditional_mixture(loadings, uniqueness, thresholds, lead_state, quadrature_order, factor_bound))",
        },
        {
            "setup": """loadings = np.array([0.5, 0.4])
uniqueness = np.array([0.75, 0.84])
thresholds = np.array([0.0, 0.0])
def run_model():
    try:
        build_conditional_mixture(loadings, uniqueness, thresholds, 2, 16, 8.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_build_conditional_mixture(loadings, uniqueness, thresholds, 2, 16, 8.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        {
            "setup": """loadings = np.array([0.8, -0.55, 0.35])
uniqueness = np.array([0.36, 0.6975, 0.8775])
thresholds = np.array([0.9, -0.4, 0.25])
quadrature_order = 12
factor_bound = 1.25
def pack_states(function):
    _, weights_0, _ = function(loadings, uniqueness, thresholds, 0, quadrature_order, factor_bound)
    _, weights_1, _ = function(loadings, uniqueness, thresholds, 1, quadrature_order, factor_bound)
    return np.concatenate((weights_0, weights_1))
""",
            "call": "pack_states(build_conditional_mixture)",
            "gold_call": "pack_states(_oracle_build_conditional_mixture)",
        },
    ]
