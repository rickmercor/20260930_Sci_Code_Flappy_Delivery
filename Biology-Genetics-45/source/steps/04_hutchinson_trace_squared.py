"""
Estimate the trace of the squared relatedness matrix stochastically.

The trace of a matrix that is never formed can be recovered from its action on

random probe vectors, as the average of z^T A z over probes z drawn with zero

mean and identity covariance. It is applied here to R^2.

Returns
-------
float, the stochastic estimate of Tr R^2 as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hutchinson_trace_squared(
    x_std: "np.ndarray",
    scale: float,
    num_probes: int = 200,
    seed: int = 101,
) -> float:
    """Estimate Tr R^2 stochastically from products with R.

    Args:
        x_std: array of shape (n, p) holding the scaled genotypes.
        scale: the divisor m in R = X X^T / m.
        num_probes: number of Gaussian probe vectors, taken as the rows of
            np.random.default_rng(seed).standard_normal((num_probes, n)).
        seed: seed passed to np.random.default_rng.

    Returns:
        float, the stochastic estimate of Tr R^2.

    Raises:
        ValueError: if num_probes is below 1 or if scale is not positive.
    """
    return trace_r_squared

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_hutchinson_trace_squared(
    x_std: "np.ndarray",
    scale: float,
    num_probes: int = 200,
    seed: int = 101,
) -> float:
    """Reference implementation."""
    x_std = np.asarray(x_std, dtype=float)
    if int(num_probes) < 1:
        raise ValueError("num_probes must be at least 1")
    if float(scale) <= 0.0:
        raise ValueError("scale must be positive")

    n_individuals = x_std.shape[0]
    rng = np.random.default_rng(seed)
    probes = rng.standard_normal((int(num_probes), n_individuals))
    applied = (x_std @ (x_std.T @ probes.T)).T / float(scale)
    return float(np.mean(np.sum(applied * applied, axis=1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _GENOTYPES = [
        [0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 2, 0, 2, 0, 1, 0, 2, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 0, 0, 0],
        [0, 1, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 0, 0],
    ]

    fixture = """import numpy as np
G = np.array(%r, dtype=float)
ac = G.sum(axis=0)
G = G[:, np.minimum(ac, 20.0 - ac) >= 2.0]
f = G.sum(axis=0) / 20.0
x_std = (G - 2.0 * f) * (f * (1.0 - f)) ** (-0.125)
scale = float(np.trace(x_std @ x_std.T) / 10.0)
""" % (_GENOTYPES,)

    return [
        {
            "setup": fixture,
            "call": "round(hutchinson_trace_squared(x_std, scale, 200, 101), 10)",
            "gold_call": "round(_oracle_hutchinson_trace_squared(x_std, scale, 200, 101), 10)",
        },
        {
            "setup": fixture,
            "call": "round(hutchinson_trace_squared(x_std, scale, 1, 7), 10)",
            "gold_call": "round(_oracle_hutchinson_trace_squared(x_std, scale, 1, 7), 10)",
        },
        {
            "setup": fixture + """
def run_model():
    try:
        hutchinson_trace_squared(x_std, scale, 0, 101)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_hutchinson_trace_squared(x_std, scale, 0, 101)
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
