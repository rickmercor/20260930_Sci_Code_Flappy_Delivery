"""
Form the regional genetic relatedness matrix from the scaled genotype matrix and

return it together with its scaling factor.

The relatedness matrix of a scaled genotype matrix X is R = X X^T / m, with the

divisor m fixed by the requirement Tr R = Tr I_n = n rather than by the number

of variants, which is what puts the genetic and residual variance components on

a common scale.

Returns
-------
tuple (np.ndarray of shape (n, n), float scaling factor m)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_arg_grm(x_std: "np.ndarray") -> tuple:
    """Build the regional relatedness matrix and its scaling factor.

    Args:
        x_std: array of shape (n, p) holding the centred, frequency-scaled
            genotypes of the retained mutations.

    Returns:
        tuple (grm, scale) where grm is an (n, n) float array and scale is the
        float divisor applied to X X^T.

    Raises:
        ValueError: if x_std is not two-dimensional or if it carries no
            variance, which leaves the scaling factor undefined.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_arg_grm(x_std: "np.ndarray") -> tuple:
    """Reference implementation."""
    x_std = np.asarray(x_std, dtype=float)
    if x_std.ndim != 2:
        raise ValueError("x_std must be a two-dimensional array")
    n_individuals = x_std.shape[0]
    cross = x_std @ x_std.T
    total = float(np.trace(cross))
    if total <= 0.0:
        raise ValueError("x_std carries no variance, the scaling factor is undefined")
    scale = total / n_individuals
    return cross / scale, scale

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

def _pin(result):
    grm, scale = result
    flat = np.asarray(grm, dtype=float).ravel()
    r = np.arange(1.0, flat.size + 1.0)
    return float(np.sum(np.sin(0.31*r)*flat) + np.sum(np.cos(0.13*r)*flat**2)) + 1000.0*float(scale)
""" % (_GENOTYPES,)

    return [
        {
            "setup": fixture,
            "call": "_pin(build_arg_grm(x_std))",
            "gold_call": "_pin(_oracle_build_arg_grm(x_std))",
        },
        {
            "setup": fixture + """x_single = x_std[:, :1]
""",
            "call": "_pin(build_arg_grm(x_single))",
            "gold_call": "_pin(_oracle_build_arg_grm(x_single))",
        },
        {
            "setup": fixture + """x_zero = np.zeros((10, 4))

def run_model():
    try:
        build_arg_grm(x_zero)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_build_arg_grm(x_zero)
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
