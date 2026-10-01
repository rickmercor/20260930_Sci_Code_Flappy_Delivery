"""
Build the quadrature rule adapted to an exponentially decaying weight on the positive half line. The rate of the weight is a free parameter, so the rule must follow the weight rather than assume the unit rate.

Gauss-Laguerre quadrature is exact for weighted polynomials on the positive half-line. Rescaling the standard nodes and weights from exp(-u) to exp(-scale*u) aligns the quadrature measure with the estimated Fourier-integrand decay and reduces the number of characteristic-function evaluations.

Returns
-------
tuple of numpy.ndarray (u, w), each a real array of shape (order,) with strictly positive entries and increasing u.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def exponential_weight_rule(order, scale):
    """Return nodes and weights of the maximal-degree rule for a decaying weight.

    Return one-dimensional arrays ``u`` and ``w`` of length ``order`` such that

        sum_{n} w[n] * P(u[n]) = integral_0^infinity exp(-scale * x) P(x) dx

    holds exactly for every polynomial ``P`` of degree at most
    ``2 * order - 1``.  The nodes must be returned in increasing order.  A
    consequence that can be used as a self-check is
    ``sum(w) == 1 / scale``.

    Parameters
    ----------
    order : int
        Strictly positive number of nodes.
    scale : float
        Strictly positive decay rate of the weight.

    Returns
    -------
    tuple of numpy.ndarray
        ``(u, w)``, each a real array of shape ``(order,)`` with strictly
        positive entries and increasing ``u``.

    Raises
    ------
    ValueError
        If ``order`` is not a positive integer or ``scale`` is not finite and
        strictly positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _unpack_model(model):
    """Validate and unpack the six model parameters."""
    np = __import__("numpy")
    arr = np.asarray(model, dtype=float).reshape(-1)
    if arr.size != 6 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("model must hold six finite numbers")
    alpha, gamma, nu, rho, v0, theta = (float(v) for v in arr)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("model[0] must satisfy 0 < alpha <= 1")
    if gamma <= 0.0 or nu <= 0.0:
        raise ValueError("model[1] and model[2] must be positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("model[3] must satisfy abs(rho) < 1")
    if v0 <= 0.0 or theta <= 0.0:
        raise ValueError("model[4] and model[5] must be positive")
    return alpha, gamma, nu, rho, v0, theta


def _unpack_market(market):
    """Validate and unpack the four market parameters."""
    np = __import__("numpy")
    arr = np.asarray(market, dtype=float).reshape(-1)
    if arr.size != 4 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("market must hold four finite numbers")
    spot, strike, rate, horizon = (float(v) for v in arr)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("market[0] and market[1] must be positive")
    if horizon <= 0.0:
        raise ValueError("market[3] must be positive")
    return spot, strike, rate, horizon


def _positive_int(value, name):
    """Return ``value`` as a positive Python int or raise ValueError."""
    np = __import__("numpy")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a positive integer")
    if int(value) < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _line_argument(xi_real, xi_imag):
    """Return the flattened complex vector xi_real + 1j*xi_imag."""
    np = __import__("numpy")
    a = np.atleast_1d(np.asarray(xi_real, dtype=float)).reshape(-1)
    b = np.atleast_1d(np.asarray(xi_imag, dtype=float)).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("xi_real and xi_imag must have the same shape")
    if not (bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))):
        raise ValueError("xi_real and xi_imag must be finite")
    return a + 1j * b
def _weight_rule(order, scale):
    """Return nodes and weights exact to degree 2*order-1 for exp(-scale*u) on (0, inf)."""
    special = __import__("scipy.special", fromlist=["roots_laguerre"])
    abscissas, coefficients = special.roots_laguerre(order)
    return abscissas / scale, coefficients / scale


def _oracle_exponential_weight_rule(order, scale):
    """Reference implementation of the scaled quadrature rule."""
    np = __import__("numpy")
    order = _positive_int(order, "order")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    nodes, weights = _weight_rule(order, scale)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return small, target-scale, unit-scale, large-order, and invalid cases."""
    flat = (
        "def flat(pair):\n"
        "    u, w = pair\n"
        "    u = np.asarray(u, dtype=float).reshape(-1)\n"
        "    w = np.asarray(w, dtype=float).reshape(-1)\n"
        "    return np.round(np.concatenate([u, np.log10(w)]), 9).tolist()\n"
    )
    return [
        {
            "setup": flat + "order = 5\nscale = 2.666571209748119\n",
            "call": "flat(exponential_weight_rule(order, scale))",
            "gold_call": "flat(_oracle_exponential_weight_rule(order, scale))",
        },
        {
            "setup": flat + "order = 16\nscale = 2.666571209748119\n",
            "call": "flat(exponential_weight_rule(order, scale))",
            "gold_call": "flat(_oracle_exponential_weight_rule(order, scale))",
        },
        {
            "setup": flat + "order = 1\nscale = 1.0\n",
            "call": "flat(exponential_weight_rule(order, scale))",
            "gold_call": "flat(_oracle_exponential_weight_rule(order, scale))",
        },
        {
            "setup": flat + "order = 64\nscale = 0.4766969758605919\n",
            "call": "flat(exponential_weight_rule(order, scale))",
            "gold_call": "flat(_oracle_exponential_weight_rule(order, scale))",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        exponential_weight_rule(8, 0.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_exponential_weight_rule(8, 0.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
