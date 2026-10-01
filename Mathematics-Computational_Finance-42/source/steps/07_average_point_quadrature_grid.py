"""
Build the fixed quadrature grid over the average point on which the whole estimator is evaluated. The grid is a Gauss-Legendre rule mapped onto a window centred on the mean of the time average of the mean-reverting process and scaled by the standard deviation that a driftless path of the same diffusion would have for its own time average. Both the centre and the scale are closed-form, so the grid is a deterministic function of the horizon and the model and costs nothing to rebuild at each maturity.

Centring matters more than it looks. The weight being integrated is concentrated where the time average of the path is likely to sit, and that is neither the initial state nor the long-run level but the exponentially weighted blend of the two that the mean-reversion produces over the interval. A window centred on the initial state wastes half its nodes at short horizons and misses the peak at long ones.

The scale is deliberately the driftless one rather than the stationary one. Mean reversion narrows the distribution of the time average relative to the driftless case, so using the driftless width gives a window that is always at least wide enough, and the Gauss-Legendre rule then converges geometrically because the integrand is analytic and decays at both ends of the window.

The rule is applied on the mapped interval, so the nodes are the standard nodes affine-mapped and the weights are the standard weights scaled by half the window length. The weights therefore sum to the window length, not to one, and a rule that forgets the Jacobian is wrong by exactly that factor.

Returns
-------
np.ndarray of shape (2*nq,) packed as nodes first and weights second, nodes increasing. The weights sum to twice the span times the scale. For a horizon of 5, initial state -3.352407217493, mean reversion 0.35, diffusion 0.62, level -3.688879454114, 81 nodes and a span of 9 the window runs from -10.730634353150 to 3.670592164010 and the weights sum to 14.407498047892.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def average_point_grid(T, x0, k, sigma, theta, nq, span):
    """Gauss-Legendre grid over the average point.

    The window is centred on the mean of the time average of the
    mean-reverting process over [0, T] started at x0, and its half-width is
    span times the standard deviation that the time average of a driftless
    path with the same diffusion coefficient would have.  A Gauss-Legendre
    rule of nq points is affine-mapped onto that window.

    Args:
        T (float): horizon, strictly positive.
        x0 (float): initial state.
        k (float): mean-reversion speed, strictly positive.
        sigma (float): diffusion coefficient, strictly positive.
        theta (float): mean-reversion level, constant in time.
        nq (int): number of quadrature nodes, at least 1.
        span (float): half-width of the window in units of the scale,
            strictly positive.

    Expected return:
        np.ndarray of shape (2*nq,) packed as [u_1..u_nq, w_1..w_nq] with
        nodes in increasing order and weights carrying the mapping Jacobian.

    Raises:
        ValueError: if T <= 0, if k <= 0, if sigma <= 0, if nq < 1, or if
        span <= 0.
    """
    return np.zeros(2 * int(nq))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_average_point_grid(T, x0, k, sigma, theta, nq, span):
    T, x0, k = float(T), float(x0), float(k)
    sigma, theta, span = float(sigma), float(theta), float(span)
    nq = int(nq)
    if T <= 0.0:
        raise ValueError("T must be positive")
    if k <= 0.0:
        raise ValueError("k must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if nq < 1:
        raise ValueError("nq must be at least 1")
    if span <= 0.0:
        raise ValueError("span must be positive")
    m = theta + (x0 - theta) * (1.0 - np.exp(-k * T)) / (k * T)
    s = sigma * np.sqrt(T / 3.0)
    lo, hi = m - span * s, m + span * s
    x, w = np.polynomial.legendre.leggauss(nq)
    return np.concatenate((0.5 * (hi - lo) * x + 0.5 * (hi + lo),
                           0.5 * (hi - lo) * w))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": ("v = average_point_grid(1.0, -3.352407217493, 0.35, 0.62, "
                      "-3.688879454114, 5, 2.0)"),
            "call": "np.round(v, 12)",
            "gold_call": ("np.array([-4.053728712094, -3.790479468804, -3.404981571390, "
                          "-3.019483673977, -2.756234430687, 0.169619353073, "
                          "0.342657125776, 0.407275709892, 0.342657125776, "
                          "0.169619353073])"),
            "tol": 1e-10,
        },
        {
            "setup": (
                "nq = 81\n"
                "v = average_point_grid(5.0, -3.352407217493, 0.35, 0.62, "
                "-3.688879454114, nq, 9.0)"
            ),
            "call": ("np.round(np.array([v[0], v[nq - 1], float(v[nq:].sum())]), 12)"),
            "gold_call": ("np.array([-10.730634353150, 3.670592164010, "
                          "14.407498047892])"),
            "tol": 1e-10,
        },
        {
            "setup": (
                "nq = 41\n"
                "v = average_point_grid(3.0, -3.0, 0.35, 0.62, -3.688879454114, nq, 6.0)\n"
                "u = v[:nq]\n"
                "mid = 0.5 * (u[0] + u[-1])"
            ),
            "call": "bool(np.all(np.diff(u) > 0.0) and abs(float(u[20] - mid)) < 1e-12)",
            "gold_call": "True",
        },
        {
            "setup": (
                "nq = 1\n"
                "v = average_point_grid(2.0, -3.0, 0.35, 0.62, -3.688879454114, nq, 4.0)\n"
                "m = -3.688879454114 + (-3.0 + 3.688879454114) * "
                "(1.0 - np.exp(-0.7)) / 0.7\n"
                "s = 0.62 * np.sqrt(2.0 / 3.0)"
            ),
            "call": "np.round(np.array([float(v[0] - m), float(v[1] - 8.0 * s)]), 12)",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "nq = 21\n"
                "v = average_point_grid(4.0, -3.0, 0.35, 0.62, -3.688879454114, nq, 5.0)\n"
                "u, w = v[:nq], v[nq:]\n"
                "poly = float(np.sum(w * (3.0 * u ** 2 - 2.0 * u + 1.0)))\n"
                "lo, hi = float(u[0]) - 0.0, float(u[-1])"
            ),
            "call": "bool(np.isfinite(poly) and float(w.sum()) > 0.0)",
            "gold_call": "True",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        average_point_grid(5.0, -3.0, 0.35, 0.62, -3.6, 0, 9.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_average_point_grid(5.0, -3.0, 0.35, 0.62, -3.6, 0, 9.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        average_point_grid(5.0, -3.0, 0.0, 0.62, -3.6, 41, 9.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_average_point_grid(5.0, -3.0, 0.0, 0.62, -3.6, 41, 9.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
