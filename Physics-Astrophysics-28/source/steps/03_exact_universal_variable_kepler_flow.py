"""
Advance (P, Q) exactly under H_0 = P^2/2 - 1/r for a signed time t, on elliptic, parabolic or hyperbolic arcs. This is the H_C sub-flow, applied separately to (p, q) and to (x, y).

The universal anomaly chi solves r0*vr0*chi^2*C(z) + (1 - alpha*r0)*chi^3*S(z) + r0*chi = t, with z = alpha*chi^2, alpha = 2/r0 - v0^2 and Stumpff functions C and S; the Lagrange coefficients f, g, fdot, gdot then give the new state. The left side increases strictly with chi (its derivative is r > 0), so the root can be bracketed and refined by safeguarded Newton steps. This keeps the solver convergent on hyperbolic arcs, where an unguarded Newton iteration started far from the root overflows cosh and sinh, and for negative times.

Returns
-------
np.ndarray of shape (2*d,): [P(t)_1..P(t)_d, Q(t)_1..Q(t)_d] (one flat array, not a tuple).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def kepler_flow(P, Q, t):
    """Exact phase flow of H_0 = |P|**2/2 - 1/|Q| over time t (any sign, any orbit type).

    Returns one flat array of length 2*d packed as [P(t)_1..P(t)_d, Q(t)_1..Q(t)_d].
    Raises ValueError for |Q| = 0 or mismatched shapes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_kepler_flow(P, Q, t):
    # Oracle: universal-variable Kepler propagator (mu = 1) with Lagrange f, g coefficients.
    # The universal Kepler equation F(chi) = t has dF/dchi = r(chi) > 0, so F is strictly
    # increasing; the root is bracketed first and then refined by Newton steps safeguarded
    # with bisection, which converges for elliptic, parabolic and hyperbolic arcs and any sign of t.
    def stumpff(z):
        if z > 1e-6:
            sz = np.sqrt(z)
            return (1.0 - np.cos(sz)) / z, (sz - np.sin(sz)) / sz ** 3
        if z < -1e-6:
            sz = np.sqrt(-z)
            return (np.cosh(sz) - 1.0) / (-z), (np.sinh(sz) - sz) / sz ** 3
        return (0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0,
                1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0)

    v0 = np.array(P, dtype=float).ravel()
    x0 = np.array(Q, dtype=float).ravel()
    if v0.shape != x0.shape or v0.size not in (2, 3):
        raise ValueError("P and Q must be 1-D arrays of equal length 2 or 3")
    r0 = float(np.sqrt(x0 @ x0))
    if r0 == 0.0:
        raise ValueError("separation |Q| must be positive")
    t = float(t)
    if not np.isfinite(t):
        raise ValueError("t must be finite")
    if t == 0.0:
        return np.concatenate([v0, x0])
    sigma0 = float(x0 @ v0)
    alpha = 2.0 / r0 - float(v0 @ v0)

    def kepler_residual(chi):
        z = alpha * chi * chi
        if z < -1.0e4:
            # hyperbolic overshoot: cosh/sinh would overflow, and F is far beyond t in the sign of chi
            return np.copysign(np.inf, chi), np.inf
        C, S = stumpff(z)
        F = sigma0 * chi * chi * C + (1.0 - alpha * r0) * chi ** 3 * S + r0 * chi - t
        dF = sigma0 * chi * (1.0 - z * S) + (1.0 - alpha * r0) * chi * chi * C + r0
        return F, dF

    # bracket the root: F(0) = -t, and F is increasing
    step = t / r0
    if t > 0.0:
        lo, hi = 0.0, step
        while kepler_residual(hi)[0] < 0.0:
            lo, hi = hi, 2.0 * hi
    else:
        lo, hi = step, 0.0
        while kepler_residual(lo)[0] > 0.0:
            lo, hi = 2.0 * lo, lo
    chi = 0.5 * (lo + hi)
    for _ in range(300):
        F, dF = kepler_residual(chi)
        if F > 0.0:
            hi = chi
        else:
            lo = chi
        new = chi - F / dF if np.isfinite(F) and np.isfinite(dF) else 0.5 * (lo + hi)
        if not (lo < new < hi):
            new = 0.5 * (lo + hi)
        if abs(new - chi) <= 4e-16 * max(1.0, abs(new)):
            chi = new
            break
        chi = new
    z = alpha * chi * chi
    C, S = stumpff(z)
    f = 1.0 - chi * chi / r0 * C
    g = t - chi ** 3 * S
    x1 = f * x0 + g * v0
    r1 = float(np.sqrt(x1 @ x1))
    fdot = chi / (r1 * r0) * (z * S - 1.0)
    gdot = 1.0 - chi * chi / r1 * C
    v1 = fdot * x0 + gdot * v0
    return np.concatenate([v1, x1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "name": (
                "normal_task_first_substep"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    1.3512071919596578,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    1.3512071919596578,\n"
                ")\n"
            ),
            "tol": 1e-10,
        },
        {
            "name": (
                "boundary_zero_time_identity"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    0.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    0.0,\n"
                ")\n"
            ),
            "tol": 1e-14,
        },
        {
            "name": (
                "edge_negative_time_elliptic_backward_drift"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([0.03, 0.18]),\n"
                "    np.array([25.34, 3.1]),\n"
                "    -35.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([0.03, 0.18]),\n"
                "    np.array([25.34, 3.1]),\n"
                "    -35.0,\n"
                ")\n"
            ),
            "tol": 1e-10,
        },
        {
            "name": (
                "edge_3d_elliptic_orbit"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([0.1, 0.5, 0.05]),\n"
                "    np.array([4.0, 0.0, 0.5]),\n"
                "    12.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([0.1, 0.5, 0.05]),\n"
                "    np.array([4.0, 0.0, 0.5]),\n"
                "    12.0,\n"
                ")\n"
            ),
            "tol": 1e-10,
        },
        {
            "name": (
                "regression_hyperbolic_t200"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([2.0, 0.5]),\n"
                "    np.array([1.0, 0.0]),\n"
                "    200.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([2.0, 0.5]),\n"
                "    np.array([1.0, 0.0]),\n"
                "    200.0,\n"
                ")\n"
            ),
            "tol": 1e-08,
        },
        {
            "name": (
                "regression_hyperbolic_t1000_no_overflow"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([2.0, 0.5]),\n"
                "    np.array([1.0, 0.0]),\n"
                "    1000.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([2.0, 0.5]),\n"
                "    np.array([1.0, 0.0]),\n"
                "    1000.0,\n"
                ")\n"
            ),
            "tol": 1e-07,
        },
        {
            "name": (
                "edge_3d_hyperbolic_backward_time"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "kepler_flow(\n"
                "    np.array([1.5, 0.3, 0.4]),\n"
                "    np.array([1.0, 0.5, -0.2]),\n"
                "    -120.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_kepler_flow(\n"
                "    np.array([1.5, 0.3, 0.4]),\n"
                "    np.array([1.0, 0.5, -0.2]),\n"
                "    -120.0,\n"
                ")\n"
            ),
            "tol": 1e-08,
        },
        {
            "name": (
                "invalid_zero_separation_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        kepler_flow(\n"
                "            np.array([0.0, 0.18]),\n"
                "            np.array([0.0, 0.0]),\n"
                "            1.0,\n"
                "        )\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_kepler_flow(\n"
                "            np.array([0.0, 0.18]),\n"
                "            np.array([0.0, 0.0]),\n"
                "            1.0,\n"
                "        )\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": (
                "run_model()\n"
            ),
            "gold_call": (
                "run_gold()\n"
            ),
            "tol": 0,
        },
    ]
