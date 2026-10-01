"""
Return the vector fields f = -dH_1/dQ and g = dH_1/dP of the PN perturbation H_1 = H_1PN + eps*H_2PN. These drive the exact cross-coupled flows A and B.

Writing H_1 as a function of p2 = P.P, s = N.P and r lets the gradients follow from the chain rule: dp2/dP = 2P, ds/dP = N, dr/dQ = N, ds/dQ = (P - sN)/r. Explicit symplectic flows of eps*H_1 evaluated on a frozen pair only need these gradients at that pair.

Returns
-------
np.ndarray of shape (2*d,): [f_1..f_d, g_1..g_d] with f = -dH_1/dQ and g = dH_1/dP (one flat array, not a tuple).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pn_perturbation_gradients(P, Q, eta, eps):
    """Force/velocity fields of the PN perturbation H_1 = H_1PN + eps*H_2PN.

    Returns one flat array of length 2*d (d = len(P)) packed as
    [f_1..f_d, g_1..g_d] with f = -dH_1/dQ and g = +dH_1/dP.
    Raises ValueError for |Q| = 0, mismatched shapes, eta outside (0, 0.25] or eps < 0.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pn_perturbation_gradients(P, Q, eta, eps):
    # Oracle: analytic gradients of H_1 = H_1PN + eps*H_2PN written through (p2, s, r),
    # p2 = P.P, s = N.P, r = |Q|; chain rule dp2/dP = 2P, ds/dP = N, dr/dQ = N, ds/dQ = (P - sN)/r.
    P = np.asarray(P, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    if P.shape != Q.shape or P.size not in (2, 3):
        raise ValueError("P and Q must be 1-D arrays of equal length 2 or 3")
    if not (0.0 < eta <= 0.25):
        raise ValueError("eta must satisfy 0 < eta <= 0.25")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    r = float(np.sqrt(Q @ Q))
    if r == 0.0:
        raise ValueError("separation |Q| must be positive")
    N = Q / r
    p2 = float(P @ P)
    s = float(N @ P)
    a1 = (3.0 * eta - 1.0) / 8.0
    b1 = (1.0 - 5.0 * eta + 5.0 * eta ** 2) / 16.0
    c1 = 5.0 - 20.0 * eta - 3.0 * eta ** 2
    e2 = eta ** 2
    d_p2 = 2.0 * a1 * p2 - (3.0 + eta) / (2.0 * r)
    d_s = -eta * s / r
    d_r = ((3.0 + eta) * p2 + eta * s ** 2) / (2.0 * r ** 2) - 1.0 / r ** 3
    d_p2 += eps * (3.0 * b1 * p2 ** 2 + (2.0 * c1 * p2 - 2.0 * e2 * s ** 2) / (8.0 * r)
                   + (5.0 + 8.0 * eta) / (2.0 * r ** 2))
    d_s += eps * ((-4.0 * e2 * s * p2 - 12.0 * e2 * s ** 3) / (8.0 * r) + 3.0 * eta * s / r ** 2)
    d_r += eps * (-(c1 * p2 ** 2 - 2.0 * e2 * s ** 2 * p2 - 3.0 * e2 * s ** 4) / (8.0 * r ** 2)
                  - ((5.0 + 8.0 * eta) * p2 + 3.0 * eta * s ** 2) / r ** 3
                  + 3.0 * (1.0 + 3.0 * eta) / (4.0 * r ** 4))
    g = 2.0 * d_p2 * P + d_s * N
    f = -(d_r * N + d_s * (P - s * N) / r)
    return np.concatenate([f, g])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "name": (
                "normal_task_initial_state"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "pn_perturbation_gradients(\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    0.1708984375,\n"
                "    0.6309573444801932,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_pn_perturbation_gradients(\n"
                "    np.array([0.0, 0.18]),\n"
                "    np.array([25.34, 0.0]),\n"
                "    0.1708984375,\n"
                "    0.6309573444801932,\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "boundary_eps_zero_pure_1pn"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "pn_perturbation_gradients(\n"
                "    np.array([0.03, 0.18]),\n"
                "    np.array([25.34, 3.1]),\n"
                "    0.1708984375,\n"
                "    0.0,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_pn_perturbation_gradients(\n"
                "    np.array([0.03, 0.18]),\n"
                "    np.array([25.34, 3.1]),\n"
                "    0.1708984375,\n"
                "    0.0,\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "edge_3d_equal_mass_radial_momentum"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "pn_perturbation_gradients(\n"
                "    np.array([0.05, 0.21, -0.03]),\n"
                "    np.array([8.0, -3.0, 1.5]),\n"
                "    0.25,\n"
                "    0.3,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_pn_perturbation_gradients(\n"
                "    np.array([0.05, 0.21, -0.03]),\n"
                "    np.array([8.0, -3.0, 1.5]),\n"
                "    0.25,\n"
                "    0.3,\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "invalid_eta_above_quarter_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        pn_perturbation_gradients(\n"
                "            np.array([0.0, 0.18]),\n"
                "            np.array([25.34, 0.0]),\n"
                "            0.3,\n"
                "            0.1,\n"
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
                "        _oracle_pn_perturbation_gradients(\n"
                "            np.array([0.0, 0.18]),\n"
                "            np.array([25.34, 0.0]),\n"
                "            0.3,\n"
                "            0.1,\n"
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
