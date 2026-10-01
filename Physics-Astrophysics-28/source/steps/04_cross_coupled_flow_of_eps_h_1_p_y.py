"""
Apply the exact flow of the sub-Hamiltonian eps*H_A = eps*H_1(p, y) to a doubled state [p, q, x, y] for scaled time tau = eps*t, given the gradients evaluated at (p, y).

Under H_1(p, y), p is the momentum conjugate to q and y is the position conjugate to x. H_A does not depend on q or x, so p and y are constant. Their conjugates then drift linearly: dq/dt = g(p,y) and dx/dt = f(p,y) (Liang and Mei, Eq. 2.16).

Returns
-------
np.ndarray of shape (4d,): updated doubled state [p, q + taug, x + tau*f, y].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def doubled_flow_A(z, tau, fg):
    """Exact flow of the cross-coupled sub-Hamiltonian eps*H_1(p, y) over scaled time tau = eps*t.

    z   : doubled state, flat array of length 4*d packed as [p, q, x, y].
    tau : scaled duration (may be negative).
    fg  : flat array of length 2*d, [f, g] = PN gradients evaluated at the invariant pair (p, y).
    Returns the updated doubled state (flat array of length 4*d).
    Raises ValueError when len(z) is not 8 or 12, or len(fg) != len(z)//2.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_doubled_flow_A(z, tau, fg):
    # Oracle: under eps*H_1(p, y) the pair (p, y) is frozen, so the flow is an exact linear drift:
    # q <- q + tau * g(p, y),  x <- x + tau * f(p, y).
    z = np.array(z, dtype=float).ravel()
    fg = np.asarray(fg, dtype=float).ravel()
    if z.size not in (8, 12):
        raise ValueError("z must have length 8 (planar) or 12 (3-D)")
    d = z.size // 4
    if fg.size != 2 * d:
        raise ValueError("fg must have length len(z)//2")
    f, g = fg[:d], fg[d:]
    out = z.copy()
    out[d:2 * d] = z[d:2 * d] + tau * g
    out[2 * d:3 * d] = z[2 * d:3 * d] + tau * f
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "name": (
                "normal_task_like_state"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "doubled_flow_A(\n"
                "    np.array(\n"
                "        [\n"
                "            0.0,\n"
                "            0.18,\n"
                "            25.34,\n"
                "            0.0,\n"
                "            0.001,\n"
                "            0.179,\n"
                "            25.3,\n"
                "            0.05,\n"
                "        ],\n"
                "    ),\n"
                "    1.0758461279590061,\n"
                "    np.array(\n"
                "        [\n"
                "            -1.2084446551113163e-05,\n"
                "            0.0,\n"
                "            0.0,\n"
                "            -0.0226974034122999,\n"
                "        ],\n"
                "    ),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_flow_A(\n"
                "    np.array(\n"
                "        [\n"
                "            0.0,\n"
                "            0.18,\n"
                "            25.34,\n"
                "            0.0,\n"
                "            0.001,\n"
                "            0.179,\n"
                "            25.3,\n"
                "            0.05,\n"
                "        ],\n"
                "    ),\n"
                "    1.0758461279590061,\n"
                "    np.array(\n"
                "        [\n"
                "            -1.2084446551113163e-05,\n"
                "            0.0,\n"
                "            0.0,\n"
                "            -0.0226974034122999,\n"
                "        ],\n"
                "    ),\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "boundary_zero_duration_identity"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "doubled_flow_A(\n"
                "    np.array([0.1, 0.2, 3.0, 4.0, 0.11, 0.19, 3.1, 3.9]),\n"
                "    0.0,\n"
                "    np.array([1.0, 2.0, 3.0, 4.0]),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_flow_A(\n"
                "    np.array([0.1, 0.2, 3.0, 4.0, 0.11, 0.19, 3.1, 3.9]),\n"
                "    0.0,\n"
                "    np.array([1.0, 2.0, 3.0, 4.0]),\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "edge_3d_negative_duration"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "doubled_flow_A(\n"
                "    np.linspace(-1.0, 1.0, 12),\n"
                "    -1.3554304,\n"
                "    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_flow_A(\n"
                "    np.linspace(-1.0, 1.0, 12),\n"
                "    -1.3554304,\n"
                "    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "invalid_gradient_length_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        doubled_flow_A(np.arange(8.0), 0.5, np.ones(6))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_doubled_flow_A(\n"
                "            np.arange(8.0),\n"
                "            0.5,\n"
                "            np.ones(6),\n"
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
