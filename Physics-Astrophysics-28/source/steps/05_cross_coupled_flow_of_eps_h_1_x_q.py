"""
Apply the exact flow of the sub-Hamiltonian eps*H_B = eps*H_1(x, q) to a doubled state [p, q, x, y] for scaled time tau = eps*t, given the gradients evaluated at (x, q).

H_B does not depend on p or y, so x and q are constant along its flow. The conjugate variables change linearly: dp/dt = f(x,q) and dy/dt = g(x,q). Together with H_C and H_A this gives an explicit three-operator splitting of the cross-coupled extension.

Returns
-------
np.ndarray of shape (4d,): updated doubled state [p + tauf, q, x, y + tau*g].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def doubled_flow_B(z, tau, fg):
    """Exact flow of the cross-coupled sub-Hamiltonian eps*H_1(x, q) over scaled time tau = eps*t.

    z   : doubled state, flat array of length 4*d packed as [p, q, x, y].
    tau : scaled duration (may be negative).
    fg  : flat array of length 2*d, [f, g] = PN gradients evaluated at the invariant pair (x, q).
    Returns the updated doubled state (flat array of length 4*d).
    Raises ValueError when len(z) is not 8 or 12, or len(fg) != len(z)//2.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_doubled_flow_B(z, tau, fg):
    # Oracle: under eps*H_1(x, q) the pair (x, q) is frozen, so the flow is an exact linear kick:
    # p <- p + tau * f(x, q),  y <- y + tau * g(x, q).
    z = np.array(z, dtype=float).ravel()
    fg = np.asarray(fg, dtype=float).ravel()
    if z.size not in (8, 12):
        raise ValueError("z must have length 8 (planar) or 12 (3-D)")
    d = z.size // 4
    if fg.size != 2 * d:
        raise ValueError("fg must have length len(z)//2")
    f, g = fg[:d], fg[d:]
    out = z.copy()
    out[0:d] = z[0:d] + tau * f
    out[3 * d:4 * d] = z[3 * d:4 * d] + tau * g
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
                "doubled_flow_B(\n"
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
                "    0.5379230639795031,\n"
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
                "_oracle_doubled_flow_B(\n"
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
                "    0.5379230639795031,\n"
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
                "doubled_flow_B(\n"
                "    np.array([0.1, 0.2, 3.0, 4.0, 0.11, 0.19, 3.1, 3.9]),\n"
                "    0.0,\n"
                "    np.array([1.0, 2.0, 3.0, 4.0]),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_flow_B(\n"
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
                "doubled_flow_B(\n"
                "    np.linspace(-1.0, 1.0, 12),\n"
                "    -0.6777152,\n"
                "    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_flow_B(\n"
                "    np.linspace(-1.0, 1.0, 12),\n"
                "    -0.6777152,\n"
                "    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "invalid_state_length_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        doubled_flow_B(np.arange(10.0), 0.5, np.ones(5))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_doubled_flow_B(\n"
                "            np.arange(10.0),\n"
                "            0.5,\n"
                "            np.ones(5),\n"
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
