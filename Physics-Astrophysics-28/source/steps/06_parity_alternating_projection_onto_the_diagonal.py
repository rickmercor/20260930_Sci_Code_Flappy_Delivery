"""
Map a doubled state back onto the submanifold x = p, y = q with weights (lam0, mu0), swapping their roles on even steps.

Doubled-phase-space integrators let the two copies separate. Liang and Mei project after every step with p~ = lambda0*p + (1-lambda0)*x and q~ = mu0*q + (1-mu0)*y, alternating which weight acts on momenta and which on positions from one step to the next. They use (lambda0, mu0) = (1/e, 1/pi).

Returns
-------
np.ndarray of shape (4*d,): projected doubled state [p~, q~, p~, q~].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def doubled_projection(z, n, lam0, mu0):
    """Parity-alternating projection of a doubled state onto the diagonal x = p, y = q.

    z    : doubled state, flat array of length 4*d packed as [p, q, x, y].
    n    : step number (integer >= 1). Odd n: p~ = lam0*p + (1-lam0)*x, q~ = mu0*q + (1-mu0)*y.
           Even n: the roles of lam0 and mu0 are swapped.
    Returns the projected doubled state [p~, q~, p~, q~] (flat array of length 4*d).
    Raises ValueError for len(z) not 8 or 12, n < 1, or lam0, mu0 outside (0, 1).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_doubled_projection(z, n, lam0, mu0):
    # Oracle: Liang & Mei projection P (their Eq. 2.6) with parity-alternating weights.
    z = np.asarray(z, dtype=float).ravel()
    if z.size not in (8, 12):
        raise ValueError("z must have length 8 (planar) or 12 (3-D)")
    if int(n) != n or int(n) < 1:
        raise ValueError("step number n must be an integer >= 1")
    if not (0.0 < lam0 < 1.0 and 0.0 < mu0 < 1.0):
        raise ValueError("lam0 and mu0 must lie in (0, 1)")
    d = z.size // 4
    p, q, x, y = z[:d], z[d:2 * d], z[2 * d:3 * d], z[3 * d:]
    if int(n) % 2 == 1:
        a, b = lam0, mu0
    else:
        a, b = mu0, lam0
    pt = a * p + (1.0 - a) * x
    qt = b * q + (1.0 - b) * y
    return np.concatenate([pt, qt, pt, qt])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "name": (
                "normal_odd_step"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "doubled_projection(\n"
                "    np.array(\n"
                "        [\n"
                "            0.01,\n"
                "            0.18,\n"
                "            25.3,\n"
                "            0.4,\n"
                "            0.012,\n"
                "            0.181,\n"
                "            25.31,\n"
                "            0.41,\n"
                "        ],\n"
                "    ),\n"
                "    1,\n"
                "    (1.0 / np.e),\n"
                "    (1.0 / np.pi),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_projection(\n"
                "    np.array(\n"
                "        [\n"
                "            0.01,\n"
                "            0.18,\n"
                "            25.3,\n"
                "            0.4,\n"
                "            0.012,\n"
                "            0.181,\n"
                "            25.31,\n"
                "            0.41,\n"
                "        ],\n"
                "    ),\n"
                "    1,\n"
                "    (1.0 / np.e),\n"
                "    (1.0 / np.pi),\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "boundary_even_step_swaps_weights"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "doubled_projection(\n"
                "    np.array(\n"
                "        [\n"
                "            0.01,\n"
                "            0.18,\n"
                "            25.3,\n"
                "            0.4,\n"
                "            0.012,\n"
                "            0.181,\n"
                "            25.31,\n"
                "            0.41,\n"
                "        ],\n"
                "    ),\n"
                "    2,\n"
                "    (1.0 / np.e),\n"
                "    (1.0 / np.pi),\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_projection(\n"
                "    np.array(\n"
                "        [\n"
                "            0.01,\n"
                "            0.18,\n"
                "            25.3,\n"
                "            0.4,\n"
                "            0.012,\n"
                "            0.181,\n"
                "            25.31,\n"
                "            0.41,\n"
                "        ],\n"
                "    ),\n"
                "    2,\n"
                "    (1.0 / np.e),\n"
                "    (1.0 / np.pi),\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "edge_3d_symmetric_average"
            ),
            "setup": (
                "import numpy as np\n"
            ),
            "call": (
                "doubled_projection(\n"
                "    np.linspace(-1.0, 1.0, 12),\n"
                "    7,\n"
                "    0.5,\n"
                "    0.5,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_doubled_projection(\n"
                "    np.linspace(-1.0, 1.0, 12),\n"
                "    7,\n"
                "    0.5,\n"
                "    0.5,\n"
                ")\n"
            ),
            "tol": 1e-13,
        },
        {
            "name": (
                "invalid_weight_outside_unit_interval_raises_valueerror"
            ),
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        doubled_projection(np.arange(8.0), 1, 1.2, 0.3)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_doubled_projection(\n"
                "            np.arange(8.0),\n"
                "            1,\n"
                "            1.2,\n"
                "            0.3,\n"
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
