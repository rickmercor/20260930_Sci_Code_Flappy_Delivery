"""
Build the diabatic proton potential energy profile of one electronic state on a grid of proton positions. Each diabatic electronic state of the assembly holds the transferring proton in an asymmetric double well whose barrier height, bias and vertical offset are properties of that electronic state.

When the transferring proton is treated quantum mechanically, every diabatic electronic state supplies its own one-dimensional potential along the proton transfer axis. In practice these profiles come from scanning the proton between its donor and acceptor heavy atoms and running an excited-state calculation at each point. Here the profile is given in closed form as a quartic double well with a linear bias, written in the reduced coordinate u = r / r_ref so that the two minima sit near u = -1 and u = +1. The sign of the bias decides which well lies lower: a positive bias favours the donor side at negative r, a negative bias favours the acceptor side. The vertical offset carries the diabatic electronic energy of the state, so differences between offsets set the energy ordering of the electronic states before any vibrational energy is added.

Returns
-------
np.ndarray of shape (N,), the diabatic proton potential energy in eV at each grid point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def proton_potential(r_grid: np.ndarray, a_barrier: float, r_ref: float,
                     b_bias: float, e_offset: float) -> np.ndarray:
    '''Diabatic proton potential energy profile on a grid.

    The profile is a symmetric quartic double well in the reduced coordinate
    u = r_grid / r_ref, scaled by a_barrier, plus a linear bias b_bias * u that
    breaks the symmetry, plus a constant vertical offset e_offset carrying the
    diabatic electronic energy.

    Parameters
    ----------
    r_grid : np.ndarray
        (N,) proton positions in angstrom, one dimensional with at least two
        points and all entries finite.
    a_barrier : float
        Positive quartic prefactor in eV.
    r_ref : float
        Positive reduced-coordinate scale in angstrom.
    b_bias : float
        Linear bias coefficient in eV; positive lowers the negative-r well.
    e_offset : float
        Constant vertical offset in eV.

    Returns
    -------
    potential : np.ndarray
        (N,) potential energy in eV at each grid point.

    Raises
    ------
    ValueError
        If r_grid is not one dimensional with at least two finite points, or if
        a_barrier or r_ref is not positive.
    '''
    return potential  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_proton_potential(r_grid: np.ndarray, a_barrier: float, r_ref: float,
                             b_bias: float, e_offset: float) -> np.ndarray:
    """Reference implementation."""
    r = np.asarray(r_grid, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("r_grid must be a one-dimensional array with at least two points")
    if not np.isfinite(r).all():
        raise ValueError("r_grid must be finite")
    if not (r_ref > 0.0):
        raise ValueError("r_ref must be positive")
    if not (a_barrier > 0.0):
        raise ValueError("a_barrier must be positive")
    u = r / r_ref
    return a_barrier * (u * u - 1.0) ** 2 + b_bias * u + e_offset

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: the reactant state, biased towards the donor side
r_grid = np.linspace(-0.90, 0.90, 201)
""",
            "call": "proton_potential(r_grid, 0.595, 0.400, 0.180, 3.250)",
            "gold_call": "_oracle_proton_potential(r_grid, 0.595, 0.400, 0.180, 3.250)",
        },
        {
            "setup": """import numpy as np
# normal: a charge-transfer state, biased towards the acceptor side
r_grid = np.linspace(-0.90, 0.90, 201)
""",
            "call": "proton_potential(r_grid, 0.290, 0.400, -0.080, 3.615)",
            "gold_call": "_oracle_proton_potential(r_grid, 0.290, 0.400, -0.080, 3.615)",
        },
        {
            "setup": """import numpy as np
# boundary: zero bias leaves the well symmetric about the origin
r_grid = np.linspace(-0.80, 0.80, 9)
""",
            "call": "proton_potential(r_grid, 0.500, 0.400, 0.0, 0.0)",
            "gold_call": "_oracle_proton_potential(r_grid, 0.500, 0.400, 0.0, 0.0)",
        },
        {
            "setup": """import numpy as np
# edge: the two-point minimum grid, evaluated exactly at the reduced minima
r_grid = np.array([-0.400, 0.400])
""",
            "call": "proton_potential(r_grid, 0.610, 0.400, -0.175, 3.100)",
            "gold_call": "_oracle_proton_potential(r_grid, 0.610, 0.400, -0.175, 3.100)",
        },
        {
            "setup": """import numpy as np
# edge: a grid that runs backwards is still a valid set of positions
r_grid = np.linspace(0.90, -0.90, 21)
""",
            "call": "proton_potential(r_grid, 0.300, 0.400, -0.185, 4.260)",
            "gold_call": "_oracle_proton_potential(r_grid, 0.300, 0.400, -0.185, 4.260)",
        },
        {
            "setup": """import numpy as np
r_grid = np.linspace(-0.9, 0.9, 11)
def run(f):
    try:
        f(r_grid, -0.5, 0.400, 0.180, 3.250); return 0      # negative barrier
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(proton_potential)",
            "gold_call": "run(_oracle_proton_potential)",
        },
        {
            "setup": """import numpy as np
r_grid = np.linspace(-0.9, 0.9, 11)
def run(f):
    try:
        f(r_grid, 0.595, 0.0, 0.180, 3.250); return 0       # zero reduced scale
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(proton_potential)",
            "gold_call": "run(_oracle_proton_potential)",
        },
        {
            "setup": """import numpy as np
grid2d = np.linspace(-0.9, 0.9, 12).reshape(3, 4)
def run(f):
    try:
        f(grid2d, 0.595, 0.400, 0.180, 3.250); return 0     # not one dimensional
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(proton_potential)",
            "gold_call": "run(_oracle_proton_potential)",
        },
    ]
