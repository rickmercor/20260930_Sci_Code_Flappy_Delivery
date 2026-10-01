"""
Construct the central difference potential by sampling it on the tensor product of a spatial grid and a grid in the dual variable. The potential is supplied as a callable and must be evaluated directly at the shifted arguments. Raise ValueError for a non-callable potential, for grids that are not one-dimensional, for empty grids, for grids containing non-finite entries, and for a potential that does not broadcast elementwise over its argument.

The Wigner equation differs from its classical counterpart in that the potential does not act pointwise. It enters through a pseudo-differential operator that convolves the quasi-distribution against a Wigner kernel, and that kernel is the Fourier transform, taken in the dual variable y at fixed x, of a difference of the potential evaluated at two arguments displaced symmetrically about x. This difference is the object built here.

Returns
-------
np.ndarray of shape (Nx, Ny), the central difference potential sampled on the tensor grid, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def difference_potential_matrix(V: "Callable[[np.ndarray], np.ndarray]",
                                x_grid: "np.ndarray",
                                y_grid: "np.ndarray") -> "np.ndarray":
    '''Sample the central difference potential on a tensor grid.

    Parameters
    ----------
    V : Callable[[np.ndarray], np.ndarray]
        Potential energy function, evaluated elementwise on a numpy array of
        arbitrary shape and returning an array of that same shape.
    x_grid : np.ndarray
        (Nx,) one-dimensional array of spatial nodes.
    y_grid : np.ndarray
        (Ny,) one-dimensional array of nodes in the dual variable.

    Returns
    -------
    D_V : np.ndarray
        (Nx, Ny) array of the central difference potential, float64.
    '''
    return D_V  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from typing import Callable


def _oracle_difference_potential_matrix(V: "Callable[[np.ndarray], np.ndarray]",
                                        x_grid: "np.ndarray",
                                        y_grid: "np.ndarray") -> "np.ndarray":
    if not callable(V):
        raise ValueError("V must be a callable accepting a numpy array")
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional")
    if x.size < 1 or y.size < 1:
        raise ValueError("x_grid and y_grid must be non-empty")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x_grid and y_grid must contain only finite values")
    Xp = x[:, None] + 0.5 * y[None, :]
    Xm = x[:, None] - 0.5 * y[None, :]
    out = np.asarray(V(Xp), dtype=float) - np.asarray(V(Xm), dtype=float)
    if out.shape != (x.size, y.size):
        raise ValueError("V must broadcast elementwise over its argument")
    return out.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: harmonic potential on the production grid ---
        {
            "setup": """import numpy as np
V = lambda t: 0.5*t**2
x_grid = np.linspace(-10.0, 10.0, 128, endpoint=False)
y_grid = np.linspace(-10.0, 10.0, 128, endpoint=False)
""",
            "call": "difference_potential_matrix(V, x_grid, y_grid)",
            "gold_call": "_oracle_difference_potential_matrix(V, x_grid, y_grid)",
        },
        # --- boundary: degenerate dual grid, single node at the origin ---
        {
            "setup": """import numpy as np
V = lambda t: 0.5*t**2
x_grid = np.linspace(-3.0, 3.0, 7)
y_grid = np.array([0.0])
""",
            "call": "difference_potential_matrix(V, x_grid, y_grid)",
            "gold_call": "_oracle_difference_potential_matrix(V, x_grid, y_grid)",
        },
        # --- edge: shifted argument ranges far outside the span of x_grid ---
        {
            "setup": """import numpy as np
V = lambda t: 0.3*np.exp(-(t-30.0)**2/2.0)
x_grid = np.linspace(0.0, 60.0, 33)
y_grid = np.linspace(-400.0, 400.0, 17)
""",
            "call": "difference_potential_matrix(V, x_grid, y_grid)",
            "gold_call": "_oracle_difference_potential_matrix(V, x_grid, y_grid)",
        },
        # --- edge: asymmetric non-square grids, odd Nx, anharmonic potential ---
        {
            "setup": """import numpy as np
V = lambda t: t**3/3.0 - 2.0*t
x_grid = np.linspace(-2.0, 5.0, 11)
y_grid = np.linspace(-1.5, 4.5, 8)
""",
            "call": "difference_potential_matrix(V, x_grid, y_grid)",
            "gold_call": "_oracle_difference_potential_matrix(V, x_grid, y_grid)",
        },
        # --- edge: catastrophic cancellation, nodes separated by ~1e-14 ---
        {
            "setup": """import numpy as np
V = lambda t: 1.0/np.sqrt(np.abs(t)**2 + 1.0)
x_grid = np.array([-1e-14, 0.0, 1e-14])
y_grid = np.array([-2e-14, 0.0, 2e-14])
""",
            "call": "difference_potential_matrix(V, x_grid, y_grid)",
            "gold_call": "_oracle_difference_potential_matrix(V, x_grid, y_grid)",
        },
        # --- invalid: potential is not callable ---
        {
            "setup": """import numpy as np
V = 3.14
x_grid = np.array([0.0, 1.0])
y_grid = np.array([0.0, 1.0])
def run_model():
    try:
        difference_potential_matrix(V, x_grid, y_grid)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_difference_potential_matrix(V, x_grid, y_grid)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: two-dimensional spatial grid ---
        {
            "setup": """import numpy as np
V = lambda t: 0.5*t**2
x_grid = np.zeros((2, 2))
y_grid = np.array([0.0, 1.0])
def run_model():
    try:
        difference_potential_matrix(V, x_grid, y_grid)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_difference_potential_matrix(V, x_grid, y_grid)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite node in the spatial grid ---
        {
            "setup": """import numpy as np
V = lambda t: 0.5*t**2
x_grid = np.array([0.0, np.inf])
y_grid = np.array([0.0, 1.0])
def run_model():
    try:
        difference_potential_matrix(V, x_grid, y_grid)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_difference_potential_matrix(V, x_grid, y_grid)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
