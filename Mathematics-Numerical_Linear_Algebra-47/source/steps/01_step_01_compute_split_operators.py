"""
Evaluate the three split right-hand-side components of the semi-discrete advected Brusselator system at a given state.

The state vector stacks the three species as y = [u; v; w], each on the same grid of n nodes with uniform spacing dx, so y has length 3n. The function returns the advective, diffusive, and reactive components as three separate rows.

Advection uses second-order centered differences at interior nodes: for a species s, the advective component at node j is -c (s_{j+1} - s_{j-1}) / (2 dx).

Diffusion uses conservative flux form with face coefficients formed as the arithmetic average of adjacent nodal values, D_{j+1/2} = (D_j + D_{j+1}) / 2. The diffusive component at interior node j is (D_{j+1/2} (s_{j+1} - s_j) - D_{j-1/2} (s_j - s_{j-1})) / dx^2.

Reaction is the Brusselator source, evaluated pointwise: r (A - (w + 1) u + v u^2) for the u species, r (w u - v u^2) for the v species, and r ((B - w) / eps - w u) for the w species.

Stationary boundary conditions are imposed by setting all three components to zero at both boundary nodes of every species. For the advective and diffusive components this follows from computing them at interior nodes only; for the reactive component the boundary entries are zeroed explicitly after evaluation.

The function raises ValueError when y is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when D_nodes is not a one-dimensional array of length y.size // 3; when y or D_nodes contains a non-finite value; when dx is not a positive scalar; when eps is not a positive scalar; or when any of c, r, A, B is not a finite scalar.

Method-of-lines treatments of advection-diffusion-reaction systems evaluate the spatial operator once and then hand a large system of ordinary differential equations to a time integrator. Integrators that exploit operator structure require the right-hand side to be supplied not as a single sum but as an additive splitting, because each component is advanced by a different sub-method chosen to match its spectral character. Here the splitting is the natural physical one: advection, whose centered discretization produces a skew-symmetric operator with purely imaginary spectrum; diffusion, whose Jacobian has large negative real eigenvalues scaling as the inverse square of the grid spacing; and reaction, which is spatially local and can be arbitrarily stiff through the parameter eps.

The conservative flux form used for diffusion matters when the coefficient varies in space. Writing the operator as a difference of fluxes evaluated at cell faces, with face coefficients obtained by averaging adjacent nodal values, keeps the discrete operator symmetric and negative semi-definite, which is what makes the real-axis stability arguments underlying stabilized explicit methods applicable. Expanding the derivative by the product rule and discretizing the resulting terms separately does not preserve these properties and yields a different operator.

Stationary boundary conditions hold the boundary values fixed for all time. In this splitting they are imposed on each component separately rather than on the assembled right-hand side, so that every sub-method inherits the same invariant boundary data. This is a stronger condition than it appears: because advection, diffusion, and reaction can each be large and of opposing sign near a fixed boundary, imposing the condition componentwise is what allows a partitioned integrator to advance any single component without violating it.

Returns
-------
np.ndarray of shape (3, 3n), rows ordered as advective, diffusive, reactive components, each a length-3n vector in the same species-stacked ordering as the input state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_split_operators(y: np.ndarray, D_nodes: np.ndarray, dx: float, c: float,
                            r: float, eps: float, A: float, B: float) -> np.ndarray:
    '''Evaluate the advective, diffusive, and reactive split components of the state.

    Parameters
    ----------
    y : np.ndarray
        State vector of shape (3n,), stacking the three species as [u; v; w],
        each sampled on the same grid of n nodes.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.
    c : float
        Advection speed.
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.

    Returns
    -------
    components : np.ndarray
        Array of shape (3, 3n). Row 0 is the advective component, row 1 the
        diffusive component, row 2 the reactive component, each a vector of
        length 3n in the same species-stacked ordering as y.
    '''
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_split_operators(y: np.ndarray, D_nodes: np.ndarray, dx: float,
                                    c: float, r: float, eps: float, A: float,
                                    B: float) -> np.ndarray:
    """Reference implementation."""
    y = np.asarray(y, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y.ndim != 1 or y.size % 3 != 0:
        raise ValueError("y must be a 1D array whose length is a multiple of 3")
    n = y.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y.size // 3")
    if not np.isfinite(y).all() or not np.isfinite(D_nodes).all():
        raise ValueError("y and D_nodes must contain only finite values")
    if not (np.isscalar(dx) or np.ndim(dx) == 0) or float(dx) <= 0.0:
        raise ValueError("dx must be a positive scalar")
    if not (np.isscalar(eps) or np.ndim(eps) == 0) or float(eps) <= 0.0:
        raise ValueError("eps must be a positive scalar")
    for _name, _val in (("c", c), ("r", r), ("A", A), ("B", B)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
    dx = float(dx)
    c = float(c)
    r = float(r)
    eps = float(eps)
    A = float(A)
    B = float(B)

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])
    f_adv = np.zeros(3 * n)
    f_dif = np.zeros(3 * n)
    for k in range(3):
        s = y[k * n:(k + 1) * n]
        adv = np.zeros(n)
        dif = np.zeros(n)
        adv[1:-1] = -c * (s[2:] - s[:-2]) / (2.0 * dx)
        dif[1:-1] = (D_face[1:] * (s[2:] - s[1:-1])
                     - D_face[:-1] * (s[1:-1] - s[:-2])) / dx ** 2
        f_adv[k * n:(k + 1) * n] = adv
        f_dif[k * n:(k + 1) * n] = dif

    u = y[:n]
    v = y[n:2 * n]
    w = y[2 * n:]
    f_rx = np.concatenate([r * (A - (w + 1.0) * u + v * u ** 2),
                           r * (w * u - v * u ** 2),
                           r * ((B - w) / eps - w * u)])
    for k in range(3):
        f_rx[k * n] = 0.0
        f_rx[(k + 1) * n - 1] = 0.0

    return np.vstack([f_adv, f_dif, f_rx])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration, initial state ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
s = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + s, 2.9 / 1.05 + s, 2.9 + s])
c, r, eps, A, B = 0.45, 1.1, 0.02, 1.05, 2.9
""",
            "call": "compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
            "gold_call": "_oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
        },
        # --- Normal: perturbed state on the locked configuration ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
s = 0.1 * np.sin(2.0 * np.pi * x)
rng = np.random.default_rng(7)
y = np.concatenate([1.05 + s, 2.9 / 1.05 + s, 2.9 + s]) + 0.05 * rng.standard_normal(3 * n)
c, r, eps, A, B = 0.45, 1.1, 0.02, 1.05, 2.9
""",
            "call": "compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
            "gold_call": "_oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
        },
        # --- Boundary: minimal admissible grid, one interior node ---
        {
            "setup": """import numpy as np
y = np.array([1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.0, 2.0, 1.0])
D_nodes = np.array([0.2, 0.5, 0.9])
dx = 0.25
c, r, eps, A, B = 0.7, 1.3, 0.05, 1.2, 2.5
""",
            "call": "compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
            "gold_call": "_oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
        },
        # --- Edge: constant diffusion coefficient ---
        {
            "setup": """import numpy as np
n = 9
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.3)
dx = 1.0 / (n - 1.0)
y = np.concatenate([1.0 + x, 2.0 - x, 3.0 + x ** 2])
c, r, eps, A, B = 0.45, 1.1, 0.02, 1.05, 2.9
""",
            "call": "compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
            "gold_call": "_oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
        },
        # --- Edge: zero advection speed with strongly varying diffusion ---
        {
            "setup": """import numpy as np
n = 11
x = np.arange(n) / (n - 1.0)
D_nodes = 0.1 * (1.0 + 0.99 * np.sin(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
y = np.concatenate([np.ones(n), 2.0 * np.ones(n), 3.0 * np.ones(n)])
c, r, eps, A, B = 0.0, 1.0, 0.001, 1.0, 3.0
""",
            "call": "compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
            "gold_call": "_oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
        },
        # --- Edge: spatially uniform state, zero advective and diffusive rows ---
        {
            "setup": """import numpy as np
n = 7
D_nodes = np.linspace(0.05, 0.4, n)
dx = 1.0 / (n - 1.0)
y = np.concatenate([np.full(n, 1.05), np.full(n, 2.9 / 1.05), np.full(n, 2.9)])
c, r, eps, A, B = 0.45, 1.1, 0.02, 1.05, 2.9
""",
            "call": "compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
            "gold_call": "_oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)",
        },
        # --- Invalid: state length not a multiple of 3 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_split_operators(np.zeros(10), np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_split_operators(np.zeros(10), np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fewer than three nodes per species ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_split_operators(np.zeros(6), np.ones(2), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_split_operators(np.zeros(6), np.ones(2), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: diffusion coefficient array length mismatch ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_split_operators(np.zeros(9), np.ones(4), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_split_operators(np.zeros(9), np.ones(4), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive grid spacing ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_split_operators(np.zeros(9), np.ones(3), 0.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_split_operators(np.zeros(9), np.ones(3), 0.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive stiffness parameter ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_split_operators(np.zeros(9), np.ones(3), 1.0, 0.45, 1.1, 0.0, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_split_operators(np.zeros(9), np.ones(3), 1.0, 0.45, 1.1, 0.0, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite state entries ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_split_operators(np.full(9, np.nan), np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_split_operators(np.full(9, np.nan), np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
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
