"""
Advance the state over one diffusion sub-step of length H using a single second-order Runge-Kutta-Legendre super-step with s internal stages and a constant forcing term.

The slope evaluated at every internal stage is the conservative-flux diffusion operator applied to that stage value, plus the constant forcing vector. The diffusion operator uses face coefficients D_{j+1/2} = (D_j + D_{j+1}) / 2, interior-node value (D_{j+1/2} (s_{j+1} - s_j) - D_{j-1/2} (s_j - s_{j-1})) / dx^2, and zero at both boundary nodes of every species. The forcing vector is added unchanged at every internal stage; it does not vary across the sub-step.

The method is the standard low-storage form of this family: the first internal stage is a damped forward step from the incoming state, and each later internal stage is a linear combination of the two preceding internal stages and the incoming state, corrected by the slope at the immediately preceding stage and by the slope evaluated once at the incoming state and reused throughout. The coefficients are those of the second-order Legendre construction for the requested stage count. The sub-step result is the last internal stage.

The function raises ValueError when y is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when forcing does not have the same shape as y; when D_nodes is not a one-dimensional array of length y.size // 3; when y, forcing or D_nodes contains a non-finite value; when H is not a finite positive scalar; when dx is not a finite positive scalar; or when s is not an integer of at least 2.

Runge-Kutta-Legendre methods belong to the family of stabilized explicit schemes built on three-term recurrences of orthogonal polynomials. Rather than storing a full Butcher tableau of size growing quadratically in the stage count, the method advances through a short recurrence in which each internal stage is a combination of the two preceding stages, the incoming state, and one new right-hand-side evaluation. This gives a low-storage implementation whose memory footprint is independent of the stage count, which is what makes counts in the tens or hundreds practical.

The stability polynomial of the second-order Legendre construction is built so that its stability interval along the negative real axis is exactly (s^2 + s - 2) / 2, with no free damping parameter to tune. This distinguishes it from Chebyshev-based constructions, which reach slightly further for a given stage count but require a small damping parameter to pull the stability function away from the unit circle at interior points of the interval, and whose recurrence coefficients are computed from Chebyshev polynomial values and derivatives at a shifted argument. The two families therefore have different coefficients and different stage-count rules, and mixing the recurrence of one with the stage-count rule of the other produces a scheme with no established stability interval.

The term proportional to the slope frozen at the incoming state is easy to overlook and is what makes the scheme second order rather than first. In the general form of these three-term recurrences the stage update carries two slope contributions: one at the immediately preceding stage and one at the incoming state, weighted by a coefficient that depends on how far the shifted polynomial at the previous index departs from unity. Dropping the frozen-slope contribution leaves a scheme that still runs, still remains bounded, and still resembles the intended method, but whose order and stability interval both differ.

The constant forcing term reflects how such a method is used inside a partitioned integrator. The inner sub-problem is not the pure diffusion equation but that equation modified by a source assembled from the outer method's other operators, held fixed across the sub-step. Because the source is constant it enters every internal slope evaluation identically, and the recurrence needs no modification beyond evaluating the augmented right-hand side.

Returns
-------
np.ndarray of shape (3n,), the state at the end of the sub-step in the same species-stacked ordering as the input state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rkl2_super_step(y: np.ndarray, H: float, s: int, forcing: np.ndarray,
                    D_nodes: np.ndarray, dx: float) -> np.ndarray:
    '''Advance one diffusion sub-step by a second-order Runge-Kutta-Legendre super-step.

    Parameters
    ----------
    y : np.ndarray
        Incoming state vector of shape (3n,), stacking the three species as
        [u; v; w].
    H : float
        Length of the sub-step, positive.
    s : int
        Number of internal stages, at least 2.
    forcing : np.ndarray
        Constant forcing vector of shape (3n,), added to the diffusion slope at
        every internal stage.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.

    Returns
    -------
    y_next : np.ndarray
        State at the end of the sub-step, shape (3n,).
    '''
    return y_next  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rkl2_super_step(y: np.ndarray, H: float, s: int, forcing: np.ndarray,
                            D_nodes: np.ndarray, dx: float) -> np.ndarray:
    """Reference implementation."""
    y = np.asarray(y, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y.ndim != 1 or y.size % 3 != 0:
        raise ValueError("y must be a 1D array whose length is a multiple of 3")
    n = y.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if forcing.shape != y.shape:
        raise ValueError("forcing must have the same shape as y")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y.size // 3")
    if not (np.isfinite(y).all() and np.isfinite(forcing).all()
            and np.isfinite(D_nodes).all()):
        raise ValueError("y, forcing and D_nodes must contain only finite values")
    if not (np.isscalar(H) or np.ndim(H) == 0) or not np.isfinite(float(H)) \
            or float(H) <= 0.0:
        raise ValueError("H must be a finite positive scalar")
    if not (np.isscalar(dx) or np.ndim(dx) == 0) or not np.isfinite(float(dx)) \
            or float(dx) <= 0.0:
        raise ValueError("dx must be a finite positive scalar")
    if int(s) != s or int(s) < 2:
        raise ValueError("s must be an integer of at least 2")
    H = float(H)
    dx = float(dx)
    s = int(s)

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])

    def _slope(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            sp = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = (D_face[1:] * (sp[2:] - sp[1:-1])
                       - D_face[:-1] * (sp[1:-1] - sp[:-2])) / dx ** 2
            out[k * n:(k + 1) * n] = g
        return out + forcing

    b = np.zeros(s + 1)
    b[0] = 1.0 / 3.0
    b[1] = 1.0 / 3.0
    for j in range(2, s + 1):
        b[j] = (j * j + j - 2.0) / (2.0 * j * (j + 1.0))
    a = 1.0 - b
    w1 = 4.0 / (s * s + s - 2.0)

    f0 = _slope(y)
    y_jm2 = y
    y_jm1 = y + b[1] * w1 * H * f0
    for j in range(2, s + 1):
        mu = (2.0 * j - 1.0) / j * b[j] / b[j - 1]
        nu = -(j - 1.0) / j * b[j] / b[j - 2]
        mu_t = mu * w1
        gamma_t = -a[j - 1] * mu_t
        y_new = (mu * y_jm1 + nu * y_jm2 + (1.0 - mu - nu) * y
                 + mu_t * H * _slope(y_jm1) + gamma_t * H * f0)
        y_jm2 = y_jm1
        y_jm1 = y_new

    return y_jm1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration, first super-time-stepping sub-step ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 73.0
H = 2.0 * gamma * h
rng = np.random.default_rng(3)
forcing = 0.4 * rng.standard_normal(3 * n)
""",
            "call": "rkl2_super_step(y, H, 9, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, H, 9, forcing, D_nodes, dx)",
        },
        # --- Normal: locked configuration, second super-time-stepping sub-step ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 73.0
H = (1.0 - 2.0 * gamma) * h
rng = np.random.default_rng(11)
forcing = 0.25 * rng.standard_normal(3 * n)
""",
            "call": "rkl2_super_step(y, H, 8, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, H, 8, forcing, D_nodes, dx)",
        },
        # --- Normal: same sub-step advanced with a different stage count ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
gamma = (2.0 - np.sqrt(2.0)) / 2.0
H = 2.0 * gamma * 1.5 / 73.0
forcing = np.zeros(3 * n)
""",
            "call": "rkl2_super_step(y, H, 14, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, H, 14, forcing, D_nodes, dx)",
        },
        # --- Boundary: minimum admissible stage count on a short sub-step ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
forcing = np.zeros(3 * n)
""",
            "call": "rkl2_super_step(y, 1e-5, 2, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, 1e-5, 2, forcing, D_nodes, dx)",
        },
        # --- Boundary: minimal grid, one interior node per species ---
        {
            "setup": """import numpy as np
y = np.array([1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.0, 2.0, 1.0])
D_nodes = np.array([0.2, 0.5, 0.9])
dx = 0.25
forcing = np.array([0.1, -0.2, 0.3, 0.0, 0.4, -0.1, 0.2, 0.1, -0.3])
""",
            "call": "rkl2_super_step(y, 0.01, 4, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, 0.01, 4, forcing, D_nodes, dx)",
        },
        # --- Edge: zero forcing on a coarse grid with constant coefficient ---
        {
            "setup": """import numpy as np
n = 9
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.3)
dx = 1.0 / (n - 1.0)
y = np.concatenate([1.0 + x, 2.0 - x, 3.0 + x ** 2])
forcing = np.zeros(3 * n)
""",
            "call": "rkl2_super_step(y, 0.005, 6, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, 0.005, 6, forcing, D_nodes, dx)",
        },
        # --- Edge: spatially uniform state so only the forcing drives the update ---
        {
            "setup": """import numpy as np
n = 11
x = np.arange(n) / (n - 1.0)
D_nodes = 0.1 * (1.0 + 0.99 * np.sin(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
y = np.concatenate([np.ones(n), 2.0 * np.ones(n), 3.0 * np.ones(n)])
forcing = np.full(3 * n, 0.5)
""",
            "call": "rkl2_super_step(y, 0.002, 5, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, 0.002, 5, forcing, D_nodes, dx)",
        },
        # --- Edge: large stage count on a strongly stiff coarse problem ---
        {
            "setup": """import numpy as np
n = 33
x = np.arange(n) / (n - 1.0)
D_nodes = 0.5 * (1.0 + 0.8 * np.cos(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
y = np.concatenate([1.0 + 0.3 * np.sin(4.0 * np.pi * x), 2.0 - 0.2 * x, 3.0 + x])
forcing = np.zeros(3 * n)
""",
            "call": "rkl2_super_step(y, 0.02, 40, forcing, D_nodes, dx)",
            "gold_call": "_oracle_rkl2_super_step(y, 0.02, 40, forcing, D_nodes, dx)",
        },
        # --- Invalid: forcing shape does not match the state ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        rkl2_super_step(np.zeros(9), 0.01, 5, np.zeros(8), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rkl2_super_step(np.zeros(9), 0.01, 5, np.zeros(8), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: stage count below the two-stage minimum ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        rkl2_super_step(np.zeros(9), 0.01, 1, np.zeros(9), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rkl2_super_step(np.zeros(9), 0.01, 1, np.zeros(9), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-integer stage count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        rkl2_super_step(np.zeros(9), 0.01, 2.5, np.zeros(9), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rkl2_super_step(np.zeros(9), 0.01, 2.5, np.zeros(9), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive sub-step length ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        rkl2_super_step(np.zeros(9), 0.0, 5, np.zeros(9), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rkl2_super_step(np.zeros(9), 0.0, 5, np.zeros(9), np.ones(3), 1.0)
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
        rkl2_super_step(np.full(9, np.inf), 0.01, 5, np.zeros(9), np.ones(3), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rkl2_super_step(np.full(9, np.inf), 0.01, 5, np.zeros(9), np.ones(3), 1.0)
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
