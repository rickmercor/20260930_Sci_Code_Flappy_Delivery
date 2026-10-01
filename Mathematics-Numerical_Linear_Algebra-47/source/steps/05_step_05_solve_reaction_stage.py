"""
Solve one implicit reaction stage of the partitioned step by nodewise Newton iteration.

The stage equation is z - h_gamma f_R(z) = rhs, where f_R is the Brusselator reaction source evaluated pointwise: r (A - (w + 1) u + v u^2) for the u species, r (w u - v u^2) for the v species, and r ((B - w) / eps - w u) for the w species, with the state stacked as [u; v; w] on a grid of n nodes. Stationary boundary conditions are imposed by zeroing the reaction source at both boundary nodes of every species, so the stage equation reduces to z = rhs at those nodes.

Because the reaction source couples only the three species values sharing a node, the Jacobian of the stage equation is block diagonal with one three by three block per node. The iteration is therefore performed nodewise: at each pass form the residual z - h_gamma f_R(z) - rhs, stop when its maximum absolute value is below tol, and otherwise solve the three by three block systems node by node and update z by subtracting the correction. The blocks are built from the exact derivative of the reaction source with respect to the three species values at that node, with the same boundary zeroing applied, so that boundary blocks reduce to the identity.

The iteration starts from guess and performs at most max_iter passes.

The function raises ValueError when rhs is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when guess does not have the same shape as rhs; when rhs or guess contains a non-finite value; when any of h_gamma, r, A, B is not a finite scalar; when h_gamma is negative; when eps is not a finite positive scalar; when tol is not a positive scalar; when max_iter is less than 1; when a nodewise block system is singular; when the iteration produces non-finite values; or when the residual criterion is not met within max_iter passes.

The reason a partitioned integrator is worth building for this class of problems is visible in this stage. Reaction terms in an advection-diffusion-reaction system are spatially local: the source at a node depends only on the state at that node. Diffusion terms are not. If both are treated implicitly together, as a standard implicit-explicit additive Runge-Kutta method does when it groups all stiff operators, the resulting algebraic system couples every node in the domain and must be solved with a global linear or nonlinear solver, requiring communication across the whole grid at every stage of every step. If instead the diffusive operator is handled by a stabilized explicit sub-method, the implicit solve that remains involves only the reaction terms, and its Jacobian is block diagonal with one small block per node. The blocks can be factored independently and in parallel, with no communication at all.

This locality is why the outer method must be solve-decoupled. In the general multirate-infinitesimal setting, a stage that simultaneously carries a nonzero implicit diagonal coefficient and a nonzero abscissa increment would require solving for the stage value while also integrating a modified sub-problem over that stage interval, which couples the two operations. Restricting the coefficients so that stages with a nonzero implicit diagonal have zero abscissa increment collapses the sub-problem to a standard backward-Euler-like algebraic equation, and it is that equation which is solved here.

The stiffness of the reaction channel enters through the parameter eps, which appears in the third species as a relaxation with rate proportional to its reciprocal. Small eps makes that channel very fast relative to the transport time scale, so an explicit treatment would restrict the step size severely while the implicit treatment absorbs it. The exact block Jacobian matters at small eps: the large diagonal entry it contributes is precisely what makes the Newton correction well scaled, and an approximate or frozen Jacobian will still converge but along a different iteration path, so agreement between implementations holds only to the residual tolerance rather than to rounding.

Returns
-------
np.ndarray of shape (3n,), the stage solution in the same species-stacked ordering as the input right-hand side
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_reaction_stage(rhs: np.ndarray, h_gamma: float, guess: np.ndarray,
                         r: float, eps: float, A: float, B: float,
                         tol: float = 1e-13, max_iter: int = 60) -> np.ndarray:
    '''Solve the implicit reaction stage equation by nodewise Newton iteration.

    Parameters
    ----------
    rhs : np.ndarray
        Right-hand side of the stage equation, shape (3n,), stacking the three
        species as [u; v; w].
    h_gamma : float
        Product of the step size and the implicit diagonal coefficient,
        non-negative.
    guess : np.ndarray
        Initial iterate for the Newton solve, shape (3n,).
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.
    tol : float
        Residual tolerance in the maximum norm, positive.
    max_iter : int
        Maximum number of Newton passes permitted, at least 1.

    Returns
    -------
    z : np.ndarray
        Stage solution of shape (3n,) satisfying the stage equation to within tol.
    '''
    return z  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_reaction_stage(rhs: np.ndarray, h_gamma: float, guess: np.ndarray,
                                 r: float, eps: float, A: float, B: float,
                                 tol: float = 1e-13,
                                 max_iter: int = 60) -> np.ndarray:
    """Reference implementation."""
    rhs = np.asarray(rhs, dtype=float)
    guess = np.asarray(guess, dtype=float)
    if rhs.ndim != 1 or rhs.size % 3 != 0:
        raise ValueError("rhs must be a 1D array whose length is a multiple of 3")
    n = rhs.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if guess.shape != rhs.shape:
        raise ValueError("guess must have the same shape as rhs")
    if not (np.isfinite(rhs).all() and np.isfinite(guess).all()):
        raise ValueError("rhs and guess must contain only finite values")
    for _name, _val in (("h_gamma", h_gamma), ("r", r), ("A", A), ("B", B)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) \
                or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
    if float(h_gamma) < 0.0:
        raise ValueError("h_gamma must be non-negative")
    if not (np.isscalar(eps) or np.ndim(eps) == 0) or not np.isfinite(float(eps)) \
            or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive scalar")
    if not (np.isscalar(tol) or np.ndim(tol) == 0) or float(tol) <= 0.0:
        raise ValueError("tol must be a positive scalar")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")
    hg = float(h_gamma)
    r = float(r)
    eps = float(eps)
    A = float(A)
    B = float(B)
    tol = float(tol)

    def _reaction(vec):
        u = vec[:n]
        v = vec[n:2 * n]
        w = vec[2 * n:]
        out = np.concatenate([r * (A - (w + 1.0) * u + v * u ** 2),
                              r * (w * u - v * u ** 2),
                              r * ((B - w) / eps - w * u)])
        for k in range(3):
            out[k * n] = 0.0
            out[(k + 1) * n - 1] = 0.0
        return out

    mask = np.ones(n)
    mask[0] = 0.0
    mask[-1] = 0.0

    z = guess.copy()
    converged = False
    for _ in range(int(max_iter)):
        resid = z - hg * _reaction(z) - rhs
        if np.max(np.abs(resid)) < tol:
            converged = True
            break
        u = z[:n]
        v = z[n:2 * n]
        w = z[2 * n:]
        J = np.zeros((n, 3, 3))
        J[:, 0, 0] = r * (-(w + 1.0) + 2.0 * v * u) * mask
        J[:, 0, 1] = r * u ** 2 * mask
        J[:, 0, 2] = -r * u * mask
        J[:, 1, 0] = r * (w - 2.0 * v * u) * mask
        J[:, 1, 1] = -r * u ** 2 * mask
        J[:, 1, 2] = r * u * mask
        J[:, 2, 0] = -r * w * mask
        J[:, 2, 2] = r * (-1.0 / eps - u) * mask
        M = np.tile(np.eye(3), (n, 1, 1)) - hg * J
        Fn = np.stack([resid[:n], resid[n:2 * n], resid[2 * n:]], axis=1)
        try:
            dz = np.linalg.solve(M, Fn[..., None])[..., 0]
        except np.linalg.LinAlgError:
            raise ValueError("nodewise Newton system is singular")
        z = z - np.concatenate([dz[:, 0], dz[:, 1], dz[:, 2]])
        if not np.isfinite(z).all():
            raise ValueError("nodewise Newton iteration produced non-finite values")
    if not converged:
        resid = z - hg * _reaction(z) - rhs
        if np.max(np.abs(resid)) >= tol:
            raise ValueError("nodewise Newton did not converge within max_iter")

    return z

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration, implicit diagonal coefficient gamma ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 73.0
rng = np.random.default_rng(5)
rhs = y0 + 0.02 * rng.standard_normal(3 * n)
h_gamma = h * gamma
""",
            "call": "solve_reaction_stage(rhs, h_gamma, y0, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, h_gamma, y0, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Normal: same configuration with a larger perturbation of the state ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
gamma = (2.0 - np.sqrt(2.0)) / 2.0
h = 1.5 / 73.0
rng = np.random.default_rng(19)
rhs = y0 + 0.15 * rng.standard_normal(3 * n)
h_gamma = h * gamma
""",
            "call": "solve_reaction_stage(rhs, h_gamma, y0, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, h_gamma, y0, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Normal: much stiffer relaxation channel with a matching tolerance ---
        {
            "setup": """import numpy as np
n = 33
x = np.arange(n) / (n - 1.0)
sn = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
rng = np.random.default_rng(23)
rhs = y0 + 0.05 * rng.standard_normal(3 * n)
""",
            "call": "solve_reaction_stage(rhs, 0.006, y0, 1.1, 1e-4, 1.05, 2.9, 1e-10)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, 0.006, y0, 1.1, 1e-4, 1.05, 2.9, 1e-10)",
        },
        # --- Boundary: zero implicit coefficient reduces the stage to the identity ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
rng = np.random.default_rng(31)
rhs = y0 + 0.02 * rng.standard_normal(3 * n)
""",
            "call": "solve_reaction_stage(rhs, 0.0, y0, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, 0.0, y0, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Boundary: minimal grid, one interior node per species ---
        {
            "setup": """import numpy as np
rhs = np.array([1.0, 1.2, 1.4, 2.6, 2.8, 3.0, 2.9, 2.7, 2.5])
guess = rhs.copy()
""",
            "call": "solve_reaction_stage(rhs, 0.01, guess, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, 0.01, guess, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Edge: guess already equals the right-hand side, no reaction scaling ---
        {
            "setup": """import numpy as np
n = 11
x = np.arange(n) / (n - 1.0)
rhs = np.concatenate([1.0 + 0.1 * x, 2.0 - 0.1 * x, 3.0 + 0.05 * x])
guess = rhs.copy()
""",
            "call": "solve_reaction_stage(rhs, 0.02, guess, 0.0, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, 0.02, guess, 0.0, 0.02, 1.05, 2.9)",
        },
        # --- Edge: large implicit coefficient far from the reference state ---
        {
            "setup": """import numpy as np
n = 17
x = np.arange(n) / (n - 1.0)
rhs = np.concatenate([0.5 + x, 3.5 - x, 2.0 + 0.5 * x])
guess = np.concatenate([np.ones(n), 2.0 * np.ones(n), 3.0 * np.ones(n)])
""",
            "call": "solve_reaction_stage(rhs, 0.5, guess, 1.3, 0.05, 1.2, 2.5)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, 0.5, guess, 1.3, 0.05, 1.2, 2.5)",
        },
        # --- Edge: looser residual tolerance with a tight iteration budget ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
rng = np.random.default_rng(41)
rhs = y0 + 0.02 * rng.standard_normal(3 * n)
""",
            "call": "solve_reaction_stage(rhs, 0.006, y0, 1.1, 0.02, 1.05, 2.9, 1e-8, 5)",
            "gold_call": "_oracle_solve_reaction_stage(rhs, 0.006, y0, 1.1, 0.02, 1.05, 2.9, 1e-8, 5)",
        },
        # --- Invalid: right-hand side length not a multiple of 3 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_reaction_stage(np.zeros(10), 0.01, np.zeros(10), 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_reaction_stage(np.zeros(10), 0.01, np.zeros(10), 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: guess shape does not match the right-hand side ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_reaction_stage(np.zeros(9), 0.01, np.zeros(8), 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_reaction_stage(np.zeros(9), 0.01, np.zeros(8), 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative implicit coefficient ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_reaction_stage(np.zeros(9), -0.01, np.zeros(9), 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_reaction_stage(np.zeros(9), -0.01, np.zeros(9), 1.1, 0.02, 1.05, 2.9)
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
        solve_reaction_stage(np.zeros(9), 0.01, np.zeros(9), 1.1, 0.0, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_reaction_stage(np.zeros(9), 0.01, np.zeros(9), 1.1, 0.0, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: residual criterion unreachable within the iteration budget ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y0 = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
rng = np.random.default_rng(5)
rhs = y0 + 0.02 * rng.standard_normal(3 * n)
def run_model():
    try:
        solve_reaction_stage(rhs, 0.006, y0, 1.1, 0.02, 1.05, 2.9, 1e-13, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_reaction_stage(rhs, 0.006, y0, 1.1, 0.02, 1.05, 2.9, 1e-13, 1)
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
