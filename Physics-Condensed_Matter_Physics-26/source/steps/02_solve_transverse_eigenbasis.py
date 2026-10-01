"""
Solve the transverse (cross-sectional) Robin eigenvalue problem for a beam cooled by Newton convection on both faces, for both eigenfunction parities at once, and return every eigenpair merged into a single array ordered by increasing eigenvalue.

The inner (localized-heating) temperature field of the beam is expanded on an eigenbasis of the one-dimensional transverse eigenvalue problem d^2(psi)/dX^2 + nu^2 psi = 0 on X in [-1/2, 1/2]. Both faces of the beam, at X = 1/2 and X = -1/2, lose heat to the surroundings by Newton (convective) cooling: at each face, the outward heat flux is proportional to how far the local temperature perturbation psi is above zero, with the same constant of proportionality (the Biot number Bi) at both faces. Because the two faces have outward normals pointing in opposite directions, writing this physical statement as a boundary condition on the derivative of psi requires working out the correct relative sign at X = 1/2 versus X = -1/2 from the physics above; neither boundary condition is given here in already-assembled form. Because the domain is symmetric about X = 0 and the resulting pair of boundary conditions are consistent with that symmetry, the general solution of the ODE splits into two independent families, one symmetric and one antisymmetric about the midplane; neither family's functional form is named here, and both must be identified directly from the ODE and the symmetry of the domain. Substituting each family's functional form into its own boundary condition (derived above) yields that family's own eigenvalue condition, an equation in nu and Bi alone; neither eigenvalue condition is given here, and both must be derived directly from the boundary conditions before being solved numerically.

Each derived eigenvalue condition has infinitely many positive roots, which must be found numerically; a root-finder that only checks a single bracket, or that mistakes a singularity of the trigonometric or hyperbolic terms arising in that condition for a root, will miss modes or return spurious ones.

Once an eigenvalue is known, its constant C must be fixed by requiring the eigenfunction to be normalized to unit L2 norm on [-1/2, 1/2], i.e. the integral of psi(X)^2 dX over that interval must equal 1; no closed-form expression for C is given here, and it must be obtained (analytically or by accurate numerical quadrature) from this normalization condition applied to whichever functional form, cosine or sine, matches the eigenvalue's own family.

The n_modes smallest eigenvalues of EACH family (2*n_modes eigenpairs in total) must be found and merged into a single table, sorted by increasing eigenvalue across both families together, with each row tagged by which family it came from: a parity value of 0.0 for a row belonging to the even (cosine) family, and 1.0 for a row belonging to the odd (sine) family. Downstream steps rely on this parity tag, not on any separation of the output into two pre-sorted blocks, to know which physical role each eigenpair plays.

Returns
-------
np.ndarray of shape (2*n_modes, 3), float: eigenvalue in column 0, normalization constant C in column 1, and parity tag (0.0 = even/cosine family, 1.0 = odd/sine family) in column 2, with rows sorted by strictly increasing eigenvalue.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_transverse_eigenbasis(bi: float, n_modes: int) -> np.ndarray:
    """Merged even+odd transverse Robin-BC eigenbasis.

    Parameters
    ----------
    bi : float
        Biot number of the transverse faces, a finite number > 0.
    n_modes : int
        Number of eigenvalues to find IN EACH family, an integer >= 1
        (so 2*n_modes rows are returned in total).

    Returns
    -------
    modes : np.ndarray
        Array of shape (2*n_modes, 3). Column 0 holds the eigenvalues nu
        (strictly positive), column 1 the normalization constants C
        (strictly positive), and column 2 the parity tag (0.0 for the even
        family, 1.0 for the odd family). Rows are sorted by strictly
        increasing eigenvalue across both families combined.

    Raises
    ------
    ValueError
        If bi is not a finite number > 0, or if n_modes is not an integer
        >= 1.
    """
    return modes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_transverse_eigenbasis(bi: float, n_modes: int) -> np.ndarray:
    import numpy as np
    from scipy.optimize import brentq
    from scipy.integrate import quad

    if not (isinstance(bi, (int, float, np.floating)) and np.isfinite(bi) and float(bi) > 0.0):
        raise ValueError("bi must be a finite number > 0")
    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")

    bi = float(bi)
    n_modes = int(n_modes)

    def _find_family(f, n):
        roots = []
        step = 0.0005
        xmax = 50.0 * (n + 4)
        x = 1e-8
        f_prev = f(x)
        while len(roots) < n and x < xmax:
            x_next = x + step
            f_next = f(x_next)
            if f_prev * f_next < 0.0:
                roots.append(brentq(f, x, x_next, xtol=1e-14))
            x, f_prev = x_next, f_next
        if len(roots) < n:
            raise ValueError("failed to bracket the requested number of eigenvalues")
        return np.array(roots[:n], dtype=float)

    nu_even = _find_family(lambda nu: nu * np.sin(nu / 2.0) - bi * np.cos(nu / 2.0), n_modes)
    nu_odd = _find_family(lambda nu: nu * np.cos(nu / 2.0) + bi * np.sin(nu / 2.0), n_modes)

    c_even = np.array([
        1.0 / np.sqrt(quad(lambda X: np.cos(nu * X) ** 2, -0.5, 0.5)[0]) for nu in nu_even
    ])
    c_odd = np.array([
        1.0 / np.sqrt(quad(lambda X: np.sin(nu * X) ** 2, -0.5, 0.5)[0]) for nu in nu_odd
    ])

    rows = np.concatenate([
        np.column_stack([nu_even, c_even, np.zeros_like(nu_even)]),
        np.column_stack([nu_odd, c_odd, np.ones_like(nu_odd)]),
    ], axis=0)

    order = np.argsort(rows[:, 0])
    return rows[order]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark Biot number (normal scenario) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
bi = 0.6818181818181818
""",
            "call": "solve_transverse_eigenbasis(bi, 4)",
            "gold_call": "_oracle_solve_transverse_eigenbasis(bi, 4)",
        },
        # --- Valid: a much larger Biot number and more modes ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
bi = 5.0
""",
            "call": "solve_transverse_eigenbasis(bi, 6)",
            "gold_call": "_oracle_solve_transverse_eigenbasis(bi, 6)",
        },
        # --- Boundary: n_modes = 1 (2 rows total), small Biot number ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
bi = 0.05
""",
            "call": "solve_transverse_eigenbasis(bi, 1)",
            "gold_call": "_oracle_solve_transverse_eigenbasis(bi, 1)",
        },
        # --- Consistency: every returned eigenvalue satisfies the equation matching its own
        # parity tag, and rows are strictly increasing in eigenvalue ---
        {
            "setup": """import numpy as np
bi = 0.6818181818181818
def check(fn):
    modes = np.asarray(fn(bi, 4), dtype=float)
    ok = int(np.all(np.diff(modes[:, 0]) > 0))
    ok &= int(modes.shape == (8, 3))
    for nu, C, parity in modes:
        if parity == 0.0:
            ok &= int(abs(nu*np.tan(nu/2.0) - bi) < 1e-6)
        else:
            ok &= int(abs(nu/np.tan(nu/2.0) + bi) < 1e-6)
    return int(bool(ok))
""",
            "call": "check(solve_transverse_eigenbasis)",
            "gold_call": "check(_oracle_solve_transverse_eigenbasis)",
        },
        # --- Consistency: each eigenfunction (using its own parity's functional form) is
        # correctly normalized to unit L2 norm ---
        {
            "setup": """import numpy as np
from scipy.integrate import quad
bi = 2.5
def check(fn):
    modes = np.asarray(fn(bi, 3), dtype=float)
    worst = 0.0
    for nu, C, parity in modes:
        shape = (lambda X: C*np.cos(nu*X)) if parity == 0.0 else (lambda X: C*np.sin(nu*X))
        val, _ = quad(lambda X: shape(X)**2, -0.5, 0.5)
        worst = max(worst, abs(val - 1.0))
    return int(worst < 1e-6)
""",
            "call": "check(solve_transverse_eigenbasis)",
            "gold_call": "check(_oracle_solve_transverse_eigenbasis)",
        },
        # --- Consistency: exactly n_modes rows of each parity must be present ---
        {
            "setup": """import numpy as np
bi = 1.3
def check(fn):
    modes = np.asarray(fn(bi, 5), dtype=float)
    n_even = int(np.sum(modes[:, 2] == 0.0))
    n_odd = int(np.sum(modes[:, 2] == 1.0))
    return int(n_even == 5 and n_odd == 5)
""",
            "call": "check(solve_transverse_eigenbasis)",
            "gold_call": "check(_oracle_solve_transverse_eigenbasis)",
        },
        # --- Invalid: non-positive Biot number ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_transverse_eigenbasis(0.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_transverse_eigenbasis(0.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: n_modes = 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_transverse_eigenbasis(1.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_transverse_eigenbasis(1.0, 0)
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
