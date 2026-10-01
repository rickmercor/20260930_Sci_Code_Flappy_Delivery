"""
Pair structure of a multicomponent dissipative particle dynamics (DPD) solvent in the hypernetted-chain (HNC) closure: the total and direct correlation functions of every species pair on the radial grid.

Beads of species mu and nu interact through the ultrasoft DPD repulsion beta phi_mu_nu(r) = A_mu_nu (1 - r)^2 / 2 for r < 1 and zero beyond, in units where the interaction range and k_B T are both one. The solvent has total bead density rho and mole fractions x_mu, so species mu has partial density rho x_mu.

The total correlation functions h_mu_nu and the direct correlation functions c_mu_nu obey the multicomponent Ornstein-Zernike relation, h_mu_nu(q) = c_mu_nu(q) + sum over lambda of rho x_lambda c_mu_lambda(q) h_lambda_nu(q) at every wavenumber, together with the HNC closure h_mu_nu(r) = exp[-beta phi_mu_nu(r) + h_mu_nu(r) - c_mu_nu(r)] - 1 for every pair. Transforms are those of the previous step on the same grid.

A species may be present at zero mole fraction. It then has no effect on the others, and its correlations are those of an infinitely dilute species in the remaining solvent, which the same equations still determine. Iterate until the indirect correlation h - c changes by less than 1e-11 anywhere between successive iterates; the solution is unique for the systems used here, so any convergent scheme gives the same functions.

Returns
-------
numpy.ndarray of shape (2, m, m, n_grid - 1): the HNC total correlation functions h (index 0) and direct correlation functions c (index 1) of every solvent species pair on the radial grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solvent_structure(rho: float, x: np.ndarray, A: np.ndarray, n_grid: int, dr: float) -> np.ndarray:
    '''HNC total and direct correlation functions of a multicomponent DPD solvent.

    Parameters
    ----------
    rho : float
        Total bead number density of the solvent, in beads per cubed range.
    x : numpy.ndarray
        Mole fractions of the m solvent species, shape (m,), non-negative and
        summing to one.
    A : numpy.ndarray
        Symmetric (m, m) array of non-negative solvent-solvent repulsion
        amplitudes A_mu_nu, in units of k_B T.
    n_grid : int
        Number of grid intervals N. Radial functions are sampled at r_i = i * dr
        for i = 1, ..., N - 1 and their transforms at q_j = j * pi / (N * dr).
    dr : float
        Grid spacing in units of the interaction range.

    Returns
    -------
    structure : numpy.ndarray
        Array of shape (2, m, m, n_grid - 1). structure[0, mu, nu] is the total
        correlation function h_mu_nu and structure[1, mu, nu] the direct
        correlation function c_mu_nu, both at r_i = i * dr, i = 1, ..., n_grid - 1.

    Raises
    ------
    ValueError
        If rho is not positive and finite, x is not a one dimensional array of
        non-negative finite mole fractions summing to one within 1e-12, A is not
        a finite, non-negative, exactly symmetric array of shape (m, m),
        n_grid is not an integer of at least 16, dr is not positive and finite,
        or (n_grid - 1) * dr is smaller than 4,
        or if the iteration fails to converge.
    '''
    return structure

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_grid(n_grid, dr):
    import numpy as np
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)) or int(n_grid) < 16:
        raise ValueError("n_grid must be an integer of at least 16")
    dr = float(dr)
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError("dr must be a positive finite number")
    if (int(n_grid) - 1) * dr < 4.0:
        raise ValueError("the grid must extend to at least r = 4")
    return int(n_grid), dr


def _check_solvent(rho, x, A):
    import numpy as np
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be a positive finite number")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 1 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("x must be a one dimensional array of non-negative mole fractions")
    if abs(float(np.sum(x)) - 1.0) > 1e-12:
        raise ValueError("the mole fractions must sum to one")
    A = np.asarray(A, dtype=float)
    if A.shape != (x.size, x.size) or not np.all(np.isfinite(A)) or np.any(A < 0.0):
        raise ValueError("A must be a finite non-negative array of shape (m, m)")
    if not np.array_equal(A, A.T):
        raise ValueError("A must be symmetric")
    return rho, x, A


def _dpd_potential(A, r):
    """beta * phi(r) = A (1 - r)^2 / 2 for r < 1 and zero beyond, for every entry of A."""
    import numpy as np
    return np.asarray(A, dtype=float)[..., None] * np.where(r < 1.0, 0.5 * (1.0 - r) ** 2, 0.0)


def _damped_iteration(step_map, gam, tol=1e-12, max_iter=20000):
    """Picard iteration gam -> step_map(gam) with a damping factor that halves whenever the error keeps rising."""
    import numpy as np
    alpha, prev, rises = 0.3, np.inf, 0
    for _ in range(max_iter):
        gnew = step_map(gam)
        err = float(np.max(np.abs(gnew - gam)))
        if not np.isfinite(err):
            raise ValueError("the HNC iteration did not converge")
        if err < tol:
            return gnew
        rises = rises + 1 if err > prev else 0
        if rises >= 3 and alpha > 0.02:
            alpha, rises = 0.5 * alpha, 0
        prev = err
        gam = alpha * gnew + (1.0 - alpha) * gam
    raise ValueError("the HNC iteration did not converge")


def _oracle_solvent_structure(rho: float, x: np.ndarray, A: np.ndarray, n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    m = x.size
    r = dr * np.arange(1, n_grid)
    bu = _dpd_potential(A, r)
    dens = rho * x
    eye = np.eye(m)

    def _update(gam):
        c = np.expm1(-bu + gam) - gam
        cq = np.moveaxis(_oracle_radial_fourier_transform(c, dr, False), -1, 0)
        hq = np.linalg.solve(eye[None] - cq * dens[None, None, :], cq)
        g = _oracle_radial_fourier_transform(np.moveaxis(hq - cq, 0, -1), dr, True)
        return 0.5 * (g + np.swapaxes(g, 0, 1))

    gam = _damped_iteration(_update, np.zeros((m, m, n_grid - 1)))
    c = np.expm1(-bu + gam) - gam
    return np.stack([gam + c, c])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: one-component water-like DPD solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Normal: three-component wet oil with unlike repulsions ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.10, 0.675, 0.225])
A = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Boundary: all repulsions zero, an ideal mixture ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.5, 0.5])
A = np.array([[0.0, 0.0], [0.0, 0.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Edge: a species at zero mole fraction, infinitely dilute inside the solver ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.8, 0.2, 0.0])
A = np.array([[25.0, 28.0, 40.0], [28.0, 25.0, 22.0], [40.0, 22.0, 25.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "gold_call": "_oracle_solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Invalid: mole fractions that do not sum to one ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.7, 0.2])
A = np.array([[25.0, 30.0], [30.0, 25.0]])
n_grid = 1024
dr = 0.02
def run_model():
    try:
        solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solvent_structure(rho, x.copy(), A.copy(), n_grid, dr)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
