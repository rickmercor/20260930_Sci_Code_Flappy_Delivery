"""
Generalised excluded-volume functions of solute beads in a multicomponent DPD solvent: the single-bead functions whose convolutions build the solvent-mediated interaction between any two beads.

For two solute beads s and t at infinite dilution, the HNC solvent-mediated potential of mean force beta W_st(r) = -[h_st(r) - c_st(r)] (the bridge function is neglected) can be written as a sum of three-dimensional convolutions of single-bead functions, beta W_st(r) = -beta p sum over mu of (psi_mu_s * psi_mu_t)(r), where p is the pressure of the pure solvent and there is one function psi_mu_s for every solvent species mu and solute bead s.

Take beta p from the virial route, beta p = rho + (2 pi / 3) rho^2 sum over mu, nu of x_mu x_nu A_mu_nu times the integral from 0 to 1 of r^3 (1 - r) g_mu_nu(r) dr, with g = 1 + h from the HNC solvent structure and the integral evaluated as a rectangle-rule sum over the grid points.

The functions are fixed uniquely by two requirements. In Fourier space, psi_mu_s(q) = -sum over nu of M_mu_nu(q) x_nu h_nu_s(q), where M(q) is a real symmetric positive-definite matrix at every wavenumber and is the same for all solute beads; and the convolution identity above must hold for every pair of beads, including beads with arbitrary solvent amplitudes. With this sign the functions are positive near the centre of a repulsive bead. Every solvent species must be present, because M is not defined when a mole fraction vanishes.

Returns
-------
numpy.ndarray of shape (m, n_s, n_grid - 1): the generalised excluded-volume function psi_mu_s of every solvent species and solute bead on the radial grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def excluded_volume_functions(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray, n_grid: int, dr: float) -> np.ndarray:
    '''Generalised excluded-volume functions psi_mu_s(r) of infinitely dilute beads.

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
    a_solute : numpy.ndarray
        Array of shape (m, n_s) of non-negative solute-solvent amplitudes, one
        column per solute bead.
    n_grid : int
        Number of grid intervals N. Radial functions are sampled at r_i = i * dr
        for i = 1, ..., N - 1 and their transforms at q_j = j * pi / (N * dr).
    dr : float
        Grid spacing in units of the interaction range.

    Returns
    -------
    psi : numpy.ndarray
        Array of shape (m, n_s, n_grid - 1); psi[mu, s] is the function
        psi_mu_s at r_i = i * dr, i = 1, ..., n_grid - 1.

    Raises
    ------
    ValueError
        If rho is not positive and finite, x is not a one dimensional array of
        non-negative finite mole fractions summing to one within 1e-12, A is not
        a finite, non-negative, exactly symmetric array of shape (m, m),
        any mole fraction is zero, a_solute is not a finite non-negative array
        of shape (m, n_s) with n_s >= 1,
        n_grid is not an integer of at least 16, dr is not positive and finite,
        or (n_grid - 1) * dr is smaller than 4,
        if an iteration fails to converge, or if the solvent's partial
        structure-factor matrix X + rho X h(q) X (X the diagonal matrix of mole
        fractions) is not positive definite at some wavenumber.
    '''
    return psi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _virial_pressure(rho, x, A, hc, dr):
    """beta p from the virial route with the DPD force, rectangle rule on the grid."""
    import numpy as np
    r = dr * np.arange(1, hc.shape[-1] + 1)
    kern = np.where(r < 1.0, r ** 3 * (1.0 - r), 0.0) * dr
    return float(rho + (2.0 * np.pi / 3.0) * rho ** 2 * np.einsum('m,n,mn,mnr,r->', x, x, A, 1.0 + hc[0], kern))


def _oracle_excluded_volume_functions(rho: float, x: np.ndarray, A: np.ndarray, a_solute: np.ndarray,
                                      n_grid: int, dr: float) -> np.ndarray:
    import numpy as np
    rho, x, A = _check_solvent(rho, x, A)
    n_grid, dr = _check_grid(n_grid, dr)
    a = _check_solute(a_solute, x.size)
    if np.any(x <= 0.0):
        raise ValueError("every solvent mole fraction must be positive")
    h, c, hc = _solute_correlations(rho, x, A, a, n_grid, dr)
    bp = _virial_pressure(rho, x, A, hc, dr)
    hsq = _oracle_radial_fourier_transform(h, dr, False)                                  # (m, ns, nq)
    h00q = np.moveaxis(_oracle_radial_fourier_transform(hc[0], dr, False), -1, 0)        # (nq, m, m)
    S = np.diag(x)[None] + rho * x[None, :, None] * h00q * x[None, None, :]
    ev, V = np.linalg.eigh(S)
    if np.any(ev <= 0.0):
        raise ValueError("the partial structure factor matrix is not positive definite")
    Sm12 = np.einsum('qab,qb,qcb->qac', V, ev ** -0.5, V)
    psiq = -np.sqrt(rho / bp) * np.einsum('qmn,n,nsq->msq', Sm12, x, hsq)
    return _oracle_radial_fourier_transform(psiq, dr, True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: one repulsive bead in the water-like solvent ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([1.0])
A = np.array([[25.0]])
a = np.array([[38.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Normal: two unlike beads in the wet oil ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.10, 0.675, 0.225])
A = np.array([[25.0, 32.0, 26.0], [32.0, 25.0, 30.0], [26.0, 30.0, 25.0]])
a = np.array([[38.0, 24.0], [26.0, 34.0], [33.0, 22.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Boundary: binary solvent with a bead identical to its first species ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.75, 0.25])
A = np.array([[25.0, 30.0], [30.0, 25.0]])
a = np.array([[25.0, 30.0], [30.0, 20.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Edge: dense one-component solvent and a bead with no solvent interaction ---
        {
            "setup": """import numpy as np
rho = 5.0
x = np.array([1.0])
A = np.array([[15.0]])
a = np.array([[0.0, 40.0]])
n_grid = 1024
dr = 0.02
""",
            "call": "excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "gold_call": "_oracle_excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)",
            "tol": 1e-07,
        },
        # --- Invalid: a solvent species at zero mole fraction ---
        {
            "setup": """import numpy as np
rho = 3.0
x = np.array([0.8, 0.2, 0.0])
A = np.array([[25.0, 28.0, 40.0], [28.0, 25.0, 22.0], [40.0, 22.0, 25.0]])
a = np.array([[30.0], [20.0], [50.0]])
n_grid = 1024
dr = 0.02
def run_model():
    try:
        excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_excluded_volume_functions(rho, x.copy(), A.copy(), a.copy(), n_grid, dr)
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
