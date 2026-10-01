"""
The full-size reference system of the embedding is a generalized Kohn-Sham (gKS) determinant: a single closed-shell Slater determinant that is the ground state of    h_gKS = h + v_Hx[gamma] + sum_i v_c[i] n_i,  where h is the one-electron matrix of the ring, v_Hx[gamma] is the Hartree-Fock (100 percent exact exchange) potential built from the reference one-electron reduced density matrix gamma itself, and v_c is a local (site diagonal) correlation potential that is an external input of this step (it is optimized in a later step).

The full-size reference system of the embedding is a generalized Kohn-Sham (gKS) determinant: a single closed-shell Slater determinant that is the ground state of

  h_gKS = h + v_Hx[gamma] + sum_i v_c[i] n_i,

where h is the one-electron matrix of the ring, v_Hx[gamma] is the Hartree-Fock (100 percent exact exchange) potential built from the reference one-electron reduced density matrix gamma itself, and v_c is a local (site diagonal) correlation potential that is an external input of this step (it is optimized in a later step). For a purely on-site repulsion U n_{i up} n_{i down} the Hartree-Fock potential is local in the site basis: the Hartree term of site i is U times the opposite-spin occupation and the exchange term vanishes between opposite spins, so that in the spin-restricted closed-shell case

  v_Hx[gamma]_{ij} = delta_{ij} U gamma_{ii},

with gamma_{ii} the occupation per spin of site i (the spin-summed density is n_i = 2 gamma_{ii}). Because v_Hx depends on gamma, the reference problem is solved self-consistently, exactly like a restricted Hartree-Fock calculation with a fixed additional external potential v_c: build the Fock matrix F = h + diag(v_c) + diag(U gamma_ii), diagonalize it, occupy the lowest n_elec/2 orbitals with two electrons each, rebuild gamma = C_occ C_occ^T, and repeat until the density matrix is stationary. Plain iteration from gamma = 0 does not converge at strong coupling, so the Fock matrix is extrapolated with Pulay's direct inversion in the iterative subspace (DIIS) using the commutator F gamma - gamma F as the error vector.

The closed-shell occupation is only defined when the highest occupied and the lowest unoccupied gKS levels are separated; a degenerate Fermi level is treated as invalid input.

Returns
-------
np.ndarray of float with shape (L, L): the converged per-spin gKS density matrix (idempotent, trace n_elec / 2) for the given h, U and v_c.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gks_density_matrix(h, U, v_c, n_elec):
    '''Self-consistent closed-shell gKS one-electron reduced density matrix.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion; the Hartree-Fock potential is diag(U gamma_ii).
    v_c : array_like of float, shape (L,)
        Local correlation potential added to the diagonal of the Fock matrix.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2; the lowest
        n_elec / 2 gKS orbitals are doubly occupied.

    Returns
    -------
    gamma : np.ndarray of float, shape (L, L)
        Converged per-spin density matrix gamma = C_occ C_occ^T (idempotent,
        trace n_elec / 2), converged so that the commutator F gamma - gamma F
        is below 1e-10 in absolute value. Raises ValueError for a non-square
        or non-symmetric h, a v_c of the wrong length, an odd or out-of-range
        n_elec, or a converged Fock matrix whose highest occupied and lowest
        unoccupied levels are degenerate (gap below 1e-8).
    '''
    return np.zeros_like(np.asarray(h, dtype=float))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_gks_density_matrix(h, U, v_c, n_elec):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("h must be a square matrix")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    L = h.shape[0]
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    v_c = np.asarray(v_c, dtype=float)
    if v_c.ndim != 1 or v_c.shape[0] != L or not np.all(np.isfinite(v_c)):
        raise ValueError("v_c must be a finite vector with L entries")
    if isinstance(n_elec, bool) or not isinstance(n_elec, (int, np.integer)):
        raise ValueError("n_elec must be an integer")
    if n_elec % 2 != 0 or n_elec < 2 or n_elec > 2 * L - 2:
        raise ValueError("n_elec must be even and satisfy 2 <= n_elec <= 2L - 2")
    n_occ = int(n_elec) // 2
    gamma = np.zeros((L, L))
    focks, errors = [], []
    ndiis = 8
    for _ in range(5000):
        fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
        err = fock @ gamma - gamma @ fock
        focks.append(fock)
        errors.append(err.ravel())
        if len(focks) > ndiis:
            focks.pop(0)
            errors.pop(0)
        fock_use = fock
        m = len(focks)
        if m > 1:
            bmat = -np.ones((m + 1, m + 1))
            bmat[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    bmat[a, b] = errors[a] @ errors[b]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coef = np.linalg.solve(bmat, rhs)[:m]
                if np.all(np.isfinite(coef)):
                    fock_use = sum(c * f for c, f in zip(coef, focks))
            except np.linalg.LinAlgError:
                pass
        _, cmat = np.linalg.eigh(fock_use)
        g_new = cmat[:, :n_occ] @ cmat[:, :n_occ].T
        if np.max(np.abs(g_new - gamma)) < 1e-12 and np.max(np.abs(err)) < 1e-10:
            gamma = g_new
            break
        gamma = g_new
    else:
        raise RuntimeError("gKS self-consistency did not converge")
    fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
    levels = np.linalg.eigvalsh(fock)
    if levels[n_occ] - levels[n_occ - 1] < 1e-8:
        raise ValueError("degenerate Fermi level: closed-shell occupation undefined")
    return gamma

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    ring = """import numpy as np
def _ring(L, t, v):
    h = np.diag(np.asarray(v, dtype=float))
    for i in range(L):
        h[i, (i + 1) % L] -= t
        h[(i + 1) % L, i] -= t
    return h
h6 = _ring(6, 1.0, [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0])
"""
    return [
        # --- Normal: the task ring at U = 7 with a vanishing correlation potential ---
        {
            "setup": ring,
            "call": "np.round(gks_density_matrix(h6, 7.0, np.zeros(6), 6), 6)",
            "gold_call": "np.round(_oracle_gks_density_matrix(h6, 7.0, np.zeros(6), 6), 6)",
        },
        # --- Normal: strong coupling with a non-trivial correlation potential ---
        {
            "setup": ring + "vc = np.array([0.75, -1.27, 1.31, -1.56, 1.55, -0.78])\n",
            "call": "np.round(gks_density_matrix(h6, 10.0, vc, 6), 6)",
            "gold_call": "np.round(_oracle_gks_density_matrix(h6, 10.0, vc, 6), 6)",
        },
        # --- Boundary: U = 0, the reference is the plain one-electron problem ---
        {
            "setup": ring + "vc = np.array([0.2, -0.1, 0.3, 0.0, -0.2, 0.1])\n",
            "call": "np.round(gks_density_matrix(h6, 0.0, vc, 6), 6)",
            "gold_call": "np.round(_oracle_gks_density_matrix(h6, 0.0, vc, 6), 6)",
        },
        # --- Edge: two electrons only (a single doubly occupied orbital) ---
        {
            "setup": ring,
            "call": "np.round(gks_density_matrix(h6, 4.0, np.zeros(6), 2), 6)",
            "gold_call": "np.round(_oracle_gks_density_matrix(h6, 4.0, np.zeros(6), 2), 6)",
        },
        # --- Invalid: odd electron number ---
        {
            "setup": ring + """
def run_model():
    try:
        gks_density_matrix(h6, 7.0, np.zeros(6), 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gks_density_matrix(h6, 7.0, np.zeros(6), 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: uniform four-site ring at half filling has a degenerate Fermi level ---
        {
            "setup": ring + """h4 = _ring(4, 1.0, [0.0, 0.0, 0.0, 0.0])
def run_model():
    try:
        gks_density_matrix(h4, 0.0, np.zeros(4), 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gks_density_matrix(h4, 0.0, np.zeros(4), 4)
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
