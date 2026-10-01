"""
One-electron and electron-repulsion matrices of a linear Pariser-Parr-Pople pi chain.

The chromophore is described by the Pariser-Parr-Pople (PPP) model: one 2p_z orbital per conjugated atom, zero differential overlap, and a Hamiltonian written in the site basis. The atoms lie on a straight line along the x axis, the first atom at the origin and each following atom displaced by the given bond length, all in angstrom.

The one-electron matrix h holds the site energy alpha of each atom on its diagonal and the resonance integral beta of each bond on the two positions that connect bonded neighbours; atoms that are not bonded have no resonance integral. Energies are in eV.

The electron repulsion between the pi electrons on sites mu and nu is the Ohno interpolation

  gamma_mu,nu = 14.397 / sqrt(r_mu,nu^2 + a_mu,nu^2),  a_mu,nu = 2 * 14.397 / (U_mu + U_nu)

with r in angstrom, gamma in eV and U the one-centre repulsion (Hubbard U) of each atom. At r = 0 the formula returns U itself, so the diagonal of gamma is the list of one-centre repulsions, and at large distance it tends to the bare Coulomb repulsion 14.397 / r.

Returns
-------
numpy.ndarray of shape (2, n_sites, n_sites): the one-electron matrix h and the Ohno repulsion matrix gamma, both in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_monomer_matrices(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                         hubbard_u: "np.ndarray") -> "np.ndarray":
    '''PPP one-electron matrix and Ohno repulsion matrix of a linear chain.

    Parameters
    ----------
    bonds : numpy.ndarray
        One-dimensional array of the n_sites - 1 bond lengths along the chain, in
        angstrom; all must be positive.
    betas : numpy.ndarray
        Resonance integral of each bond, in eV, same length as bonds.
    alphas : numpy.ndarray
        Site energy of each atom, in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsion of each atom, in eV, length n_sites; all must be positive.

    Returns
    -------
    matrices : numpy.ndarray
        Real array of shape (2, n_sites, n_sites): index 0 is the one-electron
        matrix h and index 1 is the Ohno repulsion matrix gamma.

    Raises
    ------
    ValueError
        If bonds is empty or has a non-positive entry, if betas does not have the
        length of bonds, if alphas or hubbard_u does not have length n_sites, or if
        any one-centre repulsion is not positive.
    '''
    return matrices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _chain_coordinates(bonds):
    import numpy as np
    b = np.asarray(bonds, dtype=float)
    xs = np.concatenate([[0.0], np.cumsum(b)])
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def _ohno_kernel(coords_a, u_a, coords_b, u_b):
    import numpy as np
    d = np.linalg.norm(np.asarray(coords_a)[:, None, :] - np.asarray(coords_b)[None, :, :], axis=-1)
    a = 2.0 * 14.397 / (np.asarray(u_a, dtype=float)[:, None] + np.asarray(u_b, dtype=float)[None, :])
    return 14.397 / np.sqrt(d * d + a * a)


def _oracle_ppp_monomer_matrices(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                 hubbard_u: "np.ndarray") -> "np.ndarray":
    import numpy as np
    b = np.atleast_1d(np.asarray(bonds, dtype=float))
    be = np.atleast_1d(np.asarray(betas, dtype=float))
    al = np.atleast_1d(np.asarray(alphas, dtype=float))
    u = np.atleast_1d(np.asarray(hubbard_u, dtype=float))
    if b.ndim != 1 or b.size == 0 or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("bonds must be a non-empty array of positive lengths")
    n = b.size + 1
    if be.shape != b.shape:
        raise ValueError("betas must have one entry per bond")
    if al.shape != (n,) or u.shape != (n,):
        raise ValueError("alphas and hubbard_u must have one entry per site")
    if not np.all(np.isfinite(u)) or np.any(u <= 0.0):
        raise ValueError("hubbard_u must be positive")
    h = np.diag(al)
    for k in range(b.size):
        h[k, k + 1] = be[k]
        h[k + 1, k] = be[k]
    X = _chain_coordinates(b)
    gamma = _ohno_kernel(X, u, X, u)
    return np.stack([h, gamma])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the enamine-like C=C-N chromophore of the benchmark ---
        {
            "setup": """import numpy as np
""",
            "call": "ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))",
            "gold_call": "_oracle_ppp_monomer_matrices(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]))",
            "tol": 1e-12,
        },
        # --- Normal: a four-atom butadiene chain with alternating bonds ---
        {
            "setup": """import numpy as np
""",
            "call": "ppp_monomer_matrices(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13))",
            "gold_call": "_oracle_ppp_monomer_matrices(np.array([1.35, 1.46, 1.35]), np.array([-2.6, -2.2, -2.6]), np.zeros(4), np.full(4, 11.13))",
            "tol": 1e-12,
        },
        # --- Boundary: the smallest chain, a single two-atom bond ---
        {
            "setup": """import numpy as np
""",
            "call": "ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))",
            "gold_call": "_oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]))",
            "tol": 1e-12,
        },
        # --- Edge: very unequal one-centre repulsions and a long bond ---
        {
            "setup": """import numpy as np
""",
            "call": "ppp_monomer_matrices(np.array([2.5, 0.9]), np.array([-1.1, -3.3]), np.array([1.5, 0.0, -4.0]), np.array([6.0, 11.13, 19.0]))",
            "gold_call": "_oracle_ppp_monomer_matrices(np.array([2.5, 0.9]), np.array([-1.1, -3.3]), np.array([1.5, 0.0, -4.0]), np.array([6.0, 11.13, 19.0]))",
            "tol": 1e-12,
        },
        # --- Invalid: a one-centre repulsion that is not positive ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 0.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_ppp_monomer_matrices(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 0.0]))
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
