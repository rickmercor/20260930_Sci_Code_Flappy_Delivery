"""
Build the one-electron core Hamiltonian and the scaled Ohno electron repulsion of a Pariser-Parr-Pople all-trans polyene.

In the Pariser-Parr-Pople (PPP) model every carbon atom carries one 2p pi orbital and one pi electron, the orbitals are
treated as orthogonal, and only Coulomb integrals between orbital densities survive (zero differential overlap). The
one-electron part has a valence-state energy on every site and a hopping (resonance) integral between bonded
neighbours. The electron repulsion between sites p and q follows the Ohno interpolation, which joins the on-site
Hubbard repulsion U to the bare Coulomb law at large separation. Each site also feels the attraction of the other
cores, each of charge +1, through the same repulsion function, so an isolated neutral chain stays neutral.

For an all-trans polyene with n carbon atoms the chain lies in a plane as a zigzag with all C-C-C angles equal to 120
degrees. Double (short) and single (long) bonds alternate and both chain ends carry a double bond. A dimensionless
factor scales every repulsion and core attraction together, which mimics a change in the effective screening of the
electron-electron interaction while keeping the chain neutral.

Returns
-------
numpy.ndarray of shape (2, n, n): [core Hamiltonian h, scaled Ohno repulsion gamma] in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_polyene_hamiltonian(n_sites: int, scale: float, params: "np.ndarray") -> "np.ndarray":
    '''One-electron core Hamiltonian and scaled Ohno repulsion matrix of a PPP all-trans polyene, in eV.

    Parameters
    ----------
    n_sites : int
        Number of carbon atoms n, an even integer of at least 2. Sites are numbered 0 to n-1 along the chain and the
        bond between sites k and k+1 is short (double) for even k and long (single) for odd k.
    scale : float
        Positive factor lambda that multiplies the Ohno repulsion and, through it, the core attraction.
    params : np.ndarray
        Shape (6,), [W, U, t_short, t_long, r_short, r_long]: site valence-state ionization energy W (eV), on-site
        repulsion U (eV), hopping integrals of the short and long bonds (eV, negative for bonding) and the short and long
        bond lengths (Angstrom). All C-C-C angles are 120 degrees in a planar zigzag chain.

    Returns
    -------
    hamiltonian : np.ndarray
        Shape (2, n, n). Element [1] is the repulsion matrix gamma_pq = lambda U / sqrt(1 + (U r_pq / 14.397)^2), with
        r_pq the distance in Angstrom between sites p and q (gamma_pp = lambda U). Element [0] is the core Hamiltonian:
        h_pq equals t_short or t_long for bonded neighbours and zero for other pairs, and
        h_pp = -W - sum over q != p of gamma_pq.

    Raises
    ------
    ValueError
        If n_sites is odd or smaller than 2, or scale is not positive.
    '''
    return hamiltonian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ppp_polyene_hamiltonian(n_sites: int, scale: float, params: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    n = int(n_sites)
    if n < 2 or n % 2 != 0:
        raise ValueError("n_sites must be an even integer of at least 2")
    if not scale > 0.0:
        raise ValueError("scale must be positive")
    w, u, t_short, t_long, r_short, r_long = [float(x) for x in np.asarray(params, dtype=float)]
    # planar zigzag: bond directions alternate between +30 and -30 degrees, which gives 120 degree angles
    pos = np.zeros((n, 2))
    for k in range(n - 1):
        length = r_short if k % 2 == 0 else r_long
        angle = np.deg2rad(30.0 if k % 2 == 0 else -30.0)
        pos[k + 1] = pos[k] + length * np.array([np.cos(angle), np.sin(angle)])
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    gamma = float(scale) * u / np.sqrt(1.0 + (u * dist / 14.397) ** 2)
    h = np.zeros((n, n))
    for k in range(n - 1):
        h[k, k + 1] = h[k + 1, k] = t_short if k % 2 == 0 else t_long
    h[np.diag_indices(n)] = -w - (gamma.sum(axis=1) - np.diag(gamma))
    return np.stack([h, gamma])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "params = np.array([11.16, 11.13, -2.60, -2.20, 1.35, 1.46])\n")
    return [
        # --- Normal: octatetraene at the standard interaction strength ---
        {
            "setup": base,
            "call": "ppp_polyene_hamiltonian(8, 1.0, params.copy())",
            "gold_call": "_oracle_ppp_polyene_hamiltonian(8, 1.0, params)",
            "tol": 1e-9,
        },
        # --- Boundary: ethylene, the shortest chain, with a strongly scaled interaction ---
        {
            "setup": base,
            "call": "ppp_polyene_hamiltonian(2, 1.7, params.copy())",
            "gold_call": "_oracle_ppp_polyene_hamiltonian(2, 1.7, params)",
            "tol": 1e-9,
        },
        # --- Edge: decapentaene with a weak interaction and unequal hopping and bond lengths ---
        {
            "setup": "import numpy as np\nparams = np.array([10.5, 9.0, -2.9, -1.9, 1.33, 1.49])\n",
            "call": "ppp_polyene_hamiltonian(10, 0.35, params.copy())",
            "gold_call": "_oracle_ppp_polyene_hamiltonian(10, 0.35, params)",
            "tol": 1e-9,
        },
        # --- Invalid: an odd number of sites must raise ValueError ---
        {
            "setup": base + "def run(fn):\n"
                            "    try:\n"
                            "        fn(5, 1.0, params)\n"
                            "        return 0\n"
                            "    except ValueError:\n"
                            "        return 1\n",
            "call": "run(ppp_polyene_hamiltonian)",
            "gold_call": "run(_oracle_ppp_polyene_hamiltonian)",
        },
    ]
