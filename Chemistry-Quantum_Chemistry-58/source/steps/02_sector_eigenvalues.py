"""
Lowest eigenvalues of the Pariser-Parr-Pople Hamiltonian in a sector of fixed numbers of spin-up and spin-down electrons.

The Pariser-Parr-Pople Hamiltonian of an open chain of N atoms with one pi orbital each is H = -sum over bonds k = 1..N-1 and spins s of t_k (c+(k,s) c(k+1,s) + h.c.) + U sum over atoms i of (n(i,up) - 1/2)(n(i,down) - 1/2) + sum over pairs i < j of V_ij (n_i - 1)(n_j - 1), where n(i,s) is the number operator of spin s on atom i and n_i = n(i,up) + n(i,down). Bond k joins atoms k and k+1 and carries t_k = t0 (1 + delta) when k is odd (a double bond) and t0 (1 - delta) when k is even, so the array of hopping integrals starts with a double bond; V is the interaction matrix of the previous step, or any symmetric matrix with a zero diagonal.

The Hamiltonian conserves the number of electrons of each spin, so it can be diagonalised separately in every sector (n_up, n_down). The natural basis of a sector is the set of Slater determinants with the given occupations, which for a ten-atom chain at half filling holds 63504 determinants, so a sparse or iterative eigensolver is needed where a dense one is impractical; eigenvalues are expected to be converged to about 1e-9 eV. Fermionic sign conventions do not affect the eigenvalues.

Returns
-------
numpy.ndarray of shape (n_states,): the n_states lowest eigenvalues of the sector in eV, ascending
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def sector_eigenvalues(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_up: int, n_down: int, n_states: int) -> np.ndarray:
    '''Lowest eigenvalues of the PPP chain in the (n_up, n_down) sector.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 2.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1; bond k carries t0 (1 + delta) for odd k and t0 (1 - delta) for even k.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.
    n_up : int
        Number of spin-up electrons, between 0 and n_sites.
    n_down : int
        Number of spin-down electrons, between 0 and n_sites.
    n_states : int
        Number of eigenvalues wanted, between 1 and the sector dimension.

    Returns
    -------
    energies : numpy.ndarray
        Shape (n_states,), the lowest eigenvalues in ascending order, in eV.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 2, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, n_up or n_down is outside
        [0, n_sites], or n_states is outside [1, sector dimension].
    '''
    return energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import types

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _strings(n_sites, n_occ):
    """All occupation bit-strings of n_sites sites holding n_occ electrons of one spin, ascending."""
    return [sum(1 << i for i in c) for c in itertools.combinations(range(n_sites), n_occ)]


def _hop_matrix(strings, n_sites, t):
    """One-spin hopping matrix -sum_k t_k (c+_k c_{k+1} + h.c.) on the given bit-strings (creation
    operators ordered by ascending site index)."""
    idx = {s: k for k, s in enumerate(strings)}
    rows, cols, vals = [], [], []
    for k, s in enumerate(strings):
        for bond in range(n_sites - 1):
            for p, q in ((bond, bond + 1), (bond + 1, bond)):
                if (s >> q) & 1 and not (s >> p) & 1:
                    s2 = s ^ (1 << q) ^ (1 << p)
                    lo, hi = min(p, q), max(p, q)
                    between = bin(s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))).count("1")
                    rows.append(idx[s2]); cols.append(k); vals.append(-t[bond] * (-1) ** between)
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(strings), len(strings))).tocsr()


def _sector(n_sites, n_up, n_down):
    """Determinant basis of one (n_up, n_down) sector: spin strings, index maps, occupations, as a namespace."""
    su = _strings(n_sites, n_up)
    sd = su if n_down == n_up else _strings(n_sites, n_down)
    occu = np.array([[(s >> i) & 1 for i in range(n_sites)] for s in su], dtype=float)
    occd = occu if sd is su else np.array([[(s >> i) & 1 for i in range(n_sites)] for s in sd], dtype=float)
    return types.SimpleNamespace(n=n_sites, n_up=n_up, n_down=n_down, su=su, sd=sd,
                                 iu={s: k for k, s in enumerate(su)}, id={s: k for k, s in enumerate(sd)},
                                 nu=len(su), nd=len(sd), dim=len(su) * len(sd), occu=occu, occd=occd)


def _hoppings(n_sites, t0, delta):
    return np.array([t0 * (1.0 + delta) if k % 2 == 0 else t0 * (1.0 - delta) for k in range(n_sites - 1)])


def _hamiltonian(sec, t0, delta, V, U):
    n = sec.n
    t = _hoppings(n, t0, delta)
    Hu = _hop_matrix(sec.su, n, t); Hd = _hop_matrix(sec.sd, n, t)
    H = sp.kron(Hu, sp.identity(sec.nd, format="csr"), format="csr") + sp.kron(sp.identity(sec.nu, format="csr"), Hd, format="csr")
    X = sec.occu - 0.5; Y = sec.occd - 0.5
    Vm = np.asarray(V, dtype=float)
    onsite = U * (X @ Y.T)
    xx = np.einsum("ai,ij,aj->a", X, Vm, X); yy = np.einsum("bi,ij,bj->b", Y, Vm, Y); xy = X @ Vm @ Y.T
    diag = onsite + 0.5 * (xx[:, None] + yy[None, :]) + xy
    H = H + sp.diags(diag.ravel(), format="csr")
    return H


def _lowest(H, k, dim):
    """Lowest k eigenpairs, ascending; dense below 1500, ARPACK above (fixed start vector)."""
    k = int(min(k, dim))
    if dim <= 1500:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    kk = min(k, dim - 2)
    w, v = spla.eigsh(H, k=kk, which="SA", v0=np.ones(dim), tol=1e-13, ncv=min(dim, max(3 * kk + 20, 60)))
    o = np.argsort(w)
    return w[o], v[:, o]


def _validate_model(n_sites, t0, delta, V, U):
    _check_even_chain(n_sites)
    Vm = np.asarray(V, dtype=float)
    if Vm.ndim != 2 or Vm.shape != (n_sites, n_sites):
        raise ValueError("V must be an (n_sites, n_sites) array")
    if not np.allclose(Vm, Vm.T, rtol=0.0, atol=1e-12):
        raise ValueError("V must be symmetric")
    if not (np.isfinite(t0) and t0 > 0.0):
        raise ValueError("t0 must be positive")
    if not (np.isfinite(delta) and -1.0 < delta < 1.0):
        raise ValueError("delta must lie in (-1, 1)")
    if not (np.isfinite(U) and U > 0.0):
        raise ValueError("U must be positive")
    return Vm


def _oracle_sector_eigenvalues(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_up: int, n_down: int, n_states: int) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    for name, x in (("n_up", n_up), ("n_down", n_down), ("n_states", n_states)):
        if not isinstance(x, (int, np.integer)) or isinstance(x, bool):
            raise ValueError(name + " must be an integer")
    if not (0 <= n_up <= n_sites and 0 <= n_down <= n_sites):
        raise ValueError("n_up and n_down must lie between 0 and n_sites")
    sec = _sector(n_sites, n_up, n_down)
    if not (1 <= n_states <= sec.dim):
        raise ValueError("n_states must lie between 1 and the sector dimension")
    H = _hamiltonian(sec, t0, delta, Vm, U)
    w, _ = _lowest(H, n_states, sec.dim)
    return np.array(w[:n_states], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the half-filled six-atom benchmark chain, eight lowest states ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 3, 3, 8)",
            "gold_call": "_oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 3, 3, 8)",
            "tol": 1e-08,
        },
        # --- Normal: the Sz = 1 sector of the eight-atom benchmark chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164, 0.838156692309, 0.725480211265],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900, 0.980582578949, 0.838156692309],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900],
              [1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [0.838156692309, 0.980582578949, 1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [0.725480211265, 0.838156692309, 0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 8
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 5, 3, 4)",
            "gold_call": "_oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 5, 3, 4)",
            "tol": 1e-08,
        },
        # --- Boundary: every state of the half-filled four-atom chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515],
              [4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136],
              [2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697],
              [1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000]])
n_sites = 4
t0 = 2.4
delta = 0.08333333333333333
U = 8.0
""",
            "call": "sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 2, 2, 36)",
            "gold_call": "_oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 2, 2, 36)",
            "tol": 1e-08,
        },
        # --- Normal: the strongly correlated six-atom chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994, 1.278461106161, 1.028690941455],
              [4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928, 1.640282944134, 1.278461106161],
              [2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107, 2.514470076928, 1.695139627994],
              [1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000, 4.037422412364, 2.514470076928],
              [1.278461106161, 1.640282944134, 2.514470076928, 4.037422412364, 0.000000000000, 4.325076507107],
              [1.028690941455, 1.278461106161, 1.695139627994, 2.514470076928, 4.325076507107, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.1
U = 12.0
""",
            "call": "sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 3, 3, 6)",
            "gold_call": "_oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 3, 3, 6)",
            "tol": 1e-08,
        },
        # --- Edge: a uniform chain without bond alternation ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777, 1.224092567900, 0.999724120164],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042, 1.529921801556, 1.224092567900],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [1.224092567900, 1.529921801556, 2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [0.999724120164, 1.224092567900, 1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.0
U = 4.0
""",
            "call": "sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 3, 3, 5)",
            "gold_call": "_oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 3, 3, 5)",
            "tol": 1e-08,
        },
        # --- Edge: the fully polarised sector of the four-atom chain, a single diagonal state ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 4
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 4, 0, 1)",
            "gold_call": "_oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 4, 0, 1)",
            "tol": 1e-08,
        },
        # --- Invalid: more states than the sector holds ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957, 2.163107408042, 1.574142730777],
              [3.028724843957, 0.000000000000, 2.924747953395, 2.163107408042],
              [2.163107408042, 2.924747953395, 0.000000000000, 3.028724843957],
              [1.574142730777, 2.163107408042, 3.028724843957, 0.000000000000]])
n_sites = 4
t0 = 2.5
delta = 0.1
U = 4.0
def run_model():
    try:
        sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 4, 0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_sector_eigenvalues(n_sites, t0, delta, V.copy(), U, 4, 0, 2)
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
