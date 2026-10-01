"""
Energies and covalent weights of the lowest members of the covalent triplet family of a chain, selected by their alternancy eigenvalue.

At half filling the eigenstates of a bipartite chain fall into two classes under the alternancy (particle-hole) transformation that replaces every creation operator c+(i,s) by (-1)^i c(i,s) and maps the empty chain onto the completely filled one; the transformation commutes with the Hamiltonian, so every eigenstate has eigenvalue +1 or -1. Within a spin manifold the class that contains the lowest state is called covalent, because its members are dominated by configurations with one electron on every atom, and the other class ionic. A triplet is covalent when its alternancy eigenvalue equals that of the lowest triplet of the same chain. The covalent triplets of a chain of N_d ethylene units form a family whose lowest N_d members play the role of one-particle states of a triplet excitation in the real-space analysis of the dark state, and ionic triplets can lie among them in energy.

This step works in the Sz = 0 sector (n_up = n_down = N/2), where every triplet has a component. The covalent weight of a state is the total weight of the Slater determinants in which every atom is singly occupied.

Returns
-------
numpy.ndarray of shape (n_members, 2): for the n_members lowest covalent triplets in ascending energy, the energy in eV and the covalent weight
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


def covalent_triplet_family(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_members: int) -> np.ndarray:
    '''Lowest covalent triplets of the chain in the Sz = 0 sector.

    Parameters
    ----------
    n_sites : int
        Number of atoms N, an even integer >= 2.
    t0 : float
        Mean hopping integral in eV, positive.
    delta : float
        Bond alternation, |delta| < 1, with the double bond first.
    V : numpy.ndarray
        Symmetric (n_sites, n_sites) interaction matrix in eV with a zero diagonal.
    U : float
        Hubbard parameter in eV, positive.
    n_members : int
        Number of covalent triplets wanted, a positive integer.

    Returns
    -------
    family : numpy.ndarray
        Shape (n_members, 2). Row j holds the energy in eV of the (j+1)-th lowest triplet
        whose alternancy eigenvalue equals that of the lowest triplet, and the covalent
        weight of that state (total weight of the determinants with every atom singly
        occupied). Rows are in ascending energy.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 2, V is not a symmetric (n_sites, n_sites)
        array, t0 or U is not positive, |delta| >= 1, n_members is not a positive
        integer, or the sector holds fewer than n_members covalent triplets.
    '''
    return family

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


def _site_maps(sec_from, sec_to, spin, site, create):
    """Index maps for c+_site (create=True) or c_site (create=False) acting on one spin string set.
    Returns (idx_from, idx_to, sign) for every string of sec_from's spin set that the operator does not kill."""
    strs = sec_from.su if spin == 0 else sec_from.sd
    lookup = sec_to.iu if spin == 0 else sec_to.id
    f, t_, s = [], [], []
    for k, st in enumerate(strs):
        occ = (st >> site) & 1
        if create and occ or (not create and not occ):
            continue
        sgn = (-1) ** bin(st & ((1 << site) - 1)).count("1")
        st2 = st | (1 << site) if create else st ^ (1 << site)
        f.append(k); t_.append(lookup[st2]); s.append(sgn)
    return np.array(f, dtype=int), np.array(t_, dtype=int), np.array(s, dtype=float)


def _s_plus(psi, sec):
    """S+ psi, mapping the (n_up, n_down) sector to (n_up+1, n_down-1). Returns (vector, target sector)."""
    tgt = _sector(sec.n, sec.n_up + 1, sec.n_down - 1)
    P = psi.reshape(sec.nu, sec.nd)
    out = np.zeros((tgt.nu, tgt.nd))
    cross = (-1) ** sec.n_up  # c+_{i up} written left of all up operators passes none; c_{i dn} passes n_up up-operators
    for i in range(sec.n):
        fu, tu, su = _site_maps(sec, tgt, 0, i, True)
        fd, td, sd = _site_maps(sec, tgt, 1, i, False)
        if fu.size == 0 or fd.size == 0:
            continue
        out[np.ix_(tu, td)] += cross * (su[:, None] * sd[None, :]) * P[np.ix_(fu, fd)]
    return out.ravel(), tgt


def _s2_expectation(psi, sec):
    """<S^2> for a normalised vector of a sector with S_z = (n_up - n_down)/2: S^2 = S- S+ + Sz^2 + Sz."""
    sz = 0.5 * (sec.n_up - sec.n_down)
    if sec.n_down == 0:
        return sz * sz + sz
    v, _ = _s_plus(psi, sec)
    return float(v @ v) + sz * sz + sz


def _reflect_strings(strings, n_sites):
    """Spatial inversion i -> n-1-i on one spin string set: (permutation, sign)."""
    lookup = {s: k for k, s in enumerate(strings)}
    perm = np.zeros(len(strings), dtype=int); sgn = np.zeros(len(strings))
    for k, s in enumerate(strings):
        occ = [i for i in range(n_sites) if (s >> i) & 1]
        m = len(occ)
        perm[k] = lookup[sum(1 << (n_sites - 1 - i) for i in occ)]
        sgn[k] = (-1) ** (m * (m - 1) // 2)
    return perm, sgn


def _inversion_expectation(psi, sec):
    pu, su = _reflect_strings(sec.su, sec.n)
    pd, sd = _reflect_strings(sec.sd, sec.n) if sec.sd is not sec.su else (pu, su)
    P = psi.reshape(sec.nu, sec.nd)
    Q = np.zeros_like(P)
    Q[np.ix_(pu, pd)] = (su[:, None] * sd[None, :]) * P
    return float(P.ravel() @ Q.ravel())


def _conjugate_strings(strings, n_sites):
    """Alternancy map on one spin string: apply prod_{i in s, ascending} (-1)^i c_i to the completely filled
    string (creation operators ascending). Returns (permutation to the complementary string, sign)."""
    lookup = {s: k for k, s in enumerate(strings)}
    full = (1 << n_sites) - 1
    perm = np.zeros(len(strings), dtype=int); sgn = np.zeros(len(strings))
    for k, s in enumerate(strings):
        occ = [i for i in range(n_sites) if (s >> i) & 1]
        cur = full; sign = 1
        for i in reversed(occ):  # rightmost operator acts first
            sign *= (-1) ** bin(cur & ((1 << i) - 1)).count("1") * (-1) ** i
            cur ^= (1 << i)
        perm[k] = lookup[cur]; sgn[k] = sign
    return perm, sgn


def _alternancy_expectation(psi, sec):
    pu, su = _conjugate_strings(sec.su, sec.n)
    pd, sd = _conjugate_strings(sec.sd, sec.n) if sec.sd is not sec.su else (pu, su)
    P = psi.reshape(sec.nu, sec.nd)
    Q = np.zeros_like(P)
    Q[np.ix_(pu, pd)] = (su[:, None] * sd[None, :]) * P
    return float(P.ravel() @ Q.ravel())


def _labelled_spectrum(n_sites, t0, delta, V, U, k):
    """Lowest k Sz=0 eigenpairs with labels (E, S(S+1), inversion, alternancy)."""
    sec = _sector(n_sites, n_sites // 2, n_sites // 2)
    H = _hamiltonian(sec, t0, delta, V, U)
    w, v = _lowest(H, k, sec.dim)
    lab = np.array([[w[i], _s2_expectation(v[:, i], sec), _inversion_expectation(v[:, i], sec), _alternancy_expectation(v[:, i], sec)] for i in range(len(w))])
    return sec, w, v, lab


def _covalent_triplets(n_sites, t0, delta, V, U, n_members):
    """The n_members lowest Sz=0 triplet eigenstates sharing the alternancy eigenvalue of the lowest triplet."""
    sec = _sector(n_sites, n_sites // 2, n_sites // 2)
    k = min(sec.dim, max(4 * n_members + 8, 16))
    while True:
        sec, w, v, lab = _labelled_spectrum(n_sites, t0, delta, V, U, k)
        trip = [i for i in range(len(w)) if abs(lab[i, 1] - 2.0) < 1e-6]
        if trip:
            ref = np.sign(lab[trip[0], 3])
            fam = [i for i in trip if np.sign(lab[i, 3]) == ref]
            if len(fam) >= n_members:
                return sec, w, v, lab, fam[:n_members]
        if k >= sec.dim:
            raise ValueError("the sector does not contain the requested number of covalent triplets")
        k = min(sec.dim, 2 * k)


def _oracle_covalent_triplet_family(n_sites: int, t0: float, delta: float, V: np.ndarray, U: float, n_members: int) -> np.ndarray:
    Vm = _validate_model(n_sites, t0, delta, V, U)
    if not isinstance(n_members, (int, np.integer)) or isinstance(n_members, bool) or n_members < 1:
        raise ValueError("n_members must be a positive integer")
    sec, w, v, lab, fam = _covalent_triplets(n_sites, t0, delta, Vm, U, n_members)
    cov = (np.abs(sec.occu[:, None, :] - sec.occd[None, :, :]) == 1).all(-1)  # every site singly occupied
    out = np.zeros((n_members, 2))
    for r, i in enumerate(fam):
        P = v[:, i].reshape(sec.nu, sec.nd)
        out[r, 0] = w[i]
        out[r, 1] = float((P[cov] ** 2).sum())
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four covalent triplets of the eight-atom benchmark chain, where an ionic triplet lies among them ---
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
            "call": "covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 4)",
            "gold_call": "_oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 4)",
            "tol": 1e-07,
        },
        # --- Normal: three covalent triplets of the six-atom benchmark chain ---
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
            "call": "covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 3)",
            "gold_call": "_oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 3)",
            "tol": 1e-07,
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
            "call": "covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 3)",
            "gold_call": "_oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 3)",
            "tol": 1e-07,
        },
        # --- Normal: the eight-atom straight chain of the earlier parametrisation ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237, 0.852089448374, 0.731463994105],
              [4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237, 0.852089448374],
              [2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406, 1.019964859237],
              [1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515, 1.269166956406],
              [1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136, 1.675898821515],
              [1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697, 2.447611173136],
              [0.852089448374, 1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000, 4.325422955697],
              [0.731463994105, 0.852089448374, 1.019964859237, 1.269166956406, 1.675898821515, 2.447611173136, 4.325422955697, 0.000000000000]])
n_sites = 8
t0 = 2.4
delta = 0.08333333333333333
U = 8.0
""",
            "call": "covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 4)",
            "gold_call": "_oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 4)",
            "tol": 1e-07,
        },
        # --- Edge: the weakly screened six-atom chain ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.745220485711, 3.313404468934, 2.806363405152, 2.377964623387, 2.041983858636],
              [3.745220485711, 0.000000000000, 3.706654150805, 3.313404468934, 2.758192727283, 2.377964623387],
              [3.313404468934, 3.706654150805, 0.000000000000, 3.745220485711, 3.313404468934, 2.806363405152],
              [2.806363405152, 3.313404468934, 3.745220485711, 0.000000000000, 3.706654150805, 3.313404468934],
              [2.377964623387, 2.758192727283, 3.313404468934, 3.706654150805, 0.000000000000, 3.745220485711],
              [2.041983858636, 2.377964623387, 2.806363405152, 3.313404468934, 3.745220485711, 0.000000000000]])
n_sites = 6
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 3)",
            "gold_call": "_oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 3)",
            "tol": 1e-07,
        },
        # --- Boundary: the single triplet of an ethylene unit ---
        {
            "setup": """import numpy as np
V = np.array([[0.000000000000, 3.028724843957],
              [3.028724843957, 0.000000000000]])
n_sites = 2
t0 = 2.5
delta = 0.1
U = 4.0
""",
            "call": "covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 1)",
            "gold_call": "_oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 1)",
            "tol": 1e-07,
        },
        # --- Invalid: no members requested ---
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
        covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_covalent_triplet_family(n_sites, t0, delta, V.copy(), U, 0)
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
