"""
Return the ground-state energy of the effective Hamiltonian H = sum_tu t_eff[t, u] a+_t a_u + (1/2) sum_tuvw veff[t, u, v, w] a+_t a+_v a_w a_u, where the creation and annihilation operators run over both spins of every active spatial orbital, t_eff is the (real, symmetric) one-body matrix and veff the two-body tensor in chemists' notation (tu|vw), with n_elec electrons (n_elec even) distributed over the active orbitals: the lowest eigenvalue of the full configuration-interaction Hamiltonian in the sector with equal numbers of spin-up and spin-down electrons. Return 0.0 for n_elec = 0.

Full configuration interaction in a small active space is the exact high-level solver the source pairs with its downfolded Hamiltonians; it needs nothing but the one- and two-body integrals of the effective Hamiltonian, and its result is the energy the active-space part contributes to the total.

Returns
-------
float, the active-space ground-state energy of the effective Hamiltonian (native Python float).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def active_space_fci(t_eff: "np.ndarray", veff: "np.ndarray", n_elec: int) -> float:
    """Return the ground-state energy of the effective Hamiltonian H = sum_tu t_eff[t, u] a+_t a_u + (1/2) sum_tuvw veff[t, u, v, w] a+_t a+_v a_w a_u, where the creation and annihilation operators run over both spins of every active spatial orbital, t_eff is the (real, symmetric) one-body matrix and veff the two-body tensor in chemists' notation (tu|vw), with n_elec electrons (n_elec even) distributed over the active orbitals: the lowest eigenvalue of the full configuration-interaction Hamiltonian in the sector with equal numbers of spin-up and spin-down electrons. Return 0.0 for n_elec = 0.

    Parameters
    ----------
    t_eff : numpy.ndarray
        Real symmetric (nA, nA) one-body matrix over the active orbitals.
    veff : numpy.ndarray
        Two-body tensor (nA, nA, nA, nA) in chemists' notation over the same orbitals.
    n_elec : int
        Even number of electrons, 0 <= n_elec <= 2 nA.

    Returns
    -------
    e_cas : float
        The lowest eigenvalue of H in the S_z = 0 sector (0.0 for n_elec = 0), same energy units as the inputs.

    Raises
    ------
    ValueError
        If t_eff is not square, veff does not have the matching four-index shape, or n_elec is odd, negative or larger than twice the number of orbitals.
    """
    return e_cas

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _sgn(bit, p):
    """fermionic sign (-1)^(number of occupied spin-orbitals below p) in the occupation string bit."""
    return -1.0 if bin(bit & ((1 << p) - 1)).count('1') % 2 else 1.0


def _oracle_active_space_fci(t_eff: "np.ndarray", veff: "np.ndarray", n_elec: int) -> float:
    t_eff = _check_array(t_eff, "t_eff", ndim=2)
    nA = t_eff.shape[0]
    veff = _check_array(veff, "veff", shape=(nA, nA, nA, nA))
    n_elec = _check_int(n_elec, "n_elec", 0)
    if t_eff.shape[1] != nA or n_elec % 2 or n_elec > 2 * nA:
        raise ValueError("t_eff must be square, n_elec even and at most 2 * n_orbitals")
    if n_elec == 0:
        return 0.0
    n_so = 2 * nA
    hso = np.zeros((n_so, n_so))
    hso[0::2, 0::2] = t_eff
    hso[1::2, 1::2] = t_eff
    vso = np.zeros((n_so,) * 4)
    for s in (0, 1):
        for s2 in (0, 1):
            vso[s::2, s::2, s2::2, s2::2] = veff
    # antisymmetrised <pq||rs> = (pr|qs) - (ps|qr)
    g = np.einsum('prqs->pqrs', vso) - np.einsum('psqr->pqrs', vso)
    na = n_elec // 2
    alphas = list(itertools.combinations(range(0, n_so, 2), na))
    betas = list(itertools.combinations(range(1, n_so, 2), na))
    dets = [tuple(sorted(a + b)) for a in alphas for b in betas]
    bits = [sum(1 << p for p in d) for d in dets]
    nd = len(dets)

    H = np.zeros((nd, nd))
    for I, dI in enumerate(dets):
        bI = bits[I]
        H[I, I] = sum(hso[p, p] for p in dI) + 0.5 * sum(g[p, q, p, q] for p in dI for q in dI)
        for J in range(I + 1, nd):
            bJ = bits[J]
            diff = bI ^ bJ
            nd_ = bin(diff).count('1')
            if nd_ == 2:
                p = (bI & diff).bit_length() - 1
                q = (bJ & diff).bit_length() - 1
                val = hso[p, q] + sum(g[p, r, q, r] for r in dI if r != p)
                H[I, J] = H[J, I] = _sgn(bI, p) * _sgn(bJ, q) * val
            elif nd_ == 4:
                p1, p2 = [k for k in range(n_so) if (bI & diff) >> k & 1]
                q1, q2 = [k for k in range(n_so) if (bJ & diff) >> k & 1]
                # <I| a+_p1 a+_p2 a_q2 a_q1 |J>: act on J right to left, tracking the occupation string
                bt = bJ
                s = _sgn(bt, q1); bt ^= 1 << q1
                s *= _sgn(bt, q2); bt ^= 1 << q2
                s *= _sgn(bt, p2); bt ^= 1 << p2
                s *= _sgn(bt, p1)
                H[I, J] = H[J, I] = s * g[p1, p2, q1, q2]
    return float(np.linalg.eigvalsh(H)[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nt_eff = np.array([[0.0, -1.0], [-1.0, 0.0]])\nveff = np.zeros((2, 2, 2, 2))\nveff[0, 0, 0, 0] = veff[1, 1, 1, 1] = 4.0\nn_elec = 2\n",
            "call": "active_space_fci(t_eff, veff, n_elec)",
            "gold_call": "_oracle_active_space_fci(t_eff, veff, n_elec)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nrng = np.random.default_rng(3)\nA = rng.standard_normal((3, 3)); t_eff = 0.5 * (A + A.T)\nQ = rng.standard_normal((3, 3)); Q = np.einsum('ip,iq->ipq', Q, Q)\nveff = np.einsum('ipq,irs->pqrs', Q, Q) / 3.0\nn_elec = 4\n",
            "call": "active_space_fci(t_eff, veff, n_elec)",
            "gold_call": "_oracle_active_space_fci(t_eff, veff, n_elec)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(10, 1.0, 0.07, 4.0, 6.0)\neps, C = out[0], out[1:]\ndef _fx_mo_integrals(C, U, kappa):\n    n = C.shape[0]\n    idx = np.arange(n)\n    V_mo = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)\n    Q = np.einsum('ip,iq->ipq', C, C)\n    return np.einsum('ipq,ij,jrs->pqrs', Q, V_mo, Q, optimize=True)\neri = _fx_mo_integrals(C, 4.0, 6.0)\nh = np.zeros((10, 10))\nfor k in range(9): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.07 * (-1) ** k)\nV = 4.0 / np.sqrt(1.0 + (4.0 * np.abs(np.arange(10)[:, None] - np.arange(10)[None, :]) / 6.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 5, [3, 4, 5, 6]\npairs = _oracle_constrained_ph_pairs(10, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\nt_eff = _oracle_effective_one_body(h_mo, eri, veff, n_occ, active)\nn_elec = 4\n",
            "call": "active_space_fci(t_eff, veff, n_elec)",
            "gold_call": "_oracle_active_space_fci(t_eff, veff, n_elec)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nt_eff = np.array([[-0.5, 0.2], [0.2, 0.3]])\nveff = np.zeros((2, 2, 2, 2))\nveff[0, 0, 0, 0] = veff[1, 1, 1, 1] = 1.0\nn_elec = 0\n",
            "call": "active_space_fci(t_eff, veff, n_elec)",
            "gold_call": "_oracle_active_space_fci(t_eff, veff, n_elec)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nt_eff = np.array([[0.0, -1.0], [-1.0, 0.0]])\nveff = np.zeros((2, 2, 2, 2))\nn_elec = 3\ndef run_model():\n    try:\n        active_space_fci(t_eff, veff, n_elec)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_active_space_fci(t_eff, veff, n_elec)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
