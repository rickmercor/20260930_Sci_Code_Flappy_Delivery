"""
Return the effective one-body matrix of the source's downfolded Hamiltonian, Eq. (2.20), over the active orbitals (order sorted(active)): the one-body term of the source's CASCI-type effective Hamiltonian (its Eq. (2.5)) built from the one-body integrals h_mo in the orbital basis, the bare integrals eri of step 02 and the doubly occupied orbitals outside the active space (the first n_occ orbitals are occupied), combined with the source's correction term of Eq. (2.19) for the screened two-body interaction veff of step 06 (given on the active block, same order).

When the interaction among active orbitals is replaced by a screened one, the mean-field (Hartree and exchange) interaction of the active electrons changes as well; the source's functional-derivative derivation shows that a one-body correction evaluated with the density of the reference state must accompany the screened two-body term so that no interaction is counted twice.

Returns
-------
numpy.ndarray of float64 with shape (nA, nA), nA = len(active).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_one_body(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]") -> "np.ndarray":
    """Return the effective one-body matrix of the source's downfolded Hamiltonian, Eq. (2.20), over the active orbitals (order sorted(active)): the one-body term of the source's CASCI-type effective Hamiltonian (its Eq. (2.5)) built from the one-body integrals h_mo in the orbital basis, the bare integrals eri of step 02 and the doubly occupied orbitals outside the active space (the first n_occ orbitals are occupied), combined with the source's correction term of Eq. (2.19) for the screened two-body interaction veff of step 06 (given on the active block, same order).

    Parameters
    ----------
    h_mo : numpy.ndarray
        Symmetric (n, n) one-body integrals in the orbital basis.
    eri : numpy.ndarray
        Bare two-electron integrals (n, n, n, n) in chemists' notation, same basis.
    veff : numpy.ndarray
        Screened integrals (nA, nA, nA, nA) over sorted(active), from step 06.
    n_occ : int
        Number of doubly occupied orbitals (orbitals 0 to n_occ - 1).
    active : list[int]
        Distinct indices in [0, n - 1] of the active spatial orbitals (non-empty).

    Returns
    -------
    t_eff : numpy.ndarray
        Symmetric (nA, nA) array, float64: the effective one-body matrix over sorted(active).

    Raises
    ------
    ValueError
        If the shapes are inconsistent, h_mo is not square, or active is invalid as in step 05.
    """
    return t_eff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _check_active(active, n_orb):
    act = [int(p) for p in np.asarray(active).ravel()]
    if len(act) == 0 or len(set(act)) != len(act) or min(act) < 0 or max(act) >= n_orb:
        raise ValueError("active must be a non-empty list of distinct orbital indices in range")
    return sorted(act)


def _oracle_effective_one_body(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]") -> "np.ndarray":
    h_mo = _check_array(h_mo, "h_mo", ndim=2)
    n = h_mo.shape[0]
    if h_mo.shape != (n, n):
        raise ValueError("h_mo must be square")
    eri = _check_array(eri, "eri", shape=(n, n, n, n))
    n_occ = _check_int(n_occ, "n_occ", 1)
    act = _check_active(active, n)
    nA = len(act)
    veff = _check_array(veff, "veff", shape=(nA, nA, nA, nA))
    env_occ = [i for i in range(n_occ) if i not in act]
    act_occ = [k for k, p in enumerate(act) if p < n_occ]
    # f_tu, Eq. (2.5): bare one-body term plus the mean field of the (doubly occupied) environment orbitals
    f = h_mo[np.ix_(act, act)].copy()
    for i in env_occ:
        f += 2.0 * eri[np.ix_(act, act)][:, :, i, i] - eri[np.ix_(act)][:, i, i, :][:, act]
    # double counting, Eq. (2.19), with the Hartree-Fock density (rho_vw = delta_vw on the occupied active orbitals)
    vt = veff - eri[np.ix_(act, act, act, act)]
    tdc = np.zeros((nA, nA))
    for v in act_occ:
        tdc += 2.0 * vt[:, :, v, v] - vt[:, v, v, :]
    return f - tdc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nh = np.zeros((6, 6))\nfor k in range(5): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.1 * (-1) ** k)\nV = 3.0 / np.sqrt(1.0 + (3.0 * np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 3, [2, 3]\npairs = _oracle_constrained_ph_pairs(6, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\n",
            "call": "effective_one_body(h_mo, eri, veff, n_occ, active)",
            "gold_call": "_oracle_effective_one_body(h_mo, eri, veff, n_occ, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(8, 1.0, 0.07, 4.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 4.0, 4.0)\nh = np.zeros((8, 8))\nfor k in range(7): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.07 * (-1) ** k)\nV = 4.0 / np.sqrt(1.0 + (4.0 * np.abs(np.arange(8)[:, None] - np.arange(8)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 4, [2, 3, 4, 5]\npairs = _oracle_constrained_ph_pairs(8, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\n",
            "call": "effective_one_body(h_mo, eri, veff, n_occ, active)",
            "gold_call": "_oracle_effective_one_body(h_mo, eri, veff, n_occ, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(10, 1.0, 0.07, 4.0, 6.0)\neps, C = out[0], out[1:]\ndef _fx_mo_integrals(C, U, kappa):\n    n = C.shape[0]\n    idx = np.arange(n)\n    V_mo = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)\n    Q = np.einsum('ip,iq->ipq', C, C)\n    return np.einsum('ipq,ij,jrs->pqrs', Q, V_mo, Q, optimize=True)\neri = _fx_mo_integrals(C, 4.0, 6.0)\nh = np.zeros((10, 10))\nfor k in range(9): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.07 * (-1) ** k)\nV = 4.0 / np.sqrt(1.0 + (4.0 * np.abs(np.arange(10)[:, None] - np.arange(10)[None, :]) / 6.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 5, [3, 4, 5, 6]\npairs = _oracle_constrained_ph_pairs(10, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\n",
            "call": "effective_one_body(h_mo, eri, veff, n_occ, active)",
            "gold_call": "_oracle_effective_one_body(h_mo, eri, veff, n_occ, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nh = np.zeros((6, 6))\nfor k in range(5): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.1 * (-1) ** k)\nV = 3.0 / np.sqrt(1.0 + (3.0 * np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 3, [0, 1, 2, 3, 4, 5]\nveff = eri.copy()\n",
            "call": "effective_one_body(h_mo, eri, veff, n_occ, active)",
            "gold_call": "_oracle_effective_one_body(h_mo, eri, veff, n_occ, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nh = np.zeros((6, 6))\nfor k in range(5): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.1 * (-1) ** k)\nV = 3.0 / np.sqrt(1.0 + (3.0 * np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 3, [2, 3]\nveff = np.zeros((3, 3, 3, 3))\ndef _fx_raises_valueerror(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_fx_raises_valueerror(lambda: effective_one_body(h_mo, eri, veff, n_occ, active))",
            "gold_call": "_fx_raises_valueerror(lambda: _oracle_effective_one_body(h_mo, eri, veff, n_occ, active))",
        },
    ]
