"""
Return the static (zero-frequency, vanishing broadening) screened interaction of the source, its Eqs. (2.12)-(2.13) at frequency zero, among the active orbitals: eri are the bare integrals of step 02 over all spatial orbitals, pairs the spin-orbital particle-hole pairs of the response (step 05, spin-orbital indexing as defined there), kernel the static kernel of step 04 for that pair space (same pair order), and active the active spatial orbitals. Return the four-index tensor W[t, u, v, w] of screened integrals in chemists' notation over the active orbitals in the order sorted(active), with every element screened.

The RPA screened interaction is the bare interaction plus a correction built from the transition densities and excitation energies of the response; contracting the bare integrals between an orbital pair and the particle-hole space with the static kernel of that space gives the static limit, and in the source's cRPA flavour this screening is applied to every matrix element among the active orbitals.

Returns
-------
numpy.ndarray of float64 with shape (nA, nA, nA, nA), nA = len(active).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screened_interaction(eri: "np.ndarray", pairs: "np.ndarray", kernel: "np.ndarray", active: "list[int]") -> "np.ndarray":
    """Return the static (zero-frequency, vanishing broadening) screened interaction of the source, its Eqs. (2.12)-(2.13) at frequency zero, among the active orbitals: eri are the bare integrals of step 02 over all spatial orbitals, pairs the spin-orbital particle-hole pairs of the response (step 05, spin-orbital indexing as defined there), kernel the static kernel of step 04 for that pair space (same pair order), and active the active spatial orbitals. Return the four-index tensor W[t, u, v, w] of screened integrals in chemists' notation over the active orbitals in the order sorted(active), with every element screened.

    Parameters
    ----------
    eri : numpy.ndarray
        Bare two-electron integrals (n, n, n, n) in chemists' notation over the spatial orbitals.
    pairs : numpy.ndarray
        Integer array (m, 2) of spin-orbital particle-hole pairs (i, a) as returned by step 05.
    kernel : numpy.ndarray
        Static kernel (m, m) of step 04 for exactly these pairs, in the same order.
    active : list[int]
        Distinct indices in [0, n - 1] of the active spatial orbitals (non-empty).

    Returns
    -------
    W : numpy.ndarray
        Array of shape (nA, nA, nA, nA), float64, nA = len(active): the screened integrals (tu|vw) over sorted(active).

    Raises
    ------
    ValueError
        If eri is not a four-index tensor with equal axes, pairs is not an (m, 2) array of valid spin-orbital indices, kernel is not (m, m), or active is invalid as in step 05.
    """
    return W

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


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


def _oracle_screened_interaction(eri: "np.ndarray", pairs: "np.ndarray", kernel: "np.ndarray", active: "list[int]") -> "np.ndarray":
    eri = _check_array(eri, "eri", ndim=4)
    n = eri.shape[0]
    if eri.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or (pairs.size and (pairs.min() < 0 or pairs.max() >= 2 * n)):
        raise ValueError("pairs must be an (m, 2) array of spin-orbital indices")
    pairs = [(int(i), int(a)) for i, a in pairs]
    m = len(pairs)
    K = _check_array(kernel, "kernel", shape=(m, m))
    act = _check_active(active, n)
    # W(omega = 0), Eq. (2.12)-(2.13): W = v + sum_nu w w [1/(0 - Omega) - 1/(0 + Omega)] = v - 2 v K v,
    # with w^nu_pq = sum_ia v_pq,ia (X+Y)^nu_ia summed over the (spin-orbital) pairs of the reduced space
    vpq = np.array([[[eri[p, q, i // 2, a // 2] for (i, a) in pairs] for q in act] for p in act]).reshape(len(act), len(act), m)
    return eri[np.ix_(act, act, act, act)] - 2.0 * np.einsum('pqI,IJ,rsJ->pqrs', vpq, K, vpq, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nactive = [2, 3]\npairs = _oracle_constrained_ph_pairs(6, 3, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nkernel = _oracle_rpa_static_kernel(d, v)\n",
            "call": "screened_interaction(eri, pairs, kernel, active)",
            "gold_call": "_oracle_screened_interaction(eri, pairs, kernel, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(8, 1.0, 0.07, 4.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 4.0, 4.0)\nactive = [2, 3, 4, 5]\npairs = _oracle_constrained_ph_pairs(8, 4, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nkernel = _oracle_rpa_static_kernel(d, v)\n",
            "call": "screened_interaction(eri, pairs, kernel, active)",
            "gold_call": "_oracle_screened_interaction(eri, pairs, kernel, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(10, 1.0, 0.07, 4.0, 6.0)\neps, C = out[0], out[1:]\ndef _fx_mo_integrals(C, U, kappa):\n    n = C.shape[0]\n    idx = np.arange(n)\n    V_mo = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)\n    Q = np.einsum('ip,iq->ipq', C, C)\n    return np.einsum('ipq,ij,jrs->pqrs', Q, V_mo, Q, optimize=True)\neri = _fx_mo_integrals(C, 4.0, 6.0)\nactive = [3, 4, 5, 6]\npairs = _oracle_constrained_ph_pairs(10, 5, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nkernel = _oracle_rpa_static_kernel(d, v)\n",
            "call": "screened_interaction(eri, pairs, kernel, active)",
            "gold_call": "_oracle_screened_interaction(eri, pairs, kernel, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nactive = [0, 1, 2, 3, 4, 5]\npairs = np.zeros((0, 2), dtype=np.int64)\nkernel = np.zeros((0, 0))\n",
            "call": "screened_interaction(eri, pairs, kernel, active)",
            "gold_call": "_oracle_screened_interaction(eri, pairs, kernel, active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nactive = [2, 3]\npairs = _oracle_constrained_ph_pairs(6, 3, active)\nkernel = np.zeros((3, 3))\ndef _fx_raises_valueerror(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_fx_raises_valueerror(lambda: screened_interaction(eri, pairs, kernel, active))",
            "gold_call": "_fx_raises_valueerror(lambda: _oracle_screened_interaction(eri, pairs, kernel, active))",
        },
    ]
