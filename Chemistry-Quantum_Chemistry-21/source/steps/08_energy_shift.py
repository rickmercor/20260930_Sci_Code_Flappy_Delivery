"""
Return the electronic contribution of the environment to the constant term of the source's downfolded Hamiltonian, its Eq. (2.21) (derived in its Appendix A.2), for the inputs of step 07 together with the two RPA correlation energies the expression needs, both obtained with step 03 by the caller: e_rpa_full for the full system with the bare interaction and e_rpa_active for the active orbitals alone with the screened interaction. Assemble the mean-field energy of the environment orbitals (Eq. (2.6)), the mean-field correction of the active orbitals that the screening of veff induces, and the two correlation energies with the signs the source's derivation prescribes.

The downfolded total energy is the constant shift plus the energy of the active-space solver; the source fixes the shift by demanding that this sum reproduces the total energy of its functional, which makes the shift contain a static mean-field part and the dynamical correlation of the environment, and makes it depend on the screened interaction through both a mean-field and a correlation term.

Returns
-------
float, the electronic constant energy shift of the downfolded Hamiltonian (native Python float).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def energy_shift(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]", e_rpa_full: float, e_rpa_active: float) -> float:
    """Return the electronic contribution of the environment to the constant term of the source's downfolded Hamiltonian, its Eq. (2.21) (derived in its Appendix A.2), for the inputs of step 07 together with the two RPA correlation energies the expression needs, both obtained with step 03 by the caller: e_rpa_full for the full system with the bare interaction and e_rpa_active for the active orbitals alone with the screened interaction. Assemble the mean-field energy of the environment orbitals (Eq. (2.6)), the mean-field correction of the active orbitals that the screening of veff induces, and the two correlation energies with the signs the source's derivation prescribes.

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
    e_rpa_full : float
        RPA correlation energy of the full system with the bare interaction (step 03).
    e_rpa_active : float
        RPA correlation energy of the active orbitals alone with the screened interaction (step 03).

    Returns
    -------
    e_shift : float
        The electronic energy shift E_E of the downfolded Hamiltonian in the energy units of the inputs.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, active is invalid as in step 05, or either energy is not finite.
    """
    return e_shift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_scalar(x, name, positive=False, nonneg=False):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("%s must be finite" % name)
    if positive and x <= 0:
        raise ValueError("%s must be positive" % name)
    if nonneg and x < 0:
        raise ValueError("%s must be non-negative" % name)
    return x


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


def _oracle_energy_shift(h_mo: "np.ndarray", eri: "np.ndarray", veff: "np.ndarray", n_occ: int, active: "list[int]", e_rpa_full: float, e_rpa_active: float) -> float:
    h_mo = _check_array(h_mo, "h_mo", ndim=2)
    n = h_mo.shape[0]
    if h_mo.shape != (n, n):
        raise ValueError("h_mo must be square")
    eri = _check_array(eri, "eri", shape=(n, n, n, n))
    n_occ = _check_int(n_occ, "n_occ", 1)
    act = _check_active(active, n)
    nA = len(act)
    veff = _check_array(veff, "veff", shape=(nA, nA, nA, nA))
    e_rpa_full = _check_scalar(e_rpa_full, "e_rpa_full")
    e_rpa_active = _check_scalar(e_rpa_active, "e_rpa_active")
    env_occ = [i for i in range(n_occ) if i not in act]
    act_occ = [k for k, p in enumerate(act) if p < n_occ]
    vt = veff - eri[np.ix_(act, act, act, act)]
    # Eq. (2.6): core and electron-electron Hartree-Fock energy of the environment orbitals
    E_core = 2.0 * sum(h_mo[i, i] for i in env_occ)
    E_hf_env = sum(2.0 * eri[i, i, j, j] - eri[i, j, j, i] for i in env_occ for j in env_occ)
    # Eq. (2.21), third term: Hartree-Fock energy of the occupied active orbitals with veff - v (cRPA only non-zero)
    E_hf_act = sum(2.0 * vt[t, t, u, u] - vt[t, u, u, t] for t in act_occ for u in act_occ)
    # plus the RPA correlation energy of the full system (bare v) minus that of the active orbitals with veff
    return float(E_core + E_hf_env + E_hf_act + e_rpa_full - e_rpa_active)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nh = np.zeros((6, 6))\nfor k in range(5): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.1 * (-1) ** k)\nV = 3.0 / np.sqrt(1.0 + (3.0 * np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 3, [2, 3]\npairs = _oracle_constrained_ph_pairs(6, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\ne_rpa_full, e_rpa_active = -0.0615, -0.0210\n",
            "call": "energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "gold_call": "_oracle_energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(8, 1.0, 0.07, 4.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 4.0, 4.0)\nh = np.zeros((8, 8))\nfor k in range(7): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.07 * (-1) ** k)\nV = 4.0 / np.sqrt(1.0 + (4.0 * np.abs(np.arange(8)[:, None] - np.arange(8)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 4, [2, 3, 4, 5]\nveff = eri[np.ix_(active, active, active, active)] * 0.98\ne_rpa_full, e_rpa_active = -0.2363, -0.0900\n",
            "call": "energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "gold_call": "_oracle_energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(10, 1.0, 0.07, 4.0, 6.0)\neps, C = out[0], out[1:]\ndef _fx_mo_integrals(C, U, kappa):\n    n = C.shape[0]\n    idx = np.arange(n)\n    V_mo = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)\n    Q = np.einsum('ip,iq->ipq', C, C)\n    return np.einsum('ipq,ij,jrs->pqrs', Q, V_mo, Q, optimize=True)\neri = _fx_mo_integrals(C, 4.0, 6.0)\nh = np.zeros((10, 10))\nfor k in range(9): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.07 * (-1) ** k)\nV = 4.0 / np.sqrt(1.0 + (4.0 * np.abs(np.arange(10)[:, None] - np.arange(10)[None, :]) / 6.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 5, [3, 4, 5, 6]\npairs = _oracle_constrained_ph_pairs(10, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\npf = [(i, a) for i in range(10) for a in range(10, 20) if i % 2 == a % 2]\ne_rpa_full = _oracle_rpa_correlation_energy(np.array([eps[a // 2] - eps[i // 2] for (i, a) in pf]), np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pf] for (i, a) in pf]))\npa = [(i, a) for (i, a) in pf if i // 2 in active and a // 2 in active]\nloc = {p: k for k, p in enumerate(active)}\ne_rpa_active = _oracle_rpa_correlation_energy(np.array([eps[a // 2] - eps[i // 2] for (i, a) in pa]), np.array([[veff[loc[i // 2], loc[a // 2], loc[j // 2], loc[b // 2]] for (j, b) in pa] for (i, a) in pa]))\n",
            "call": "energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "gold_call": "_oracle_energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nh = np.zeros((6, 6))\nfor k in range(5): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.1 * (-1) ** k)\nV = 3.0 / np.sqrt(1.0 + (3.0 * np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 3, [0, 1, 2, 3, 4, 5]\nveff = eri.copy()\ne_rpa_full, e_rpa_active = -0.0615, -0.0615\n",
            "call": "energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "gold_call": "_oracle_energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nout = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)\neps, C = out[0], out[1:]\neri = _oracle_mo_integrals(C, 3.0, 4.0)\nh = np.zeros((6, 6))\nfor k in range(5): h[k, k + 1] = h[k + 1, k] = -(1.0 + 0.1 * (-1) ** k)\nV = 3.0 / np.sqrt(1.0 + (3.0 * np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]) / 4.0) ** 2)\nh -= np.diag(V.sum(1) - np.diag(V))\nh_mo = C.T @ h @ C\nn_occ, active = 3, [2, 3]\npairs = _oracle_constrained_ph_pairs(6, n_occ, active)\nd = np.array([eps[a // 2] - eps[i // 2] for (i, a) in pairs])\nv = np.array([[eri[i // 2, a // 2, j // 2, b // 2] for (j, b) in pairs] for (i, a) in pairs])\nveff = _oracle_screened_interaction(eri, pairs, _oracle_rpa_static_kernel(d, v), active)\ne_rpa_full, e_rpa_active = float('nan'), -0.0210\ndef _fx_raises_valueerror(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "_fx_raises_valueerror(lambda: energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active))",
            "gold_call": "_fx_raises_valueerror(lambda: _oracle_energy_shift(h_mo, eri, veff, n_occ, active, e_rpa_full, e_rpa_active))",
        },
    ]
