"""
Inactive orbital energies of the Dyall zeroth-order Hamiltonian.

This step returns one orbital energy eps of the occupied-inactive (core) or

unoccupied-inactive (virtual) block of the Dyall partition, obtained by canonicalising the

two inactive blocks separately against the generalised Fock operator of the interacting

active-space reference.



Scientific background.  In the Dyall partition the inactive orbitals enter the

zeroth-order Hamiltonian only quadratically, H0 = eps_P P^+ P + H0(active), so each of them

must carry a single orbital energy.  The operator whose eigenvalues those energies are is

the generalised Fock operator built on the one-body density matrix of the interacting

active-space reference, not on a closed-shell determinant,



F_ij = h_ij + sum_{k in core} <ik||jk> + sum_{x,y in act} <ix||jy> <x^+ y> ,



with <pq||rs> = <pq|rs> - <pq|sr>.  The active density <x^+ y> is the one-body reduced

density matrix of the lowest eigenvector of the active-space configuration interaction

problem built from the core-dressed effective one-electron operator

heff_xy = h_xy + sum_{k in core} <xk||yk> together with the all-active two-electron block,

solved in the n_act_elec-electron determinant space of the active spin orbitals.  The

determinants are occupation bitstrings anchored on their ascending occupation lists, so

every creation and annihilation operator carries the fermionic sign (-1) raised to the

number of occupied orbitals below its index.



The decisive structural point is that the occupied-inactive and unoccupied-inactive blocks

are canonicalised SEPARATELY: F is diagonalised inside the core block and inside the

virtual block, and the core-virtual block of F is never touched.  Diagonalising one joint

inactive Fock matrix instead would mix occupied and unoccupied inactive orbitals, changing

which orbitals are occupied in the zeroth-order reference and therefore changing the

partition itself.  The returned list of energies is ordered as the core eigenvalues in

ascending order followed by the virtual eigenvalues in ascending order, so `index` runs

from 0 to len(core) + len(vir) - 1 in that order; the energies themselves are eigenvalues

and are therefore independent of any phase or ordering convention of the eigenvectors.



Raises ValueError when h is not a non-empty square array, when g does not match h in

shape, when either holds a non-finite entry, when core, act and vir do not partition the

spin orbitals exactly once, when n_act_elec is not an integer between 0 and the number of

active spin orbitals, when the active electron count admits no determinant, and when index

is not an integer between 0 and len(core) + len(vir) - 1.

Returns
-------
float, one inactive orbital energy eps of the Dyall zeroth-order Hamiltonian in hartree, the core block eigenvalues in ascending order followed by the virtual block eigenvalues in ascending order, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dyall_inactive_orbital_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_act_elec: int, index: int) -> float:
    """One inactive orbital energy of the Dyall zeroth-order Hamiltonian.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core : sequence of int
        Occupied inactive spin-orbital indices.
    act : sequence of int
        Active spin-orbital indices.
    vir : sequence of int
        Unoccupied inactive spin-orbital indices.  core, act and vir together must
        partition range(n) exactly once.
    n_act_elec : int
        Number of electrons in the active space.
    index : int
        Position in the concatenated list of inactive orbital energies, core block first
        in ascending order, then virtual block in ascending order.

    Returns
    -------
    float
        The requested orbital energy in hartree.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools

import numpy as np

def _check_int(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

def _validate_integrals(h, g):
    """Check h and g are finite real arrays of matching shape; return them as floats."""
    try:
        h_arr = np.asarray(h, dtype=float)
        g_arr = np.asarray(g, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("h and g must be real numeric arrays") from exc
    if h_arr.ndim != 2 or h_arr.shape[0] != h_arr.shape[1] or h_arr.shape[0] < 1:
        raise ValueError("h must be a non-empty square two-dimensional array")
    n_orb = h_arr.shape[0]
    if g_arr.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("g must have shape (n, n, n, n) matching h")
    if not (np.all(np.isfinite(h_arr)) and np.all(np.isfinite(g_arr))):
        raise ValueError("h and g must contain only finite numbers")
    return h_arr, g_arr

def _validate_partition(core, act, vir, n_orb):
    """Check that core, act and vir partition range(n_orb) exactly once."""
    try:
        core_l = [int(i) for i in core]
        act_l = [int(i) for i in act]
        vir_l = [int(i) for i in vir]
    except (TypeError, ValueError) as exc:
        raise ValueError("core, act and vir must be sequences of integers") from exc
    if sorted(core_l + act_l + vir_l) != list(range(n_orb)):
        raise ValueError("core, act and vir must partition the spin orbitals exactly once")
    return core_l, act_l, vir_l

def _determinants(orbitals, n_elec):
    """Occupation bitmasks of every determinant with n_elec electrons in `orbitals`."""
    return [sum(1 << i for i in combo)
            for combo in itertools.combinations(orbitals, n_elec)]

def _annihilate(det, p):
    """Apply the annihilation operator p to a determinant; return (sign, determinant)."""
    if not (det >> p) & 1:
        return 0, None
    sign = -1 if bin(det & ((1 << p) - 1)).count('1') % 2 else 1
    return sign, det & ~(1 << p)

def _create(det, p):
    """Apply the creation operator p+ to a determinant; return (sign, determinant)."""
    if (det >> p) & 1:
        return 0, None
    sign = -1 if bin(det & ((1 << p) - 1)).count('1') % 2 else 1
    return sign, det | (1 << p)

def _ci_matrix(h, g, dets):
    """Matrix of H = h_pq p+ q + (1/2) <pq|rs> p+ q+ s r over a determinant list."""
    n_orb = h.shape[0]
    index = {d: i for i, d in enumerate(dets)}
    mat = np.zeros((len(dets), len(dets)))
    for col, det in enumerate(dets):
        occ = [i for i in range(n_orb) if (det >> i) & 1]
        for q in occ:
            s1, d1 = _annihilate(det, q)
            for p in range(n_orb):
                s2, d2 = _create(d1, p)
                if s2 == 0:
                    continue
                row = index.get(d2)
                if row is not None:
                    mat[row, col] += s1 * s2 * h[p, q]
        for r in occ:
            sr, dr = _annihilate(det, r)
            occ2 = [i for i in range(n_orb) if (dr >> i) & 1]
            for s in occ2:
                ss, ds = _annihilate(dr, s)
                for q in range(n_orb):
                    sq, dq = _create(ds, q)
                    if sq == 0:
                        continue
                    for p in range(n_orb):
                        sp, dp = _create(dq, p)
                        if sp == 0:
                            continue
                        row = index.get(dp)
                        if row is not None:
                            mat[row, col] += 0.5 * sr * ss * sq * sp * g[p, q, r, s]
    return mat

def _effective_one_electron(h, gaa, core, act):
    """Eq. (15): heff_xy = h_xy + sum_k <xk||yk>, the CORE-ONLY active mean field."""
    return np.array([[h[x, y] + sum(gaa[x, k, y, k] for k in core) for y in act]
                     for x in act])

def _active_ci(h, g, act, heff, n_act_elec):
    """Diagonalise heff plus the all-active two-electron block in the n_act_elec sector."""
    n_orb = h.shape[0]
    h_act = np.zeros((n_orb, n_orb))
    g_act = np.zeros_like(g)
    if act:
        h_act[np.ix_(act, act)] = heff
        g_act[np.ix_(act, act, act, act)] = g[np.ix_(act, act, act, act)]
    dets = _determinants(act, n_act_elec)
    if not dets:
        raise ValueError("the active electron count admits no determinant")
    vals, vecs = np.linalg.eigh(_ci_matrix(h_act, g_act, dets))
    return dets, vals, vecs

def _active_rdm(dets, coeff, act, n_orb):
    """One-body density matrix <x+ y> of the active reference state."""
    index = {d: i for i, d in enumerate(dets)}
    rdm = np.zeros((n_orb, n_orb))
    for col, det in enumerate(dets):
        for y in act:
            s1, d1 = _annihilate(det, y)
            if s1 == 0:
                continue
            for x in act:
                s2, d2 = _create(d1, x)
                if s2 == 0:
                    continue
                row = index.get(d2)
                if row is not None:
                    rdm[x, y] += s1 * s2 * coeff[col] * coeff[row]
    return rdm

def _fock_block(h, gaa, core, act, rdm, orbitals):
    """Eq. (14): F_ij = h_ij + sum_k <ik||jk> + sum_xy <ix||jy> <x+ y>, one block only."""
    return np.array([[h[i, j] + sum(gaa[i, k, j, k] for k in core)
                      + sum(gaa[i, x, j, y] * rdm[x, y] for x in act for y in act)
                      for j in orbitals] for i in orbitals])

def _dyall_setup(h, g, core, act, vir, n_act_elec):
    """Assemble the Dyall zeroth-order Hamiltonian in the canonical inactive basis.

    Returns (h_can, g_can, h0, g0, eps, heff, rdm).  The core and virtual blocks are
    canonicalised SEPARATELY, eps holds the core orbital energies in ascending order
    followed by the virtual ones in ascending order, h0 carries eps on its inactive
    diagonal and heff on its active block, and g0 keeps only the all-active block.
    """
    n_orb = h.shape[0]
    gaa = g - np.transpose(g, (0, 1, 3, 2))
    heff = _effective_one_electron(h, gaa, core, act)
    dets, _vals, vecs = _active_ci(h, g, act, heff, n_act_elec)
    rdm = _active_rdm(dets, vecs[:, 0], act, n_orb) if act else np.zeros((n_orb, n_orb))
    rot = np.eye(n_orb)
    eps_core, u_core = np.linalg.eigh(_fock_block(h, gaa, core, act, rdm, core))
    eps_vir, u_vir = np.linalg.eigh(_fock_block(h, gaa, core, act, rdm, vir))
    rot[np.ix_(core, core)] = u_core
    rot[np.ix_(vir, vir)] = u_vir
    eps = np.concatenate([eps_core, eps_vir])
    h_can = rot.T @ h @ rot
    g_can = np.einsum('ap,bq,cr,ds,abcd->pqrs', rot, rot, rot, rot, g, optimize=True)
    h0 = np.zeros((n_orb, n_orb))
    for k, orb in enumerate(list(core) + list(vir)):
        h0[orb, orb] = eps[k]
    g0 = np.zeros_like(g_can)
    if act:
        h0[np.ix_(act, act)] = heff
        g0[np.ix_(act, act, act, act)] = g_can[np.ix_(act, act, act, act)]
    return h_can, g_can, h0, g0, eps, heff, rdm

def _oracle_dyall_inactive_orbital_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_act_elec: int, index: int) -> float:
    """Reference implementation: Eq. (14) Fock blocks, canonicalised separately."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l, vir_l = _validate_partition(core, act, vir, n_orb)
    n_act_elec = _check_int("n_act_elec", n_act_elec, 0, len(act_l))
    index = _check_int("index", index, 0, len(core_l) + len(vir_l) - 1)
    _hc, _gc, _h0, _g0, eps, _heff, _rdm = _dyall_setup(
        h_arr, g_arr, core_l, act_l, vir_l, n_act_elec)
    return float(eps[index])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n",
            'call': 'dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 1, 0)',
            'gold_call': '_oracle_dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 1, 0)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n",
            'call': 'dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 1, 1)',
            'gold_call': '_oracle_dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 1, 1)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n",
            'call': 'dyall_inactive_orbital_energy(H5.copy(), G5.copy(), [2], [1, 3], [0, 4], 2, 2)',
            'gold_call': '_oracle_dyall_inactive_orbital_energy(H5.copy(), G5.copy(), [2], [1, 3], [0, 4], 2, 2)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n\ndef run_model():\n    return [dyall_inactive_orbital_energy(H6, G6, [2, 3], [1, 4], [0, 5], 2, k)\n            for k in range(4)]\ndef run_gold():\n    return [_oracle_dyall_inactive_orbital_energy(H6, G6, [2, 3], [1, 4], [0, 5], 2, k)\n            for k in range(4)]\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n\ndef run_model():\n    return dyall_inactive_orbital_energy(H12, G12, CORE12, ACT12, VIR12, 3, 0)\ndef run_gold():\n    value = _oracle_dyall_inactive_orbital_energy(H12, G12, CORE12, ACT12, VIR12, 3, 0)\n    assert abs(value - (-0.19911202013020635)) <= 1e-10 * max(1.0, abs(-0.19911202013020635)),         "the frozen anchor -0.19911202013020635 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n\ndef run_model():\n    return dyall_inactive_orbital_energy(H12, G12, CORE12, ACT12, VIR12, 3, 5)\ndef run_gold():\n    value = _oracle_dyall_inactive_orbital_energy(H12, G12, CORE12, ACT12, VIR12, 3, 5)\n    assert abs(value - (0.45093322797787183)) <= 1e-10 * max(1.0, abs(0.45093322797787183)),         "the frozen anchor 0.45093322797787183 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n",
            'call': 'dyall_inactive_orbital_energy(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 3, 2)',
            'gold_call': '_oracle_dyall_inactive_orbital_energy(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 3, 2)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n\ndef run_model():\n    try:    dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 1, 2); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 1, 2); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n\ndef run_model():\n    try:    dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [], 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [], 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n\ndef run_model():\n    try:    dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 3, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_dyall_inactive_orbital_energy(H4, G4, [1], [0, 2], [3], 3, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH5, G5 = _fx_system(5)\nH6, G6 = _fx_system(6)\n\nbad = G4.copy(); bad[0, 0, 0, 0] = np.inf\ndef run_model():\n    try:    dyall_inactive_orbital_energy(H4, bad, [1], [0, 2], [3], 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_dyall_inactive_orbital_energy(H4, bad, [1], [0, 2], [3], 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
