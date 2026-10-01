"""
Reference energy of the interacting active space of the Dyall partition.

This step returns the lowest eigenvalue of the active-space configuration interaction

problem that the Dyall zeroth-order Hamiltonian carries, i.e. the energy of the

n_act_elec-electron reference state inside the active spin orbitals.



Scientific background.  What distinguishes the Dyall partition from an ordinary

Moller-Plesset or Epstein-Nesbet split is that the zeroth-order Hamiltonian keeps the FULL

two-electron interaction inside the active space,



H0(active) = heff_xy x^+ y + (1/2) <xy|zw> x^+ y^+ w z ,



so its reference state is not a single determinant but a correlated multideterminantal

state.  The one-electron part is the core-dressed operator

heff_xy = h_xy + sum_{k in core} <xk||yk>; the two-electron part is the all-active block of

the integrals, untouched.  The problem is solved exactly, by building the matrix of that

operator over all C(len(act), n_act_elec) determinants of the active spin orbitals and

diagonalising it; the requested energy is the lowest eigenvalue.



Two conventions decide the digits.  Every determinant is stored as an occupation

bitstring and anchored on its ascending occupation list, so the operator string p^+ q^+ s r

acting on it carries the fermionic sign (-1)^m for each elementary creation or

annihilation, m being the number of occupied spin orbitals strictly below the acting

index at that moment.  And the two-electron integrals are in physicists' notation,

<pq|rs>, entering the Hamiltonian as (1/2) <pq|rs> p^+ q^+ s r with the last two indices

reversed relative to the bra; using chemists' notation without transposing the middle

indices silently produces a different, still symmetric, operator.



The state this energy belongs to is the reference of the whole construction, and it is the

lowest state of the PARTITIONED Hamiltonian in the active space, not the lowest state of

the full Hamiltonian in that space and not a variational ground state of the molecule.



Raises ValueError when h is not a non-empty square array, when g does not match h in

shape, when either holds a non-finite entry, when core and act are not disjoint integer

index sets inside the spin-orbital range, when n_act_elec is not an integer between 0 and

the number of active spin orbitals, and when the active electron count admits no

determinant.

Returns
-------
float, the lowest eigenvalue of the active-space configuration interaction problem of the Dyall zeroth-order Hamiltonian, in hartree, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def active_reference_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        n_act_elec: int) -> float:
    """Energy of the interacting active-space reference of the Dyall partition.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core : sequence of int
        Occupied inactive spin-orbital indices.
    act : sequence of int
        Active spin-orbital indices; must be disjoint from core.
    n_act_elec : int
        Number of electrons in the active space.

    Returns
    -------
    float
        The lowest eigenvalue, in hartree, of the active-space configuration interaction
        problem built from heff and the all-active two-electron block.
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

def _validate_two_sets(core, act, n_orb):
    """Check core and act are disjoint integer index sets inside range(n_orb)."""
    try:
        core_l = [int(i) for i in core]
        act_l = [int(i) for i in act]
    except (TypeError, ValueError) as exc:
        raise ValueError("core and act must be sequences of integers") from exc
    joined = core_l + act_l
    if len(set(joined)) != len(joined):
        raise ValueError("core and act must be disjoint and free of repeats")
    if any(not 0 <= i < n_orb for i in joined):
        raise ValueError("core and act indices must lie inside the spin-orbital range")
    return core_l, act_l

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

def _oracle_active_reference_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        n_act_elec: int) -> float:
    """Reference implementation: heff plus the all-active block, diagonalised exactly."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l = _validate_two_sets(core, act, n_orb)
    n_act_elec = _check_int("n_act_elec", n_act_elec, 0, len(act_l))
    gaa = g_arr - np.transpose(g_arr, (0, 1, 3, 2))
    heff = _effective_one_electron(h_arr, gaa, core_l, act_l)
    _dets, vals, _vecs = _active_ci(h_arr, g_arr, act_l, heff, n_act_elec)
    return float(vals[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'active_reference_energy(H4.copy(), G4.copy(), [1], [0, 2], 1)',
            'gold_call': '_oracle_active_reference_energy(H4.copy(), G4.copy(), [1], [0, 2], 1)',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    return active_reference_energy(H4, G4, [1], [0, 2], 0)\ndef run_gold():\n    value = _oracle_active_reference_energy(H4, G4, [1], [0, 2], 0)\n    assert abs(value - (0.0)) <= 1e-10 * max(1.0, abs(0.0)),         "the frozen anchor 0.0 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'active_reference_energy(H6, G6, [2, 3], [1, 4], 2)',
            'gold_call': '_oracle_active_reference_energy(H6, G6, [2, 3], [1, 4], 2)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'active_reference_energy(H8, G8, [3, 4], [1, 2, 5, 6], 3)',
            'gold_call': '_oracle_active_reference_energy(H8, G8, [3, 4], [1, 2, 5, 6], 3)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'active_reference_energy(H6.copy(), G6.copy(), [], [1, 4], 2)',
            'gold_call': '_oracle_active_reference_energy(H6.copy(), G6.copy(), [], [1, 4], 2)',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n\ndef run_model():\n    return active_reference_energy(H12, G12, CORE12, ACT12, 3)\ndef run_gold():\n    value = _oracle_active_reference_energy(H12, G12, CORE12, ACT12, 3)\n    assert abs(value - (-1.9638514217145964)) <= 1e-10 * max(1.0, abs(-1.9638514217145964)),         "the frozen anchor -1.9638514217145964 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    active_reference_energy(H4, G4, [1], [0, 2], 3); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_reference_energy(H4, G4, [1], [0, 2], 3); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    active_reference_energy(H4, G4, [0], [0, 2], 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_reference_energy(H4, G4, [0], [0, 2], 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    active_reference_energy(H4, G4, [1], [0, 9], 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_reference_energy(H4, G4, [1], [0, 9], 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    active_reference_energy(H4, G4, [1], [0, 2], -1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_active_reference_energy(H4, G4, [1], [0, 2], -1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
