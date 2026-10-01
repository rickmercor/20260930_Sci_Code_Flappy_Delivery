"""
Charged-sector channel energies of the zeroth-order Green's function.

This step returns one electron-addition or electron-removal energy of the interacting

zeroth-order (Dyall) Hamiltonian, i.e. one pole position of the zeroth-order Green's

function G0, selected by branch and by its position in the retained list.



Scientific background.  The multi-reference construction dresses an INTERACTING zeroth-order

propagator rather than a mean-field one, so G0 is not a set of orbital energies.  Its

Lehmann representation runs over the exact eigenstates of the Dyall zeroth-order

Hamiltonian in the two neighbouring particle-number sectors,



[G0(w)]_pq = sum_m <Phi_0| p |Phi^{N+1}_m> <Phi^{N+1}_m| q^+ |Phi_0> / (w - w^+_m)

               + sum_m <Phi_0| q^+ |Phi^{N-1}_m> <Phi^{N-1}_m| p |Phi_0> / (w - w^-_m) ,



with w^+_m = E^{N+1}_m - E^N_0 the addition energies and w^-_m = -(E^{N-1}_m - E^N_0) the

removal ones; this step returns the unsigned excitation energies E^{N+-1}_m - E^N_0 of

whichever branch is asked for.  H0 is built as before: the inactive orbital energies of the

separately canonicalised core and virtual Fock blocks on the inactive diagonal, the

core-dressed effective one-electron operator on the active block, and the all-active

two-electron block; it is then diagonalised over the COMPLETE determinant space of all n

spin orbitals in the N, N+1 and N-1 electron sectors, and |Phi_0> is its lowest N-electron

eigenstate.  Every determinant is anchored on its ascending occupation list, so the single

annihilation operator that connects two sectors carries the fermionic sign (-1) raised to

the number of occupied spin orbitals below its index.



A channel is kept only when it actually couples to the reference: the amplitude vector

a_m = (<Phi_0| p |Phi^{N+1}_m>)_p, or its removal counterpart, must have Euclidean norm

above 1e-10.  Most eigenstates of a sector fail this outright, because no single creation

or annihilation operator connects them to |Phi_0>; on the frozen twelve-orbital model the

smallest retained norm is 8.1e-3 and the largest discarded one is 6.3e-14, so the cut sits

eleven orders of magnitude away from anything a correct implementation can produce and is

not a tunable parameter.  The retained channels are listed in ascending order of their

sector eigenvalue, which is the order a symmetric eigensolver returns, and `index` selects

within that list.  Because only energy differences are asked for, the answer is

independent of every eigenvector phase.



Raises ValueError when h is not a non-empty square array, when g does not match h in

shape, when either holds a non-finite entry, when core, act and vir do not partition the

spin orbitals exactly once, when n_elec is not n_act_elec plus the number of core spin

orbitals, when n_act_elec exceeds the number of active spin orbitals, when branch is

neither +1 nor -1, when the requested neighbouring sector has no determinant space, and

when index falls outside the retained list of that branch.

Returns
-------
float, one charged-sector excitation energy E^{N+-1}_m - E^N_0 of the Dyall zeroth-order Hamiltonian in hartree, taken from the retained channels of the requested branch in ascending energy order, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def charged_channel_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, branch: int,
        index: int) -> float:
    """One retained charged-sector excitation energy of the Dyall zeroth-order Hamiltonian.

    Parameters
    ----------
    h : array_like, shape (n, n)
        One-electron Hamiltonian in an orthonormal spin-orbital basis, in hartree.
    g : array_like, shape (n, n, n, n)
        Two-electron integrals in physicists' notation, g[p, q, r, s] = <pq|rs>.
    core, act, vir : sequence of int
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n) exactly once.
    n_elec : int
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int
        Number of electrons in the active space.
    branch : int
        +1 for the electron-addition (N + 1) sector, -1 for the electron-removal (N - 1)
        sector.
    index : int
        Position in the retained channel list of that branch, ordered by increasing
        sector eigenvalue.

    Returns
    -------
    float
        E^{N+1}_m - E^N_0 for branch +1, or E^{N-1}_m - E^N_0 for branch -1, in hartree.
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

def _validate_counts(n_elec, n_act_elec, core, act):
    """Check the electron counts against the partition; return them as ints."""
    n_elec = _check_int("n_elec", n_elec, 1, 64)
    n_act_elec = _check_int("n_act_elec", n_act_elec, 0, 64)
    if n_act_elec > len(act):
        raise ValueError("n_act_elec cannot exceed the number of active spin orbitals")
    if n_elec != n_act_elec + len(core):
        raise ValueError("n_elec must equal n_act_elec plus the number of core spin orbitals")
    return n_elec, n_act_elec

def _cache_key(h, g, core, act, vir, n_elec, n_act_elec):
    """Hashable digest of the whole specification, so repeated calls reuse one solve."""
    return (h.tobytes(), g.tobytes(), tuple(core), tuple(act), tuple(vir),
            int(n_elec), int(n_act_elec))

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

def _zeroth_order_states(h0, g0, n_orb, n_elec):
    """Spectrum and eigenvectors of the Dyall H0 over the whole n_elec determinant space."""
    if n_elec < 0 or n_elec > n_orb:
        raise ValueError("the electron count leaves no determinant space")
    dets = _determinants(range(n_orb), n_elec)
    vals, vecs = np.linalg.eigh(_ci_matrix(h0, g0, dets))
    return dets, vals, vecs

def _charged_channels(h0, g0, n_orb, n_elec, dets, vals, vecs):
    """Retained addition and removal channels of the zeroth-order Green's function.

    Returns (add_amplitude, add_energy, rem_amplitude, rem_energy).  A channel is
    retained only when its amplitude vector has Euclidean norm above 1e-10, which
    discards the zeroth-order states that no single creation or annihilation operator
    connects to the reference; the measured gap is eleven orders wide on the frozen
    model, so the cut is not a tunable parameter.
    """
    coeff0 = vecs[:, 0]
    index0 = {d: i for i, d in enumerate(dets)}
    dets_p, vals_p, vecs_p = _zeroth_order_states(h0, g0, n_orb, n_elec + 1)
    add = np.zeros((len(dets_p), n_orb))
    for ket, det in enumerate(dets_p):
        for p in range(n_orb):
            sign, rest = _annihilate(det, p)
            if sign == 0:
                continue
            row = index0.get(rest)
            if row is not None:
                add[:, p] += sign * coeff0[row] * vecs_p[ket, :]
    dets_m, vals_m, vecs_m = _zeroth_order_states(h0, g0, n_orb, n_elec - 1)
    index_m = {d: i for i, d in enumerate(dets_m)}
    rem = np.zeros((len(dets_m), n_orb))
    for ket, det in enumerate(dets):
        for p in range(n_orb):
            sign, rest = _annihilate(det, p)
            if sign == 0:
                continue
            row = index_m.get(rest)
            if row is not None:
                rem[:, p] += sign * coeff0[ket] * vecs_m[row, :]
    omega_add = vals_p - vals[0]
    omega_rem = vals_m - vals[0]
    keep_add = np.linalg.norm(add, axis=1) > 1e-10
    keep_rem = np.linalg.norm(rem, axis=1) > 1e-10
    return add[keep_add], omega_add[keep_add], rem[keep_rem], omega_rem[keep_rem]

def _solve_channels(h, g, core, act, vir, n_elec, n_act_elec, *, _cache={}):
    """Retained addition and removal channels, cached on the whole specification.

    The keyword-only ``_cache`` memoises the result on the whole model
    specification; it is pure memoisation of a deterministic function and
    changes no result.
    """
    key = _cache_key(h, g, core, act, vir, n_elec, n_act_elec)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    n_orb = h.shape[0]
    _hc, _gc, h0, g0, _eps, _heff, _rdm = _dyall_setup(h, g, core, act, vir, n_act_elec)
    dets, vals, vecs = _zeroth_order_states(h0, g0, n_orb, n_elec)
    out = _charged_channels(h0, g0, n_orb, n_elec, dets, vals, vecs)
    _cache[key] = out
    return out

def _oracle_charged_channel_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, branch: int,
        index: int) -> float:
    """Reference implementation of the retained G0 channel energies."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l, vir_l = _validate_partition(core, act, vir, n_orb)
    n_elec, n_act_elec = _validate_counts(n_elec, n_act_elec, core_l, act_l)
    branch = _check_int("branch", branch, -1, 1)
    if branch not in (-1, 1):
        raise ValueError("branch must be +1 (addition) or -1 (removal)")
    if n_elec + 1 > n_orb or n_elec - 1 < 0:
        raise ValueError("a neighbouring particle-number sector is empty")
    _add, omega_add, _rem, omega_rem = _solve_channels(
        h_arr, g_arr, core_l, act_l, vir_l, n_elec, n_act_elec)
    energies = omega_add if branch == 1 else omega_rem
    index = _check_int("index", index, 0, len(energies) - 1)
    return float(energies[index])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, 1, 0)',
            'gold_call': '_oracle_charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, 1, 0)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, -1, 0)',
            'gold_call': '_oracle_charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, -1, 0)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    return [charged_channel_energy(H6, G6, [2, 3], [1, 4], [0, 5], 4, 2, -1, k)\n            for k in range(4)]\ndef run_gold():\n    return [_oracle_charged_channel_energy(H6, G6, [2, 3], [1, 4], [0, 5], 4, 2, -1, k)\n            for k in range(4)]\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'charged_channel_energy(H6.copy(), G6.copy(), [2, 3], [1, 4], [0, 5], 4, 2, -1, 3)',
            'gold_call': '_oracle_charged_channel_energy(H6.copy(), G6.copy(), [2, 3], [1, 4], [0, 5], 4, 2, -1, 3)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'charged_channel_energy(H8, G8, [3, 4], [1, 2, 5, 6], [0, 7], 5, 3, 1, 2)',
            'gold_call': '_oracle_charged_channel_energy(H8, G8, [3, 4], [1, 2, 5, 6], [0, 7], 5, 3, 1, 2)',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n\ndef run_model():\n    return charged_channel_energy(H12, G12, CORE12, ACT12, VIR12, 5, 3, 1, 0)\ndef run_gold():\n    value = _oracle_charged_channel_energy(H12, G12, CORE12, ACT12, VIR12, 5, 3, 1, 0)\n    assert abs(value - (0.001632385690201854)) <= 1e-10 * max(1.0, abs(0.001632385690201854)),         "the frozen anchor 0.001632385690201854 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n\ndef run_model():\n    return charged_channel_energy(H12, G12, CORE12, ACT12, VIR12, 5, 3, -1, 0)\ndef run_gold():\n    value = _oracle_charged_channel_energy(H12, G12, CORE12, ACT12, VIR12, 5, 3, -1, 0)\n    assert abs(value - (0.05342079928514032)) <= 1e-10 * max(1.0, abs(0.05342079928514032)),         "the frozen anchor 0.05342079928514032 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n",
            'call': 'charged_channel_energy(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 5, 3, 1, 18)',
            'gold_call': '_oracle_charged_channel_energy(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 5, 3, 1, 18)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n",
            'call': 'charged_channel_energy(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 5, 3, -1, 16)',
            'gold_call': '_oracle_charged_channel_energy(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 5, 3, -1, 16)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, 1, 40); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_charged_channel_energy(H4, G4, [1], [0, 2], [3], 2, 1, 1, 40); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    charged_channel_energy(H4, G4, [1], [0, 2], [3], 3, 1, 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_charged_channel_energy(H4, G4, [1], [0, 2], [3], 3, 1, 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
