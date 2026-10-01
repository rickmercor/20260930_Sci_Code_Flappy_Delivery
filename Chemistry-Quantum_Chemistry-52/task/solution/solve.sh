#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math

import numpy as np

from scipy.special import erf

def _check_int(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

def _boys0(x):
    """Zeroth Boys function F_0(x), with the removable singularity at x = 0 handled."""
    if x <= 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))

def _basis(n_basis):
    """Exponents, centres and normalisation constants of the n_basis primitives."""
    angle_step = 2.4
    radius = 1.35
    pitch = 0.62
    exponent_base = 0.42
    exponent_step = 0.06
    idx = np.arange(n_basis, dtype=float)
    angle = angle_step * idx
    centres = np.stack([radius * np.cos(angle),
                        radius * np.sin(angle),
                        pitch * idx], axis=1)
    alpha = exponent_base + exponent_step * idx
    norm = (2.0 * alpha / np.pi) ** 0.75
    return alpha, centres, norm

def _primitive_one_electron(n_basis):
    """Overlap, kinetic and nuclear-attraction matrices over the raw primitives."""
    point_charge = 0.55
    alpha, centres, norm = _basis(n_basis)
    ovlp = np.zeros((n_basis, n_basis))
    kin = np.zeros((n_basis, n_basis))
    nuc = np.zeros((n_basis, n_basis))
    for a in range(n_basis):
        for b in range(n_basis):
            pa, pb = alpha[a], alpha[b]
            tot = pa + pb
            dist2 = float(np.sum((centres[a] - centres[b]) ** 2))
            pref = math.exp(-pa * pb / tot * dist2)
            prod = (pa * centres[a] + pb * centres[b]) / tot
            base = (np.pi / tot) ** 1.5 * pref
            ovlp[a, b] = norm[a] * norm[b] * base
            kin[a, b] = norm[a] * norm[b] * (
                pa * pb / tot * (3.0 - 2.0 * pa * pb / tot * dist2)) * base
            acc = 0.0
            for centre in centres:
                arg = tot * float(np.sum((prod - centre) ** 2))
                acc += -point_charge * 2 * np.pi / tot * pref * _boys0(arg)
            nuc[a, b] = norm[a] * norm[b] * acc
    return ovlp, kin, nuc

def _loewdin_transform(ovlp):
    """Symmetric orthonormaliser X = S^{-1/2}, formed through the spectral resolution."""
    val, vec = np.linalg.eigh(ovlp)
    return vec @ np.diag(val ** -0.5) @ vec.T

def _orthonormal_one_electron(n_basis, *, _cache={}):
    """The full Loewdin-orthonormal one-electron Hamiltonian of the model.

    The keyword-only ``_cache`` memoises the matrix per basis size; it is pure
    memoisation of a deterministic function and changes no result.
    """
    cached = _cache.get(n_basis)
    if cached is not None:
        return cached
    ovlp, kin, nuc = _primitive_one_electron(n_basis)
    x_mat = _loewdin_transform(ovlp)
    h_mat = x_mat.T @ (kin + nuc) @ x_mat
    _cache[n_basis] = h_mat
    return h_mat

def orthonormal_one_electron_element(n_basis: int, p: int, q: int) -> float:
    """Reference implementation: primitive integrals, S^{-1/2}, congruence transform."""
    n_basis = _check_int("n_basis", n_basis, 1, 24)
    p = _check_int("p", p, 0, n_basis - 1)
    q = _check_int("q", q, 0, n_basis - 1)
    return float(_orthonormal_one_electron(n_basis)[p, q])

import math

import numpy as np

from scipy.special import erf

def _check_index(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

def _boys_zero(x):
    """Zeroth Boys function F_0(x), with the removable singularity at x = 0 handled."""
    if x <= 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))

def _helix_basis(n_basis):
    """Exponents, centres and normalisation constants of the n_basis primitives."""
    angle_step = 2.4
    radius = 1.35
    pitch = 0.62
    exponent_base = 0.42
    exponent_step = 0.06
    idx = np.arange(n_basis, dtype=float)
    angle = angle_step * idx
    centres = np.stack([radius * np.cos(angle),
                        radius * np.sin(angle),
                        pitch * idx], axis=1)
    alpha = exponent_base + exponent_step * idx
    norm = (2.0 * alpha / np.pi) ** 0.75
    return alpha, centres, norm

def _primitive_repulsion(n_basis):
    """The n_basis^4 chemists'-notation (ab|cd) integrals and the overlap matrix."""
    alpha, centres, norm = _helix_basis(n_basis)
    ovlp = np.zeros((n_basis, n_basis))
    tot = np.zeros((n_basis, n_basis))
    pref = np.zeros((n_basis, n_basis))
    prod = np.zeros((n_basis, n_basis, 3))
    for a in range(n_basis):
        for b in range(n_basis):
            pa, pb = alpha[a], alpha[b]
            summ = pa + pb
            dist2 = float(np.sum((centres[a] - centres[b]) ** 2))
            kab = math.exp(-pa * pb / summ * dist2)
            tot[a, b] = summ
            pref[a, b] = kab
            prod[a, b] = (pa * centres[a] + pb * centres[b]) / summ
            base = (np.pi / summ) ** 1.5 * kab
            ovlp[a, b] = norm[a] * norm[b] * base
    eri = np.zeros((n_basis, n_basis, n_basis, n_basis))
    for a in range(n_basis):
        for b in range(n_basis):
            for c in range(n_basis):
                for d in range(n_basis):
                    s1, s2 = tot[a, b], tot[c, d]
                    arg = s1 * s2 / (s1 + s2) * float(np.sum((prod[a, b] - prod[c, d]) ** 2))
                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \
                        * pref[a, b] * pref[c, d] * _boys_zero(arg)
                    eri[a, b, c, d] = norm[a] * norm[b] * norm[c] * norm[d] * val
    return ovlp, eri

def _symmetric_orthonormaliser(ovlp):
    """Symmetric orthonormaliser X = S^{-1/2}, formed through the spectral resolution."""
    val, vec = np.linalg.eigh(ovlp)
    return vec @ np.diag(val ** -0.5) @ vec.T

def _orthonormal_two_electron(n_basis, *, _cache={}):
    """The full physicists'-notation <pq|rs> array of the model.

    The keyword-only ``_cache`` memoises the array per basis size; it is pure
    memoisation of a deterministic function and changes no result.
    """
    cached = _cache.get(n_basis)
    if cached is not None:
        return cached
    ovlp, eri = _primitive_repulsion(n_basis)
    x_mat = _symmetric_orthonormaliser(ovlp)
    chem = np.einsum('ap,bq,cr,ds,abcd->pqrs', x_mat, x_mat, x_mat, x_mat, eri,
                     optimize=True)
    phys = np.transpose(chem, (0, 2, 1, 3))
    _cache[n_basis] = phys
    return phys

def orthonormal_two_electron_element(n_basis: int, p: int, q: int, r: int, s: int) -> float:
    """Reference implementation: Boys-function ERIs, four-index transform, transposition."""
    n_basis = _check_index("n_basis", n_basis, 1, 16)
    p = _check_index("p", p, 0, n_basis - 1)
    q = _check_index("q", q, 0, n_basis - 1)
    r = _check_index("r", r, 0, n_basis - 1)
    s = _check_index("s", s, 0, n_basis - 1)
    return float(_orthonormal_two_electron(n_basis)[p, q, r, s])

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

def active_effective_one_electron_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        x: int, y: int) -> float:
    """Reference implementation of the core-only active mean field."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l = _validate_two_sets(core, act, n_orb)
    x = _check_int("x", x, 0, n_orb - 1)
    y = _check_int("y", y, 0, n_orb - 1)
    if x not in act_l or y not in act_l:
        raise ValueError("x and y must both be active spin orbitals")
    gaa = g_arr - np.transpose(g_arr, (0, 1, 3, 2))
    return float(h_arr[x, y] + sum(gaa[x, k, y, k] for k in core_l))

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

def dyall_inactive_orbital_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
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

def active_reference_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
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

def charged_channel_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
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

def _residual_interaction(g_can, act, n_orb):
    """Eq. (18): v_pr,qs = (1 - d_pA d_rA d_qA d_sA) <pq|rs>; the all-active block is cut."""
    mask = np.zeros(n_orb, dtype=bool)
    mask[list(act)] = True
    proj = np.einsum('p,r,q,s->prqs', mask, mask, mask, mask).astype(float)
    return np.transpose(g_can, (0, 2, 1, 3)) * (1.0 - proj)

def _transition_densities(dets, vecs, n_orb):
    """T[m, p, r] = <Phi_m| p+ r |Phi_0> over the zeroth-order states of one sector."""
    coeff0 = vecs[:, 0]
    index = {d: i for i, d in enumerate(dets)}
    work = np.zeros((n_orb, n_orb, len(dets)))
    for ket, det in enumerate(dets):
        for r in range(n_orb):
            s1, d1 = _annihilate(det, r)
            if s1 == 0:
                continue
            for p in range(n_orb):
                s2, d2 = _create(d1, p)
                if s2 == 0:
                    continue
                bra = index.get(d2)
                if bra is not None:
                    work[p, r, bra] += s1 * s2 * coeff0[ket]
    return np.einsum('mI,prI->mpr', vecs.T, work, optimize=True)

def _mrrpa(h0, g0, g_can, act, n_orb, dets, vals, vecs):
    """Solve the paired MR-RPA problem of Eqs. (S30)-(S33) and (S43).

    Returns (omega, x_plus_y, resid, t_kept): the excitation energies in ascending
    order, the metric-normalised X + Y combination, the residual interaction of
    Eq. (18) and the retained one-body transition densities.
    """
    tall = _transition_densities(dets, vecs, n_orb)
    norms = np.linalg.norm(tall.reshape(len(dets), -1), axis=1)
    keep = [m for m in range(1, len(dets)) if norms[m] > 1e-9]
    if not keep:
        raise ValueError("no excited zeroth-order state carries a transition density")
    omega0 = vals[np.array(keep)] - vals[0]
    t_kept = tall[np.array(keep)]
    resid = _residual_interaction(g_can, act, n_orb)
    a_mat = np.einsum('mpr,prqs,nsq->mn', t_kept, resid, t_kept, optimize=True) \
        + np.diag(omega0)
    b_mat = np.einsum('mpr,prqs,nqs->mn', t_kept, resid, t_kept, optimize=True)
    w1, u1 = np.linalg.eigh(a_mat - b_mat)
    if w1.min() <= 0:
        raise ValueError("the MR-RPA solution is unstable: A - B is not positive definite")
    root = u1 @ np.diag(np.sqrt(w1)) @ u1.T
    w2, tvec = np.linalg.eigh(root @ (a_mat + b_mat) @ root)
    if w2.min() <= 0:
        raise ValueError("the MR-RPA solution is unstable: Omega^2 is not positive")
    omega = np.sqrt(w2)
    x_plus_y = (root @ tvec) / np.sqrt(omega)
    return omega, x_plus_y, resid, t_kept

def _solve_mrrpa(h, g, core, act, vir, n_elec, n_act_elec, *, _cache={}):
    """Solve the MR-RPA problem once and cache it on the whole specification.

    The keyword-only ``_cache`` memoises the result on the whole model
    specification; it is pure memoisation of a deterministic function and
    changes no result.
    """
    key = _cache_key(h, g, core, act, vir, n_elec, n_act_elec)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    n_orb = h.shape[0]
    _hc, g_can, h0, g0, _eps, _heff, _rdm = _dyall_setup(h, g, core, act, vir, n_act_elec)
    dets, vals, vecs = _zeroth_order_states(h0, g0, n_orb, n_elec)
    out = _mrrpa(h0, g0, g_can, act, n_orb, dets, vals, vecs)
    _cache[key] = out
    return out

def mrrpa_excitation_energy(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, index: int) -> float:
    """Reference implementation of the MR-RPA eigenvalue problem, Eqs. (S30)-(S33)."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l, vir_l = _validate_partition(core, act, vir, n_orb)
    n_elec, n_act_elec = _validate_counts(n_elec, n_act_elec, core_l, act_l)
    omega, _xpy, _resid, _tkept = _solve_mrrpa(
        h_arr, g_arr, core_l, act_l, vir_l, n_elec, n_act_elec)
    index = _check_int("index", index, 0, len(omega) - 1)
    return float(omega[index])

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

def _residual_interaction(g_can, act, n_orb):
    """Eq. (18): v_pr,qs = (1 - d_pA d_rA d_qA d_sA) <pq|rs>; the all-active block is cut."""
    mask = np.zeros(n_orb, dtype=bool)
    mask[list(act)] = True
    proj = np.einsum('p,r,q,s->prqs', mask, mask, mask, mask).astype(float)
    return np.transpose(g_can, (0, 2, 1, 3)) * (1.0 - proj)

def _transition_densities(dets, vecs, n_orb):
    """T[m, p, r] = <Phi_m| p+ r |Phi_0> over the zeroth-order states of one sector."""
    coeff0 = vecs[:, 0]
    index = {d: i for i, d in enumerate(dets)}
    work = np.zeros((n_orb, n_orb, len(dets)))
    for ket, det in enumerate(dets):
        for r in range(n_orb):
            s1, d1 = _annihilate(det, r)
            if s1 == 0:
                continue
            for p in range(n_orb):
                s2, d2 = _create(d1, p)
                if s2 == 0:
                    continue
                bra = index.get(d2)
                if bra is not None:
                    work[p, r, bra] += s1 * s2 * coeff0[ket]
    return np.einsum('mI,prI->mpr', vecs.T, work, optimize=True)

def _mrrpa(h0, g0, g_can, act, n_orb, dets, vals, vecs):
    """Solve the paired MR-RPA problem of Eqs. (S30)-(S33) and (S43).

    Returns (omega, x_plus_y, resid, t_kept): the excitation energies in ascending
    order, the metric-normalised X + Y combination, the residual interaction of
    Eq. (18) and the retained one-body transition densities.
    """
    tall = _transition_densities(dets, vecs, n_orb)
    norms = np.linalg.norm(tall.reshape(len(dets), -1), axis=1)
    keep = [m for m in range(1, len(dets)) if norms[m] > 1e-9]
    if not keep:
        raise ValueError("no excited zeroth-order state carries a transition density")
    omega0 = vals[np.array(keep)] - vals[0]
    t_kept = tall[np.array(keep)]
    resid = _residual_interaction(g_can, act, n_orb)
    a_mat = np.einsum('mpr,prqs,nsq->mn', t_kept, resid, t_kept, optimize=True) \
        + np.diag(omega0)
    b_mat = np.einsum('mpr,prqs,nqs->mn', t_kept, resid, t_kept, optimize=True)
    w1, u1 = np.linalg.eigh(a_mat - b_mat)
    if w1.min() <= 0:
        raise ValueError("the MR-RPA solution is unstable: A - B is not positive definite")
    root = u1 @ np.diag(np.sqrt(w1)) @ u1.T
    w2, tvec = np.linalg.eigh(root @ (a_mat + b_mat) @ root)
    if w2.min() <= 0:
        raise ValueError("the MR-RPA solution is unstable: Omega^2 is not positive")
    omega = np.sqrt(w2)
    x_plus_y = (root @ tvec) / np.sqrt(omega)
    return omega, x_plus_y, resid, t_kept

def _solve_screened(h, g, core, act, vir, n_elec, n_act_elec, *, _cache={}):
    """MR-RPA solution plus the screened couplings M, cached on the specification.

    The keyword-only ``_cache`` memoises the result on the whole model
    specification; it is pure memoisation of a deterministic function and
    changes no result.
    """
    key = _cache_key(h, g, core, act, vir, n_elec, n_act_elec)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    n_orb = h.shape[0]
    _hc, g_can, h0, g0, _eps, _heff, _rdm = _dyall_setup(h, g, core, act, vir, n_act_elec)
    dets, vals, vecs = _zeroth_order_states(h0, g0, n_orb, n_elec)
    omega, x_plus_y, resid, t_kept = _mrrpa(h0, g0, g_can, act, n_orb, dets, vals, vecs)
    contracted = np.einsum('mqs,mI->qsI', t_kept, x_plus_y, optimize=True)
    coupling = np.einsum('prqs,qsI->prI', resid, contracted, optimize=True)
    out = (omega, coupling)
    _cache[key] = out
    return out

def static_screened_interaction_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, p: int,
        r: int) -> float:
    """Reference implementation of the statically screened residual interaction."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l, vir_l = _validate_partition(core, act, vir, n_orb)
    n_elec, n_act_elec = _validate_counts(n_elec, n_act_elec, core_l, act_l)
    p = _check_int("p", p, 0, n_orb - 1)
    r = _check_int("r", r, 0, n_orb - 1)
    omega, coupling = _solve_screened(
        h_arr, g_arr, core_l, act_l, vir_l, n_elec, n_act_elec)
    return float(2.0 * np.sum(coupling[p, r, :] ** 2 / omega))

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

def _residual_interaction(g_can, act, n_orb):
    """Eq. (18): v_pr,qs = (1 - d_pA d_rA d_qA d_sA) <pq|rs>; the all-active block is cut."""
    mask = np.zeros(n_orb, dtype=bool)
    mask[list(act)] = True
    proj = np.einsum('p,r,q,s->prqs', mask, mask, mask, mask).astype(float)
    return np.transpose(g_can, (0, 2, 1, 3)) * (1.0 - proj)

def _static_self_energy(h, g, core, act, vir, n_elec, n_act_elec, *, _cache={}):
    """Assemble Sigma_static; cached on the whole specification.

    The keyword-only ``_cache`` memoises the result on the whole model
    specification; it is pure memoisation of a deterministic function and
    changes no result.
    """
    key = _cache_key(h, g, core, act, vir, n_elec, n_act_elec)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    n_orb = h.shape[0]
    h_can, g_can, h0, _g0, _eps, _heff, rdm = _dyall_setup(
        h, g, core, act, vir, n_act_elec)
    resid = _residual_interaction(g_can, act, n_orb)
    vbar = resid - np.transpose(resid, (0, 3, 2, 1))
    dens = np.zeros((n_orb, n_orb))
    for i in core:
        dens[i, i] = 1.0
    if act:
        dens[np.ix_(act, act)] = rdm[np.ix_(act, act)]
    out = (h_can - h0) + np.einsum('prqs,qs->pr', vbar, dens, optimize=True)
    _cache[key] = out
    return out

def static_self_energy_norm(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int) -> float:
    """Reference implementation of the static self-energy and its Frobenius norm."""
    h_arr, g_arr = _validate_integrals(h, g)
    n_orb = h_arr.shape[0]
    core_l, act_l, vir_l = _validate_partition(core, act, vir, n_orb)
    n_elec, n_act_elec = _validate_counts(n_elec, n_act_elec, core_l, act_l)
    stat = _static_self_energy(h_arr, g_arr, core_l, act_l, vir_l, n_elec, n_act_elec)
    return float(np.linalg.norm(stat))

import itertools

import numpy as np

def _pole_cache(key, ordered=None, *, _cache={}):
    """Store or fetch the ordered negative-frequency poles of one specification.

    Pure memoisation of a deterministic result, keyed on the whole model
    specification, so it changes no value the pipeline produces.
    """
    if ordered is not None:
        _cache[key] = ordered
    return _cache.get(key)

def _check_int(name, value, low, high):
    """Return int(value) after checking it is an integer with low <= value <= high."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must lie between {low} and {high}")
    return value

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

def _canonical_rotation(h, gaa, core, act, vir, rdm):
    """Canonicalise the core and virtual Fock blocks SEPARATELY; return (rot, eps)."""
    n_orb = h.shape[0]
    rot = np.eye(n_orb)
    eps_core, u_core = np.linalg.eigh(_fock_block(h, gaa, core, act, rdm, core))
    eps_vir, u_vir = np.linalg.eigh(_fock_block(h, gaa, core, act, rdm, vir))
    rot[np.ix_(core, core)] = u_core
    rot[np.ix_(vir, vir)] = u_vir
    return rot, np.concatenate([eps_core, eps_vir])

def _zeroth_order_states(h0, g0, n_orb, n_elec):
    """Spectrum and eigenvectors of the Dyall H0 over the whole n_elec determinant space."""
    if n_elec < 0 or n_elec > n_orb:
        raise ValueError("the electron count leaves no determinant space")
    dets = _determinants(range(n_orb), n_elec)
    vals, vecs = np.linalg.eigh(_ci_matrix(h0, g0, dets))
    return dets, vals, vecs

def _residual_interaction(g_can, act, n_orb):
    """Eq. (18): v_pr,qs = (1 - d_pA d_rA d_qA d_sA) <pq|rs>; the all-active block is cut."""
    mask = np.zeros(n_orb, dtype=bool)
    mask[list(act)] = True
    proj = np.einsum('p,r,q,s->prqs', mask, mask, mask, mask).astype(float)
    return np.transpose(g_can, (0, 2, 1, 3)) * (1.0 - proj)

def _transition_densities(dets, vecs, n_orb):
    """T[m, p, r] = <Phi_m| p+ r |Phi_0> over the zeroth-order states of one sector."""
    coeff0 = vecs[:, 0]
    index = {d: i for i, d in enumerate(dets)}
    work = np.zeros((n_orb, n_orb, len(dets)))
    for ket, det in enumerate(dets):
        for r in range(n_orb):
            s1, d1 = _annihilate(det, r)
            if s1 == 0:
                continue
            for p in range(n_orb):
                s2, d2 = _create(d1, p)
                if s2 == 0:
                    continue
                bra = index.get(d2)
                if bra is not None:
                    work[p, r, bra] += s1 * s2 * coeff0[ket]
    return np.einsum('mI,prI->mpr', vecs.T, work, optimize=True)

def _mrrpa(h0, g0, g_can, act, n_orb, dets, vals, vecs):
    """Solve the paired MR-RPA problem of Eqs. (S30)-(S33) and (S43).

    Returns (omega, x_plus_y, resid, t_kept): the excitation energies in ascending
    order, the metric-normalised X + Y combination, the residual interaction of
    Eq. (18) and the retained one-body transition densities.
    """
    tall = _transition_densities(dets, vecs, n_orb)
    norms = np.linalg.norm(tall.reshape(len(dets), -1), axis=1)
    keep = [m for m in range(1, len(dets)) if norms[m] > 1e-9]
    if not keep:
        raise ValueError("no excited zeroth-order state carries a transition density")
    omega0 = vals[np.array(keep)] - vals[0]
    t_kept = tall[np.array(keep)]
    resid = _residual_interaction(g_can, act, n_orb)
    a_mat = np.einsum('mpr,prqs,nsq->mn', t_kept, resid, t_kept, optimize=True) \
        + np.diag(omega0)
    b_mat = np.einsum('mpr,prqs,nqs->mn', t_kept, resid, t_kept, optimize=True)
    w1, u1 = np.linalg.eigh(a_mat - b_mat)
    if w1.min() <= 0:
        raise ValueError("the MR-RPA solution is unstable: A - B is not positive definite")
    root = u1 @ np.diag(np.sqrt(w1)) @ u1.T
    w2, tvec = np.linalg.eigh(root @ (a_mat + b_mat) @ root)
    if w2.min() <= 0:
        raise ValueError("the MR-RPA solution is unstable: Omega^2 is not positive")
    omega = np.sqrt(w2)
    x_plus_y = (root @ tvec) / np.sqrt(omega)
    return omega, x_plus_y, resid, t_kept

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

def _dyson_spectrum(stat, coupling, omega, add, w_add, rem, w_rem, n_orb):
    """Poles and spectral weights of G, as the resolvent of one real symmetric matrix.

    The 12-by-12 corner of ([G0]^-1 - Sigma)^-1 is the corner of the resolvent of a
    supermatrix that carries the spin orbitals, the orthogonal complement holding the
    G0 channels, and one rank-one channel per (MR-RPA mode, G0 pole) pair.
    """
    chan_add = np.einsum('prI,mr->pIm', coupling, add, optimize=True).reshape(n_orb, -1)
    pole_add = (omega[:, None] + w_add[None, :]).reshape(-1)
    chan_rem = np.einsum('prI,mr->pIm', coupling, rem, optimize=True).reshape(n_orb, -1)
    pole_rem = -(omega[:, None] + w_rem[None, :]).reshape(-1)
    chan = np.concatenate([chan_add, chan_rem], axis=1)
    pole = np.concatenate([pole_add, pole_rem])
    amp = np.concatenate([add.T, rem.T], axis=1)
    nu0 = np.concatenate([w_add, -w_rem])
    perp = np.eye(len(nu0)) - amp.T @ amp
    ev_perp, vec_perp = np.linalg.eigh(perp)
    comp = vec_perp[:, ev_perp > 0.5].T
    basis = np.concatenate([amp, comp], axis=0)
    g0_mat = basis @ np.diag(nu0) @ basis.T
    blk_a = g0_mat[:n_orb, :n_orb]
    blk_b = g0_mat[:n_orb, n_orb:]
    blk_c = g0_mat[n_orb:, n_orb:]
    n_comp = blk_c.shape[0]
    n_chan = len(pole)
    dim = n_orb + n_comp + n_chan
    sup = np.zeros((dim, dim))
    sup[:n_orb, :n_orb] = blk_a + stat
    sup[:n_orb, n_orb:n_orb + n_comp] = blk_b
    sup[n_orb:n_orb + n_comp, :n_orb] = blk_b.T
    sup[n_orb:n_orb + n_comp, n_orb:n_orb + n_comp] = blk_c
    sup[:n_orb, n_orb + n_comp:] = chan
    sup[n_orb + n_comp:, :n_orb] = chan.T
    sup[n_orb + n_comp:, n_orb + n_comp:] = np.diag(pole)
    vals, vecs = np.linalg.eigh(sup)
    return vals, np.sum(vecs[:n_orb, :] ** 2, axis=0)

def mrgw_satellite_energy(n_basis: int | None = None, core: list[int] | None = None,
        act: list[int] | None = None, vir: list[int] | None = None,
        n_elec: int | None = None, n_act_elec: int | None = None,
        rank: int | None = None) -> float:
    """Reference implementation: every earlier sub-problem, assembled into the answer."""
    hartree_to_ev = 27.211386245988
    frozen_model = (12, [4, 5], [1, 2, 3, 6, 7, 8], [0, 9, 10, 11], 5, 3)
    spec = (n_basis, core, act, vir, n_elec, n_act_elec)
    if all(item is None for item in spec):
        n_basis, core, act, vir, n_elec, n_act_elec = frozen_model
    elif any(item is None for item in spec):
        raise ValueError("give the whole model specification or none of it")
    rank = 3 if rank is None else rank
    n_basis = _check_int("n_basis", n_basis, 2, 12)
    core, act, vir = _validate_partition(core, act, vir, n_basis)
    n_elec, n_act_elec = _validate_counts(n_elec, n_act_elec, core, act)
    rank = _check_int("rank", rank, 1, 10 ** 7)
    if n_elec + 1 > n_basis or n_elec - 1 < 0:
        raise ValueError("a neighbouring particle-number sector is empty")
    key = (n_basis, tuple(core), tuple(act), tuple(vir), n_elec, n_act_elec)
    hit = _pole_cache(key)
    if hit is not None:
        if len(hit) < rank:
            raise ValueError("fewer negative-frequency poles than the requested rank")
        return float(hit[rank - 1] * hartree_to_ev)

    # (a) the integrals of the generated model, element by element
    one_e = np.array([[orthonormal_one_electron_element(n_basis, p, q)
                       for q in range(n_basis)] for p in range(n_basis)])
    two_e = np.array([[[[orthonormal_two_electron_element(n_basis, p, q, r, s)
                         for s in range(n_basis)] for r in range(n_basis)]
                       for q in range(n_basis)] for p in range(n_basis)])
    gaa = two_e - np.transpose(two_e, (0, 1, 3, 2))

    # (b) Eq. (15) active effective one-electron operator and the interacting reference
    heff = np.array([[active_effective_one_electron_element(
        one_e, two_e, core, act, x, y) for y in act] for x in act])
    dets_a, vals_a, vecs_a = _active_ci(one_e, two_e, act, heff, n_act_elec)
    rdm = (_active_rdm(dets_a, vecs_a[:, 0], act, n_basis) if act
           else np.zeros((n_basis, n_basis)))
    e_act = active_reference_energy(one_e, two_e, core, act, n_act_elec)
    if abs(e_act - float(vals_a[0])) > 1e-10:
        raise ValueError("the active reference energy disagrees with the active CI")

    # Eq. (14) inactive orbital energies, and the separate core/virtual canonicalisation
    eps = np.array([dyall_inactive_orbital_energy(
        one_e, two_e, core, act, vir, n_act_elec, k)
        for k in range(len(core) + len(vir))])
    rot, eps_check = _canonical_rotation(one_e, gaa, core, act, vir, rdm)
    if np.max(np.abs(eps_check - eps)) > 1e-10:
        raise ValueError("the inactive orbital energies disagree with the Fock blocks")
    h_can = rot.T @ one_e @ rot
    g_can = np.einsum('ap,bq,cr,ds,abcd->pqrs', rot, rot, rot, rot, two_e, optimize=True)
    h0 = np.zeros((n_basis, n_basis))
    for k, orb in enumerate(core + vir):
        h0[orb, orb] = eps[k]
    g0 = np.zeros_like(g_can)
    if act:
        h0[np.ix_(act, act)] = heff
        g0[np.ix_(act, act, act, act)] = g_can[np.ix_(act, act, act, act)]

    dets, vals, vecs = _zeroth_order_states(h0, g0, n_basis, n_elec)
    if abs(float(vals[0]) - (float(np.sum(eps[:len(core)])) + e_act)) > 1e-8:
        raise ValueError("the zeroth-order reference energy is not the sum of the core "
                         "orbital energies and the active reference energy")

    # (c) the retained G0 channels, with their energies taken from the channel step
    add, w_add_ref, rem, w_rem_ref = _charged_channels(
        h0, g0, n_basis, n_elec, dets, vals, vecs)
    w_add = np.array([charged_channel_energy(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, 1, k)
        for k in range(add.shape[0])])
    w_rem = np.array([charged_channel_energy(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, -1, k)
        for k in range(rem.shape[0])])
    if (np.max(np.abs(w_add - w_add_ref)) > 1e-10
            or np.max(np.abs(w_rem - w_rem_ref)) > 1e-10):
        raise ValueError("the charged-sector channel energies disagree with the spectrum")

    # (d) the MR-RPA screening, with its spectrum taken from the excitation step
    omega_ref, x_plus_y, resid, t_kept = _mrrpa(
        h0, g0, g_can, act, n_basis, dets, vals, vecs)
    omega = np.array([mrrpa_excitation_energy(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, k)
        for k in range(len(omega_ref))])
    if np.max(np.abs(omega - omega_ref)) > 1e-10:
        raise ValueError("the MR-RPA excitation energies disagree with the spectrum")
    contracted = np.einsum('mqs,mI->qsI', t_kept, x_plus_y, optimize=True)
    coupling = np.einsum('prqs,qsI->prI', resid, contracted, optimize=True)
    probe = core[0] if core else 0
    screened = static_screened_interaction_element(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, probe, probe)
    if abs(screened - float(2.0 * np.sum(coupling[probe, probe, :] ** 2 / omega))) > 1e-8:
        raise ValueError("the screened interaction disagrees with the assembled couplings")

    # (e) the static self-energy, checked against the static self-energy step
    vbar = resid - np.transpose(resid, (0, 3, 2, 1))
    dens = np.zeros((n_basis, n_basis))
    for i in core:
        dens[i, i] = 1.0
    if act:
        dens[np.ix_(act, act)] = rdm[np.ix_(act, act)]
    stat = (h_can - h0) + np.einsum('prqs,qs->pr', vbar, dens, optimize=True)
    if abs(static_self_energy_norm(one_e, two_e, core, act, vir, n_elec,
                                           n_act_elec)
           - float(np.linalg.norm(stat))) > 1e-8:
        raise ValueError("the static self-energy disagrees with its own norm")

    # the Dyson equation, and the selection rule
    poles, weights = _dyson_spectrum(stat, coupling, omega, add, w_add, rem, w_rem,
                                     n_basis)
    negative = poles < 0
    order = np.argsort(-weights[negative])
    ordered = poles[negative][order]
    _pole_cache(key, ordered)
    if len(ordered) < rank:
        raise ValueError("fewer negative-frequency poles than the requested rank")
    return float(ordered[rank - 1] * hartree_to_ev)
SCICODE_GOLD_EOF
