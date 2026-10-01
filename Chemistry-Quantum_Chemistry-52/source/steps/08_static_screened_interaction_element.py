"""
Static limit of the screened interaction built from the MR-RPA couplings.

This step contracts the screened couplings M of the multi-reference construction with the

MR-RPA excitation energies to give one element of the statically screened interaction,

W[p, r] = 2 sum_I M[p, r, I]^2 / Omega_I.



Scientific background.  Once the MR-RPA problem is solved, the residual interaction is

dressed by the response of the correlated reference.  The screened coupling of mode I is



M[p, r, I] = sum_{q,s} v_{pr,qs} sum_u T[u, q, s] R[u, I] ,



with v the residual interaction of the Dyall partition, T[u, q, s] = <Phi_u| q^+ s |Phi_0>

the one-body transition densities of the retained zeroth-order states, and R = X + Y the

sum of the two MR-RPA amplitude blocks in the metric normalisation

(X + Y)^T (X - Y) = 1.  The collapse of the general two-term expression, one term in X and

one in Y, into the single contraction with X + Y holds only for real orbitals, where the

residual interaction obeys v_{pr,qs} = v_{pr,sq}; keeping X alone instead, as an

independent-particle picture would suggest, is a different and measurably worse

approximation.



The quantity returned here is the zero-frequency limit of the dynamically screened

interaction that the self-energy carries,



W[p, r] = 2 sum_I M[p, r, I]^2 / Omega_I ,



the factor two coming from the two resonant and antiresonant halves of the paired

spectrum.  It is a sum of squares divided by positive eigenvalues, so it is invariant

under the arbitrary sign each MR-RPA eigenvector carries and is non-negative on the

diagonal; its trace measures the total screening strength the residual interaction

acquires from the correlated reference.  Because M inherits the residual projector, the

all-active part of the interaction contributes nothing to it.



Raises ValueError when h is not a non-empty square array, when g does not match h in

shape, when either holds a non-finite entry, when core, act and vir do not partition the

spin orbitals exactly once, when n_elec is not n_act_elec plus the number of core spin

orbitals, when n_act_elec exceeds the number of active spin orbitals, when no zeroth-order

state carries a transition density, when the MR-RPA solution is unstable, and when p or r

is not an integer in the range 0 to n - 1.

Returns
-------
float, the element W[p, r] = 2 sum_I M[p, r, I]^2 / Omega_I of the statically screened residual interaction, in hartree, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_screened_interaction_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
        vir: list[int], n_elec: int, n_act_elec: int, p: int,
        r: int) -> float:
    """One element of the statically screened residual interaction.

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
    p : int
        First spin-orbital index of the requested element, 0 <= p < n.
    r : int
        Second spin-orbital index of the requested element, 0 <= r < n.

    Returns
    -------
    float
        W[p, r] = 2 sum_I M[p, r, I]^2 / Omega_I in hartree, with M the screened couplings
        of the MR-RPA modes and Omega their excitation energies.
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

def _oracle_static_screened_interaction_element(h: np.ndarray, g: np.ndarray, core: list[int], act: list[int],
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 2, 1, 1, 1)',
            'gold_call': '_oracle_static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 2, 1, 1, 1)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 2, 1, 1, 3)',
            'gold_call': '_oracle_static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 2, 1, 1, 3)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'static_screened_interaction_element(H6.copy(), G6.copy(), [2, 3], [1, 4], [0, 5], 4, 2, 5, 5)',
            'gold_call': '_oracle_static_screened_interaction_element(H6.copy(), G6.copy(), [2, 3], [1, 4], [0, 5], 4, 2, 5, 5)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n",
            'call': 'static_screened_interaction_element(H6.copy(), G6.copy(), [3], [1, 2, 4], [0, 5], 3, 2, 0, 5)',
            'gold_call': '_oracle_static_screened_interaction_element(H6.copy(), G6.copy(), [3], [1, 2, 4], [0, 5], 3, 2, 0, 5)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    return sum(static_screened_interaction_element(H8, G8, [3, 4], [1, 2, 5, 6], [0, 7],\n                                                   5, 3, k, k) for k in range(8))\ndef run_gold():\n    return sum(_oracle_static_screened_interaction_element(H8, G8, [3, 4], [1, 2, 5, 6],\n                                                           [0, 7], 5, 3, k, k)\n               for k in range(8))\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum(\'ap,bq,cr,ds,abcd->pqrs\', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n\ndef run_model():\n    return static_screened_interaction_element(H12, G12, CORE12, ACT12, VIR12, 5, 3, 4, 4)\ndef run_gold():\n    value = _oracle_static_screened_interaction_element(H12, G12, CORE12, ACT12, VIR12, 5, 3, 4, 4)\n    assert abs(value - (0.022872824718724493)) <= 1e-10 * max(1.0, abs(0.022872824718724493)),         "the frozen anchor 0.022872824718724493 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH12, G12 = _fx_system(12)\nCORE12 = [4, 5]\nACT12 = [1, 2, 3, 6, 7, 8]\nVIR12 = [0, 9, 10, 11]\n",
            'call': 'static_screened_interaction_element(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 5, 3, 11, 11)',
            'gold_call': '_oracle_static_screened_interaction_element(H12.copy(), G12.copy(), CORE12.copy(), ACT12.copy(), VIR12.copy(), 5, 3, 11, 11)',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 2, 1, 4, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 2, 1, 4, 0)\n    except ValueError: return 1\n    except Exception:  return 2\n    return 0\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    static_screened_interaction_element(H4, G4, [1], [0, 2], [2, 3], 2, 1, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_static_screened_interaction_element(H4, G4, [1], [0, 2], [2, 3], 2, 1, 0, 0)\n    except ValueError: return 1\n    except Exception:  return 2\n    return 0\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': "import math\nimport numpy as np\nfrom scipy.special import erf\n\n\ndef _fx_boys(x):\n    if x <= 1e-12:\n        return 1.0\n    return 0.5 * math.sqrt(math.pi / x) * float(erf(math.sqrt(x)))\n\n\ndef _fx_system(n):\n    idx = np.arange(n, dtype=float)\n    ang = 2.4 * idx\n    cen = np.stack([1.35 * np.cos(ang), 1.35 * np.sin(ang), 0.62 * idx], axis=1)\n    alp = 0.42 + 0.06 * idx\n    nrm = (2.0 * alp / np.pi) ** 0.75\n    ovl = np.zeros((n, n)); kin = np.zeros((n, n)); nuc = np.zeros((n, n))\n    tot = np.zeros((n, n)); pre = np.zeros((n, n)); prd = np.zeros((n, n, 3))\n    for a in range(n):\n        for b in range(n):\n            pa, pb = alp[a], alp[b]\n            s = pa + pb\n            d2 = float(np.sum((cen[a] - cen[b]) ** 2))\n            k = math.exp(-pa * pb / s * d2)\n            pab = (pa * cen[a] + pb * cen[b]) / s\n            tot[a, b] = s; pre[a, b] = k; prd[a, b] = pab\n            base = (np.pi / s) ** 1.5 * k\n            ovl[a, b] = nrm[a] * nrm[b] * base\n            kin[a, b] = nrm[a] * nrm[b] * (pa * pb / s * (3.0 - 2.0 * pa * pb / s * d2)) * base\n            acc = 0.0\n            for cc in cen:\n                acc += -0.55 * 2 * np.pi / s * k * _fx_boys(s * float(np.sum((pab - cc) ** 2)))\n            nuc[a, b] = nrm[a] * nrm[b] * acc\n    eri = np.zeros((n, n, n, n))\n    for a in range(n):\n        for b in range(n):\n            for c in range(n):\n                for d in range(n):\n                    s1, s2 = tot[a, b], tot[c, d]\n                    t = s1 * s2 / (s1 + s2) * float(np.sum((prd[a, b] - prd[c, d]) ** 2))\n                    val = 2 * np.pi ** 2.5 / (s1 * s2 * np.sqrt(s1 + s2)) \\\n                        * pre[a, b] * pre[c, d] * _fx_boys(t)\n                    eri[a, b, c, d] = nrm[a] * nrm[b] * nrm[c] * nrm[d] * val\n    w, u = np.linalg.eigh(ovl)\n    x = u @ np.diag(w ** -0.5) @ u.T\n    hh = x.T @ (kin + nuc) @ x\n    ch = np.einsum('ap,bq,cr,ds,abcd->pqrs', x, x, x, x, eri, optimize=True)\n    return hh, np.transpose(ch, (0, 2, 1, 3))\n\nH4, G4 = _fx_system(4)\nH6, G6 = _fx_system(6)\nH8, G8 = _fx_system(8)\n\ndef run_model():\n    try:    static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 4, 3, 0, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_static_screened_interaction_element(H4, G4, [1], [0, 2], [3], 4, 3, 0, 0)\n    except ValueError: return 1\n    except Exception:  return 2\n    return 0\n",
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
