"""
Multi-reference GW satellite energy: the closed pipeline and the final answer.

This final step assembles every earlier sub-problem into the generalised Dyson equation of

the multi-reference GW construction, collects the poles of the interacting Green's

function together with their spectral weights, and returns the frequency in electronvolts

of the pole that is `rank`-th by spectral weight among the poles at negative frequency.

Called with no arguments it returns that value for the frozen twelve-spin-orbital model at

rank three.



Scientific background.  The construction replaces the single-determinant reference of

ordinary GW by an interacting one.  The pipeline has five stages and each is an earlier

step.  (a) The model is generated from its own rule -- twelve normalised spherical

Gaussians on a helix, a one-electron operator of kinetic energy plus attraction to point

charges at all twelve centres, the ordinary Coulomb repulsion integrals, symmetric Loewdin

orthonormalisation with the input order kept -- so the one- and two-electron integrals come

element by element from the two integral steps.  (b) The Dyall partition is built: the

core-dressed effective one-electron operator heff on the active block, and the orbital

energies of the separately canonicalised core and virtual generalised Fock blocks on the

inactive diagonal.  Its lowest N-electron eigenstate is the reference, and the identity

E_0 = sum over core of eps + E(active reference) is checked here before anything else is

assembled.  (c) The zeroth-order Green's function G0 is the Lehmann sum over the retained

charged-sector channels of that same zeroth-order Hamiltonian, so it has 19 addition and

17 removal poles on the frozen model rather than twelve orbital energies.  (d) The

residual interaction is screened at the MR-RPA level over all four blocks and contracted

back to the screened couplings M.  (e) The self-energy is the static piece plus the

frequency-dependent piece built from M and the G0 channels.



Because G0 and Sigma are both finite sums of simple poles, the Dyson equation

G(w) = ([G0(w)]^(-1) - Sigma(w))^(-1) is solved exactly, with no broadening, no iteration

and no root search: G is the n-by-n corner of the resolvent of one real symmetric

supermatrix whose blocks are the n spin orbitals, the orthogonal complement carrying the

G0 channels, and one channel per (MR-RPA mode, G0 pole) pair.  Its eigenvalues are the

poles of G and the summed squares of the first n components of each eigenvector are the

spectral weights, which sum to n exactly.  On the frozen model that supermatrix is 4248 by

4248 and the sum rule returns 12.



The selection rule is the part that decides the answer.  The poles at negative frequency

are ordered by DECREASING spectral weight and the rank-th of that ordering is taken.  The

two criteria do not commute: ranking the whole spectrum first and only then restricting to

negative frequency picks a different pole, because the global weight leaders are not all on

the removal side.  And the wanted pole is not extremal in either sense: on the frozen model

it is neither the lowest-lying pole nor the heaviest one, it carries weight 0.4749, and its

neighbours in the weight ordering sit 0.87 above and 0.37 below.



Raises ValueError when the model specification is given only in part, when n_basis is not

an integer between 2 and 12, when core, act and vir do not partition the spin orbitals

exactly once, when n_elec is not n_act_elec plus the number of core spin orbitals, when

n_act_elec exceeds the number of active spin orbitals, when a neighbouring

particle-number sector is empty, when the active reference energy, the inactive orbital

energies, the channel energies, the MR-RPA spectrum, the screened interaction or the

static self-energy disagree with the value the corresponding sub-problem returns, when the

zeroth-order reference energy is not the sum of the core orbital energies and the active

reference energy, when the MR-RPA solution is unstable, when rank is not a positive

integer, and when fewer than rank poles lie at negative frequency.

Returns
-------
float, the frequency in electronvolts of the pole of the interacting Green's function that is rank-th by spectral weight among the poles at negative frequency, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mrgw_satellite_energy(n_basis: int | None = None, core: list[int] | None = None,
        act: list[int] | None = None, vir: list[int] | None = None,
        n_elec: int | None = None, n_act_elec: int | None = None,
        rank: int | None = None) -> float:
    """Frequency of a selected pole of the multi-reference GW Green's function, in eV.

    Parameters
    ----------
    n_basis : int, optional
        Number of spherical Gaussian primitives in the helix, 2 <= n_basis <= 12.  Leave
        every model argument unset to use the frozen twelve-spin-orbital model.
    core, act, vir : sequence of int, optional
        Occupied inactive, active and unoccupied inactive spin-orbital indices; together
        they must partition range(n_basis) exactly once.
    n_elec : int, optional
        Total number of electrons, equal to n_act_elec plus len(core).
    n_act_elec : int, optional
        Number of electrons in the active space.
    rank : int, optional
        Position in the weight ordering of the negative-frequency poles, counting from
        one.  Defaults to 3, the satellite the frozen model asks for.

    Returns
    -------
    float
        The frequency in electronvolts of the rank-th heaviest pole of the interacting
        Green's function among those at negative frequency.  With no arguments this is
        the multi-reference GW satellite energy of the frozen model.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_mrgw_satellite_energy(n_basis: int | None = None, core: list[int] | None = None,
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
    one_e = np.array([[_oracle_orthonormal_one_electron_element(n_basis, p, q)
                       for q in range(n_basis)] for p in range(n_basis)])
    two_e = np.array([[[[_oracle_orthonormal_two_electron_element(n_basis, p, q, r, s)
                         for s in range(n_basis)] for r in range(n_basis)]
                       for q in range(n_basis)] for p in range(n_basis)])
    gaa = two_e - np.transpose(two_e, (0, 1, 3, 2))

    # (b) Eq. (15) active effective one-electron operator and the interacting reference
    heff = np.array([[_oracle_active_effective_one_electron_element(
        one_e, two_e, core, act, x, y) for y in act] for x in act])
    dets_a, vals_a, vecs_a = _active_ci(one_e, two_e, act, heff, n_act_elec)
    rdm = (_active_rdm(dets_a, vecs_a[:, 0], act, n_basis) if act
           else np.zeros((n_basis, n_basis)))
    e_act = _oracle_active_reference_energy(one_e, two_e, core, act, n_act_elec)
    if abs(e_act - float(vals_a[0])) > 1e-10:
        raise ValueError("the active reference energy disagrees with the active CI")

    # Eq. (14) inactive orbital energies, and the separate core/virtual canonicalisation
    eps = np.array([_oracle_dyall_inactive_orbital_energy(
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
    w_add = np.array([_oracle_charged_channel_energy(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, 1, k)
        for k in range(add.shape[0])])
    w_rem = np.array([_oracle_charged_channel_energy(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, -1, k)
        for k in range(rem.shape[0])])
    if (np.max(np.abs(w_add - w_add_ref)) > 1e-10
            or np.max(np.abs(w_rem - w_rem_ref)) > 1e-10):
        raise ValueError("the charged-sector channel energies disagree with the spectrum")

    # (d) the MR-RPA screening, with its spectrum taken from the excitation step
    omega_ref, x_plus_y, resid, t_kept = _mrrpa(
        h0, g0, g_can, act, n_basis, dets, vals, vecs)
    omega = np.array([_oracle_mrrpa_excitation_energy(
        one_e, two_e, core, act, vir, n_elec, n_act_elec, k)
        for k in range(len(omega_ref))])
    if np.max(np.abs(omega - omega_ref)) > 1e-10:
        raise ValueError("the MR-RPA excitation energies disagree with the spectrum")
    contracted = np.einsum('mqs,mI->qsI', t_kept, x_plus_y, optimize=True)
    coupling = np.einsum('prqs,qsI->prI', resid, contracted, optimize=True)
    probe = core[0] if core else 0
    screened = _oracle_static_screened_interaction_element(
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
    if abs(_oracle_static_self_energy_norm(one_e, two_e, core, act, vir, n_elec,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    return mrgw_satellite_energy()\ndef run_gold():\n    value = _oracle_mrgw_satellite_energy()\n    assert abs(value - (-4.094284780510972)) <= 1e-10 * max(1.0, abs(-4.094284780510972)),         "the frozen anchor -4.094284780510972 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    return mrgw_satellite_energy(12, [4, 5], [1, 2, 3, 6, 7, 8], [0, 9, 10, 11], 5, 3, 3)\ndef run_gold():\n    value = _oracle_mrgw_satellite_energy(12, [4, 5], [1, 2, 3, 6, 7, 8], [0, 9, 10, 11], 5, 3, 3)\n    assert abs(value - (-4.094284780510972)) <= 1e-10 * max(1.0, abs(-4.094284780510972)),         "the frozen anchor -4.094284780510972 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n\ndef run_model():\n    return mrgw_satellite_energy(12, [4, 5], [1, 2, 3, 6, 7, 8], [0, 9, 10, 11], 5, 3, 1)\ndef run_gold():\n    value = _oracle_mrgw_satellite_energy(12, [4, 5], [1, 2, 3, 6, 7, 8], [0, 9, 10, 11], 5, 3, 1)\n    assert abs(value - (-2.4060884905318525)) <= 1e-10 * max(1.0, abs(-2.4060884905318525)),         "the frozen anchor -2.4060884905318525 is not reproduced"\n    return value\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'mrgw_satellite_energy(4, [0], [1, 2], [3], 2, 1, 3)',
            'gold_call': '_oracle_mrgw_satellite_energy(4, [0], [1, 2], [3], 2, 1, 3)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'mrgw_satellite_energy(6, [0, 1], [2, 3, 4], [5], 4, 2, 2)',
            'gold_call': '_oracle_mrgw_satellite_energy(6, [0, 1], [2, 3, 4], [5], 4, 2, 2)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'mrgw_satellite_energy(6, [0], [1, 2, 3], [4, 5], 3, 2, 1)',
            'gold_call': '_oracle_mrgw_satellite_energy(6, [0], [1, 2, 3], [4, 5], 3, 2, 1)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'mrgw_satellite_energy(8, [0, 1], [2, 3, 4, 5], [6, 7], 5, 3, 3)',
            'gold_call': '_oracle_mrgw_satellite_energy(8, [0, 1], [2, 3, 4, 5], [6, 7], 5, 3, 3)',
        },
        {
            'setup': 'import numpy as np\n',
            'call': 'mrgw_satellite_energy(5, [0], [1, 2, 3], [4], 3, 2, 4)',
            'gold_call': '_oracle_mrgw_satellite_energy(5, [0], [1, 2, 3], [4], 3, 2, 4)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:    mrgw_satellite_energy(6); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_mrgw_satellite_energy(6); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:    mrgw_satellite_energy(6, [2, 3], [1, 4], [0, 5], 4, 2, 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_mrgw_satellite_energy(6, [2, 3], [1, 4], [0, 5], 4, 2, 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:    mrgw_satellite_energy(4, [1], [0, 2], [2], 2, 1, 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_mrgw_satellite_energy(4, [1], [0, 2], [2], 2, 1, 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:    mrgw_satellite_energy(4, [0], [1, 2], [3], 2, 1, 100000); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_mrgw_satellite_energy(4, [0], [1, 2], [3], 2, 1, 100000); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:    mrgw_satellite_energy(4, [0], [1, 2], [3], 4, 1, 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_mrgw_satellite_energy(4, [0], [1, 2], [3], 4, 1, 1); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try:    mrgw_satellite_energy(4, [0], [1, 2], [3], 2, 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\ndef run_gold():\n    try:    _oracle_mrgw_satellite_energy(4, [0], [1, 2], [3], 2, 1, 0); return 0\n    except ValueError: return 1\n    except Exception:  return 2\n',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
