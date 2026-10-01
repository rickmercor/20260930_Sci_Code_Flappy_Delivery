#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def _switch(d, r1, r2):
    """Smooth 1 -> 0 coordination switch and its derivative with respect to d."""
    x = np.clip((d - r1) / (r2 - r1), 0.0, 1.0)
    f = 0.5 * (1.0 + np.cos(np.pi * x))
    inside = (d > r1) & (d < r2)
    df = np.where(inside,
                  -0.5 * np.pi / (r2 - r1) * np.sin(np.pi * x), 0.0)
    return f, df


def surface_energy_forces(ads: "np.ndarray", slab: "np.ndarray",
                                  ads_species: "np.ndarray",
                                  slab_species: "np.ndarray",
                                  cell: "np.ndarray", de: "np.ndarray",
                                  alpha: "np.ndarray", r0: "np.ndarray",
                                  r_bond: "np.ndarray",
                                  valence: "np.ndarray", k_coord: "np.ndarray",
                                  f1: float = 1.3, f2: float = 2.0,
                                  r_in: float = 4.0,
                                  r_out: float = 5.0) -> tuple:
    ads = np.asarray(ads, dtype=float)
    slab = np.asarray(slab, dtype=float)
    ads_species = np.asarray(ads_species, dtype=int)
    slab_species = np.asarray(slab_species, dtype=int)
    de = np.asarray(de, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    r_bond = np.asarray(r_bond, dtype=float)
    valence = np.asarray(valence, dtype=float).reshape(-1)
    k_coord = np.asarray(k_coord, dtype=float).reshape(-1)
    cell = np.asarray(cell, dtype=float).reshape(-1)

    if ads.ndim not in (2, 3) or ads.shape[-1] != 3 or ads.shape[-2] < 1:
        raise ValueError("ads must have shape (N, 3) or (M, N, 3) with N >= 1")
    if slab.ndim != 2 or slab.shape[-1] != 3:
        raise ValueError("slab must have shape (F, 3)")
    if ads_species.shape != (ads.shape[-2],):
        raise ValueError("ads_species must have one entry per adsorbate atom")
    if slab_species.shape != (slab.shape[0],):
        raise ValueError("slab_species must have one entry per slab atom")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    for name, arr in (("de", de), ("alpha", alpha), ("r0", r0),
                      ("r_bond", r_bond)):
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(name + " must be a square matrix")
        if not np.allclose(arr, arr.T, rtol=0.0, atol=1e-12):
            raise ValueError(name + " must be symmetric")
    if not (de.shape == alpha.shape == r0.shape == r_bond.shape):
        raise ValueError("de, alpha, r0 and r_bond must have the same shape")
    if np.any(alpha <= 0.0) or np.any(r0 <= 0.0):
        raise ValueError("alpha and r0 must be positive")
    if np.any(r_bond < 0.0):
        raise ValueError("r_bond must be non-negative")
    nsp = de.shape[0]
    if valence.shape != (nsp,) or k_coord.shape != (nsp,):
        raise ValueError("valence and k_coord must have one entry per species")
    if np.any(k_coord < 0.0) or np.any(valence < 0.0):
        raise ValueError("valence and k_coord must be non-negative")
    if not (f2 > f1 > 0.0):
        raise ValueError("f2 must exceed f1 and both must be positive")
    if not (r_out > r_in > 0.0):
        raise ValueError("r_out must exceed r_in and both must be positive")
    if ads_species.size and (ads_species.min() < 0 or ads_species.max() >= nsp):
        raise ValueError("ads_species out of range of the parameter matrices")
    if slab_species.size and (slab_species.min() < 0 or slab_species.max() >= nsp):
        raise ValueError("slab_species out of range of the parameter matrices")

    flat = ads.reshape(-1, ads.shape[-2], 3)
    nimg, n = flat.shape[0], flat.shape[1]
    lx, ly = float(cell[0]), float(cell[1])

    energy = np.zeros(nimg)
    forces = np.zeros((nimg, n, 3))

    def _wrap(dvec):
        dvec[..., 0] -= np.round(dvec[..., 0] / lx) * lx
        dvec[..., 1] -= np.round(dvec[..., 1] / ly) * ly
        return dvec

    def _morse(d, sp_a, sp_b):
        """Morse pair energy tapered to zero between r_in and r_out."""
        ex = np.exp(-alpha[sp_a, sp_b] * (d - r0[sp_a, sp_b]))
        raw = de[sp_a, sp_b] * (ex * ex - 2.0 * ex)
        draw = 2.0 * de[sp_a, sp_b] * alpha[sp_a, sp_b] * (ex - ex * ex)
        cut, dcut = _switch(d, r_in, r_out)
        return raw * cut, draw * cut + raw * dcut

    if n > 1:
        ii, jj = np.triu_indices(n, k=1)
        dvec = _wrap(flat[:, ii, :] - flat[:, jj, :])
        d = np.maximum(np.sqrt(np.sum(dvec * dvec, axis=-1)), 1e-12)
        sa, sb = ads_species[ii], ads_species[jj]
        e, dedr = _morse(d, sa, sb)
        energy += e.sum(axis=1)

        rb = r_bond[sa, sb]
        live = rb > 0.0
        rb = np.where(live, rb, 1.0)
        f, df = _switch(d, f1 * rb, f2 * rb)
        f = np.where(live, f, 0.0)
        df = np.where(live, df, 0.0)
        coord = np.zeros((nimg, n))
        np.add.at(coord, (slice(None), ii), f)
        np.add.at(coord, (slice(None), jj), f)
        excess = np.maximum(coord - valence[ads_species][None, :], 0.0)
        energy += np.sum(k_coord[ads_species][None, :] * excess ** 2, axis=1)
        dedc = 2.0 * k_coord[ads_species][None, :] * excess
        dedr = dedr + (dedc[:, ii] + dedc[:, jj]) * df

        grad = (dedr / d)[..., None] * dvec
        np.add.at(forces, (slice(None), ii), -grad)
        np.add.at(forces, (slice(None), jj), grad)

    if slab.shape[0] > 0:
        dvec = _wrap(flat[:, :, None, :] - slab[None, None, :, :])
        d = np.maximum(np.sqrt(np.sum(dvec * dvec, axis=-1)), 1e-12)
        e, dedr = _morse(d, ads_species[:, None], slab_species[None, :])
        energy += e.sum(axis=(1, 2))
        forces -= np.sum((dedr / d)[..., None] * dvec, axis=2)

    if ads.ndim == 2:
        return float(energy[0]), forces[0]
    return energy.reshape(ads.shape[:-2]), forces.reshape(ads.shape)

import numpy as np


def _mic_delta(vec, cell):
    """Minimum-image displacement in the two periodic in-plane directions."""
    out = np.array(vec, dtype=float, copy=True)
    out[..., 0] -= np.round(out[..., 0] / cell[0]) * cell[0]
    out[..., 1] -= np.round(out[..., 1] / cell[1]) * cell[1]
    return out


def _image_offsets(cell):
    """The nine in-plane periodic offsets, ordered i = 3 * (na + 1) + (nb + 1)."""
    na = np.repeat(np.arange(-1, 2), 3)
    nb = np.tile(np.arange(-1, 2), 3)
    off = np.zeros((9, 3))
    off[:, 0] = na * cell[0]
    off[:, 1] = nb * cell[1]
    return off


def pbc_displacements(pos_is: "np.ndarray", pos_fs: "np.ndarray",
                              cell: "np.ndarray", moiety: "np.ndarray",
                              alpha: float = 0.68) -> tuple:
    pos_is = np.asarray(pos_is, dtype=float)
    pos_fs = np.asarray(pos_fs, dtype=float)
    moiety = np.asarray(moiety, dtype=int)
    cell = np.asarray(cell, dtype=float).reshape(-1)

    if pos_is.ndim != 2 or pos_is.shape[1] != 3 or pos_is.shape[0] < 1:
        raise ValueError("pos_is must have shape (N, 3) with N >= 1")
    if pos_fs.shape != pos_is.shape:
        raise ValueError("pos_fs must have the same shape as pos_is")
    if moiety.shape != (pos_is.shape[0],):
        raise ValueError("moiety must have one label per atom")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    if not (alpha >= 0.0):
        raise ValueError("alpha must be non-negative")

    n = pos_is.shape[0]
    disp = (pos_fs[:, None, :] + _image_offsets(cell)[None, :, :]
            - pos_is[:, None, :])
    norm = np.sqrt(np.sum(disp * disp, axis=-1))
    mic = np.argmin(norm, axis=1)
    chosen = mic.copy()

    for label in np.unique(moiety):
        grp = np.flatnonzero(moiety == label)
        if grp.size < 2:
            continue
        counts = np.bincount(mic[grp], minlength=9)
        majority = int(np.argmax(counts))
        inside = grp[mic[grp] == majority]
        outside = grp[mic[grp] != majority]
        if outside.size == 0:
            continue
        v_avg = disp[inside, mic[inside], :].mean(axis=0)
        n_avg = float(np.sqrt(np.dot(v_avg, v_avg)))
        for k in outside:
            nv = norm[k]
            if n_avg <= 1e-12:
                theta = np.zeros(9)
            else:
                safe = np.where(nv < 1e-12, np.inf, nv)
                cosang = (disp[k] @ v_avg) / safe / n_avg
                theta = np.where(nv < 1e-12, 0.0,
                                 np.arccos(np.clip(cosang, -1.0, 1.0)))
            score = theta + alpha * nv / max(float(nv.min()), 1e-12)
            chosen[k] = int(np.argmin(score))

    return disp[np.arange(n), chosen, :], chosen.astype(float)

import numpy as np


def _idpp_value_gradient(pos, frozen, d_mov, d_fix, cell):
    """Image-dependent pair potential of a stack of images and its gradient."""
    n = pos.shape[-2]
    eye = np.eye(n, dtype=bool)
    dvec = pos[..., :, None, :] - pos[..., None, :, :]
    d = np.where(eye, 1.0, np.sqrt(np.sum(dvec * dvec, axis=-1)))
    u = d_mov - d
    value = 0.5 * np.sum(np.where(eye, 0.0, u * u / d ** 4), axis=(-2, -1))
    coeff = np.where(eye, 0.0, -2.0 * u / d ** 4 - 4.0 * u * u / d ** 5)
    grad = np.sum((coeff / d)[..., None] * dvec, axis=-2)
    if frozen.shape[0] > 0:
        fvec = _mic_delta(pos[..., :, None, :] - frozen[None, None, :, :], cell)
        fd = np.maximum(np.sqrt(np.sum(fvec * fvec, axis=-1)), 1e-12)
        fu = d_fix - fd
        value = value + np.sum(fu * fu / fd ** 4, axis=(-2, -1))
        fcoeff = -2.0 * fu / fd ** 4 - 4.0 * fu * fu / fd ** 5
        grad = grad + np.sum((fcoeff / fd)[..., None] * fvec, axis=-2)
    return value, grad


def build_idpp_band(pos_is: "np.ndarray", disp: "np.ndarray",
                            frozen: "np.ndarray", cell: "np.ndarray",
                            n_images: int, n_iter: int = 50,
                            step: float = 0.01,
                            max_step: float = 0.05) -> tuple:
    pos_is = np.asarray(pos_is, dtype=float)
    disp = np.asarray(disp, dtype=float)
    frozen = np.asarray(frozen, dtype=float).reshape(-1, 3)
    cell = np.asarray(cell, dtype=float).reshape(-1)
    if pos_is.ndim != 2 or pos_is.shape[1] != 3 or pos_is.shape[0] < 2:
        raise ValueError("pos_is must have shape (N, 3) with N >= 2")
    if disp.shape != pos_is.shape:
        raise ValueError("disp must have the same shape as pos_is")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    if int(n_images) < 2:
        raise ValueError("n_images must be at least 2")
    if int(n_iter) < 0:
        raise ValueError("n_iter must be non-negative")
    if not (step > 0.0 and max_step > 0.0):
        raise ValueError("step and max_step must be positive")

    m = int(n_images)
    lam = np.linspace(0.0, 1.0, m)
    band = pos_is[None, :, :] + lam[:, None, None] * disp[None, :, :]
    e_idpp = np.zeros(m)
    if m == 2:
        return band, e_idpp

    pos_fs = pos_is + disp

    def _moving_distances(p):
        dv = p[:, None, :] - p[None, :, :]
        return np.sqrt(np.sum(dv * dv, axis=-1))

    def _fixed_distances(p):
        dv = _mic_delta(p[:, None, :] - frozen[None, :, :], cell)
        return np.maximum(np.sqrt(np.sum(dv * dv, axis=-1)), 1e-12)

    w = lam[1:-1][:, None, None]
    d_mov = ((1.0 - w) * _moving_distances(pos_is)[None, :, :]
             + w * _moving_distances(pos_fs)[None, :, :])
    if frozen.shape[0] > 0:
        d_fix = ((1.0 - w) * _fixed_distances(pos_is)[None, :, :]
                 + w * _fixed_distances(pos_fs)[None, :, :])
    else:
        d_fix = np.zeros((m - 2, pos_is.shape[0], 0))

    pos = band[1:-1].copy()
    for _ in range(int(n_iter)):
        dr = -float(step) * _idpp_value_gradient(pos, frozen, d_mov, d_fix,
                                                 cell)[1]
        nrm = np.sqrt(np.sum(dr * dr, axis=-1))
        scale = np.where(nrm > max_step, max_step / np.maximum(nrm, 1e-30), 1.0)
        pos = pos + dr * scale[..., None]
    band[1:-1] = pos
    e_idpp[1:-1] = _idpp_value_gradient(pos, frozen, d_mov, d_fix, cell)[0]
    return band, e_idpp

import numpy as np


def reaction_distance_metrics(band: "np.ndarray", e_idpp: "np.ndarray",
                                      donor: int, transfer: int,
                                      acceptor: int) -> tuple:
    band = np.asarray(band, dtype=float)
    e_idpp = np.asarray(e_idpp, dtype=float)
    if band.ndim != 3 or band.shape[2] != 3 or band.shape[0] < 2:
        raise ValueError("band must have shape (M, N, 3) with M >= 2")
    if e_idpp.shape != (band.shape[0],):
        raise ValueError("e_idpp must have one entry per image")
    n = band.shape[1]
    for name, idx in (("donor", donor), ("transfer", transfer),
                      ("acceptor", acceptor)):
        if not (0 <= int(idx) < n):
            raise ValueError(name + " index is outside the atom range")

    seg = band[1:] - band[:-1]
    ds = np.sqrt(np.sum(seg * seg, axis=(1, 2)))
    r = np.concatenate(([0.0], np.cumsum(ds)))
    mu = float(np.sum(0.5 * (e_idpp[1:] + e_idpp[:-1]) * np.diff(r)))

    d_id, t_id, a_id = int(donor), int(transfer), int(acceptor)
    dt = np.sqrt(np.sum((band[:, d_id] - band[:, t_id]) ** 2, axis=1))
    ta = np.sqrt(np.sum((band[:, t_id] - band[:, a_id]) ** 2, axis=1))
    da = np.sqrt(np.sum((band[:, d_id] - band[:, a_id]) ** 2, axis=1))
    tau = float(np.mean(dt + ta + da))
    return mu, tau

import numpy as np


def select_pareto_geometries(energies: "np.ndarray",
                                     distances: "np.ndarray",
                                     e_window: float = 1.2,
                                     d_window: float = 4.5,
                                     tol_e: float = 0.1,
                                     tol_d: float = 0.1,
                                     n_fronts: int = 2) -> "np.ndarray":
    energies = np.asarray(energies, dtype=float).reshape(-1)
    distances = np.asarray(distances, dtype=float).reshape(-1)
    if energies.size < 1:
        raise ValueError("energies must not be empty")
    if distances.shape != energies.shape:
        raise ValueError("distances must have the same length as energies")
    if not (e_window >= 0.0 and d_window >= 0.0):
        raise ValueError("e_window and d_window must be non-negative")
    if not (tol_e >= 0.0 and tol_d >= 0.0):
        raise ValueError("tol_e and tol_d must be non-negative")
    if int(n_fronts) < 1:
        raise ValueError("n_fronts must be at least 1")

    remaining = np.flatnonzero((energies <= energies.min() + e_window)
                               & (distances <= d_window))
    selected = []
    for _ in range(int(n_fronts)):
        if remaining.size == 0:
            break
        e, d = energies[remaining], distances[remaining]
        beaten = ((e[None, :] <= e[:, None]) & (d[None, :] <= d[:, None])
                  & ((e[None, :] < e[:, None]) | (d[None, :] < d[:, None])))
        core = remaining[~np.any(beaten, axis=1)]
        near = np.any((np.abs(e[:, None] - energies[core][None, :]) <= tol_e)
                      & (np.abs(d[:, None] - distances[core][None, :]) <= tol_d),
                      axis=1)
        selected.extend(remaining[near].tolist())
        remaining = remaining[~near]
    return np.array(sorted(selected), dtype=int)

import numpy as np


def _anchor_index(slab, ads, masses, fragments, cell):
    """Slab atom closest to the midpoint of the two fragment centres of mass."""
    coms = []
    for lab in np.unique(fragments):
        sel = fragments == lab
        w = masses[sel]
        coms.append(np.sum(ads[sel] * w[:, None], axis=0) / w.sum())
    mid = np.mean(np.asarray(coms), axis=0)
    d = np.sqrt(np.sum(_mic_delta(slab - mid[None, :], cell) ** 2, axis=1))
    tied = np.flatnonzero(d <= d.min() + 1e-3)
    return int(tied.max())


def align_final_state(slab: "np.ndarray", ads_is: "np.ndarray",
                              ads_fs: "np.ndarray", masses: "np.ndarray",
                              frag_is: "np.ndarray", frag_fs: "np.ndarray",
                              cell: "np.ndarray", sym_ops: "np.ndarray",
                              r_cut: float = 4.0, r_tol: float = 1.0) -> tuple:
    slab = np.asarray(slab, dtype=float)
    ads_is = np.asarray(ads_is, dtype=float)
    ads_fs = np.asarray(ads_fs, dtype=float)
    masses = np.asarray(masses, dtype=float)
    frag_is = np.asarray(frag_is, dtype=int)
    frag_fs = np.asarray(frag_fs, dtype=int)
    cell = np.asarray(cell, dtype=float).reshape(-1)
    sym_ops = np.asarray(sym_ops, dtype=float)

    if slab.ndim != 2 or slab.shape[1] != 3 or slab.shape[0] < 1:
        raise ValueError("slab must have shape (F, 3) with F >= 1")
    if ads_is.ndim != 2 or ads_is.shape[1] != 3 or ads_is.shape[0] < 1:
        raise ValueError("ads_is must have shape (N, 3) with N >= 1")
    if ads_fs.shape != ads_is.shape:
        raise ValueError("ads_fs must have the same shape as ads_is")
    if masses.shape != (ads_is.shape[0],) or np.any(masses <= 0.0):
        raise ValueError("masses must be one positive value per adsorbate atom")
    if frag_is.shape != (ads_is.shape[0],) or frag_fs.shape != (ads_is.shape[0],):
        raise ValueError("frag_is and frag_fs must have one label per atom")
    if np.unique(frag_is).size != 2 or np.unique(frag_fs).size != 2:
        raise ValueError("frag_is and frag_fs must each define two fragments")
    if cell.shape != (2,) or not np.all(cell > 0.0):
        raise ValueError("cell must be two positive in-plane lengths")
    if sym_ops.ndim != 3 or sym_ops.shape[1:] != (3, 3) or sym_ops.shape[0] < 1:
        raise ValueError("sym_ops must have shape (S, 3, 3) with S >= 1")
    if not (r_cut > 0.0 and r_tol > 0.0):
        raise ValueError("r_cut and r_tol must be positive")

    r_target = slab[_anchor_index(slab, ads_is, masses, frag_is, cell)]
    r_source = slab[_anchor_index(slab, ads_fs, masses, frag_fs, cell)]
    t = r_target - r_source

    def _tiled_environment(anchor, radius):
        # every in-plane periodic copy of the slab within radius of the anchor
        reach = [int(np.ceil(radius / cell[k])) + 1 for k in range(2)]
        shifts = np.array([[i * cell[0], j * cell[1], 0.0]
                           for i in range(-reach[0], reach[0] + 1)
                           for j in range(-reach[1], reach[1] + 1)])
        rel = (slab[None, :, :] + shifts[:, None, :]
               - anchor[None, None, :]).reshape(-1, 3)
        return rel[np.sqrt(np.sum(rel ** 2, axis=1)) <= radius]

    rc = float(r_cut)
    env_s = env_t = np.zeros((0, 3))
    for _ in range(64):
        env_s = _tiled_environment(r_source, rc)
        env_t = _tiled_environment(r_target, rc)
        if env_s.shape[0] == env_t.shape[0]:
            break
        rc += 0.5
    if env_s.shape[0] != env_t.shape[0]:
        raise ValueError("could not match the two anchoring environments")

    def _registration_distance(rot):
        turned = env_s @ np.asarray(rot, dtype=float).T
        gap = np.sqrt(np.sum((turned[:, None, :] - env_t[None, :, :]) ** 2,
                             axis=2))
        return float(np.sqrt(np.sum(np.min(gap, axis=1) ** 2)))

    if _registration_distance(np.eye(3)) < r_tol:
        r_ref = np.eye(3)
    else:
        scores = [_registration_distance(op) for op in sym_ops]
        r_ref = np.asarray(sym_ops[int(np.argmin(scores))], dtype=float)

    aligned = ((ads_fs + t[None, :] - r_target[None, :]) @ r_ref.T
               + r_target[None, :])
    candidates = np.einsum("sij,nj->sni", sym_ops,
                           aligned - r_target[None, :]) + r_target[None, None, :]
    return t, r_ref, candidates

import numpy as np


def variable_spring_constants(energies: "np.ndarray",
                                      k_min: float = 0.1,
                                      k_max: float = 4.0) -> "np.ndarray":
    energies = np.asarray(energies, dtype=float).reshape(-1)
    if energies.size < 2:
        raise ValueError("energies must hold at least two images")
    if not (k_max > k_min > 0.0):
        raise ValueError("k_max must exceed k_min and both must be positive")

    e_ref = max(float(energies[0]), float(energies[-1]))
    e_max = float(energies.max())
    e_pair = np.maximum(energies[:-1], energies[1:])
    dk = float(k_max) - float(k_min)
    k = np.full(e_pair.shape, float(k_max) - dk)
    hot = e_pair > e_ref
    if np.any(hot):
        k[hot] = k_max - dk * (e_max - e_pair[hot]) / (e_max - e_ref)
    return k

import numpy as np


def neb_band_forces(band: "np.ndarray", energies: "np.ndarray",
                            forces: "np.ndarray", k: "np.ndarray",
                            climb: int = -1) -> "np.ndarray":
    band = np.asarray(band, dtype=float)
    energies = np.asarray(energies, dtype=float).reshape(-1)
    forces = np.asarray(forces, dtype=float)
    k = np.asarray(k, dtype=float).reshape(-1)
    if band.ndim != 3 or band.shape[2] != 3 or band.shape[0] < 3:
        raise ValueError("band must have shape (M, N, 3) with M >= 3")
    if energies.shape != (band.shape[0],):
        raise ValueError("energies must have one entry per image")
    if forces.shape != band.shape:
        raise ValueError("forces must have the shape of band")
    if k.shape != (band.shape[0] - 1,):
        raise ValueError("k must hold one spring constant per adjacent pair")
    if np.any(k <= 0.0):
        raise ValueError("spring constants must be positive")

    m = band.shape[0]
    if int(climb) != -1 and not (1 <= int(climb) <= m - 2):
        raise ValueError("climb must be -1 or an interior image index")
    out = np.zeros_like(band)
    top = int(climb)

    for i in range(1, m - 1):
        tp = band[i + 1] - band[i]
        tm = band[i] - band[i - 1]
        if energies[i + 1] > energies[i] > energies[i - 1]:
            tan = tp
        elif energies[i + 1] < energies[i] < energies[i - 1]:
            tan = tm
        else:
            de_p = abs(energies[i + 1] - energies[i])
            de_m = abs(energies[i - 1] - energies[i])
            hi, lo = max(de_p, de_m), min(de_p, de_m)
            if energies[i + 1] > energies[i - 1]:
                tan = tp * hi + tm * lo
            else:
                tan = tp * lo + tm * hi
        nrm = float(np.sqrt(np.sum(tan * tan)))
        tan = tan / nrm if nrm > 1e-12 else np.zeros_like(tan)

        proj = float(np.sum(forces[i] * tan))
        if i == top:
            out[i] = forces[i] - 2.0 * proj * tan
            continue
        spring = (k[i] * float(np.sqrt(np.sum(tp * tp)))
                  - k[i - 1] * float(np.sqrt(np.sum(tm * tm))))
        out[i] = forces[i] - proj * tan + spring * tan
    return out

import numpy as np


def _fire_relax(positions, force_fn, fmax, max_steps, batch=False, dt0=0.1,
                dt_max=0.3, n_min=5, f_inc=1.1, f_dec=0.5, a_start=0.1,
                f_a=0.99, max_step=0.1):
    """FIRE relaxation; with batch=True axis 0 holds independent systems."""
    pos = np.array(positions, dtype=float, copy=True)
    vel = np.zeros_like(pos)
    axes = tuple(range(1, pos.ndim)) if batch else tuple(range(pos.ndim))
    nsys = pos.shape[0] if batch else 1
    tail = (1,) * (pos.ndim - 1)
    dt = np.full(nsys, float(dt0))
    a = np.full(nsys, float(a_start))
    n_pos = np.zeros(nsys, dtype=int)

    def _cast(v):
        return np.asarray(v, dtype=float).reshape((nsys,) + tail)

    for _ in range(int(max_steps)):
        f = force_fn(pos)
        if float(np.max(np.sqrt(np.sum(f * f, axis=-1)))) < fmax:
            break
        vf = np.sum(f * vel, axis=axes).reshape(nsys)
        nf = np.maximum(np.sqrt(np.sum(f * f, axis=axes)).reshape(nsys), 1e-300)
        nv = np.sqrt(np.sum(vel * vel, axis=axes)).reshape(nsys)
        up = vf > 0.0
        vel = np.where(_cast(up), (1.0 - _cast(a)) * vel + _cast(a / nf * nv) * f,
                       0.0)
        grow = up & (n_pos > n_min)
        dt = np.where(up, np.where(grow, np.minimum(dt * f_inc, dt_max), dt),
                      dt * f_dec)
        a = np.where(up, np.where(grow, a * f_a, a), a_start)
        n_pos = np.where(up, n_pos + 1, 0)
        vel = vel + _cast(dt) * f
        dr = _cast(dt) * vel
        nrm = np.sqrt(np.sum(dr * dr, axis=axes)).reshape(nsys)
        scale = _cast(np.where(nrm > max_step,
                              max_step / np.maximum(nrm, 1e-300), 1.0))
        vel = vel * scale
        pos = pos + dr * scale
    return pos


def _build_slab(lattice, n_cell, z_layer, n_layer=3):
    """Square-lattice (100) slab, ABAB stacked, whose top layer lies at z = 0."""
    idx = np.arange(n_cell, dtype=float)
    gx, gy = np.meshgrid(idx, idx, indexing="ij")
    out = []
    for lay in range(int(n_layer)):
        off = 0.5 * (lay % 2)
        out.append(np.stack([(gx.ravel() + off) * lattice,
                             (gy.ravel() + off) * lattice,
                             np.full(gx.size, -lay * float(z_layer))], axis=1))
    return np.concatenate(out, axis=0)


def _c4v_operations():
    """The eight point-group operations of the square surface lattice."""
    ops = []
    for r in range(4):
        ang = 0.5 * np.pi * r
        c, s = float(round(np.cos(ang))), float(round(np.sin(ang)))
        ops.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    mirror = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    for r in range(4):
        ops.append(ops[r] @ mirror)
    return np.asarray(ops, dtype=float)


def _candidate_pool(state, lattice, z_c, z_o, cells, tilt):
    """Deterministic enumeration of adsorbate placements on the square lattice.

    Atom order is C, H1, H2, H3, O, H4 in both states; H3 belongs to the methyl
    group in the initial state and to the water molecule in the final state.
    Every fragment is rotated about the vertical axis through its own site by
    ``tilt`` degrees before placement, which takes each one off every mirror
    plane of the square lattice; without it eight of the sixteen initial-state
    candidates would be invariant under the mirror that exchanges H2 and H3,
    and their relaxation would have to break an exact degeneracy.
    """
    sites = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    ch3 = np.array([[0.0, 0.0, 0.0], [1.03, 0.0, 0.36],
                    [-0.515, 0.892, 0.36], [-0.515, -0.892, 0.36]])
    ch2 = np.array([[0.0, 0.0, 0.0], [1.03, 0.0, 0.36], [-0.515, 0.892, 0.36]])
    oh = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
    h2o = np.array([[-0.757, 0.0, 0.586], [0.0, 0.0, 0.0], [0.757, 0.0, 0.586]])
    ang = np.radians(float(tilt))
    rot = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                    [np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0]])
    ch3, ch2, oh, h2o = (f @ rot.T for f in (ch3, ch2, oh, h2o))
    out = []
    for p in sites:
        for q in sites:
            for cx, cy in cells:
                pos = np.zeros((6, 3))
                c_site = np.array([p[0] * lattice, p[1] * lattice, z_c])
                o_site = np.array([(q[0] + cx) * lattice,
                                   (q[1] + cy) * lattice, z_o])
                if state == "is":
                    pos[[0, 1, 2, 3]] = ch3 + c_site
                    pos[[4, 5]] = oh + o_site
                else:
                    pos[[0, 1, 2]] = ch2 + c_site
                    pos[[3, 4, 5]] = h2o + o_site
                out.append(pos)
    return np.asarray(out)


def _is_intact(pos, cell, bonds, z_max, heavy):
    """Structure filter: bonding pattern, intercalation and desorption."""
    if pos[:, 2].min() <= 0.0 or pos[:, 2].max() >= z_max:
        return False
    heavy = np.asarray(heavy)
    for light, host in bonds:
        d = np.sqrt(np.sum(_mic_delta(pos[heavy] - pos[light], cell) ** 2,
                           axis=1))
        if heavy[int(np.argmin(d))] != host:
            return False
    return True


def _distance_fingerprint(pos, cell):
    """Sorted adsorbate pair distances, a translation and rotation invariant."""
    ii, jj = np.triu_indices(pos.shape[0], k=1)
    d = _mic_delta(pos[ii] - pos[jj], cell)
    return np.sort(np.sqrt(np.sum(d * d, axis=1)))


def _dedupe(positions, energies, cell, tol_e=1e-3, tol_d=1e-2):
    """Drop structures whose energy and distance fingerprint repeat an earlier one."""
    keep, prints = [], []
    for i in range(positions.shape[0]):
        fp = _distance_fingerprint(positions[i], cell)
        if any(abs(energies[i] - energies[j]) < tol_e
               and float(np.max(np.abs(fp - q))) < tol_d
               for j, q in zip(keep, prints)):
            continue
        keep.append(i)
        prints.append(fp)
    return np.array(keep, dtype=int)


def _neb_optimize(band, _energy_forces, k_min, k_max, f_switch, f_conv,
                  steps_pre, steps_ci):
    """Non-climbing FIRE pass, then a climbing-image pass on a frozen top image.

    The spring constants are rebuilt at every force evaluation.  Holding them
    fixed over a block of evaluations makes the band force a discontinuous
    function of the images and puts a floor under the residual force that no
    step budget can get below; rebuilding them each time removes that floor.
    """
    state = {"climb": -1}

    def _band_force(pos):
        ener, raw = _energy_forces(pos)
        k = variable_spring_constants(ener, k_min, k_max)
        return neb_band_forces(pos, ener, raw, k, state["climb"])

    band = _fire_relax(band, _band_force, f_switch, steps_pre, batch=True)
    state["climb"] = int(np.argmax(np.asarray(_energy_forces(band)[0])[1:-1])) + 1
    return _fire_relax(band, _band_force, f_conv, steps_ci, batch=True)


def run_nebscape(de: "np.ndarray", alpha: "np.ndarray",
                         r0: "np.ndarray", r_bond: "np.ndarray",
                         valence: "np.ndarray", k_coord: "np.ndarray",
                         lattice: float = 2.8, n_cell: int = 4,
                         n_layer: int = 3, z_layer: float = 1.4,
                         z_c: float = 2.05, z_o: float = 1.95,
                         tilt: float = 25.0,
                         z_max: float = 6.0, n_rep: int = 3,
                         n_coarse: int = 6, n_fine: int = 20, n_neb: int = 20,
                         n_pre: int = 10,
                         e_window: float = 1.2, d_window: float = 4.5,
                         de_window: float = 0.1, mic_alpha: float = 0.68,
                         k_min: float = 0.1, k_max: float = 4.0,
                         f_switch: float = 0.2, f_conv: float = 1e-4,
                         f_relax: float = 1e-5,
                         steps_relax: int = 10000, steps_pre: int = 30000,
                         steps_ci: int = 30000) -> float:
    de = np.asarray(de, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    r_bond = np.asarray(r_bond, dtype=float)
    valence = np.asarray(valence, dtype=float)
    k_coord = np.asarray(k_coord, dtype=float)
    if not (lattice > 0.0 and z_layer > 0.0):
        raise ValueError("lattice and z_layer must be positive")
    if int(n_cell) < 2 or int(n_layer) < 1:
        raise ValueError("n_cell must be at least 2 and n_layer at least 1")
    if min(int(n_coarse), int(n_fine), int(n_neb)) < 3:
        raise ValueError("image counts must be at least 3")
    if min(int(n_rep), int(n_pre)) < 1:
        raise ValueError("n_rep and n_pre must be at least 1")

    lattice = float(lattice)
    n_cell = int(n_cell)
    cell = np.array([n_cell * lattice, n_cell * lattice])
    slab = _build_slab(lattice, n_cell, float(z_layer), int(n_layer))
    slab_species = np.zeros(slab.shape[0], dtype=int)
    ads_species = np.array([1, 3, 3, 3, 2, 3])
    masses = np.array([12.011, 1.008, 1.008, 1.008, 15.999, 1.008])
    moiety = np.array([0, 0, 0, 2, 1, 1])
    frag_is = np.array([0, 0, 0, 0, 1, 1])
    frag_fs = np.array([0, 0, 0, 1, 1, 1])
    heavy = np.array([0, 4])
    bonds_is = ((1, 0), (2, 0), (3, 0), (5, 4))
    bonds_fs = ((1, 0), (2, 0), (3, 4), (5, 4))
    donor, transfer, acceptor = 0, 3, 4
    sym_ops = _c4v_operations()

    def _energy_forces(p):
        return surface_energy_forces(p, slab, ads_species, slab_species,
                                             cell, de, alpha, r0, r_bond,
                                             valence, k_coord)

    def _relax(p):
        return _fire_relax(p, lambda q: _energy_forces(q)[1], float(f_relax),
                           int(steps_relax), batch=True)

    raw = {state: _candidate_pool(state, lattice, float(z_c), float(z_o),
                                  ((1, 0),), float(tilt))
           for state in ("is", "fs")}
    split = raw["is"].shape[0]
    relaxed_pool = _relax(np.concatenate([raw["is"], raw["fs"]], axis=0))
    pools = {}
    for state, bonds, block in (("is", bonds_is, relaxed_pool[:split]),
                                ("fs", bonds_fs, relaxed_pool[split:])):
        ok = np.array([i for i in range(block.shape[0])
                       if _is_intact(block[i], cell, bonds, float(z_max),
                                     heavy)],
                      dtype=int)
        if ok.size == 0:
            raise ValueError("no candidate of the %s ensemble survived the "
                             "structure filter" % state)
        rel = block[ok]
        ener = np.asarray(_energy_forces(rel)[0])
        dist = np.sqrt(np.sum(_mic_delta(rel[:, donor, :] - rel[:, acceptor, :],
                                         cell) ** 2, axis=1))
        keep = select_pareto_geometries(ener, dist, e_window, d_window)
        keep = keep[_dedupe(rel[keep], ener[keep], cell)]
        keep = keep[np.argsort(ener[keep], kind="stable")[:int(n_rep)]]
        keep = np.sort(keep)
        pools[state] = (rel[keep], ener[keep])

    is_pos = pools["is"][0]
    fs_pos, fs_ener = pools["fs"]

    cand, augmented = [], {}
    for a_i in range(is_pos.shape[0]):
        for b_i in range(fs_pos.shape[0]):
            aug = align_final_state(slab, is_pos[a_i], fs_pos[b_i],
                                            masses, frag_is, frag_fs, cell,
                                            sym_ops)[2]
            augmented[(a_i, b_i)] = aug
            for s_i in range(aug.shape[0]):
                disp = pbc_displacements(is_pos[a_i], aug[s_i], cell,
                                                 moiety, mic_alpha)[0]
                band, e_idpp = build_idpp_band(
                    is_pos[a_i], disp, slab, cell, int(n_coarse))
                mu, tau = reaction_distance_metrics(
                    band, e_idpp, donor, transfer, acceptor)
                cand.append((mu, tau, a_i, b_i, s_i))

    cand.sort(key=lambda c: (c[0], c[2], c[3], c[4]))
    pre = sorted(cand[:int(n_pre)], key=lambda c: (c[1], c[2], c[3], c[4]))
    relaxed = _relax(np.asarray([augmented[(c[2], c[3])][c[4]] for c in pre]))
    e_relaxed = np.asarray(_energy_forces(relaxed)[0])

    survivors = []
    for j, (_, _, a_i, b_i, _) in enumerate(pre):
        if abs(float(e_relaxed[j]) - float(fs_ener[b_i])) >= de_window:
            continue
        disp = pbc_displacements(is_pos[a_i], relaxed[j], cell, moiety,
                                         mic_alpha)[0]
        band, e_idpp = build_idpp_band(is_pos[a_i], disp, slab, cell,
                                               int(n_fine))
        mu2, tau2 = reaction_distance_metrics(band, e_idpp, donor,
                                                      transfer, acceptor)
        survivors.append((mu2, tau2, j, a_i, relaxed[j]))

    if not survivors:
        raise ValueError("no interpolation survived the selection pipeline")
    survivors.sort(key=lambda c: (c[0], c[1], c[2]))

    a_i, fs_final = survivors[0][3], survivors[0][4]
    disp = pbc_displacements(is_pos[a_i], fs_final, cell, moiety,
                                     mic_alpha)[0]
    band = build_idpp_band(is_pos[a_i], disp, slab, cell, int(n_neb))[0]
    band = _neb_optimize(band, _energy_forces, k_min, k_max, f_switch, f_conv,
                         int(steps_pre), int(steps_ci))
    ener = np.asarray(_energy_forces(band)[0])
    return float(np.max(ener) - ener[0])
SCICODE_GOLD_EOF
