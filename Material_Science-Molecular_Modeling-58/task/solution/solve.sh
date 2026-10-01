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

def build_error_panel(seed: int = 58031, n_configs: int = 96) -> object:
    ENERGY_MAE = np.array([0.128, 0.069, 0.161, 0.159, 0.184, 0.116, 0.146, 0.158, 0.151], dtype=np.float64)
    FORCE_MAE = np.array([0.762, 0.848, 0.414, 0.501, 1.043, 0.562, 0.695, 0.627, 0.377], dtype=np.float64)
    if not isinstance(seed, (int, np.integer)) or not isinstance(n_configs, (int, np.integer)) or n_configs < 24:
        raise ValueError("seed must be an integer and n_configs must be an integer >= 24")
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=(int(n_configs), 9, 4))
    i = np.arange(int(n_configs), dtype=np.float64)[:, None]
    j = np.arange(9, dtype=np.float64)[None, :]
    energy_scale = ENERGY_MAE / 96.48533212
    force_scale = FORCE_MAE / 96.48533212
    e = energy_scale[None, :] * (1.0 + 0.16 * z[:, :, 0]) + 0.0003 * np.sin((i + 1.0) * (j + 2.0) / 13.0)
    f = force_scale[None, :] * (1.0 + 0.13 * z[:, :, 1]) + 0.0012 * np.cos((i + 2.0) * (j + 1.0) / 17.0)
    e = np.abs(e)
    f = np.abs(f)
    selector = ((np.arange(int(n_configs))[:, None] * 7 + np.arange(9)[None, :] * 11 + int(seed)) % 53) == 0
    e = e + selector * (0.225 + 0.015 * (j % 3))
    f = f + selector * (10.2 + 0.3 * (j % 2))
    joint = np.sqrt((e / energy_scale[None, :]) ** 2 + (f / force_scale[None, :]) ** 2)
    leverage = 0.55 * z[:, :, 2] + 0.25 * z[:, :, 3] + 0.2 * np.sin((i + 1) * (j + 1) / 7.0)
    return np.stack([e, f, joint, leverage], axis=-1).astype(np.float64)

import numpy as np

def _main_density_cluster(points: np.ndarray, eps: float) -> np.ndarray:
    distance = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    neighbor = distance <= eps
    core = neighbor.sum(axis=1) >= 5
    labels = np.full(len(points), -1, dtype=int)
    cluster_id = 0
    for start in range(len(points)):
        if not core[start] or labels[start] >= 0:
            continue
        labels[start] = cluster_id
        queue = [start]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for nxt in np.flatnonzero(neighbor[current] & core):
                if labels[nxt] < 0:
                    labels[nxt] = cluster_id
                    queue.append(int(nxt))
        cluster_id += 1
    # Assign border points to the first core cluster reaching them.
    for i in range(len(points)):
        if labels[i] < 0:
            adjacent = labels[neighbor[i] & core]
            if adjacent.size:
                labels[i] = int(adjacent.min())
    if cluster_id == 0:
        return np.zeros(len(points), dtype=bool)
    sizes = np.bincount(labels[labels >= 0], minlength=cluster_id)
    best = int(np.argmax(sizes))
    return labels == best

def curate_active_pool(panel: object, radius: float = 3.4, train_fraction: float = 0.85) -> object:
    x = np.asarray(panel, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (9, 4) or x.shape[0] < 24 or not np.isfinite(x).all():
        raise ValueError("panel must be finite with shape (n, 9, 4), n >= 24")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(train_fraction) or not (0 < train_fraction < 1):
        raise ValueError("radius and train_fraction are invalid")
    out = np.empty((9, 7), dtype=np.float64)
    for s in range(9):
        eligible = np.flatnonzero((x[:, s, 0] <= 0.2) & (x[:, s, 1] <= 10.0))
        if eligible.size < 8:
            raise ValueError("curation retained too few configurations")
        joint = x[eligible, s, :2]
        med = np.median(joint, axis=0)
        mad = np.median(np.abs(joint - med), axis=0)
        scale = np.where(mad > 0, 1.4826 * mad, 1.0)
        standardized = (joint - med) / scale
        density_keep = _main_density_cluster(standardized, float(radius) / 4.0)
        idx = eligible[density_keep]
        if idx.size < 8:
            raise ValueError("curation retained too few configurations")
        order = idx[np.argsort(x[idx, s, 3], kind="stable")]
        n_train = int(np.floor(float(train_fraction) * order.size))
        n_train = min(max(n_train, 1), order.size - 1)
        vals = x[order, s, :2]
        cov = np.cov(vals.T, ddof=0)
        out[s] = [order.size, n_train, order.size - n_train, vals[:, 0].mean(), vals[:, 1].mean(), cov[0, 1], np.linalg.norm((vals - med) / scale, axis=1).max()]
    return out

import numpy as np

def build_relative_landscapes(summary: object, seed: int = 8123, n_forms: int = 6) -> object:
    s = np.asarray(summary, dtype=np.float64)
    if s.shape != (9, 7) or not np.isfinite(s).all() or not isinstance(seed, (int, np.integer)) or not isinstance(n_forms, (int, np.integer)) or n_forms < 4:
        raise ValueError("summary, seed, or n_forms is invalid")
    rng = np.random.default_rng(int(seed))
    k = np.arange(int(n_forms), dtype=np.float64)[None, :]
    q = np.arange(9, dtype=np.float64)[:, None]
    base = 0.33 * (k - (q % int(n_forms))) ** 2 + 0.21 * np.sin((q + 1) * (k + 2) / 3.0)
    base += 0.018 * s[:, 3, None] * (k + 1) + rng.normal(0, 0.018, size=base.shape)
    tuned = base + rng.normal(0, 0.065 + 0.025 * s[:, 3, None], size=base.shape)
    foundation = base + rng.normal(0, 0.24 + 0.11 * s[:, 4, None], size=base.shape)
    foundation[4] = foundation[4, ::-1]
    foundation[6] = foundation[6].mean() + 0.015 * (k[0] - k.mean())
    stack = np.stack([base, tuned, foundation], axis=-1)
    return stack - np.min(stack, axis=1, keepdims=True)

import numpy as np

def rank_polymorphs(landscapes: object) -> object:
    x = np.asarray(landscapes, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] != 9 or x.shape[2] != 3 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError("landscapes must be finite with shape (9, n_forms, 3)")
    out = np.empty((9, 2, 5), dtype=np.float64)
    for s in range(9):
        ref = x[s, :, 0]
        true_order = np.argsort(ref, kind="stable")
        for m, col in enumerate((1, 2)):
            pred = x[s, :, col]
            mae = np.mean(np.abs(pred - ref))
            concordance = 0.0
            for i in range(ref.size - 1):
                for j in range(i + 1, ref.size):
                    concordance += np.sign((ref[i] - ref[j]) * (pred[i] - pred[j]))
            tau = concordance / (ref.size * (ref.size - 1) / 2.0)
            penalty = ref[int(np.argmin(pred))]
            top2 = len(set(true_order[:2]).intersection(np.argsort(pred, kind="stable")[:2])) / 2.0
            gap_error = abs(np.partition(pred, 1)[1] - np.partition(ref, 1)[1])
            out[s, m] = [mae, tau, penalty, top2, gap_error]
    return out

import numpy as np

def _whole_molecules(frac):
    anchor = frac[:, :1, :]
    delta = frac - anchor
    whole = anchor + delta - np.rint(delta)
    return whole, whole.mean(axis=1)

def _frame_rmsd15(ref, cand, cref, ccand):
    ref_whole, ref_centers = _whole_molecules(ref)
    cand_whole, cand_centers = _whole_molecules(cand)
    origin = ref_centers[13]
    offset = ref_centers - origin
    offset -= np.rint(offset)
    distance = np.linalg.norm(offset @ cref, axis=1)
    selected = np.lexsort((np.arange(27), distance))[:15]
    difference = ref_centers[selected, None, :] - cand_centers[None, :, :]
    difference -= np.rint(difference)
    matched = np.argmin(np.linalg.norm(difference @ ccand, axis=2), axis=1)
    if np.unique(matched).size != 15:
        raise ValueError("molecule correspondence is not unique")
    x = (ref_whole[selected] - np.rint(ref_centers[selected] - origin)[:, None, :]) @ cref
    cand_origin = cand_centers[matched[np.where(selected == 13)[0][0]]]
    y = (cand_whole[matched] - np.rint(cand_centers[matched] - cand_origin)[:, None, :]) @ ccand
    x = x.reshape(-1, 3)
    y = y.reshape(-1, 3)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    u, _, vt = np.linalg.svd(y.T @ x)
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(np.mean(np.sum((x - y @ rotation) ** 2, axis=1))))

def compute_packing_fidelity(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    s = np.asarray(summary, dtype=np.float64)
    if s.shape != (9, 7) or not np.isfinite(s).all() or not isinstance(seed, (int, np.integer)) or not isinstance(n_frames, (int, np.integer)) or n_frames < 60:
        raise ValueError("summary, seed, or n_frames is invalid")
    rng = np.random.default_rng(int(seed))
    offsets = np.stack(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2), indexing="ij"), axis=-1).reshape(-1, 3)
    motif = np.array([[0.0, 0.0, 0.0], [0.71, 0.21, 0.05], [-0.26, 0.63, -0.08]])
    out = np.empty((9, 2), dtype=np.float64)
    for k in range(9):
        centers = np.mod(np.array([0.93, 0.07, 0.89]) + 0.16 * offsets + np.array([0.003 * k, 0.002 * k, -0.001 * k]), 1.0)
        cell = np.diag([9.1 + 0.07 * k, 8.7 + 0.04 * k, 9.4 + 0.03 * k])
        ref_cart = centers[:, None, :] @ cell + motif[None, :, :]
        ref = np.mod(ref_cart @ np.linalg.inv(cell), 1.0)
        rmsd = np.empty(int(n_frames), dtype=np.float64)
        for t in range(int(n_frames)):
            strain = rng.normal(0.0, 0.012, size=(3, 3))
            other_cell = cell @ (np.eye(3) + strain)
            theta = 0.16 * np.sin((t + 1) * (k + 2) / 37.0)
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
            sigma = 0.060 + 0.014 * k + 0.003 * s[k, 3]
            other_cart = centers[:, None, :] @ other_cell + motif[None, :, :] @ rotation
            other_cart += rng.normal(0.0, sigma, size=other_cart.shape)
            other = np.mod(other_cart @ np.linalg.inv(other_cell), 1.0)
            other = other[rng.permutation(27)]
            rmsd[t] = _frame_rmsd15(ref, other, cell, other_cell)
        out[k] = [rmsd.mean(), np.mean(rmsd < 0.3)]
    return out

import numpy as np

def compute_dynamic_stability(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    s = np.asarray(summary, dtype=np.float64)
    if s.shape != (9, 7) or not np.isfinite(s).all() or not isinstance(seed, (int, np.integer)) or not isinstance(n_frames, (int, np.integer)) or n_frames < 60:
        raise ValueError("summary, seed, or n_frames is invalid")
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(n_frames), dtype=np.float64)
    out = np.empty((9, 5), dtype=np.float64)
    for k in range(9):
        rng.normal(0.135 + 0.012 * (k % 4) + 0.02 * s[k, 3], 0.052, size=int(n_frames))
        theta = 0.18 * np.sin((t + 1) * (k + 2) / 37.0) + rng.normal(0, 0.075 + 0.006 * k, size=int(n_frames))
        p2 = 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
        r = np.linspace(0.0, 6.0, 96)
        g_ref = np.exp(-0.5 * ((r - (2.2 + 0.04 * k)) / 0.34) ** 2) + 0.48 * np.exp(-0.5 * ((r - 4.1) / 0.55) ** 2)
        g_model = g_ref * (1.0 + 0.025 * np.sin((k + 1) * r)) + rng.normal(0, 0.006, size=r.size)
        rdf = np.trapezoid(np.abs(g_model - g_ref), r) / np.trapezoid(np.abs(g_ref), r)
        energy = 0.003 * np.sin(2 * np.pi * t / 53.0 + k) + rng.normal(0, 0.00045, size=int(n_frames)) + (k - 4) * 2e-6 * t
        volume = 1.0 + 0.008 * np.sin(2 * np.pi * t / 71.0 + 0.3 * k) + rng.normal(0, 0.0012, size=int(n_frames)) + (k - 4) * 1.5e-6 * t
        h = int(n_frames) // 2
        out[k] = [p2.mean(), p2.std(ddof=0), rdf, abs(energy[h:].mean() - energy[:h].mean()), abs(volume[h:].mean() - volume[:h].mean())]
    return out

import numpy as np

def _fuse_system_graph_reference(summary: object, ranks: object, packing: object, dynamics: object, iterations: int = 29) -> object:
    s = np.asarray(summary, dtype=np.float64)
    r = np.asarray(ranks, dtype=np.float64)
    p = np.asarray(packing, dtype=np.float64)
    y = np.asarray(dynamics, dtype=np.float64)
    if s.shape != (9, 7) or r.shape != (9, 2, 5) or p.shape != (9, 2) or y.shape != (9, 5) or not np.isfinite(s).all() or not np.isfinite(r).all() or not np.isfinite(p).all() or not np.isfinite(y).all() or not isinstance(iterations, (int, np.integer)) or iterations < 3:
        raise ValueError("graph inputs or iterations are invalid")
    d = np.column_stack([p, y])
    feat = np.column_stack([s[:, 3], s[:, 4], r[:, 0, 0], r[:, 0, 1], r[:, 0, 2], d[:, 0], 1.0 - d[:, 1], 1.0 - d[:, 2], d[:, 4], d[:, 5], d[:, 6]])
    med = np.median(feat, axis=0)
    mad = np.median(np.abs(feat - med), axis=0)
    z = (feat - med) / np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    dist = np.sqrt(np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2))
    w = np.exp(-dist / (np.median(dist[dist > 0]) + 1e-12))
    np.fill_diagonal(w, 0.0)
    mask = np.zeros_like(w, dtype=bool)
    for i in range(9):
        mask[i, np.argsort(dist[i], kind="stable")[1:4]] = True
    w = np.where(mask | mask.T, w, 0.0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    h = 0.31 * z[:, 2] + 0.24 * (1.0 - r[:, 0, 1]) + 0.18 * z[:, 5] + 0.15 * z[:, 8] + 0.12 * z[:, 10]
    for q in range(int(iterations)):
        msg = w @ h
        gate = 1.0 / (1.0 + np.exp(-(0.7 * z[:, 0] - 0.45 * z[:, 3] + 0.03 * (q + 1))))
        h = np.tanh((0.68 - 0.09 * gate) * h + (0.27 + 0.11 * gate) * msg + 0.017 * np.sin((q + 1) * (np.arange(9) + 1)))
        h -= h.mean()
    eig = np.linalg.eigvalsh(0.5 * (w + w.T))
    anchors = np.array([np.linalg.norm(h), np.sum(h * (np.arange(9) + 1)), np.max(h) - np.min(h), np.sum(w * dist), eig[-1], eig[-2]], dtype=np.float64)
    return np.concatenate([h, anchors])

def _robustness_tensor_reference(fused: object, ranks: object, packing: object, dynamics: object, iterations: int = 41) -> object:
    f = np.asarray(fused, dtype=np.float64)
    r = np.asarray(ranks, dtype=np.float64)
    p = np.asarray(packing, dtype=np.float64)
    y = np.asarray(dynamics, dtype=np.float64)
    if f.shape != (15,) or r.shape != (9, 2, 5) or p.shape != (9, 2) or y.shape != (9, 5) or not np.isfinite(f).all() or not np.isfinite(r).all() or not np.isfinite(p).all() or not np.isfinite(y).all() or not isinstance(iterations, (int, np.integer)) or iterations < 5:
        raise ValueError("tensor inputs or iterations are invalid")
    d = np.column_stack([p, y])
    a0 = np.array([0.72, 0.86, 1.00, 1.14], dtype=np.float64)
    a1 = np.array([0.55, 0.75, 0.95, 1.15, 1.35], dtype=np.float64)
    a2 = np.array([0.03, 0.07, 0.11, 0.17], dtype=np.float64)
    a3 = np.array([0.82, 0.94, 1.06], dtype=np.float64)
    vals = np.empty((4, 5, 4, 3), dtype=np.float64)
    base = f[:9]
    rank_gap = np.mean(r[:, 0, 0] + 0.5 * (1.0 - r[:, 0, 1]) + r[:, 0, 2])
    dyn_gap = np.mean(d[:, 0] + d[:, 4] + 8.0 * d[:, 5] + 20.0 * d[:, 6])
    for i, x0 in enumerate(a0):
        for j, x1 in enumerate(a1):
            for k, x2 in enumerate(a2):
                for m, x3 in enumerate(a3):
                    v = base.copy() * x0 + (np.arange(9) - 4) * x2
                    mom = np.zeros(9, dtype=np.float64)
                    for q in range(int(iterations)):
                        rolled = np.roll(v, 1) - 2.0 * v + np.roll(v, -1)
                        force = np.tanh(x1 * rolled + x3 * base - 0.13 * rank_gap + 0.09 * dyn_gap)
                        mom = 0.73 * mom + (0.19 + 0.002 * q) * force
                        v = v + mom / (1.0 + 0.025 * q) - 0.04 * v.mean()
                    vals[i, j, k, m] = np.sqrt(np.mean(v * v)) + 0.08 * np.max(np.abs(v)) + 0.03 * abs(np.sum(v * (np.arange(9) + 1)))
    flat = vals.reshape(-1, order="C")
    stats = np.array([np.median(flat), np.std(flat, ddof=0), np.quantile(flat, 0.9), np.max(flat) - np.min(flat), np.sum(flat * (np.arange(flat.size) + 1)), np.linalg.svd(vals.reshape(20, 12, order="C"), compute_uv=False)[0]], dtype=np.float64)
    return np.concatenate([flat, stats])

def compute_molcryst_audit(seed: int = 58031, n_configs: int = 96, rank_seed: int = 8123, dyn_seed: int = 9917) -> object:
    if not all(isinstance(v, (int, np.integer)) for v in (seed, n_configs, rank_seed, dyn_seed)) or n_configs < 24:
        raise ValueError("all seeds and n_configs must be valid integers")
    p = build_error_panel(seed, n_configs)
    c = curate_active_pool(p)
    l = build_relative_landscapes(c, rank_seed)
    r = rank_polymorphs(l)
    pack = compute_packing_fidelity(c, dyn_seed)
    dyn = compute_dynamic_stability(c, dyn_seed)
    f = _fuse_system_graph_reference(c, r, pack, dyn)
    t = _robustness_tensor_reference(f, r, pack, dyn)
    panel_checksum = np.sum(p * ((np.arange(p.size).reshape(p.shape, order="C") % 17) + 1))
    kept = np.sum(c[:, 0])
    tuned_mae = np.mean(r[:, 0, 0])
    tuned_tau = np.mean(r[:, 0, 1])
    tuned_penalty = np.mean(r[:, 0, 2])
    rmsd = np.mean(pack[:, 0])
    packing = np.mean(pack[:, 1])
    vol = np.max(dyn[:, 4])
    graph_checksum = f[-5]
    med, std, q90, span, checksum, sv = t[-6:]
    energy_mae = np.array([0.128, 0.069, 0.161, 0.159, 0.184, 0.116, 0.146, 0.158, 0.151], dtype=np.float64)
    force_mae = np.array([0.762, 0.848, 0.414, 0.501, 1.043, 0.562, 0.695, 0.627, 0.377], dtype=np.float64)
    source_gap = abs(energy_mae.mean() - 0.141) + abs(force_mae.mean() - 0.648)
    J = (np.log1p(abs(checksum)) / (1.0 + sv) + med + std + q90 / (1.0 + span)) * (1.0 + tuned_mae + max(0.0, 0.4 - tuned_tau) + tuned_penalty + rmsd + (1.0 - packing) + 20.0 * vol + source_gap)
    return np.array([panel_checksum, kept, tuned_mae, tuned_tau, tuned_penalty, rmsd, packing, vol, graph_checksum, med, std, q90, span, checksum, sv, source_gap, J], dtype=np.float64)
SCICODE_GOLD_EOF
