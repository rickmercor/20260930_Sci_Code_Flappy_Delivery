"""
Orchestrates the six preceding public steps and returns the complete molecular-crystal audit.

The final record preserves the source's curation, ranking, packing, and dynamical diagnostics
before applying the disclosed graph, robustness, and scalar-reduction benchmark conventions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_molcryst_audit(seed: int = 58031, n_configs: int = 96, rank_seed: int = 8123, dyn_seed: int = 9917) -> object:
    """Call Steps 1 through 6 and return seventeen audit values ending in J.

    Parameters
    ----------
    seed : int
        Error-panel seed, also used by the modular outlier selector.
    n_configs : int
        Number of error-panel configurations, at least 24.
    rank_seed : int
        Relative-landscape seed.
    dyn_seed : int
        Shared packing and dynamics seed.

    Returns
    -------
    object
        Float64 array containing panel checksum, retained count, three ranking
        means, two packing anchors, maximum volume drift, the index-weighted
        checksum of the nine final centered graph risks, six
        tensor anchors, source gap, and J.

    Raises
    ------
    ValueError
        If a seed or n_configs is not an integer, or n_configs is below 24.

    Notes
    -----
    Output index 8 (zero-based) is sum((i+1)*h[i] for i=0..8), where h is
    the nine final centered graph risks. It is not the weighted-distance
    checksum sum(W*distance) also named in the graph diagnostics.
    Use the graph, tensor, checksum, and J conventions stated in the task prompt.
    The panel checksum uses C-order flattening with repeating weights 1 through 17.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_compute_molcryst_audit(seed: int = 58031, n_configs: int = 96, rank_seed: int = 8123, dyn_seed: int = 9917) -> object:
    if not all(isinstance(v, (int, np.integer)) for v in (seed, n_configs, rank_seed, dyn_seed)) or n_configs < 24:
        raise ValueError("all seeds and n_configs must be valid integers")
    p = _oracle_build_error_panel(seed, n_configs)
    c = _oracle_curate_active_pool(p)
    l = _oracle_build_relative_landscapes(c, rank_seed)
    r = _oracle_rank_polymorphs(l)
    pack = _oracle_compute_packing_fidelity(c, dyn_seed)
    dyn = _oracle_compute_dynamic_stability(c, dyn_seed)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"compute_molcryst_audit(58031,96,8123,9917)", "gold_call":"_oracle_compute_molcryst_audit(58031,96,8123,9917)", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58079,84,8177,9967)", "gold_call":"_oracle_compute_molcryst_audit(58079,84,8177,9967)", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58121,72,8219,10009)", "gold_call":"_oracle_compute_molcryst_audit(58121,72,8219,10009)", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58163,108,8263,10061)", "gold_call":"_oracle_compute_molcryst_audit(58163,108,8263,10061)", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58031,96,8123,9917)[-1]", "gold_call":"_oracle_compute_molcryst_audit(58031,96,8123,9917)[-1]", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58079,84,8177,9967)[-1]", "gold_call":"_oracle_compute_molcryst_audit(58079,84,8177,9967)[-1]", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58121,72,8219,10009)[-1]", "gold_call":"_oracle_compute_molcryst_audit(58121,72,8219,10009)[-1]", "tol":1e-9},
        {"setup":"", "call":"compute_molcryst_audit(58163,108,8263,10061)[-1]", "gold_call":"_oracle_compute_molcryst_audit(58163,108,8263,10061)[-1]", "tol":1e-9},
        {"setup":"def _candidate_probe():\n    try: compute_molcryst_audit(1,23,2,3); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_compute_molcryst_audit(1,23,2,3); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
