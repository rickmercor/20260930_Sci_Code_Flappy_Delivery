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

def embed_localized_orbitals(coefficients: "np.ndarray", shell_counts: "np.ndarray", weights: "np.ndarray") -> tuple:
    coefficients = np.asarray(coefficients, dtype=float)
    shell_counts = np.asarray(shell_counts)
    weights = np.asarray(weights, dtype=float)
    if coefficients.ndim != 4 or min(coefficients.shape) < 1:
        raise ValueError("coefficients must have shape (P,A,R,M) with nonzero axes")
    p, a, r, m = coefficients.shape
    if shell_counts.shape != (a,) or not np.issubdtype(shell_counts.dtype, np.integer):
        raise ValueError("shell_counts must be an integer array of shape (A,)")
    if np.any(shell_counts < 1) or np.any(shell_counts > r):
        raise ValueError("each shell count must lie in [1,R]")
    if weights.ndim != 2 or weights.shape[1] != r or weights.shape[0] < 1:
        raise ValueError("weights must have shape (K,R), K>=1")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(weights)):
        raise ValueError("numeric inputs must be finite")
    mask = np.arange(r)[None, :] < shell_counts[:, None]
    padded = coefficients * mask[None, :, :, None]
    embedded = np.einsum("kr,parm->pakm", weights, padded, optimize=True)
    flat = embedded.ravel(order="C")
    checksum = float(np.dot(np.arange(1, flat.size + 1, dtype=float), flat) / flat.size)
    return padded, embedded, checksum

import numpy as np

def apply_signed_mo_attention(embedded: "np.ndarray", fragment_ids: "np.ndarray", q_weights: "np.ndarray", k_weights: "np.ndarray", v_weights: "np.ndarray", head_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    x = np.asarray(embedded, dtype=float)
    f = np.asarray(fragment_ids)
    qw = np.asarray(q_weights, dtype=float)
    kw = np.asarray(k_weights, dtype=float)
    vw = np.asarray(v_weights, dtype=float)
    hw = np.asarray(head_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("embedded must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if f.shape != (p,) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("fragment_ids must be an integer vector of length P")
    if qw.ndim != 3 or qw.shape[1:] != (k, k) or kw.shape != qw.shape or vw.shape != qw.shape:
        raise ValueError("q, k, and v weights must all have shape (H,K,K)")
    h = qw.shape[0]
    if h < 1 or hw.shape != (h,):
        raise ValueError("head_weights must have shape (H,)")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, qw, kw, vw, hw)):
        raise ValueError("all numeric inputs must be finite")
    q = np.einsum("hjk,pakm->hpajm", qw, x, optimize=True)
    key = np.einsum("hjk,pakm->hpajm", kw, x, optimize=True)
    value = np.einsum("hjk,pakm->hpajm", vw, x, optimize=True)
    qn = np.sqrt(np.sum(q * q, axis=(2, 3, 4))) + abs(float(epsilon))
    kn = np.sqrt(np.sum(key * key, axis=(2, 3, 4))) + abs(float(epsilon))
    q_unit = q / qn[:, :, None, None, None]
    k_unit = key / kn[:, :, None, None, None]
    scores = np.einsum("hpajm,hqajm->hpq", q_unit, k_unit, optimize=True)
    same_fragment = f[:, None] == f[None, :]
    scores = scores * same_fragment[None, :, :]
    raw = np.einsum("hpq,hqajm->hpajm", scores, value, optimize=True)
    output_norms = np.sqrt(np.sum(raw * raw, axis=(2, 3, 4)))
    heads = raw / (output_norms[:, :, None, None, None] + abs(float(epsilon)))
    mixed = np.einsum("h,hpajm->pajm", hw, heads, optimize=True)
    return mixed, scores, output_norms

import numpy as np

def apply_odd_local_mixing(features: "np.ndarray", positions: "np.ndarray", cutoff: float, linear_weights: "np.ndarray", cubic_weights: "np.ndarray") -> tuple:
    x = np.asarray(features, dtype=float)
    pos = np.asarray(positions, dtype=float)
    w1 = np.asarray(linear_weights, dtype=float)
    w3 = np.asarray(cubic_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("features must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if pos.shape != (a, 3) or w1.shape != (k, k) or w3.shape != (k, k):
        raise ValueError("positions or channel maps have invalid shape")
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, pos, w1, w3)):
        raise ValueError("numeric inputs must be finite")
    aggregate = np.zeros_like(x)
    count = 0
    for center in range(a):
        for neighbor in range(a):
            if center == neighbor:
                continue
            distance = float(np.linalg.norm(pos[center] - pos[neighbor]))
            if distance < cutoff:
                aggregate[:, center] += np.exp(-distance / cutoff) * x[:, neighbor]
                count += 1
    if count == 0:
        raise ValueError("the cutoff graph must contain at least one directed edge")
    linear = np.einsum("jk,pakm->pajm", w1, aggregate, optimize=True)
    cubic_input = aggregate * np.sum(aggregate * aggregate, axis=-1, keepdims=True)
    cubic = np.einsum("jk,pakm->pajm", w3, cubic_input, optimize=True)
    updated = x + linear + cubic
    return updated, aggregate, int(count)

import numpy as np

def readout_single_amplitudes(features: "np.ndarray", occupied_count: int, pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    x = np.asarray(features, dtype=float)
    pw = np.asarray(pair_weights, dtype=float)
    wh = np.asarray(hidden_weights, dtype=float)
    wo = np.asarray(output_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("features must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if not isinstance(occupied_count, (int, np.integer)) or not 1 <= occupied_count < p:
        raise ValueError("occupied_count must be an integer in [1,P-1]")
    if pw.shape != (k, k) or wh.ndim != 2 or wh.shape[0] != k or wo.shape != (wh.shape[1],):
        raise ValueError("readout weight shapes are inconsistent")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, pw, wh, wo)):
        raise ValueError("numeric inputs must be finite")
    norms = np.sqrt(np.sum(x * x, axis=(1, 2, 3))) + abs(float(epsilon))
    unit = x / norms[:, None, None, None]
    projected = np.einsum("jk,pakm->pajm", pw, unit, optimize=True)
    occ = projected[:occupied_count]
    virt = projected[occupied_count:]
    pair_features = np.einsum("iakm,vakm->ivk", occ, virt, optimize=True)
    hidden = np.tanh(np.einsum("ivk,kh->ivh", pair_features, wh, optimize=True))
    t1 = np.einsum("ivh,h->iv", hidden, wo, optimize=True)
    flat = t1.ravel(order="C")
    checksum = float(np.dot(np.arange(1, flat.size + 1, dtype=float), flat))
    return t1, pair_features, checksum

import numpy as np

def readout_double_amplitudes(features: "np.ndarray", occupied_count: int, mp2_amplitudes: "np.ndarray", pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    x = np.asarray(features, dtype=float)
    mp2 = np.asarray(mp2_amplitudes, dtype=float)
    pw = np.asarray(pair_weights, dtype=float)
    wh = np.asarray(hidden_weights, dtype=float)
    wo = np.asarray(output_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("features must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if not isinstance(occupied_count, (int, np.integer)) or not 1 <= occupied_count < p:
        raise ValueError("occupied_count must be an integer in [1,P-1]")
    nv = p - occupied_count
    if mp2.shape != (occupied_count, occupied_count, nv, nv):
        raise ValueError("mp2_amplitudes has the wrong shape")
    if not np.allclose(mp2, -mp2.swapaxes(0, 1), rtol=0.0, atol=1e-12) or not np.allclose(mp2, -mp2.swapaxes(2, 3), rtol=0.0, atol=1e-12):
        raise ValueError("mp2_amplitudes must be antisymmetric in occupied and virtual pairs")
    if pw.shape != (k, k) or wh.ndim != 2 or wh.shape[0] != k or wo.shape != (wh.shape[1],):
        raise ValueError("readout weight shapes are inconsistent")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, mp2, pw, wh, wo)):
        raise ValueError("numeric inputs must be finite")
    norms = np.sqrt(np.sum(x * x, axis=(1, 2, 3))) + abs(float(epsilon))
    unit = x / norms[:, None, None, None]
    projected = np.einsum("jk,pakm->pajm", pw, unit, optimize=True)
    occ = projected[:occupied_count]
    virt = projected[occupied_count:]
    pair = np.einsum("iakm,vakm->ivk", occ, virt, optimize=True)
    four = pair[:, None, :, None, :] * pair[None, :, None, :, :]
    hidden = np.tanh(np.einsum("ijabk,kh->ijabh", four, wh, optimize=True))
    raw = np.einsum("ijabh,h->ijab", hidden, wo, optimize=True)
    correction = 0.25 * (raw - raw.swapaxes(0, 1) - raw.swapaxes(2, 3) + raw.transpose(1, 0, 3, 2))
    t2 = mp2 + correction
    residual = float(max(np.max(np.abs(t2 + t2.swapaxes(0, 1))), np.max(np.abs(t2 + t2.swapaxes(2, 3)))))
    return t2, correction, residual

import numpy as np

def compute_cc_correlation_audit(t1: "np.ndarray", t2: "np.ndarray", antisymmetrized_integrals: "np.ndarray") -> tuple:
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    g = np.asarray(antisymmetrized_integrals, dtype=float)
    if t1.ndim != 2 or min(t1.shape) < 1:
        raise ValueError("t1 must have shape (n_occ,n_virt)")
    no, nv = t1.shape
    if t2.shape != (no, no, nv, nv) or g.shape != t2.shape:
        raise ValueError("t2 and integrals must have shape (n_occ,n_occ,n_virt,n_virt)")
    if not all(np.all(np.isfinite(z)) for z in (t1, t2, g)):
        raise ValueError("all inputs must be finite")
    for name, tensor in (("t2", t2), ("antisymmetrized_integrals", g)):
        if not np.allclose(tensor, -tensor.swapaxes(0, 1), rtol=0.0, atol=1e-12) or not np.allclose(tensor, -tensor.swapaxes(2, 3), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be antisymmetric in occupied and virtual pairs")
    direct = np.einsum("ia,jb->ijab", t1, t1, optimize=True)
    product = direct - direct.swapaxes(2, 3)
    doubles = float(0.25 * np.sum(t2 * g))
    singles = float(0.25 * np.sum(product * g))
    energy = float(doubles + singles)
    norm = float(np.sqrt(np.sum(t1 * t1) + np.sum(t2 * t2)))
    raw = (t2 + product) * g
    flat = raw.ravel(order="C")
    checksum = float(np.dot(np.arange(1, flat.size + 1, dtype=float), flat) / flat.size)
    return energy, singles, doubles, norm, checksum

import numpy as np

def compute_molecular_orbital_audit(seed: int = 33027, atom_count: int = 6, occupied_count: int = 3, virtual_count: int = 3, radial_channels: int = 4, hidden_channels: int = 4, heads: int = 3, cutoff: float = 2.25, epsilon: float = 1e-8) -> tuple:
    integer_values = (atom_count, occupied_count, virtual_count, radial_channels, hidden_channels, heads)
    if not all(isinstance(z, (int, np.integer)) for z in integer_values):
        raise ValueError("count parameters must be integers")
    if not 3 <= atom_count <= 9 or not 1 <= occupied_count <= 5 or not 1 <= virtual_count <= 5:
        raise ValueError("atom and orbital counts are outside their supported ranges")
    if not 2 <= radial_channels <= 6 or not 2 <= hidden_channels <= 6 or not 1 <= heads <= 4:
        raise ValueError("channel or head count is outside its supported range")
    if not np.isfinite(cutoff) or cutoff <= 0.0 or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("cutoff and epsilon must be positive and finite")
    rng = np.random.default_rng(int(seed))
    orbital_count = occupied_count + virtual_count
    magnetic_channels = 3
    coefficients = rng.normal(0.0, 0.42, size=(orbital_count, atom_count, radial_channels, magnetic_channels))
    shell_counts = 1 + ((3 * np.arange(atom_count) + 1) % radial_channels)
    embed_weights = np.fromfunction(lambda k, r: (np.sin((k + 1) * (r + 2) / 5.0) + 0.21 * np.cos((k + 2) * (r + 1) / 7.0)) / np.sqrt(radial_channels), (hidden_channels, radial_channels), dtype=float)
    padded, embedded, embedding_checksum = embed_localized_orbitals(coefficients, shell_counts, embed_weights)
    fragment_ids = np.zeros(orbital_count, dtype=int)
    fragment_ids[(orbital_count + 1) // 2:] = 1
    q_weights = np.fromfunction(lambda h, j, k: 0.19 * np.sin((h + 1) * (j + 2) * (k + 1) / 9.0), (heads, hidden_channels, hidden_channels), dtype=float)
    k_weights = np.fromfunction(lambda h, j, k: 0.17 * np.cos((h + 2) * (j + 1) * (k + 2) / 11.0), (heads, hidden_channels, hidden_channels), dtype=float)
    v_weights = np.fromfunction(lambda h, j, k: 0.15 * np.sin((h + 3) * (j + 1) + (k + 2) / 8.0), (heads, hidden_channels, hidden_channels), dtype=float)
    head_weights = 0.5 + 0.2 * np.cos(np.arange(heads, dtype=float) + 1.0)
    mixed, scores, output_norms = apply_signed_mo_attention(embedded, fragment_ids, q_weights, k_weights, v_weights, head_weights, epsilon)
    angle = 2.0 * np.pi * np.arange(atom_count, dtype=float) / atom_count
    radius = 1.15 + 0.08 * np.sin(3.0 * angle)
    positions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), 0.22 * np.sin(2.0 * angle + 0.3)))
    linear_weights = np.fromfunction(lambda j, k: 0.075 * np.cos((j + 1) * (k + 2) / 4.0), (hidden_channels, hidden_channels), dtype=float)
    cubic_weights = np.fromfunction(lambda j, k: 0.012 * np.sin((j + 2) * (k + 1) / 5.0), (hidden_channels, hidden_channels), dtype=float)
    updated, aggregate, directed_edges = apply_odd_local_mixing(mixed, positions, cutoff, linear_weights, cubic_weights)
    pair_weights_1 = np.fromfunction(lambda j, k: 0.23 * np.cos((j + 1) * (k + 2) / 6.0), (hidden_channels, hidden_channels), dtype=float)
    hidden_width = hidden_channels + 2
    hidden_weights_1 = np.fromfunction(lambda j, h: 0.31 * np.sin((j + 2) * (h + 1) / 7.0), (hidden_channels, hidden_width), dtype=float)
    output_weights_1 = 0.18 * np.cos(np.arange(hidden_width, dtype=float) + 1.0)
    t1, t1_pairs, t1_checksum = readout_single_amplitudes(updated, occupied_count, pair_weights_1, hidden_weights_1, output_weights_1, epsilon)
    orbital_energies = np.concatenate((-1.10 - 0.17 * np.arange(occupied_count), 0.25 + 0.21 * np.arange(virtual_count)))
    raw_integrals = rng.normal(0.0, 0.09, size=(occupied_count, occupied_count, virtual_count, virtual_count))
    integrals = 0.25 * (raw_integrals - raw_integrals.swapaxes(0, 1) - raw_integrals.swapaxes(2, 3) + raw_integrals.transpose(1, 0, 3, 2))
    denominators = orbital_energies[:occupied_count, None, None, None] + orbital_energies[None, :occupied_count, None, None] - orbital_energies[None, None, occupied_count:, None] - orbital_energies[None, None, None, occupied_count:]
    mp2 = integrals / denominators
    pair_weights_2 = np.fromfunction(lambda j, k: 0.20 * np.sin((j + 2) * (k + 1) / 5.0), (hidden_channels, hidden_channels), dtype=float)
    hidden_weights_2 = np.fromfunction(lambda j, h: 0.27 * np.cos((j + 1) * (h + 2) / 8.0), (hidden_channels, hidden_width), dtype=float)
    output_weights_2 = 0.14 * np.sin(np.arange(hidden_width, dtype=float) + 1.0)
    t2, correction, exchange_residual = readout_double_amplitudes(updated, occupied_count, mp2, pair_weights_2, hidden_weights_2, output_weights_2, epsilon)
    energy, singles_energy, doubles_energy, amplitude_norm, contraction_checksum = compute_cc_correlation_audit(t1, t2, integrals)
    cross = fragment_ids[:, None] != fragment_ids[None, :]
    cross_score_max = float(np.max(np.abs(scores[:, cross]))) if np.any(cross) else 0.0
    valid_score_mean = float(np.mean(np.abs(scores[:, ~cross])))
    t2_flat = t2.ravel(order="C")
    t2_checksum = float(np.dot(np.arange(1, t2_flat.size + 1, dtype=float), np.abs(t2_flat)) / t2_flat.size)
    correction_ratio = float(np.linalg.norm(correction) / (np.linalg.norm(mp2) + epsilon))
    state_norm = float(np.linalg.norm(updated))
    attention_norm = float(np.mean(output_norms))
    padding_zero = float(np.max(np.abs(padded * (np.arange(radial_channels)[None, None, :, None] >= shell_counts[None, :, None, None]))))
    diagnostics = {"amplitude_norm": amplitude_norm, "attention_norm": attention_norm, "contraction_checksum": contraction_checksum, "correlation_energy": energy, "correction_ratio": correction_ratio, "cross_score_max": cross_score_max, "directed_edges": float(directed_edges), "doubles_energy": doubles_energy, "embedding_checksum": float(embedding_checksum), "exchange_residual": exchange_residual, "pair_norm": float(np.linalg.norm(t1_pairs)), "padding_zero": padding_zero, "singles_energy": singles_energy, "state_norm": state_norm, "t1_checksum": float(t1_checksum), "t2_checksum": t2_checksum, "valid_score_mean": valid_score_mean}
    raw_j = float(np.exp(-abs(energy)) * (1.0 + valid_score_mean + 0.25 * correction_ratio) / (1.0 + amplitude_norm + 0.02 * state_norm + 0.01 * attention_norm))
    return float(np.round(raw_j, 8)), diagnostics, t1, t2, scores
SCICODE_GOLD_EOF
