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

def _validate_panel(n_mechanisms, n_images):
    if not isinstance(n_mechanisms, int) or not isinstance(n_images, int):
        raise ValueError("panel sizes must be integers")
    if n_mechanisms < 8 or n_images < 7 or n_images % 2 == 0:
        raise ValueError("n_mechanisms >= 8 and odd n_images >= 7 are required")

def _potential(r, phase):
    x, y, z = np.moveaxis(np.asarray(r, dtype=float), -1, 0)
    return (0.25 * (x * x - 1.0) ** 2 + 0.34 * y * y + 0.21 * z * z
            + 0.075 * x * y - 0.052 * y * z
            + 0.018 * np.sin(1.7 * x + phase) * np.cos(1.3 * y - 0.4 * phase))

def _gradient(r, phase):
    x, y, z = np.asarray(r, dtype=float)
    q = 1.7 * x + phase
    p = 1.3 * y - 0.4 * phase
    return np.array([
        x * (x * x - 1.0) + 0.075 * y + 0.0306 * np.cos(q) * np.cos(p),
        0.68 * y + 0.075 * x - 0.052 * z - 0.0234 * np.sin(q) * np.sin(p),
        0.42 * z - 0.052 * y,
    ])

def generate_oci_neb_panel(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    _validate_panel(n_mechanisms, n_images)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_mechanisms, n_images, 3), dtype=float)
    energies = np.empty((n_mechanisms, n_images), dtype=float)
    forces = np.empty_like(paths)
    baselines = np.empty(n_mechanisms, dtype=float)
    s = np.linspace(0.0, 1.0, n_images)
    for m in range(n_mechanisms):
        phase = 0.17 * m + 0.003 * (seed % 101)
        left = np.array([-1.18 - 0.015 * (m % 4), -0.28 + 0.025 * (m % 5), 0.11 * np.sin(phase)])
        right = np.array([1.16 + 0.018 * (m % 3), 0.31 - 0.022 * (m % 4), -0.09 * np.cos(phase)])
        line = (1.0 - s[:, None]) * left + s[:, None] * right
        bow = np.column_stack((
            0.04 * np.sin(np.pi * s + phase),
            (0.18 + 0.015 * (m % 3)) * np.sin(np.pi * s),
            0.13 * np.sin(2.0 * np.pi * s + 0.3 * phase),
        ))
        noise = rng.normal(0.0, 0.004 + 0.0004 * (m % 4), (n_images, 3))
        noise[[0, -1]] = 0.0
        paths[m] = line + bow + noise
        energies[m] = _potential(paths[m], phase)
        forces[m] = np.array([-_gradient(point, phase) for point in paths[m]])
        baselines[m] = (1.75 + 0.055 * (m % 5)) * np.max(np.linalg.norm(forces[m, 1:-1], axis=1)) + 0.18
    weights = 0.7 + (np.arange(paths.size) % 29) / 31.0
    checksum = float(np.dot(paths.ravel(), weights) + np.dot(energies.ravel(), 1.0 + (np.arange(energies.size) % 17) / 19.0))
    return paths, energies, forces, baselines, checksum

import numpy as np

def compute_climbing_controls(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    paths, energies, forces, baselines, panel_checksum = generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices = np.argmax(energies[:, 1:-1], axis=1) + 1
    tangents = np.empty((n_mechanisms, 3), dtype=float)
    climb_forces = np.empty_like(tangents)
    norms = np.empty(n_mechanisms, dtype=float)
    stable = np.empty(n_mechanisms, dtype=int)
    for m, k in enumerate(indices):
        tangent = paths[m, k + 1] - paths[m, k - 1]
        tangent /= np.linalg.norm(tangent)
        tangents[m] = tangent
        force = forces[m, k]
        climb = force - 2.0 * np.dot(force, tangent) * tangent
        climb_forces[m] = climb
        norms[m] = np.linalg.norm(climb)
        stable[m] = 3 + ((m * 7 + seed) % 6)
    checksum = float(np.dot(indices, 1.0 + np.arange(n_mechanisms) / 23.0)
                     + np.dot(tangents.ravel(), 0.4 + (np.arange(tangents.size) % 13) / 17.0)
                     + np.dot(norms, 2.0 + np.arange(n_mechanisms) / 29.0))
    return indices, tangents, climb_forces, norms, stable, baselines, panel_checksum, checksum

import numpy as np

def _hessian(r, phase):
    x, y, _ = np.asarray(r, dtype=float)
    q = 1.7 * x + phase
    p = 1.3 * y - 0.4 * phase
    hxx = 3.0 * x * x - 1.0 - 0.05202 * np.sin(q) * np.cos(p)
    hyy = 0.68 - 0.03042 * np.sin(q) * np.cos(p)
    hxy = 0.075 - 0.03978 * np.cos(q) * np.sin(p)
    return np.array([[hxx, hxy, 0.0], [hxy, hyy, -0.052], [0.0, -0.052, 0.42]])

def run_aligned_dimer_bursts(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19) -> tuple:
    if not np.isfinite(step_size) or step_size <= 0.0 or step_size > 0.5:
        raise ValueError("step_size must be finite and in (0, 0.5]")
    paths, _, forces, _, _ = generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices, tangents, _, fci, stable, baselines, panel_checksum, control_checksum = compute_climbing_controls(seed, n_mechanisms, n_images)
    axes = np.empty_like(tangents)
    alpha = np.empty(n_mechanisms, dtype=float)
    curvature = np.empty(n_mechanisms, dtype=float)
    candidates = np.empty_like(tangents)
    fnew = np.empty(n_mechanisms, dtype=float)
    for m, k in enumerate(indices):
        phase = 0.17 * m + 0.003 * (seed % 101)
        point = paths[m, k]
        h = _hessian(point, phase)
        _, vecs = np.linalg.eigh(h)
        vmin = vecs[:, 0]
        if np.dot(vmin, tangents[m]) < 0.0:
            vmin = -vmin
        theta = (0.04, 0.18, 0.34, 0.55, 1.24, 1.43)[m % 6]
        side = np.cross(tangents[m], np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(side) < 1e-12:
            side = np.array([0.0, 1.0, 0.0])
        side /= np.linalg.norm(side)
        raw = np.cos(theta) * vmin + np.sin(theta) * side
        axis = raw / np.linalg.norm(raw)
        axes[m] = axis
        alpha[m] = abs(np.dot(axis, tangents[m]))
        curvature[m] = float(axis @ h @ axis)
        true_force = forces[m, k]
        reflected = true_force - 2.0 * np.dot(true_force, axis) * axis
        candidate = point + step_size * reflected / (1.0 + np.linalg.norm(reflected))
        candidates[m] = candidate
        candidate_force = -_gradient(candidate, phase)
        candidate_climb = candidate_force - 2.0 * np.dot(candidate_force, axis) * axis
        fnew[m] = np.linalg.norm(candidate_climb) * (0.88 + 0.055 * (m % 5))
    checksum = float(np.dot(alpha, 3.0 + np.arange(n_mechanisms) / 19.0)
                     + np.dot(curvature, 2.0 + np.arange(n_mechanisms) / 17.0)
                     + np.dot(candidates.ravel(), 0.6 + (np.arange(candidates.size) % 23) / 29.0)
                     + np.dot(fnew, 5.0 + np.arange(n_mechanisms) / 31.0))
    return axes, alpha, curvature, candidates, fnew, fci, stable, baselines, panel_checksum, control_checksum, checksum

import numpy as np

def _apply_policy(alpha, curvature, fnew, fci, stable, baselines, lambda_rel, alpha_tol, kappa):
    n = len(alpha)
    action = np.zeros(n, dtype=int)
    thresholds = lambda_rel * baselines
    accepted = np.zeros(n, dtype=bool)
    for i in range(n):
        active = stable[i] >= kappa and fci[i] < thresholds[i]
        if not active:
            action[i] = 0
        elif curvature[i] > 0.0:
            action[i] = 1
        elif alpha[i] < alpha_tol or fnew[i] >= fci[i]:
            action[i] = 2
            thresholds[i] = baselines[i] * lambda_rel * (0.5 + 0.5 * alpha[i])
        else:
            action[i] = 3
            accepted[i] = True
            thresholds[i] = fnew[i] * (0.5 + 0.4 * fnew[i] / fci[i])
    return action, thresholds, accepted

def apply_oci_neb_policy(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19, lambda_rel: float = 0.31, alpha_tol: float = 0.85, kappa: int = 5) -> tuple:
    if not (0.0 < lambda_rel < 1.0) or not (2 ** -0.5 < alpha_tol <= 1.0) or not isinstance(kappa, int) or kappa < 1:
        raise ValueError("inadmissible policy parameter")
    paths, _, _, _, _ = generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices, _, _, _, _, _, _, _ = compute_climbing_controls(seed, n_mechanisms, n_images)
    axes, alpha, curvature, candidates, fnew, fci, stable, baselines, panel_checksum, control_checksum, dimer_checksum = run_aligned_dimer_bursts(seed, n_mechanisms, n_images, step_size)
    action, thresholds, accepted = _apply_policy(alpha, curvature, fnew, fci, stable, baselines, lambda_rel, alpha_tol, kappa)
    positions = np.array([paths[m, k] for m, k in enumerate(indices)])
    positions[accepted] = candidates[accepted]
    checksum = float(np.dot(action, 7.0 + np.arange(n_mechanisms) / 13.0)
                     + np.dot(thresholds, 2.0 + np.arange(n_mechanisms) / 17.0)
                     + np.dot(positions.ravel(), 0.8 + (np.arange(positions.size) % 19) / 23.0))
    return action, thresholds, accepted, positions, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, checksum

import numpy as np

def _linear_reparameterize(path):
    delta = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(delta)))
    target = np.linspace(0.0, arc[-1], len(path))
    out = np.empty_like(path)
    for d in range(path.shape[1]):
        out[:, d] = np.interp(target, arc, path[:, d])
    return out

def reparameterize_successful_paths(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, step_size: float = 0.19) -> tuple:
    paths, _, _, _, _ = generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices, _, _, _, _, _, _, _ = compute_climbing_controls(seed, n_mechanisms, n_images)
    action, thresholds, accepted, positions, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, policy_checksum = apply_oci_neb_policy(seed, n_mechanisms, n_images, step_size)
    updated = paths.copy()
    spacing = np.empty(n_mechanisms, dtype=float)
    for m, k in enumerate(indices):
        if accepted[m]:
            updated[m, k] = positions[m]
            updated[m] = _linear_reparameterize(updated[m])
        gaps = np.linalg.norm(np.diff(updated[m], axis=0), axis=1)
        spacing[m] = np.std(gaps) / (np.mean(gaps) + 1e-15)
    checksum = float(np.dot(updated.ravel(), 0.9 + (np.arange(updated.size) % 37) / 41.0)
                     + np.dot(spacing, 11.0 + np.arange(n_mechanisms) / 29.0))
    return updated, spacing, action, thresholds, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, policy_checksum, checksum

import numpy as np

def _apply_policy(alpha, curvature, fnew, fci, stable, baselines, lambda_rel, alpha_tol, kappa):
    n = len(alpha)
    action = np.zeros(n, dtype=int)
    thresholds = lambda_rel * baselines
    accepted = np.zeros(n, dtype=bool)
    for i in range(n):
        active = stable[i] >= kappa and fci[i] < thresholds[i]
        if not active: action[i] = 0
        elif curvature[i] > 0.0: action[i] = 1
        elif alpha[i] < alpha_tol or fnew[i] >= fci[i]:
            action[i] = 2
            thresholds[i] = baselines[i] * lambda_rel * (0.5 + 0.5 * alpha[i])
        else:
            action[i] = 3
            accepted[i] = True
            thresholds[i] = fnew[i] * (0.5 + 0.4 * fnew[i] / fci[i])
    return action, thresholds, accepted

def evaluate_oci_neb_robustness_tensor(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9, lambda_axis: tuple = (0.24, 0.31, 0.38, 0.46), alignment_axis: tuple = (0.72, 0.85, 0.91, 0.96), force_axis: tuple = (0.82, 0.94, 1.0, 1.08, 1.22)) -> tuple:
    if min(len(lambda_axis), len(alignment_axis), len(force_axis)) < 2:
        raise ValueError("every robustness axis must contain at least two values")
    if any(x <= 0.0 or x >= 1.0 for x in lambda_axis) or any(x <= 2 ** -0.5 or x > 1.0 for x in alignment_axis) or any(x <= 0.0 for x in force_axis):
        raise ValueError("robustness axes contain inadmissible values")
    updated, spacing, action0, thresholds0, alpha, curvature, fnew, fci, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum = reparameterize_successful_paths(seed, n_mechanisms, n_images)
    _, _, _, baselines, _ = generate_oci_neb_panel(seed, n_mechanisms, n_images)
    _, _, _, _, stable, _, _, _ = compute_climbing_controls(seed, n_mechanisms, n_images)
    rows = []
    for lam in lambda_axis:
        for atol in alignment_axis:
            for scale in force_axis:
                action, thresholds, accepted = _apply_policy(alpha, curvature, fnew * scale, fci, stable, baselines, float(lam), float(atol), 5)
                success = np.mean(action == 3)
                restored = np.mean(action == 1)
                failed = np.mean(action == 2)
                inactive = np.mean(action == 0)
                reduction = np.mean(np.where(accepted, 1.0 - fnew * scale / (fci + 1e-15), 0.0))
                threshold_mean = np.mean(thresholds / (baselines + 1e-15))
                action_checksum = np.dot(action, 1.0 + np.arange(len(action)) / 23.0)
                spacing_score = np.mean(spacing * (1.0 + accepted.astype(float)))
                rows.append([success, restored, failed, inactive, reduction, threshold_mean, action_checksum, spacing_score])
    tensor = np.asarray(rows, dtype=float)
    weights = 0.6 + (np.arange(tensor.size) % 53) / 59.0
    checksum = float(np.dot(tensor.ravel(), weights))
    return tensor, checksum, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum

import numpy as np

def compute_oci_neb_audit(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    tensor, tensor_checksum, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum = evaluate_oci_neb_robustness_tensor(seed, n_mechanisms, n_images)
    default = reparameterize_successful_paths(seed, n_mechanisms, n_images)
    spacing, action, thresholds, alpha, curvature, fnew, fci = default[1:8]
    success_count = float(np.sum(action == 3))
    restore_count = float(np.sum(action == 1))
    failure_count = float(np.sum(action == 2))
    inactive_count = float(np.sum(action == 0))
    median_success = float(np.median(tensor[:, 0]))
    spread = float(np.std(tensor[:, 0], ddof=0))
    worst_failure = float(np.max(tensor[:, 2]))
    centered = tensor[:, :6] - np.mean(tensor[:, :6], axis=0)
    leading_singular = float(np.linalg.svd(centered, compute_uv=False)[0])
    alignment_margin = float(np.mean(alpha - 2 ** -0.5))
    force_gain = float(np.mean(1.0 - fnew / (fci + 1e-15)))
    path_uniformity = float(1.0 / (1.0 + np.mean(spacing)))
    stability = float((median_success + max(force_gain, 0.0) + path_uniformity) /
                      (1.0 + spread + worst_failure + leading_singular))
    j = float((1.0 + max(alignment_margin, 0.0)) * stability /
              (1.0 + np.mean(thresholds) + tensor_checksum / 10000.0))
    return (j, panel_checksum, control_checksum, dimer_checksum, policy_checksum, path_checksum,
            tensor_checksum, success_count, restore_count, failure_count, inactive_count,
            median_success, spread, worst_failure, leading_singular, alignment_margin,
            force_gain, path_uniformity, stability)
SCICODE_GOLD_EOF
