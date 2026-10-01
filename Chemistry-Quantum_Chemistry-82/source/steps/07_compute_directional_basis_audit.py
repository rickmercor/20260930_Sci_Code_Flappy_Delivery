"""
Construct the fixture with default_rng(seed). For n=direction_count and j=0,...,n-1, set phi=(1+sqrt(5))/2, z=1-2(j+0.5)/n, rho=sqrt(1-z^2), azimuth=2pij/phi, normalize [rho cos(azimuth),rho sin(azimuth),z], and assign weight 4pi/n. Draw reference then mode arrays of shape (molecule_count,3,3,3) from normal distributions with scales 0.42 and 1.0, symmetrize their final two axes, add 1.05+0.06m, -0.72+0.025m, and 0.48+0.035(m mod 5) to reference entries [0,0,0], [1,1,1], and [2,2,2], and Frobenius-normalize each mode. Use base_bias=[0.24,0.075,0.12,0.026,0.052,0.006], anis_scale=[0.19,0.105,0.12,0.065,0.07,0.028], group_anisotropy=[0.38,0.42,0.72,2.25], groups m mod 4, and the row-major group_sign matrix [[1,-0.85,1,0.15],[0.75,-0.65,-0.45,0.05],[0.85,-0.72,0.8,0.10],[0.55,-0.52,-0.35,0.02],[0.68,-0.60,0.55,0.08],[0.34,-0.28,-0.22,0.01]]. At frequency index f, let x = omega[f]/omega_max, scale = 1 + 0.24 × x + 0.11 × x^2, beta_mra = scale × reference + 0.035 × f × mode, bias = base_bias[b] × group_sign[b,group], modulation = 1 + 0.09 × sin((m+1)(f+1)), and beta_basis = (1 + bias × modulation) × beta_mra + anis_scale[b] × group_anisotropy[group] × (0.58 + 0.07 × f) × mode. Call Steps 1 through 5 for every molecule, family, and frequency; set the twelve features to frequency means of alternating positive-region and negative-region normalized vector-error RMS values; and call Step 6. Define R, P, N, and A as full-cube means, C as relative[molecule_count//2,3,2], K as the row-major weighted feature checksum, D as the last-frequency mean relative error minus the static mean, and G, H, B from Step 6. Return J and all diagnostics, where J=R+0.35(P+N)+0.025A+0.02H+0.0005B+0.1|D|+0.005G.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
tuple: Final scalar J followed by diagnostics, frequencies, error cube, and feature matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_directional_basis_audit(seed: int = 82026, molecule_count: int = 68, direction_count: int = 86, omega_max: float = 0.36, max_clusters: int = 4) -> tuple:
    """Run the complete deterministic directional basis audit.

    Parameters
    ----------
    seed : int
        Seed for the disclosed tensor fixture.
    molecule_count : int
        Number of synthetic molecules, at least eight.
    direction_count : int
        Number of equal-weight spherical directions, at least 24.
    omega_max : float
        Positive first-excitation estimate.
    max_clusters : int
        Largest mixture count accepted by Step 6.
    Returns
    -------
    tuple
        J, ten scalar diagnostics, frequency vector, error cube, feature matrix, labels, and BIC vector.
    Raises
    ------
    ValueError
        If a top-level input violates its documented contract.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _fixture(seed: int, molecule_count: int, direction_count: int) -> tuple:
    rng = np.random.default_rng(int(seed))
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    indices = np.arange(direction_count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / direction_count
    radius = np.sqrt(1.0 - z * z)
    azimuth = 2.0 * np.pi * indices / golden_ratio
    directions = np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    weights = np.full(direction_count, 4.0 * np.pi / direction_count, dtype=float)
    reference = rng.normal(0.0, 0.42, size=(molecule_count, 3, 3, 3))
    reference = 0.5 * (reference + np.swapaxes(reference, 2, 3))
    modes = rng.normal(0.0, 1.0, size=(molecule_count, 3, 3, 3))
    modes = 0.5 * (modes + np.swapaxes(modes, 2, 3))
    for molecule in range(molecule_count):
        reference[molecule, 0, 0, 0] += 1.05 + 0.06 * molecule
        reference[molecule, 1, 1, 1] -= 0.72 - 0.025 * molecule
        reference[molecule, 2, 2, 2] += 0.48 + 0.035 * (molecule % 5)
        modes[molecule] /= np.linalg.norm(modes[molecule])
    return directions, weights, reference, modes


def _oracle_compute_directional_basis_audit(
    seed: int = 82026,
    molecule_count: int = 68,
    direction_count: int = 86,
    omega_max: float = 0.36,
    max_clusters: int = 4,
) -> tuple:
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not isinstance(molecule_count, (int, np.integer)) or int(molecule_count) < 8:
        raise ValueError("molecule_count must be an integer at least 8")
    if not isinstance(direction_count, (int, np.integer)) or int(direction_count) < 24:
        raise ValueError("direction_count must be an integer at least 24")
    frequencies = _oracle_build_shg_frequency_grid(omega_max)
    directions, weights, reference, modes = _fixture(seed, int(molecule_count), int(direction_count))
    basis_count = 6
    relative = np.empty((molecule_count, basis_count, frequencies.size), dtype=float)
    positive = np.empty_like(relative)
    negative = np.empty_like(relative)
    anisotropy = np.empty_like(relative)
    groups = np.arange(molecule_count) % 4
    base_bias = np.array([0.24, 0.075, 0.12, 0.026, 0.052, 0.006], dtype=float)
    anis_scale = np.array([0.19, 0.105, 0.12, 0.065, 0.07, 0.028], dtype=float)
    group_sign = np.array(
        [
            [1.0, -0.85, 1.0, 0.15],
            [0.75, -0.65, -0.45, 0.05],
            [0.85, -0.72, 0.8, 0.10],
            [0.55, -0.52, -0.35, 0.02],
            [0.68, -0.60, 0.55, 0.08],
            [0.34, -0.28, -0.22, 0.01],
        ],
        dtype=float,
    )
    group_anisotropy = np.array([0.38, 0.42, 0.72, 2.25], dtype=float)
    frequency_scale = 1.0 + 0.24 * frequencies / float(omega_max) + 0.11 * (frequencies / float(omega_max)) ** 2
    for molecule in range(int(molecule_count)):
        for basis in range(basis_count):
            for frequency_index, scale in enumerate(frequency_scale):
                beta_mra = reference[molecule] * scale + 0.035 * frequency_index * modes[molecule]
                bias = base_bias[basis] * group_sign[basis, groups[molecule]]
                modulation = 1.0 + 0.09 * math.sin((molecule + 1) * (frequency_index + 1))
                beta_basis = (1.0 + bias * modulation) * beta_mra + anis_scale[basis] * group_anisotropy[
                    groups[molecule]
                ] * (0.58 + 0.07 * frequency_index) * modes[molecule]
                relative[molecule, basis, frequency_index] = _oracle_compute_relative_rms_total(
                    beta_basis, beta_mra, directions, weights
                )[0]
                signed = _oracle_compute_signed_projection_metrics(beta_basis, beta_mra, directions, weights)
                positive[molecule, basis, frequency_index] = signed[0]
                negative[molecule, basis, frequency_index] = signed[1]
                anisotropy[molecule, basis, frequency_index] = _oracle_compute_component_rms_anisotropy(
                    beta_basis, beta_mra, directions, weights
                )[1]
    features = np.empty((molecule_count, 2 * basis_count), dtype=float)
    features[:, 0::2] = np.mean(positive, axis=2)
    features[:, 1::2] = np.mean(negative, axis=2)
    best_clusters, labels, bics, entropy, bic_gap = _oracle_select_convergence_clusters(
        features, max_clusters=max_clusters, regularization=1e-6
    )
    mean_relative = float(np.mean(relative))
    mean_positive = float(np.mean(positive))
    mean_negative = float(np.mean(negative))
    mean_anisotropy = float(np.mean(anisotropy))
    central_relative = float(relative[molecule_count // 2, 3, 2])
    coefficients = np.arange(1, features.size + 1, dtype=float).reshape(features.shape)
    feature_checksum = float(np.sum(coefficients * features))
    frequency_drift = float(np.mean(relative[:, :, -1]) - np.mean(relative[:, :, 0]))
    j_value = (
        mean_relative
        + 0.35 * (mean_positive + mean_negative)
        + 0.025 * mean_anisotropy
        + 0.02 * entropy
        + 0.0005 * bic_gap
        + 0.1 * abs(frequency_drift)
        + 0.005 * best_clusters
    )
    return (
        float(j_value),
        mean_relative,
        mean_positive,
        mean_negative,
        mean_anisotropy,
        central_relative,
        feature_checksum,
        int(best_clusters),
        float(entropy),
        float(bic_gap),
        frequency_drift,
        frequencies,
        relative,
        features,
        labels,
        bics,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common="""import numpy as np
def _flat(v):
 p=[]
 for x in v if isinstance(v,tuple) else (v,):p.extend(np.asarray(x,dtype=float).ravel().tolist())
 return np.asarray(p)
def _e(g,*a):
 try:g(*a)
 except ValueError:return 1.0
 return 0.0"""
    return [
        {"setup":common,"call":"_flat(compute_directional_basis_audit())","gold_call":"_flat(_oracle_compute_directional_basis_audit())","tol":1e-7},
        {"setup":common,"call":"_flat(compute_directional_basis_audit(11,24,42,.4,3))","gold_call":"_flat(_oracle_compute_directional_basis_audit(11,24,42,.4,3))","tol":1e-7},
        {"setup":common,"call":"_flat(compute_directional_basis_audit(907,32,60,.28,4))","gold_call":"_flat(_oracle_compute_directional_basis_audit(907,32,60,.28,4))","tol":1e-7},
        {"setup":common,"call":"_e(compute_directional_basis_audit,1,7,24,.3,3)","gold_call":"_e(_oracle_compute_directional_basis_audit,1,7,24,.3,3)","tol":1e-12}
    ]
