"""
Run the complete seven-step probabilistic force-matching pipeline and compare the fitted symmetric precision with the exact Gaussian-broadened precision.

For a harmonic atomistic reference with precision K and linear coordinate map M, the deterministic mapped covariance is M K^-1 M^T. Gaussian coarse-graining with variance sigma squared adds sigma^2 I. The exact noised-PMF precision is therefore the inverse of M K^-1 M^T + sigma^2 I. This orchestrator must call all seven earlier steps in sequence and return the relative Frobenius error of the fitted precision.

Returns
-------
Return one finite float equal to the relative Frobenius error between the fitted and exact noised-PMF precision matrices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def noised_pmf_precision_error(
    atomistic_precision: np.ndarray,
    mapping: np.ndarray,
    atomistic_positions: np.ndarray,
    atomistic_forces: np.ndarray,
    standard_normals: np.ndarray,
    sigma: float,
    force_map_ridge: float,
    stage_ridge: float,
    fit_ridge: float,
) -> float:
    """Call the seven preceding steps for a harmonic CG reference.

    Parameters
    ----------
    atomistic_precision
        Symmetric positive-definite atomistic precision matrix.
    mapping
        Nonnegative coordinate map with unit row sums.
    atomistic_positions
        Equilibrium frames with shape (frames, atoms, components).
    atomistic_forces
        Harmonic forces with the same shape.
    standard_normals
        Fixed Gaussian draws with shape
        (replicates, frames, beads, components).
    sigma
        Positive Gaussian standard deviation.
    force_map_ridge, stage_ridge, fit_ridge
        Nonnegative regularizers for the three fitted linear systems.

    Returns
    -------
    float
        Relative Frobenius error of the fitted noised-PMF precision.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_noised_pmf_precision_error(
    atomistic_precision,
    mapping,
    atomistic_positions,
    atomistic_forces,
    standard_normals,
    sigma,
    force_map_ridge,
    stage_ridge,
    fit_ridge,
):
    import numpy as np

    precision = np.asarray(atomistic_precision, dtype=float)
    positions = np.asarray(atomistic_positions, dtype=float)
    forces = np.asarray(atomistic_forces, dtype=float)
    coordinate_map = np.asarray(mapping, dtype=float)
    if (
        precision.ndim != 2
        or precision.shape[0] != precision.shape[1]
        or positions.ndim != 3
        or precision.shape[0] != positions.shape[1]
    ):
        raise ValueError("atomistic precision has incompatible shape")
    if np.any(~np.isfinite(precision)):
        raise ValueError("atomistic precision must be finite")
    if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-12):
        raise ValueError("atomistic precision must be symmetric")
    try:
        np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError("atomistic precision must be positive definite") from exc
    expected_forces = -np.einsum("ab,fbc->fac", precision, positions)
    if forces.shape != positions.shape or not np.allclose(
        forces, expected_forces, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("forces do not match the harmonic reference")

    mapped = _oracle_map_atomistic_coordinates(positions, coordinate_map)
    base_map = _oracle_minimum_variance_force_map(
        coordinate_map, forces, force_map_ridge
    )
    noised = _oracle_sample_probabilistic_cg(
        mapped, standard_normals, sigma
    )
    joint = _oracle_build_extended_forces(
        forces, coordinate_map, mapped, noised, sigma
    )
    extended_map = _oracle_fit_staged_force_map(
        joint, base_map, stage_ridge
    )
    targets = _oracle_project_extended_forces(joint, extended_map)
    fitted = _oracle_fit_conservative_precision(noised, targets, fit_ridge)

    unnoised_covariance = coordinate_map @ np.linalg.solve(
        precision, coordinate_map.T
    )
    noised_covariance = unnoised_covariance + sigma * sigma * np.eye(
        coordinate_map.shape[0]
    )
    exact = np.linalg.inv(noised_covariance)
    return float(np.linalg.norm(fitted - exact) / np.linalg.norm(exact))


def _draw_harmonic_fixture(seed, precision, mapping, frames, replicas, sigma):
    import numpy as np

    rng = np.random.default_rng(seed)
    n_atoms = len(precision)
    components = 3
    cholesky = np.linalg.cholesky(precision)
    white = rng.normal(size=(frames, components, n_atoms))
    flattened = np.linalg.solve(
        cholesky.T, white.reshape(-1, n_atoms).T
    ).T
    positions = flattened.reshape(frames, components, n_atoms).transpose(
        0, 2, 1
    )
    forces = -np.einsum("ab,fbc->fac", precision, positions)
    normals = rng.normal(
        size=(replicas, frames, components, len(mapping))
    ).transpose(0, 1, 3, 2)
    return (
        precision,
        mapping,
        positions,
        forces,
        normals,
        sigma,
        1e-4,
        0.02,
        1e-5,
    )


def _benchmark_fixture():
    import numpy as np

    fixture_rng = np.random.default_rng(7129)
    matrix = fixture_rng.normal(size=(6, 6))
    precision = matrix.T @ matrix / 6.0 + 1.25 * np.eye(6)
    sites = np.arange(6, dtype=float)
    centers = 0.4 + 2.1 * np.arange(3, dtype=float)
    raw_map = np.exp(
        -0.55 * (sites[None, :] - centers[:, None]) ** 2
        + 0.08 * fixture_rng.normal(size=(3, 6))
    )
    mapping = raw_map / raw_map.sum(axis=1, keepdims=True)
    return _draw_harmonic_fixture(
        20260822, precision, mapping, 8, 16, 0.22
    )


def _one_bead_fixture():
    import numpy as np

    precision = np.array([[2.4, -0.3], [-0.3, 1.8]])
    mapping = np.array([[0.4, 0.6]])
    return _draw_harmonic_fixture(17, precision, mapping, 5, 6, 0.15)


def _overlap_fixture():
    import numpy as np

    precision = np.array(
        [[2.8, -0.5, 0.1], [-0.5, 2.5, -0.4], [0.1, -0.4, 2.1]]
    )
    mapping = np.array([[0.7, 0.3, 0.0], [0.0, 0.35, 0.65]])
    return _draw_harmonic_fixture(311, precision, mapping, 7, 4, 0.18)


def _four_atom_fixture():
    import numpy as np

    precision = np.array(
        [
            [3.2, -0.6, 0.15, 0.0],
            [-0.6, 2.7, -0.35, 0.1],
            [0.15, -0.35, 2.4, -0.45],
            [0.0, 0.1, -0.45, 2.2],
        ]
    )
    mapping = np.array(
        [[0.55, 0.30, 0.15, 0.0], [0.0, 0.10, 0.35, 0.55]]
    )
    return _draw_harmonic_fixture(902, precision, mapping, 6, 5, 0.27)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nargs=_benchmark_fixture()",
            "call": "noised_pmf_precision_error(*args)",
            "gold_call": "_oracle_noised_pmf_precision_error(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_one_bead_fixture()",
            "call": "noised_pmf_precision_error(*args)",
            "gold_call": "_oracle_noised_pmf_precision_error(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_overlap_fixture()",
            "call": "noised_pmf_precision_error(*args)",
            "gold_call": "_oracle_noised_pmf_precision_error(*args)",
        },
        {
            "setup": "import numpy as np\nargs=_four_atom_fixture()",
            "call": "noised_pmf_precision_error(*args)",
            "gold_call": "_oracle_noised_pmf_precision_error(*args)",
        },
    ]
