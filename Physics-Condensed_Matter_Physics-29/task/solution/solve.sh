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
from numbers import Real

import numpy as np


def compute_hubbard_drift(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    hbar: float = 1.0,
) -> "np.ndarray":
    """Reference implementation."""
    up = np.asarray(n_up, dtype=complex)
    down = np.asarray(n_down, dtype=complex)
    hop = np.asarray(hopping, dtype=complex)
    if (
        up.ndim != 2
        or up.shape[0] != up.shape[1]
        or up.shape[0] < 1
        or down.shape != up.shape
        or hop.shape != up.shape
        or not np.all(np.isfinite(up))
        or not np.all(np.isfinite(down))
        or not np.all(np.isfinite(hop))
    ):
        raise ValueError("n_up, n_down, and hopping must be finite compatible square arrays")
    if isinstance(hbar, bool) or not isinstance(hbar, Real):
        raise ValueError("hbar must be a positive finite real scalar")
    hbar_value = float(hbar)
    if not math.isfinite(hbar_value) or hbar_value <= 0.0:
        raise ValueError("hbar must be a positive finite real scalar")

    drift_up = (1j / hbar_value) * (hop.T @ up - up @ hop.T)
    drift_down = (1j / hbar_value) * (hop.T @ down - down @ hop.T)
    return np.stack((drift_up, drift_down)).astype(complex)

import math
from numbers import Real

import numpy as np


def compute_opposite_spin_diffusion(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    interaction: float,
) -> "np.ndarray":
    """Reference implementation."""
    up = np.asarray(n_up, dtype=complex)
    down = np.asarray(n_down, dtype=complex)
    if (
        up.ndim != 2
        or up.shape[0] != up.shape[1]
        or up.shape[0] < 1
        or down.shape != up.shape
        or not np.all(np.isfinite(up))
        or not np.all(np.isfinite(down))
    ):
        raise ValueError("n_up and n_down must be finite compatible square arrays")
    if isinstance(interaction, bool) or not isinstance(interaction, Real):
        raise ValueError("interaction must be a finite real scalar")
    interaction_value = float(interaction)
    if not math.isfinite(interaction_value):
        raise ValueError("interaction must be a finite real scalar")

    n_sites = up.shape[0]
    hole_up = np.eye(n_sites, dtype=complex) - up
    hole_down = np.eye(n_sites, dtype=complex) - down
    block = np.zeros((n_sites * n_sites, n_sites * n_sites), dtype=complex)
    for i in range(n_sites):
        for j in range(n_sites):
            row = i * n_sites + j
            for k in range(n_sites):
                for ell in range(n_sites):
                    column = k * n_sites + ell
                    contraction = 0.0j
                    for a in range(n_sites):
                        contraction += (
                            hole_up[a, j]
                            * up[i, a]
                            * hole_down[a, ell]
                            * down[k, a]
                            - up[a, j]
                            * hole_up[i, a]
                            * down[a, ell]
                            * hole_down[k, a]
                        )
                    block[row, column] = 1j * interaction_value * contraction
    return block

import math
from numbers import Integral, Real

import numpy as np


def compute_randomized_svd_factors(
    diffusion_block: "np.ndarray",
    rank: int,
    seed: int,
    tolerance_factor: float = 64.0,
) -> "np.ndarray":
    """Reference implementation."""
    block = np.asarray(diffusion_block, dtype=complex)
    if (
        block.ndim != 2
        or block.shape[0] != block.shape[1]
        or block.shape[0] < 1
        or not np.all(np.isfinite(block))
    ):
        raise ValueError("diffusion_block must be a finite nonempty square array")
    if isinstance(rank, bool) or not isinstance(rank, Integral):
        raise ValueError("rank must be an integer")
    rank_value = int(rank)
    if rank_value < 1 or rank_value > block.shape[0]:
        raise ValueError("rank must satisfy 1 <= rank <= matrix dimension")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(tolerance_factor, bool) or not isinstance(tolerance_factor, Real):
        raise ValueError("tolerance_factor must be positive and finite")
    tolerance_value = float(tolerance_factor)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance_factor must be positive and finite")

    scaled = -0.5j * block
    rng = np.random.default_rng(int(seed))
    sketch = rng.standard_normal((block.shape[1], rank_value))
    image = scaled @ sketch
    basis, _ = np.linalg.qr(image, mode="reduced")
    projected = basis.conj().T @ scaled
    reduced_left, singular_values, right_adjoint = np.linalg.svd(
        projected, full_matrices=False
    )
    left = basis @ reduced_left

    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.empty((2 * block.shape[0] + 1, 0), dtype=complex)

    cutoff = (
        tolerance_value
        * np.finfo(float).eps
        * max(scaled.shape)
        * singular_values[0]
    )
    keep = singular_values > cutoff
    left = left[:, keep]
    singular_values = singular_values[keep]
    right_adjoint = right_adjoint[keep, :]

    for column in range(left.shape[1]):
        magnitudes = np.abs(left[:, column])
        maximum = float(np.max(magnitudes))
        candidates = np.flatnonzero(magnitudes >= maximum * (1.0 - 1.0e-10))
        pivot = int(candidates[0])
        phase = np.exp(-1j * np.angle(left[pivot, column]))
        left[:, column] *= phase
        right_adjoint[column, :] *= np.conj(phase)

    right = right_adjoint.conj().T
    return np.vstack(
        (left, singular_values.astype(complex)[None, :], right)
    )

import numpy as np


def construct_diffusion_gauge(factors: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    packed = np.asarray(factors, dtype=complex)
    if (
        packed.ndim != 2
        or packed.shape[0] < 3
        or packed.shape[0] % 2 != 1
        or not np.all(np.isfinite(packed))
    ):
        raise ValueError("factors must have finite shape (2*m + 1, k)")
    m = (packed.shape[0] - 1) // 2
    k = packed.shape[1]
    if k > m:
        raise ValueError("the retained rank cannot exceed m")
    if k == 0:
        return np.empty((2 * m, 0), dtype=complex)

    left = packed[:m, :]
    singular_row = packed[m, :]
    right = packed[m + 1 :, :]
    if np.max(np.abs(singular_row.imag)) > 1.0e-12:
        raise ValueError("singular values must be real")
    singular_values = singular_row.real
    if np.any(singular_values < 0.0):
        raise ValueError("singular values must be nonnegative")

    root = np.sqrt(singular_values)
    upper = left * root[None, :]
    lower = np.conj(right) * root[None, :]
    return np.block(
        [[upper, 1j * upper], [1j * lower, lower]]
    )

import numpy as np


def compute_factorization_residual(
    diffusion_block: "np.ndarray",
    gauge: "np.ndarray",
) -> float:
    """Reference implementation."""
    block = np.asarray(diffusion_block, dtype=complex)
    noise = np.asarray(gauge, dtype=complex)
    if (
        block.ndim != 2
        or block.shape[0] != block.shape[1]
        or block.shape[0] < 1
        or noise.ndim != 2
        or noise.shape[0] != 2 * block.shape[0]
        or not np.all(np.isfinite(block))
        or not np.all(np.isfinite(noise))
    ):
        raise ValueError("diffusion_block and gauge have incompatible finite shapes")

    zeros = np.zeros_like(block)
    target = np.block([[zeros, block], [block.T, zeros]])
    error = float(np.linalg.norm(target - noise @ noise.T, ord="fro"))
    scale = float(np.linalg.norm(target, ord="fro"))
    return error if scale == 0.0 else error / scale

import math
from numbers import Integral, Real

import numpy as np


def propagate_phase_space_ensemble(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    drift: "np.ndarray",
    gauge: "np.ndarray",
    dt: float,
    n_trajectories: int,
    seed: int,
) -> "np.ndarray":
    """Reference implementation."""
    up = np.asarray(n_up, dtype=complex)
    down = np.asarray(n_down, dtype=complex)
    drift_array = np.asarray(drift, dtype=complex)
    noise = np.asarray(gauge, dtype=complex)
    if (
        up.ndim != 2
        or up.shape[0] != up.shape[1]
        or up.shape[0] < 1
        or down.shape != up.shape
        or drift_array.shape != (2, up.shape[0], up.shape[1])
        or noise.ndim != 2
        or noise.shape[0] != 2 * up.size
        or not np.all(np.isfinite(up))
        or not np.all(np.isfinite(down))
        or not np.all(np.isfinite(drift_array))
        or not np.all(np.isfinite(noise))
    ):
        raise ValueError("phase-space arrays, drift, and gauge have incompatible finite shapes")
    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise ValueError("dt must be positive and finite")
    dt_value = float(dt)
    if not math.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be positive and finite")
    if (
        isinstance(n_trajectories, bool)
        or not isinstance(n_trajectories, Integral)
        or int(n_trajectories) < 1
    ):
        raise ValueError("n_trajectories must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")

    count = int(n_trajectories)
    initial = np.concatenate((up.ravel(order="C"), down.ravel(order="C")))
    rng = np.random.default_rng(int(seed))
    normals = rng.standard_normal((noise.shape[1], count))
    wiener = math.sqrt(dt_value) * normals
    return (
        initial[:, None]
        + drift_array.reshape(-1, 1, order="C") * dt_value
        + noise @ wiener
    )

import math
from numbers import Integral, Real

import numpy as np


def compute_interaction_energy_density(
    trajectories: "np.ndarray",
    interaction: float,
    n_sites: int,
) -> float:
    """Reference implementation."""
    samples = np.asarray(trajectories, dtype=complex)
    if (
        isinstance(n_sites, bool)
        or not isinstance(n_sites, Integral)
        or int(n_sites) < 1
    ):
        raise ValueError("n_sites must be a positive integer")
    site_count = int(n_sites)
    if (
        samples.ndim != 2
        or samples.shape[0] != 2 * site_count * site_count
        or samples.shape[1] < 1
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("trajectories have an incompatible finite shape")
    if isinstance(interaction, bool) or not isinstance(interaction, Real):
        raise ValueError("interaction must be a finite real scalar")
    interaction_value = float(interaction)
    if not math.isfinite(interaction_value):
        raise ValueError("interaction must be a finite real scalar")

    estimates = np.empty(samples.shape[1], dtype=complex)
    split = site_count * site_count
    for trajectory in range(samples.shape[1]):
        up = samples[:split, trajectory].reshape(site_count, site_count, order="C")
        down = samples[split:, trajectory].reshape(site_count, site_count, order="C")
        estimates[trajectory] = (
            interaction_value
            * np.sum(np.diag(up) * np.diag(down))
            / site_count
        )
    return float(np.real(np.mean(estimates)))

import math
from numbers import Integral, Real

import numpy as np


def run_diffusion_gauge_benchmark(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    interaction: float,
    hbar: float,
    rank: int,
    sketch_seed: int,
    tolerance_factor: float,
    dt: float,
    n_trajectories: int,
    noise_seed: int,
    residual_tolerance: float = 1.0e-10,
    decimals: int = 12,
) -> float:
    """Reference implementation."""
    if (
        isinstance(residual_tolerance, bool)
        or not isinstance(residual_tolerance, Real)
        or not math.isfinite(float(residual_tolerance))
        or float(residual_tolerance) <= 0.0
    ):
        raise ValueError("residual_tolerance must be positive and finite")
    if (
        isinstance(decimals, bool)
        or not isinstance(decimals, Integral)
        or int(decimals) < 0
        or int(decimals) > 15
    ):
        raise ValueError("decimals must be an integer from 0 to 15")

    drift = compute_hubbard_drift(n_up, n_down, hopping, hbar)
    diffusion_block = compute_opposite_spin_diffusion(
        n_up, n_down, interaction
    )
    factors = compute_randomized_svd_factors(
        diffusion_block, rank, sketch_seed, tolerance_factor
    )
    gauge = construct_diffusion_gauge(factors)
    residual = compute_factorization_residual(diffusion_block, gauge)
    if not np.isfinite(residual) or residual > float(residual_tolerance):
        raise ValueError("diffusion gauge does not satisfy the required covariance tolerance")
    trajectories = propagate_phase_space_ensemble(
        n_up,
        n_down,
        drift,
        gauge,
        dt,
        n_trajectories,
        noise_seed,
    )
    energy = compute_interaction_energy_density(
        trajectories, interaction, np.asarray(n_up).shape[0]
    )
    return float(round(energy, int(decimals)))
SCICODE_GOLD_EOF
