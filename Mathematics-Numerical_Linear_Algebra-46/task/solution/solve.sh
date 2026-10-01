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
def compute_critical_driving_force(
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    fracture_energy: float,
    horizon: float,
) -> tuple:
    """Reference implementation (exact moments of the piecewise-polynomial profile)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _moment(breaks, coeffs, power):
        # int_0^1 p(rho) rho**power drho, integrated exactly piece by piece.
        total = 0.0
        for j in range(coeffs.shape[0]):
            low, high = breaks[j], breaks[j + 1]
            for m, a in enumerate(coeffs[j]):
                q = m + power + 1
                total += a * (high ** q - low ** q) / q
        return total

    if not (_is_number(fracture_energy) and fracture_energy > 0.0):
        raise ValueError("fracture_energy must be a finite positive number")
    if not (_is_number(horizon) and horizon > 0.0):
        raise ValueError("horizon must be a finite positive number")
    try:
        breaks = np.asarray(profile_breaks, dtype=float)
        coeffs = np.asarray(profile_coeffs, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("profile arrays must be numeric") from None
    if breaks.ndim != 1 or breaks.size < 2 or not np.all(np.isfinite(breaks)):
        raise ValueError("profile_breaks must be a finite 1D array of length >= 2")
    if breaks[0] != 0.0 or breaks[-1] != 1.0 or np.any(np.diff(breaks) <= 0.0):
        raise ValueError("profile_breaks must increase strictly from 0 to 1")
    if coeffs.ndim != 2 or coeffs.shape[0] != breaks.size - 1 or coeffs.shape[1] < 1:
        raise ValueError("profile_coeffs needs one coefficient row per interval")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("profile_coeffs must be finite")

    # A point at height z = xi * delta above the crack sees the cap of its
    # neighbourhood below the plane; the cap weight fraction is
    # f(xi) = int_xi^1 p rho (rho - xi) drho / (2 int_0^1 p rho^2 drho).
    # Both sides of the plane contribute, so c_0 = 2 int_0^1 f(xi) dxi, and
    # exchanging the order of integration leaves a ratio of two moments.
    second = _moment(breaks, coeffs, 2)
    third = _moment(breaks, coeffs, 3)
    if not (second > 0.0 and third > 0.0):
        raise ValueError("the profile moments must be positive")
    c0 = third / (2.0 * second)
    critical = float(fracture_energy) / (2.0 * c0 * float(horizon))
    return float(c0), float(critical)

import numpy as np
def build_bond_family(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
) -> tuple:
    """Reference implementation (integer offset stencil with a sorted bond list)."""
    import itertools
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    try:
        shape = tuple(grid_shape)
    except TypeError:
        raise ValueError("grid_shape must hold three integers") from None
    if len(shape) != 3 or not all(_is_integer(n) and n > 0 for n in shape):
        raise ValueError("grid_shape must hold three positive integers")
    if not (_is_number(spacing) and spacing > 0.0):
        raise ValueError("spacing must be a finite positive number")
    if not (_is_number(horizon) and horizon >= spacing):
        raise ValueError("horizon must be finite and at least the spacing")
    breaks = np.asarray(profile_breaks, dtype=float)
    coeffs = np.asarray(profile_coeffs, dtype=float)
    if breaks.ndim != 1 or breaks.size < 2 or not np.all(np.isfinite(breaks)):
        raise ValueError("profile_breaks must be a finite 1D array of length >= 2")
    if breaks[0] != 0.0 or breaks[-1] != 1.0 or np.any(np.diff(breaks) <= 0.0):
        raise ValueError("profile_breaks must increase strictly from 0 to 1")
    if coeffs.ndim != 2 or coeffs.shape[0] != breaks.size - 1 or not np.all(np.isfinite(coeffs)):
        raise ValueError("profile_coeffs needs one finite coefficient row per interval")
    powers = np.arange(coeffs.shape[1])
    m2 = float(np.sum(coeffs * (breaks[1:, None] ** (powers + 3)
                                - breaks[:-1, None] ** (powers + 3)) / (powers + 3)))
    if not m2 > 0.0:
        raise ValueError("the profile must have a positive second moment")

    nx, ny, nz = (int(n) for n in shape)
    grid = np.array(list(itertools.product(range(nx), range(ny), range(nz))), dtype=int)
    positions = float(spacing) * grid.astype(float)
    reach = int(np.floor(horizon / spacing)) + 1
    starts, ends, lengths = [], [], []
    for offset in itertools.product(range(-reach, reach + 1), repeat=3):
        length = float(spacing) * float(np.sqrt(np.dot(offset, offset)))
        if length == 0.0 or length > horizon:
            continue
        target = grid + np.array(offset)
        inside = np.all((target >= 0) & (target < np.array([nx, ny, nz])), axis=1)
        source = np.nonzero(inside)[0]
        starts.append(source)
        ends.append((target[inside, 0] * ny + target[inside, 1]) * nz + target[inside, 2])
        lengths.append(np.full(source.size, length))
    start = np.concatenate(starts).astype(np.int64)
    end = np.concatenate(ends).astype(np.int64)
    length = np.concatenate(lengths)
    order = np.lexsort((end, start))
    start, end, length = start[order], end[order], length[order]

    rho = length / float(horizon)
    piece = np.clip(np.searchsorted(breaks, rho, side="right") - 1, 0, coeffs.shape[0] - 1)
    profile = np.sum(coeffs[piece] * rho[:, None] ** powers, axis=1)
    weight = profile / (4.0 * np.pi * float(horizon) ** 3 * m2)
    return positions, start, end, weight

import numpy as np
def assemble_nodal_deformation_gradients(
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
    bond_weight: "np.ndarray",
    kinematic_weight: "np.ndarray",
    nodal_volume: "float | np.ndarray",
    basis: str = "C1",
) -> tuple:
    """Reference implementation (normal equations of the weighted fit)."""
    import numpy as np

    ref = np.asarray(ref_positions, dtype=float)
    cur = np.asarray(cur_positions, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or cur.shape != ref.shape:
        raise ValueError("positions must be two (N, 3) arrays of equal shape")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(cur))):
        raise ValueError("positions must be finite")
    start = np.asarray(bond_start)
    end = np.asarray(bond_end)
    weight = np.asarray(bond_weight, dtype=float)
    kin = np.asarray(kinematic_weight, dtype=float)
    if start.ndim != 1 or not (start.shape == end.shape == weight.shape == kin.shape):
        raise ValueError("bond arrays must share one 1D length")
    if not (np.issubdtype(start.dtype, np.integer) and np.issubdtype(end.dtype, np.integer)):
        raise ValueError("bond indices must be integers")
    count = ref.shape[0]
    if start.size and (min(start.min(), end.min()) < 0 or max(start.max(), end.max()) >= count):
        raise ValueError("bond indices must lie in [0, N)")
    if not (np.all(np.isfinite(weight)) and np.all(np.isfinite(kin))):
        raise ValueError("weights must be finite")
    if np.any(weight < 0.0) or np.any(kin < 0.0):
        raise ValueError("weights must be nonnegative")
    volume = np.asarray(nodal_volume, dtype=float)
    if volume.ndim == 0:
        volume = np.full(count, float(volume))
    if volume.shape != (count,) or not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("nodal_volume must be positive and scalar or shape (N,)")
    if basis not in ("C1", "RK1", "RK2"):
        raise ValueError("unsupported polynomial basis")

    ref_bond = ref[end] - ref[start]
    cur_bond = cur[end] - cur[start]
    coef = weight * kin * volume[end]
    shape = np.zeros((count, 3, 3))
    moment = np.zeros((count, 3, 3))
    for a in range(3):
        for b in range(3):
            shape[:, a, b] = np.bincount(start, coef * ref_bond[:, a] * ref_bond[:, b], minlength=count)
            moment[:, a, b] = np.bincount(start, coef * cur_bond[:, a] * ref_bond[:, b], minlength=count)
    try:
        np.linalg.cholesky(shape)
    except np.linalg.LinAlgError:
        raise ValueError("every shape tensor must be positive definite") from None
    # Normal equations: F K = sum c dx dX^T, so F^T = K^{-1} (sum c dX dx^T).
    gradients = np.linalg.solve(shape, moment.transpose(0, 2, 1)).transpose(0, 2, 1)
    if basis != "C1":
        # Dimensionless monomials keep the mixed polynomial orders balanced.
        scale = np.zeros(count)
        np.maximum.at(scale, start, np.linalg.norm(ref_bond, axis=1))
        q = ref_bond / scale[start, None]
        columns = [np.ones(start.size), q[:, 0], q[:, 1], q[:, 2]]
        if basis == "RK2":
            columns.extend([q[:, 0] ** 2, q[:, 1] ** 2, q[:, 2] ** 2,
                            q[:, 0] * q[:, 1], q[:, 0] * q[:, 2], q[:, 1] * q[:, 2]])
        design = np.column_stack(columns)
        width = design.shape[1]
        moments = np.zeros((count, width, width))
        rhs = np.zeros((count, width, 3))
        for a in range(width):
            for b in range(width):
                moments[:, a, b] = np.bincount(start, coef * design[:, a] * design[:, b], minlength=count)
            for b in range(3):
                rhs[:, a, b] = np.bincount(start, coef * design[:, a] * cur_bond[:, b], minlength=count)
        if np.any(np.linalg.matrix_rank(moments) < width):
            raise ValueError("every weighted polynomial fit must have full column rank")
        coefficients = np.linalg.solve(moments, rhs)
        gradients = coefficients[:, 1:4, :].transpose(0, 2, 1) / scale[:, None, None]
    return shape, gradients

import numpy as np
def compute_bond_deformation_gradients(
    nodal_gradients: "np.ndarray",
    ref_positions: "np.ndarray",
    cur_positions: "np.ndarray",
    bond_start: "np.ndarray",
    bond_end: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation (rank-one correction of the averaged gradient)."""
    import numpy as np

    grads = np.asarray(nodal_gradients, dtype=float)
    ref = np.asarray(ref_positions, dtype=float)
    cur = np.asarray(cur_positions, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or cur.shape != ref.shape:
        raise ValueError("positions must be two (N, 3) arrays of equal shape")
    if grads.shape != (ref.shape[0], 3, 3):
        raise ValueError("nodal_gradients must have shape (N, 3, 3)")
    if not (np.all(np.isfinite(grads)) and np.all(np.isfinite(ref)) and np.all(np.isfinite(cur))):
        raise ValueError("inputs must be finite")
    start = np.asarray(bond_start)
    end = np.asarray(bond_end)
    if start.ndim != 1 or start.shape != end.shape:
        raise ValueError("bond index arrays must share one 1D length")
    if not (np.issubdtype(start.dtype, np.integer) and np.issubdtype(end.dtype, np.integer)):
        raise ValueError("bond indices must be integers")
    count = ref.shape[0]
    if start.size and (min(start.min(), end.min()) < 0 or max(start.max(), end.max()) >= count):
        raise ValueError("bond indices must lie in [0, N)")

    ref_bond = ref[end] - ref[start]
    cur_bond = cur[end] - cur[start]
    length_sq = np.einsum("bi,bi->b", ref_bond, ref_bond)
    if np.any(length_sq <= 0.0):
        raise ValueError("every bond needs a positive reference length")
    average = 0.5 * (grads[start] + grads[end])
    # Replace the averaged gradient's action along the bond by the bond's own
    # image: F_b = F_avg + (dx - F_avg dX) dX^T / |dX|^2.
    mismatch = cur_bond - np.einsum("bij,bj->bi", average, ref_bond)
    correction = mismatch[:, :, None] * ref_bond[:, None, :] / length_sq[:, None, None]
    return average + correction

import numpy as np
def compute_crack_driving_force(
    bond_gradients: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
) -> "np.ndarray":
    """Reference implementation (Cauchy stress of the undamaged response)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    grads = np.asarray(bond_gradients, dtype=float)
    if grads.ndim != 3 or grads.shape[1:] != (3, 3):
        raise ValueError("bond_gradients must have shape (N_b, 3, 3)")
    if not np.all(np.isfinite(grads)):
        raise ValueError("bond_gradients must be finite")
    if not (_is_number(youngs_modulus) and youngs_modulus > 0.0):
        raise ValueError("youngs_modulus must be a finite positive number")
    if not (_is_number(poisson_ratio) and -1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    jacobian = np.linalg.det(grads)
    if np.any(jacobian <= 0.0):
        raise ValueError("every bond gradient needs a positive determinant")

    modulus = float(youngs_modulus)
    nu = float(poisson_ratio)
    lame = modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    shear = modulus / (2.0 * (1.0 + nu))
    identity = np.eye(3)
    green = 0.5 * (np.einsum("bki,bkj->bij", grads, grads) - identity)
    trace = np.trace(green, axis1=1, axis2=2)
    second_pk = lame * trace[:, None, None] * identity + 2.0 * shear * green
    cauchy = np.einsum("bij,bjk,blk->bil", grads, second_pk, grads) / jacobian[:, None, None]
    # Symmetrize against round-off before the symmetric eigen-solver.
    cauchy = 0.5 * (cauchy + cauchy.transpose(0, 2, 1))
    largest = np.linalg.eigvalsh(cauchy)[:, -1]
    tensile = np.maximum(largest, 0.0)
    return tensile ** 2 / (2.0 * modulus)

import numpy as np
def update_bond_phase_field(
    driving_force: "np.ndarray",
    history: "np.ndarray",
    critical_driving_force: float,
    kinematic_threshold: float,
) -> tuple:
    """Reference implementation (history maximum, closed-form law, delayed weight)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    try:
        force = np.asarray(driving_force, dtype=float)
        previous = np.asarray(history, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("driving_force and history must be numeric") from None
    if force.ndim != 1 or previous.shape != force.shape:
        raise ValueError("driving_force and history must be 1D arrays of equal length")
    if not (np.all(np.isfinite(force)) and np.all(np.isfinite(previous))):
        raise ValueError("driving_force and history must be finite")
    if np.any(force < 0.0) or np.any(previous < 0.0):
        raise ValueError("driving_force and history must be nonnegative")
    if not (_is_number(critical_driving_force) and critical_driving_force > 0.0):
        raise ValueError("critical_driving_force must be a finite positive number")
    if not (_is_number(kinematic_threshold) and 0.0 <= kinematic_threshold < 1.0):
        raise ValueError("kinematic_threshold must lie in [0, 1)")

    critical = float(critical_driving_force)
    threshold = float(kinematic_threshold)
    # Irreversibility: the phase field follows the largest force seen so far.
    new_history = np.maximum(previous, force)
    phase = np.minimum(1.0, new_history / (new_history + critical))
    # The kinematic weight stays at one until s passes s_c, then falls
    # quadratically to zero at s = 1; it is continuous at s = s_c.
    remaining = (1.0 - phase) / (1.0 - threshold)
    weight = np.where(phase <= threshold, 1.0, remaining ** 2)
    return new_history, phase, weight

import numpy as np
def run_separation_history(
    grid_shape: tuple,
    spacing: float,
    horizon: float,
    profile_breaks: "np.ndarray",
    profile_coeffs: "np.ndarray",
    youngs_modulus: float,
    poisson_ratio: float,
    fracture_energy: float,
    kinematic_threshold: float,
    separation_layer: int,
    slip_angle: float,
    openings: "np.ndarray",
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> tuple:
    """Reference implementation (one lagged pass of the force-evaluation algorithm per state)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    try:
        loads = np.asarray(openings, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("openings must be numeric") from None
    if loads.size == 0 or not np.all(np.isfinite(loads)):
        raise ValueError("openings must be nonempty and finite")
    if loads.ndim == 1:
        if np.any(loads < 0.0):
            raise ValueError("1D openings must be nonnegative")
        loads = np.column_stack([loads, np.zeros_like(loads), np.zeros_like(loads)])
    elif loads.ndim != 2 or loads.shape[1] != 3:
        raise ValueError("openings must have shape (T,) or (T, 3)")
    if not _is_number(slip_angle):
        raise ValueError("slip_angle must be a finite number")

    _, critical = compute_critical_driving_force(
        profile_breaks, profile_coeffs, fracture_energy, horizon
    )
    positions, start, end, weight = build_bond_family(
        grid_shape, spacing, horizon, profile_breaks, profile_coeffs
    )
    ny, nz = int(grid_shape[1]), int(grid_shape[2])
    if not (_is_integer(separation_layer) and 0 <= separation_layer <= ny - 2):
        raise ValueError("separation_layer must be an integer in [0, N_y - 2]")
    layer = (np.arange(positions.shape[0]) // nz) % ny
    side = np.where(layer > separation_layer, 1.0, -1.0)[:, None]
    angle = np.radians(float(slip_angle))
    frame = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    factors = np.ones(positions.shape[0]) if volume_factors is None else np.asarray(volume_factors, dtype=float)
    if factors.shape != (positions.shape[0],) or not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("volume_factors must be positive finite shape (N,)")
    volume = float(spacing) ** 3 * factors
    history = np.zeros(start.size) if initial_history is None else np.asarray(initial_history, dtype=float).copy()
    if history.shape != (start.size,):
        raise ValueError("initial_history must have shape (N_b,)")
    history, phase, kinematic = update_bond_phase_field(
        np.zeros(start.size), history, critical, kinematic_threshold
    )
    for local_translation in loads:
        current = positions + side * (0.5 * (frame @ local_translation))
        # Kinematics first, with the weights of the previous state.
        _, nodal = assemble_nodal_deformation_gradients(
            positions, current, start, end, weight, kinematic, volume, basis
        )
        bond_gradients = compute_bond_deformation_gradients(
            nodal, positions, current, start, end
        )
        force = compute_crack_driving_force(bond_gradients, youngs_modulus, poisson_ratio)
        # Damage update last; its kinematic weight enters the next state.
        history, phase, kinematic = update_bond_phase_field(
            force, history, critical, kinematic_threshold
        )
    return history, phase, kinematic

import numpy as np
def estimate_flank_conditioning(
    grid_shape: tuple = (10, 8, 5),
    spacing: float = 4.0e-4,
    horizon: float = 1.206e-3,
    profile_breaks: tuple = (0.0, 0.5, 1.0),
    profile_coeffs: tuple = ((1.0, 0.0, -6.0, 6.0), (2.0, -6.0, 6.0, -2.0)),
    youngs_modulus: float = 3.2e10,
    poisson_ratio: float = 0.25,
    fracture_energy: float = 3.0,
    kinematic_threshold: float = 0.95,
    separation_layer: int = 3,
    slip_angle: float = 60.0,
    openings: "tuple | np.ndarray" = (8.0e-7, 1.6e-6, 2.4e-6, 1.0e-6),
    probe_node: tuple = (5, 3, 0),
    basis: str = "C1",
    volume_factors: "np.ndarray | None" = None,
    initial_history: "np.ndarray | None" = None,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    _, _, kinematic = run_separation_history(
        grid_shape, spacing, horizon, profile_breaks, profile_coeffs,
        youngs_modulus, poisson_ratio, fracture_energy, kinematic_threshold,
        separation_layer, slip_angle, openings, basis, volume_factors, initial_history,
    )
    positions, start, end, weight = build_bond_family(
        grid_shape, spacing, horizon, profile_breaks, profile_coeffs
    )
    try:
        probe = tuple(probe_node)
    except TypeError:
        raise ValueError("probe_node must hold three integers") from None
    shape = tuple(int(n) for n in grid_shape)
    if len(probe) != 3 or not all(_is_integer(p) and 0 <= p < n for p, n in zip(probe, shape)):
        raise ValueError("probe_node must hold three integers inside grid_shape")
    index = (probe[0] * shape[1] + probe[1]) * shape[2] + probe[2]

    # Only the shape tensors are needed, so the current configuration is
    # taken equal to the reference one.
    tensors, _ = assemble_nodal_deformation_gradients(
        positions, positions, start, end, weight, kinematic,
        float(spacing) ** 3 * (np.ones(positions.shape[0]) if volume_factors is None else np.asarray(volume_factors)),
        basis,
    )
    tensor = 0.5 * (tensors[index] + tensors[index].T)
    eigenvalues = np.linalg.eigvalsh(tensor)
    if not eigenvalues[0] > 0.0:
        raise ValueError("the probe shape tensor must be positive definite")
    return float(eigenvalues[-1] / eigenvalues[0])
SCICODE_GOLD_EOF
