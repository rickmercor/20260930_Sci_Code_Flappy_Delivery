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
def build_kinetic_root_seeds(kappa: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    values = np.asarray(kappa, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("kappa must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("kappa values must be finite and positive")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("kappa values must be strictly increasing")

    omega_real = np.sqrt(1.0 + 3.0 * values**2)
    gamma = -np.sqrt(np.pi / 8.0) * np.exp(
        -0.5 / values**2 - 1.5
    ) / values**3
    zeta_real = omega_real / (np.sqrt(2.0) * values)
    zeta_imag = gamma / (np.sqrt(2.0) * values)
    seeds = np.column_stack((values, zeta_real, zeta_imag))
    if not np.all(np.isfinite(seeds)):
        raise ValueError("the asymptotic seed is not finite")
    return seeds.astype(float)

from scipy.special import wofz
import numpy as np
def solve_least_damped_roots(
    seed_table: np.ndarray, residual_tol: float = 1e-12
) -> np.ndarray:
    """Reference implementation."""
    seeds = np.asarray(seed_table, dtype=float)
    tol = float(residual_tol)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or seeds.shape[0] == 0:
        raise ValueError("seed_table must have shape (m, 3)")
    if not np.all(np.isfinite(seeds)) or not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("seeds must be finite and residual_tol positive")
    output = np.empty((seeds.shape[0], 4), dtype=float)
    for row, (kappa, real_seed, imag_seed) in enumerate(seeds):
        zeta = complex(real_seed, imag_seed)
        for _ in range(64):
            plasma_z = 1j * np.sqrt(np.pi) * wofz(zeta)
            response = 1.0 + zeta * plasma_z
            residual = response + kappa**2
            if abs(residual) <= tol:
                break
            derivative = plasma_z - 2.0 * zeta * response
            if abs(derivative) <= np.finfo(float).eps:
                raise ValueError("singular Newton derivative")
            step = residual / derivative
            if abs(step) > 0.75:
                step *= 0.75 / abs(step)
            zeta -= step
        final_residual = abs(1.0 + zeta * 1j * np.sqrt(np.pi) * wofz(zeta) + kappa**2)
        if final_residual > tol or zeta.real <= 0.0 or zeta.imag >= 0.0:
            raise ValueError("requested least-damped root did not converge")
        output[row] = (kappa, zeta.real, zeta.imag, final_residual)
    return output

import numpy as np
def match_pade_coefficients(root_table: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    roots = np.asarray(root_table, dtype=float)
    if roots.ndim != 2 or roots.shape[1] != 4 or roots.shape[0] == 0:
        raise ValueError("root_table must have shape (m, 4)")
    if not np.all(np.isfinite(roots)):
        raise ValueError("root_table must be finite")
    matched = np.empty((roots.shape[0], 5), dtype=float)
    for row, (kappa, real_part, imag_part, _) in enumerate(roots):
        if kappa <= 0.0 or real_part <= 0.0 or imag_part >= 0.0:
            raise ValueError("roots must lie on the requested kinetic branch")
        zeta = complex(real_part, imag_part)
        a_column = zeta - 2.0 * kappa**2 * zeta**3
        b_column = kappa**2 * zeta
        right_side = -(1.0 + kappa**2 - 2.0 * kappa**2 * zeta**2)
        system = np.array(
            [[(1j * a_column).real, (1j * b_column).real],
             [(1j * a_column).imag, (1j * b_column).imag]],
            dtype=float,
        )
        rhs = np.array([right_side.real, right_side.imag], dtype=float)
        if abs(np.linalg.det(system)) <= 1e-18:
            raise ValueError("root-matching system is singular")
        alpha, beta = np.linalg.solve(system, rhs)
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError("matched coefficients must be finite")
        matched[row] = (kappa, real_part, imag_part, alpha, beta)
    return matched

import numpy as np
def derive_closure_coefficients(pade_table: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    table = np.asarray(pade_table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 5 or table.shape[0] == 0:
        raise ValueError("pade_table must have shape (m, 5)")
    if not np.all(np.isfinite(table)):
        raise ValueError("pade_table must be finite")
    alpha = table[:, 3]
    beta = table[:, 4]
    if np.any(np.abs(alpha) <= np.finfo(float).eps):
        raise ValueError("a1 cannot vanish")
    q1 = beta / alpha - 3.0
    q3 = -1.0 / alpha
    result = np.column_stack(
        (table[:, 0], table[:, 1], table[:, 2], q1, q3)
    )
    if not np.all(np.isfinite(result)) or np.any(q1 <= 0.0) or np.any(q3 <= 0.0):
        raise ValueError("the requested branch must yield positive real closure coefficients")
    return result.astype(float)

import numpy as np
def assemble_moment_generators(closure_table: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    closure = np.asarray(closure_table, dtype=float)
    if closure.ndim != 2 or closure.shape[1] != 5 or closure.shape[0] == 0:
        raise ValueError("closure_table must have shape (m, 5)")
    if not np.all(np.isfinite(closure)):
        raise ValueError("closure_table must be finite")
    matrices = np.zeros((closure.shape[0], 3, 3), dtype=complex)
    sqrt_two = np.sqrt(2.0)
    for row, values in enumerate(closure):
        kappa, _, _, q1, q3 = values
        if kappa <= 0.0 or q1 <= 0.0 or q3 <= 0.0:
            raise ValueError("kappa, Q1, and Q3 must be positive")
        matrices[row, 0, 1] = -1j * sqrt_two * kappa
        matrices[row, 1, 0] = -1j / (sqrt_two * kappa)
        matrices[row, 1, 2] = -1j * kappa / sqrt_two
        matrices[row, 2, 0] = sqrt_two * kappa * q3
        matrices[row, 2, 1] = -1j * sqrt_two * kappa * (3.0 + q1)
        matrices[row, 2, 2] = -sqrt_two * kappa * q3
    if not np.all(np.isfinite(matrices)):
        raise ValueError("moment generators must be finite")
    return matrices

import numpy as np
def initialize_isothermal_modes(
    generators: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    """Reference implementation."""
    matrices = np.asarray(generators, dtype=complex)
    amp = np.asarray(amplitudes, dtype=float)
    phase = np.asarray(phases, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or matrices.shape[0] == 0:
        raise ValueError("generators must have shape (m, 3, 3)")
    if amp.ndim != 1 or phase.ndim != 1 or amp.shape != phase.shape:
        raise ValueError("amplitudes and phases must be aligned vectors")
    if amp.size != matrices.shape[0]:
        raise ValueError("mode controls must align with generators")
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(amp)):
        raise ValueError("generators and amplitudes must be finite")
    if not np.all(np.isfinite(phase)) or np.any(amp < 0.0):
        raise ValueError("phases must be finite and amplitudes nonnegative")
    density = 0.5 * amp * np.exp(1j * phase)
    modes = np.zeros((amp.size, 3), dtype=complex)
    modes[:, 0] = density
    modes[:, 2] = density
    return modes

from numbers import Real
import numpy as np
def propagate_midpoint_modes(
    generators: np.ndarray,
    initial_modes: np.ndarray,
    delta_t: float,
    final_time: float,
) -> np.ndarray:
    """Reference implementation."""
    matrices = np.asarray(generators, dtype=complex)
    state = np.asarray(initial_modes, dtype=complex).copy()
    if isinstance(delta_t, bool) or not isinstance(delta_t, Real):
        raise ValueError("delta_t must be a real scalar")
    if isinstance(final_time, bool) or not isinstance(final_time, Real):
        raise ValueError("final_time must be a real scalar")
    step = float(delta_t)
    stop = float(final_time)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("generators must have shape (m, 3, 3)")
    if state.shape != (matrices.shape[0], 3) or matrices.shape[0] == 0:
        raise ValueError("initial_modes must have shape (m, 3)")
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(state)):
        raise ValueError("mode data must be finite")
    if not np.isfinite(step) or step <= 0.0 or not np.isfinite(stop) or stop < 0.0:
        raise ValueError("time controls are outside their finite domains")
    count = round(stop / step)
    if not np.isclose(count * step, stop, rtol=0.0, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of delta_t")
    for _ in range(count):
        slope_one = np.einsum("mij,mj->mi", matrices, state)
        midpoint = state + 0.5 * step * slope_one
        slope_two = np.einsum("mij,mj->mi", matrices, midpoint)
        state += step * slope_two
    if not np.all(np.isfinite(state)):
        raise ValueError("midpoint propagation produced non-finite modes")
    return state

import numpy as np
def compute_log_field_energy(
    kappa: np.ndarray, initial_modes: np.ndarray, final_modes: np.ndarray
) -> float:
    """Reference implementation."""
    wave_numbers = np.asarray(kappa, dtype=float)
    initial = np.asarray(initial_modes, dtype=complex)
    final = np.asarray(final_modes, dtype=complex)
    if wave_numbers.ndim != 1 or wave_numbers.size == 0:
        raise ValueError("kappa must be a nonempty vector")
    expected = (wave_numbers.size, 3)
    if initial.shape != expected or final.shape != expected:
        raise ValueError("mode arrays must have shape (len(kappa), 3)")
    if not np.all(np.isfinite(wave_numbers)) or np.any(wave_numbers <= 0.0):
        raise ValueError("wave numbers must be finite and positive")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
        raise ValueError("mode arrays must be finite")
    initial_energy = float(np.sum(np.abs(initial[:, 0] / wave_numbers) ** 2))
    final_energy = float(np.sum(np.abs(final[:, 0] / wave_numbers) ** 2))
    if initial_energy <= 0.0 or final_energy <= 0.0:
        raise ValueError("both field energies must be strictly positive")
    result = float(np.log10(final_energy / initial_energy))
    if not np.isfinite(result):
        raise ValueError("logarithmic field-energy ratio must be finite")
    return result

import numpy as np
def run_root_matched_energy(
    kappa: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    delta_t: float,
    final_time: float,
    residual_tol: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    seeds = build_kinetic_root_seeds(kappa)
    roots = solve_least_damped_roots(seeds, residual_tol)
    pade = match_pade_coefficients(roots)
    closure = derive_closure_coefficients(pade)
    generators = assemble_moment_generators(closure)
    initial = initialize_isothermal_modes(generators, amplitudes, phases)
    final = propagate_midpoint_modes(generators, initial, delta_t, final_time)
    result = float(compute_log_field_energy(kappa, initial, final))
    if not np.isfinite(result):
        raise ValueError("final energy diagnostic must be finite")
    return result
SCICODE_GOLD_EOF
