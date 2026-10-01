#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_dimensionless_groups(
    length: float,
    thickness: float,
    thermal_conductivity: float,
    specific_heat: float,
    density: float,
    thermal_expansion: float,
    attenuation_coefficient: float,
    heat_conversion_fraction: float,
    heat_transfer_coefficient: float,
    laser_radius: float,
    incident_intensity_prefactor: float,
    elapsed_time: float,
) -> np.ndarray:
    import numpy as np

    for name, value in (("length", length), ("thickness", thickness),
                        ("thermal_conductivity", thermal_conductivity),
                        ("specific_heat", specific_heat), ("density", density),
                        ("attenuation_coefficient", attenuation_coefficient),
                        ("heat_transfer_coefficient", heat_transfer_coefficient),
                        ("laser_radius", laser_radius),
                        ("incident_intensity_prefactor", incident_intensity_prefactor)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(thermal_expansion, (int, float, np.floating)) and np.isfinite(thermal_expansion)
            and float(thermal_expansion) != 0.0):
        raise ValueError("thermal_expansion must be finite and nonzero")
    if not (isinstance(heat_conversion_fraction, (int, float, np.floating))
            and np.isfinite(heat_conversion_fraction) and 0.0 < float(heat_conversion_fraction) <= 1.0):
        raise ValueError("heat_conversion_fraction must be in (0, 1]")
    if not (isinstance(elapsed_time, (int, float, np.floating)) and np.isfinite(elapsed_time)
            and float(elapsed_time) >= 0.0):
        raise ValueError("elapsed_time must be a finite number >= 0")

    L = float(length)
    h = float(thickness)
    k = float(thermal_conductivity)
    c_theta = float(specific_heat)
    rho0 = float(density)
    gamma1 = float(thermal_expansion)
    beta_dim = float(attenuation_coefficient)
    eta_th = float(heat_conversion_fraction)
    H = float(heat_transfer_coefficient)
    w_dim = float(laser_radius)
    i0_pref = float(incident_intensity_prefactor)
    t_phys = float(elapsed_time)

    bi = H * h / k
    beta = beta_dim * h
    w = w_dim / h
    gamma1_dimless = gamma1 * eta_th * beta_dim * h * L * i0_pref / k
    t_diff = rho0 * c_theta * h ** 2 / k
    t = t_phys / t_diff

    return np.array([bi, beta, w, gamma1_dimless, t], dtype=float)

def solve_transverse_eigenbasis(bi: float, n_modes: int) -> np.ndarray:
    import numpy as np
    from scipy.optimize import brentq
    from scipy.integrate import quad

    if not (isinstance(bi, (int, float, np.floating)) and np.isfinite(bi) and float(bi) > 0.0):
        raise ValueError("bi must be a finite number > 0")
    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")

    bi = float(bi)
    n_modes = int(n_modes)

    def _find_family(f, n):
        roots = []
        step = 0.0005
        xmax = 50.0 * (n + 4)
        x = 1e-8
        f_prev = f(x)
        while len(roots) < n and x < xmax:
            x_next = x + step
            f_next = f(x_next)
            if f_prev * f_next < 0.0:
                roots.append(brentq(f, x, x_next, xtol=1e-14))
            x, f_prev = x_next, f_next
        if len(roots) < n:
            raise ValueError("failed to bracket the requested number of eigenvalues")
        return np.array(roots[:n], dtype=float)

    nu_even = _find_family(lambda nu: nu * np.sin(nu / 2.0) - bi * np.cos(nu / 2.0), n_modes)
    nu_odd = _find_family(lambda nu: nu * np.cos(nu / 2.0) + bi * np.sin(nu / 2.0), n_modes)

    c_even = np.array([
        1.0 / np.sqrt(quad(lambda X: np.cos(nu * X) ** 2, -0.5, 0.5)[0]) for nu in nu_even
    ])
    c_odd = np.array([
        1.0 / np.sqrt(quad(lambda X: np.sin(nu * X) ** 2, -0.5, 0.5)[0]) for nu in nu_odd
    ])

    rows = np.concatenate([
        np.column_stack([nu_even, c_even, np.zeros_like(nu_even)]),
        np.column_stack([nu_odd, c_odd, np.ones_like(nu_odd)]),
    ], axis=0)

    order = np.argsort(rows[:, 0])
    return rows[order]

def project_heat_source_onto_modes(modes: np.ndarray, beta: float, w: float) -> np.ndarray:
    import numpy as np
    from scipy.integrate import quad

    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] < 2:
        raise ValueError("modes must have shape (N, 3) with N >= 2")
    nu_col, c_col, parity_col = modes[:, 0], modes[:, 1], modes[:, 2]
    if not np.all(np.isfinite(nu_col)) or np.any(nu_col <= 0.0):
        raise ValueError("every eigenvalue in modes must be finite and > 0")
    if not np.all(np.isfinite(c_col)) or np.any(c_col <= 0.0):
        raise ValueError("every normalization constant in modes must be finite and > 0")
    if not np.all(np.isin(parity_col, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")
    if not (isinstance(beta, (int, float, np.floating)) and np.isfinite(beta) and float(beta) > 0.0):
        raise ValueError("beta must be a finite number > 0")
    if not (isinstance(w, (int, float, np.floating)) and np.isfinite(w) and float(w) > 0.0):
        raise ValueError("w must be a finite number > 0")

    beta = float(beta)
    w = float(w)

    proj = np.empty(modes.shape[0], dtype=float)
    for i in range(modes.shape[0]):
        nu, C, parity = nu_col[i], c_col[i], parity_col[i]
        if parity == 1.0:
            val, _ = quad(lambda X: np.exp(-beta * (X + 0.5)) * C * np.sin(nu * X), -0.5, 0.5)
        else:
            val, _ = quad(lambda X: np.exp(-X ** 2 / w ** 2) * C * np.cos(nu * X), -0.5, 0.5)
        proj[i] = val

    return proj

def compute_cross_sectional_moment_weights(modes: np.ndarray) -> np.ndarray:
    import numpy as np
    from scipy.integrate import quad

    modes = np.asarray(modes, dtype=float)

    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] < 2:
        raise ValueError("modes must have shape (N, 3) with N >= 2")
    nu_col, c_col, parity_col = modes[:, 0], modes[:, 1], modes[:, 2]
    if not np.all(np.isfinite(nu_col)) or np.any(nu_col <= 0.0):
        raise ValueError("every eigenvalue in modes must be finite and > 0")
    if not np.all(np.isfinite(c_col)) or np.any(c_col <= 0.0):
        raise ValueError("every normalization constant in modes must be finite and > 0")
    if not np.all(np.isin(parity_col, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")

    weights = np.empty(modes.shape[0], dtype=float)
    for i in range(modes.shape[0]):
        nu, C, parity = nu_col[i], c_col[i], parity_col[i]
        if parity == 1.0:
            val, _ = quad(lambda X: X * C * np.sin(nu * X), -0.5, 0.5)
        else:
            val, _ = quad(lambda X: C * np.cos(nu * X), -0.5, 0.5)
        weights[i] = val

    return weights

def compute_transient_modal_temperatures(modes: np.ndarray, proj: np.ndarray,
                                                  w: float, t: float) -> np.ndarray:
    import numpy as np

    modes = np.asarray(modes, dtype=float)
    proj = np.asarray(proj, dtype=float).ravel()

    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes must have shape (N, 3)")
    if proj.size != modes.shape[0]:
        raise ValueError("proj must have exactly one entry per row of modes")
    if not np.all(np.isfinite(proj)):
        raise ValueError("proj must contain only finite values")

    parity = modes[:, 2]
    if not np.all(np.isin(parity, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")
    is_odd = parity == 1.0
    is_even = parity == 0.0
    if not np.any(is_odd) or not np.any(is_even):
        raise ValueError("modes must contain at least one row of each parity")

    if not (isinstance(w, (int, float, np.floating)) and np.isfinite(w) and float(w) > 0.0):
        raise ValueError("w must be a finite number > 0")
    if not (isinstance(t, (int, float, np.floating)) and np.isfinite(t) and float(t) >= 0.0):
        raise ValueError("t must be a finite number >= 0")

    w = float(w)
    t = float(t)

    nu_odd = modes[is_odd, 0]
    a_odd = proj[is_odd]
    nu_even = modes[is_even, 0]
    b_even = proj[is_even]

    ibar = np.outer(a_odd, b_even) * w * np.sqrt(np.pi)
    kappa2 = np.add.outer(nu_odd ** 2, nu_even ** 2)
    theta = (ibar / kappa2) * (1.0 - np.exp(-kappa2 * t))

    return theta

def assemble_thermal_moment(modes: np.ndarray, weights: np.ndarray, theta: np.ndarray) -> float:
    import numpy as np

    modes = np.asarray(modes, dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    theta = np.asarray(theta, dtype=float)

    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError("modes must have shape (N, 3)")
    parity = modes[:, 2]
    if not np.all(np.isin(parity, [0.0, 1.0])):
        raise ValueError("every parity tag in modes must be exactly 0.0 or 1.0")
    is_odd = parity == 1.0
    is_even = parity == 0.0
    if not np.any(is_odd) or not np.any(is_even):
        raise ValueError("modes must contain at least one row of each parity")

    if weights.size != modes.shape[0]:
        raise ValueError("weights must have exactly one entry per row of modes")
    if theta.ndim != 2 or theta.shape != (int(np.sum(is_odd)), int(np.sum(is_even))):
        raise ValueError("theta must have shape (n_odd, n_even) matching modes")
    for name, arr in (("modes", modes), ("weights", weights), ("theta", theta)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")

    x_arm = weights[is_odd]
    plain_avg = weights[is_even]

    # M2(t) = sum over odd/even pairs of Theta_ij * (odd mode's depth moment arm) * (even mode's lateral average).
    m = float(np.sum(theta * np.outer(x_arm, plain_avg)))
    return m

def compute_fold_angle(
    length: float = 0.025,
    thickness: float = 0.0025,
    thermal_conductivity: float = 0.55,
    specific_heat: float = 4.0e3,
    density: float = 1000.0,
    thermal_expansion: float = -0.015,
    attenuation_coefficient: float = 220.0,
    heat_conversion_fraction: float = 0.8,
    heat_transfer_coefficient: float = 150.0,
    laser_radius: float = 0.005,
    incident_intensity_prefactor: float = 2.5e4,
    elapsed_time: float = 3.0,
    n_modes: int = 4,
) -> float:
    import numpy as np

    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool) and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")
    n_modes = int(n_modes)

    # Step 1: dimensionless groups.
    groups = compute_dimensionless_groups(
        length, thickness, thermal_conductivity, specific_heat, density,
        thermal_expansion, attenuation_coefficient, heat_conversion_fraction,
        heat_transfer_coefficient, laser_radius, incident_intensity_prefactor, elapsed_time,
    )
    bi, beta, w, gamma1_dimless, t = groups

    # Step 2: merged transverse eigenbasis.
    modes = solve_transverse_eigenbasis(bi, n_modes)

    # Step 3: per-mode heat-source projection.
    proj = project_heat_source_onto_modes(modes, beta, w)

    # Step 4: per-mode cross-sectional moment-arm weights.
    weights = compute_cross_sectional_moment_weights(modes)

    # Step 5: transient modal temperature amplitudes.
    theta = compute_transient_modal_temperatures(modes, proj, w, t)

    # Step 6: assemble the total thermal moment M2(t) (first moment of the inner temperature field).
    m_t = assemble_thermal_moment(modes, weights, theta)

    # Step 7 (this orchestrator): the physical fold angle.
    # The reduced model's hinge condition, in its scaled variables X1 = x1/L and u2 = v/h, is
    # [d u2 / d X1] = -12 * Gamma1 * M2 (the factor 12 = h^4 / I, I = h^4/12 the second moment of area).
    # The physical slope is dv/dx1 = (h/L) * d u2/d X1, so the physical slope jump is
    # -12 * (h/L) * Gamma1 * M2 = -gamma1 * M*, with M* = 12 * Delta_theta * M2 the thermal moment in kelvin
    # (gamma1 * Delta_theta = (h/L) * Gamma1). With free ends each arm is straight, and in the small-rotation
    # approximation each arm's rotation equals its slope, so the fold angle is the magnitude of the slope jump.
    # (The reduced form with v~ = sqrt(12) u2 and M~ = 12 sqrt(12) M2 gives the same: (h/L) |Gamma1 M~| / sqrt(12).)
    aspect = float(thickness) / float(length)
    phi = float(12.0 * aspect * abs(gamma1_dimless * m_t))

    return phi
SCICODE_GOLD_EOF
