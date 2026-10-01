#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def scale_staging_lattice(energy_gev, exponent, reference):
    import math
    import numpy as np
    energy_gev = float(energy_gev)
    exponent = float(exponent)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (7,) or not np.all(np.isfinite(ref)):
        raise ValueError("reference must be a finite length-7 array")
    if not math.isfinite(energy_gev) or energy_gev <= 0 or not math.isfinite(exponent):
        raise ValueError("energy must be positive and exponent finite")
    er, lr, ellr, betar, br, taur, r56r = ref
    if min(er, lr, ellr, betar, br) <= 0 or taur == 0:
        raise ValueError("invalid physical reference")
    ratio = energy_gev / er
    try:
        root = math.sqrt(ratio)
        length = lr * root
        ell = ellr * root
        beta = betar * root
        field = br * ratio ** (-exponent)
        tau = taur * ratio ** exponent
        dispersion = 1.0 / tau
        r56 = r56r * ratio ** (-(2.0 * exponent + 0.5))
    except (OverflowError, ZeroDivisionError):
        raise ValueError("scaled values exceed the supported finite domain")
    out = np.array([length, ell, beta, field, tau, dispersion, r56], dtype=float)
    if (not np.all(np.isfinite(out)) or min(length, ell, beta, field) <= 0
            or tau == 0 or dispersion == 0):
        raise ValueError("scaled values exceed the supported finite domain")
    return out

def second_order_chromatic_growth(sigma_delta, length_m, half_gap_m, beta_m):
    import math
    sigma_delta = float(sigma_delta)
    length_m = float(length_m)
    half_gap_m = float(half_gap_m)
    beta_m = float(beta_m)
    if not all(math.isfinite(x) for x in (sigma_delta, length_m, half_gap_m, beta_m)):
        raise ValueError("inputs must be finite")
    if sigma_delta < 0 or min(length_m, half_gap_m, beta_m) <= 0:
        raise ValueError("spread is nonnegative and lengths are positive")
    try:
        shape_root = math.hypot(math.sqrt(2.0) * length_m / beta_m,
                                math.sqrt(3.0) * (half_gap_m / length_m + 1.0))
        result = 2.0 * (1.0 + length_m / half_gap_m) * sigma_delta ** 2 * shape_root
    except (OverflowError, ZeroDivisionError):
        raise ValueError("result exceeds the supported finite domain")
    if not math.isfinite(result):
        raise ValueError("result exceeds the supported finite domain")
    return result

def nonlinear_geometric_growth(energy_gev, length_m, half_gap_m, beta_m,
                                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad):
    import math
    import numpy as np
    vals = [float(x) for x in (energy_gev, length_m, half_gap_m, beta_m,
                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad)]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("inputs must be finite")
    energy_gev, length_m, half_gap_m, beta_m, tau_per_m, eps_nx_m_rad, eps_ny_m_rad = vals
    if min(energy_gev, length_m, half_gap_m, beta_m, eps_nx_m_rad, eps_ny_m_rad) <= 0:
        raise ValueError("energy, lengths and emittances must be positive")
    gamma = energy_gev / 0.000511
    try:
        coeff = (tau_per_m ** 2 * length_m ** 3 /
                 (beta_m ** 2 * gamma) * (1.0 + length_m / half_gap_m) *
                 (half_gap_m / length_m + 1.0))
        gx = coeff * math.hypot(math.sqrt(6.0) * eps_nx_m_rad,
                                math.sqrt(18.0) * eps_ny_m_rad)
        gy = coeff * math.hypot(math.sqrt(18.0) * eps_nx_m_rad,
                                math.sqrt(6.0) * eps_ny_m_rad)
    except (OverflowError, ZeroDivisionError):
        raise ValueError("result exceeds the supported finite domain")
    out = np.array([gx, gy], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("result exceeds the supported finite domain")
    return out

def incoherent_radiation_growth(energy_gev, exponent, reference_energy_gev,
                                                reference_growth):
    import math
    energy_gev = float(energy_gev)
    exponent = float(exponent)
    reference_energy_gev = float(reference_energy_gev)
    reference_growth = float(reference_growth)
    if not all(math.isfinite(x) for x in (energy_gev, exponent, reference_energy_gev, reference_growth)):
        raise ValueError("inputs must be finite")
    if min(energy_gev, reference_energy_gev) <= 0 or reference_growth < 0:
        raise ValueError("energies are positive and growth is nonnegative")
    if reference_growth == 0.0:
        return 0.0
    log_result = (math.log(reference_growth) +
                  (3.0 - 5.0 * exponent) * math.log(energy_gev / reference_energy_gev))
    if log_result > math.log(float.fromhex('0x1.fffffffffffffp+1023')):
        raise ValueError("result exceeds the supported finite domain")
    if log_result < math.log(float.fromhex('0x0.0000000000001p-1022')):
        return 0.0
    result = math.exp(log_result)
    if not math.isfinite(result):
        raise ValueError("result exceeds the supported finite domain")
    return result

def combine_emittance_increments(chromatic_growth, geometric_growth, isr_growth):
    import math
    import numpy as np
    chromatic_growth = float(chromatic_growth)
    geometric_growth = np.asarray(geometric_growth, dtype=float)
    isr_growth = float(isr_growth)
    if geometric_growth.shape != (2,) or not np.all(np.isfinite(geometric_growth)):
        raise ValueError("geometric_growth must be a finite length-2 array")
    if not all(math.isfinite(x) for x in (chromatic_growth, isr_growth)):
        raise ValueError("increments must be finite")
    if chromatic_growth < 0 or isr_growth < 0 or np.any(geometric_growth < 0):
        raise ValueError("increments must be nonnegative")
    rx = math.hypot(1.0, chromatic_growth, geometric_growth[0], isr_growth)
    ry = math.hypot(1.0, chromatic_growth, geometric_growth[1])
    if not all(math.isfinite(x) for x in (rx, ry)):
        raise ValueError("quadrature result exceeds the supported finite domain")
    return np.array([rx, ry], dtype=float)

def robust_energy_profile(exponent, energies_gev, scenarios, reference):
    import math
    import numpy as np
    exponent = float(exponent)
    energies = np.asarray(energies_gev, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if energies.ndim != 1 or len(energies) == 0 or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a nonempty finite vector")
    if np.any(energies <= 0) or np.any(np.diff(energies) <= 0):
        raise ValueError("energies must be positive and strictly increasing")
    if scenarios.ndim != 2 or scenarios.shape[1] != 4 or len(scenarios) == 0 or not np.all(np.isfinite(scenarios)):
        raise ValueError("scenarios must have finite shape (S,4)")
    if np.any(scenarios < 0) or np.any(scenarios[:, 1:3] <= 0):
        raise ValueError("scenario spreads are nonnegative and emittances positive")
    out = np.empty((len(energies), 4), dtype=float)
    for i, energy in enumerate(energies):
        scaled = scale_staging_lattice(energy, exponent, reference)
        length, ell, beta, field, tau = scaled[:5]
        worst_x = -math.inf
        worst_y = -math.inf
        for sigma_delta, eps_nx, eps_ny, amplitude in scenarios:
            chromatic = second_order_chromatic_growth(sigma_delta, length, ell, beta)
            geometric = nonlinear_geometric_growth(
                energy, length, ell, beta, tau, eps_nx, eps_ny)
            isr = incoherent_radiation_growth(
                energy, exponent, reference[0], amplitude)
            ratios = combine_emittance_increments(chromatic, geometric, isr)
            worst_x = max(worst_x, ratios[0])
            worst_y = max(worst_y, ratios[1])
        out[i] = [worst_x, worst_y, abs(tau), field]
    if not np.all(np.isfinite(out)):
        raise ValueError("profile exceeds the supported finite domain")
    return out

def select_minimax_exponent(exponents, energies_gev, scenarios, reference, limits):
    import math
    import numpy as np
    exponents = np.asarray(exponents, dtype=float)
    energies = np.asarray(energies_gev, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if exponents.ndim != 1 or len(exponents) == 0 or not np.all(np.isfinite(exponents)):
        raise ValueError("exponents must be a nonempty finite vector")
    if limits.shape != (4,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("limits must be a positive finite length-4 array")
    best = None
    best_key = None
    for exponent in exponents:
        profile = robust_energy_profile(exponent, energies, scenarios, reference)
        last = -1
        for i, row in enumerate(profile):
            loads = np.array([row[0] / limits[0], row[1] / limits[1],
                              row[2] / limits[2], limits[3] / row[3]], dtype=float)
            if np.all(loads <= 1.0):
                last = i
            else:
                break
        if last < 0:
            continue
        row = profile[last]
        loads = np.array([row[0] / limits[0], row[1] / limits[1],
                          row[2] / limits[2], limits[3] / row[3]], dtype=float)
        load = float(np.max(loads))
        if last + 1 < len(energies):
            next_row = profile[last + 1]
            next_loads = np.array([next_row[0] / limits[0], next_row[1] / limits[1],
                                   next_row[2] / limits[2], limits[3] / next_row[3]], dtype=float)
            next_code = int(np.argmax(next_loads)) + 1
        else:
            next_code = 0
        key = (float(energies[last]), -load, -float(exponent))
        if best_key is None or key > best_key:
            best_key = key
            best = np.array([exponent, energies[last], float(last), load, float(next_code)], dtype=float)
    if best is None:
        raise ValueError("no exponent is feasible at the first energy")
    return best

def transport_nonlinear_lens_pair(initial_slopes, length_m, half_gap_m, tau_per_m):
    import math
    import numpy as np
    rays = np.asarray(initial_slopes, dtype=float)
    length_m = float(length_m)
    half_gap_m = float(half_gap_m)
    tau_per_m = float(tau_per_m)
    if rays.ndim < 1 or rays.shape[-1] != 2 or rays.size == 0:
        raise ValueError("initial_slopes must have a nonempty final axis of length 2")
    if not np.all(np.isfinite(rays)) or not all(math.isfinite(x) for x in
            (length_m, half_gap_m, tau_per_m)):
        raise ValueError("inputs must be finite")
    if length_m <= 0 or half_gap_m <= 0:
        raise ValueError("drift lengths must be positive")
    flat = rays.reshape(-1, 2)
    xp0 = flat[:, 0]
    yp0 = flat[:, 1]
    focal = 1.0 / (1.0 / length_m + 1.0 / half_gap_m)
    x1 = length_m * xp0
    y1 = length_m * yp0
    xp1 = xp0 - (x1 + 0.5 * tau_per_m * (x1 * x1 + y1 * y1)) / focal
    yp1 = yp0 - (y1 + tau_per_m * x1 * y1) / focal
    x2 = x1 + 2.0 * half_gap_m * xp1
    y2 = y1 + 2.0 * half_gap_m * yp1
    xp2 = xp1 - (x2 + 0.5 * tau_per_m * (x2 * x2 + y2 * y2)) / focal
    yp2 = yp1 - (y2 + tau_per_m * x2 * y2) / focal
    x3 = x2 + length_m * xp2
    y3 = y2 + length_m * yp2
    out = np.column_stack((x3, xp2, y3, yp2))
    if not np.all(np.isfinite(out)):
        raise ValueError("mapped rays exceed the supported finite domain")
    return out.reshape(rays.shape[:-1] + (4,))

def nonlinear_lens_validity_certificate(energy_gev, length_m, half_gap_m, beta_m,
                                                        tau_per_m, eps_nx_m_rad, eps_ny_m_rad,
                                                        offset_fraction, offset_correlation,
                                                        quadrature_order):
    import math
    import numpy as np
    vals = [float(x) for x in (energy_gev, length_m, half_gap_m, beta_m,
                               tau_per_m, eps_nx_m_rad, eps_ny_m_rad,
                               offset_fraction, offset_correlation)]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("inputs must be finite")
    (energy_gev, length_m, half_gap_m, beta_m, tau_per_m,
     eps_nx_m_rad, eps_ny_m_rad, offset_fraction, offset_correlation) = vals
    order_float = float(quadrature_order)
    if not math.isfinite(order_float):
        raise ValueError("quadrature_order must be finite")
    order = int(order_float)
    if order_float != order or order < 5 or order > 15 or order % 2 == 0:
        raise ValueError("quadrature_order must be an odd integer from 5 through 15")
    if min(energy_gev, length_m, half_gap_m, beta_m,
           eps_nx_m_rad, eps_ny_m_rad, offset_fraction) <= 0 or tau_per_m == 0:
        raise ValueError("physical scales and offset_fraction are positive; tau is nonzero")
    if offset_correlation < -1.0 or offset_correlation > 1.0:
        raise ValueError("offset correlation must lie in [-1,1]")
    taper_scale = abs(tau_per_m) * max(length_m, half_gap_m, beta_m)
    if taper_scale < 1e-12 or taper_scale > 1e6:
        raise ValueError("taper is outside the resolved finite probe range")
    gamma = energy_gev / 0.000511
    eps_x = eps_nx_m_rad / gamma
    eps_y = eps_ny_m_rad / gamma
    sigma_xp = math.sqrt(eps_x / beta_m)
    sigma_yp = math.sqrt(eps_y / beta_m)
    nodes, one_weights = np.polynomial.hermite.hermgauss(order)
    xp = (math.sqrt(2.0) * sigma_xp * nodes[:, None] +
          np.zeros((order, order), dtype=float))
    yp = (math.sqrt(2.0) * sigma_yp * nodes[None, :] +
          np.zeros((order, order), dtype=float))
    rays = np.column_stack((xp.ravel(), yp.ravel()))
    weights = np.outer(one_weights, one_weights).ravel() / math.pi

    def mapped(tau):
        return transport_nonlinear_lens_pair(
            rays, length_m, half_gap_m, tau)

    full = mapped(tau_per_m)
    half = mapped(0.5 * tau_per_m)

    def centered_moment(a, b=None):
        a = np.asarray(a, dtype=float)
        ac = a - np.dot(weights, a)
        if b is None:
            return float(np.dot(weights, ac * ac))
        b = np.asarray(b, dtype=float)
        bc = b - np.dot(weights, b)
        return float(np.dot(weights, ac * bc))

    def exact_growth(pos, angle, eps_geo):
        var_pos = eps_geo * beta_m + centered_moment(pos)
        var_angle = centered_moment(angle)
        cov = centered_moment(pos, angle)
        determinant = max(var_pos * var_angle - cov * cov, 0.0)
        ratio_sq = determinant / (eps_geo * eps_geo)
        return math.sqrt(max(ratio_sq - 1.0, 0.0))

    gx_exact = exact_growth(full[:, 0], full[:, 1], eps_x)
    gy_exact = exact_growth(full[:, 2], full[:, 3], eps_y)
    asym = nonlinear_geometric_growth(
        energy_gev, length_m, half_gap_m, beta_m, tau_per_m,
        eps_nx_m_rad, eps_ny_m_rad)
    sx_full = math.sqrt(centered_moment(full[:, 0]))
    sx_half = math.sqrt(centered_moment(half[:, 0]))
    sy_full = math.sqrt(centered_moment(full[:, 2]))
    sy_half = math.sqrt(centered_moment(half[:, 2]))
    if (not all(math.isfinite(x) for x in (sx_full, sx_half, sy_full, sy_half))
            or min(sx_full, sx_half, sy_full, sy_half) <= 0):
        raise ValueError("degenerate cancellation-order probe")
    nx = math.log(sx_full / sx_half, 2.0)
    ny = math.log(sy_full / sy_half, 2.0)

    sigma_x = math.sqrt(eps_x * beta_m)
    sigma_offset = offset_fraction * sigma_x
    factor = 1.0 / length_m + 1.0 / half_gap_m
    matrix = factor * np.array([
        [-length_m, length_m],
        [-(1.0 + 2.0 * half_gap_m / length_m), 1.0]], dtype=float)
    offset_cov = sigma_offset ** 2 * np.array(
        [[1.0, offset_correlation], [offset_correlation, 1.0]], dtype=float)
    final_cov = matrix @ offset_cov @ matrix.T
    mean_action = 0.5 * gamma * (final_cov[0, 0] / beta_m +
                                 beta_m * final_cov[1, 1])
    action_ratio = mean_action / eps_nx_m_rad
    if not math.isfinite(action_ratio) or action_ratio <= 0:
        raise ValueError("degenerate offset-action probe")
    tolerance_sigma = offset_fraction / math.sqrt(action_ratio)
    out = np.array([gx_exact, gy_exact, asym[0], asym[1], nx, ny,
                    action_ratio, tolerance_sigma], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("diagnostics exceed the supported finite domain")
    return out

def run_robust_scaling_benchmark(exponents=None, energies_gev=None, scenarios=None,
                                                 reference=None, limits=None):
    import itertools
    import math
    import numpy as np
    supplied = [exponents, energies_gev, scenarios, reference, limits]
    if all(x is None for x in supplied):
        exponents = np.arange(0.25, 0.7501, 0.005)
        energies_gev = 50.0 * np.exp(np.arange(401) * np.log(100.0) / 400.0)
        scenarios = np.array(list(itertools.product(
            [0.018, 0.032], [8e-6, 12e-6], [0.10e-6, 0.18e-6], [0.025, 0.040])), dtype=float)
        reference = np.array([50.0, np.sqrt(5.0), 2.0 * np.sqrt(5.0),
                              0.015 * np.sqrt(5.0), 1.0, -61.42, -1e-4], dtype=float)
        limits = np.array([1.25, 1.25, 520.0, 0.040], dtype=float)
    elif any(x is None for x in supplied):
        raise ValueError("either omit every argument or supply all five")
    exponents = np.asarray(exponents, dtype=float)
    energies_gev = np.asarray(energies_gev, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    reference = np.asarray(reference, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (exponents.ndim != 1 or len(exponents) == 0 or
            energies_gev.ndim != 1 or len(energies_gev) == 0 or
            scenarios.ndim != 2 or scenarios.shape[1] != 4 or
            reference.shape != (7,) or limits.shape != (4,) or
            not all(np.all(np.isfinite(x)) for x in
                    (exponents, energies_gev, scenarios, reference, limits))):
        raise ValueError("malformed or non-finite benchmark inputs")
    selection = select_minimax_exponent(exponents, energies_gev, scenarios, reference, limits)
    exponent = float(selection[0])
    energy = float(selection[1])
    index = int(selection[2])
    next_code = float(selection[4])
    profile = robust_energy_profile(exponent, energies_gev, scenarios, reference)
    scaled = scale_staging_lattice(energy, exponent, reference)
    length, ell, beta, field, tau = scaled[:5]
    direct_x = -math.inf
    direct_y = -math.inf
    for sigma_delta, eps_nx, eps_ny, amplitude in np.asarray(scenarios, dtype=float):
        chromatic = second_order_chromatic_growth(sigma_delta, length, ell, beta)
        geometric = nonlinear_geometric_growth(
            energy, length, ell, beta, tau, eps_nx, eps_ny)
        isr = incoherent_radiation_growth(energy, exponent, reference[0], amplitude)
        ratios = combine_emittance_increments(chromatic, geometric, isr)
        direct_x = max(direct_x, ratios[0])
        direct_y = max(direct_y, ratios[1])
    worst_x = max(direct_x, profile[index, 0])
    worst_y = max(direct_y, profile[index, 1])
    max_eps_nx = float(np.max(scenarios[:, 1]))
    max_eps_ny = float(np.max(scenarios[:, 2]))
    validity = nonlinear_lens_validity_certificate(
        energy, length, ell, beta, tau, max_eps_nx, max_eps_ny,
        0.20, -0.35, 7)
    gamma = energy / 0.000511
    probe = np.array([
        [math.sqrt(max_eps_nx / (gamma * beta)), 0.0],
        [0.0, math.sqrt(max_eps_ny / (gamma * beta))],
        [math.sqrt(max_eps_nx / (gamma * beta)),
         -math.sqrt(max_eps_ny / (gamma * beta))]], dtype=float)
    mapped_probe = transport_nonlinear_lens_pair(probe, length, ell, tau)
    linear_probe = transport_nonlinear_lens_pair(probe, length, ell, 0.0)
    asym_direct = nonlinear_geometric_growth(
        energy, length, ell, beta, tau, max_eps_nx, max_eps_ny)
    if (not np.allclose([worst_x, worst_y, abs(tau), field], profile[index], rtol=1e-12, atol=1e-14)
            or next_code not in (0.0, 1.0, 2.0, 3.0, 4.0)
            or not np.all(np.isfinite(validity))
            or not np.allclose(validity[2:4], asym_direct, rtol=1e-12, atol=1e-14)
            or np.any(validity[:4] < 0.0)
            or np.any(validity[4:6] <= 0.5) or np.any(validity[4:6] >= 3.5)
            or validity[6] <= 0.0 or validity[7] <= 0.0
            or not np.allclose(linear_probe[:, [0, 2]], 0.0, rtol=0.0, atol=1e-14)
            or not np.allclose(linear_probe[:, [1, 3]], probe, rtol=1e-12, atol=1e-14)
            or not np.all(np.isfinite(mapped_probe))):
        raise RuntimeError("inconsistent end-to-end certificate")
    return float(np.round(energy, 1))
SCICODE_GOLD_EOF
