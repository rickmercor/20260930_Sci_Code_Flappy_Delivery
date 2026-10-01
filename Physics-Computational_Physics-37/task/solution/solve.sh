#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def lumped_current_projection(lumped_mass, current_load):
    """Apply the paper's lumped L2 projection (30) to vector curl loads."""
    import numpy as np
    mass = np.asarray(lumped_mass, dtype=float)
    load = np.asarray(current_load, dtype=float)
    if mass.ndim != 1 or mass.size == 0 or load.shape != (mass.size, 3):
        raise ValueError("mass must have shape (n,) and curl_load shape (n,3)")
    if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(load)) or np.any(mass <= 0):
        raise ValueError("inputs must be finite and masses positive")
    return (load / mass[:, None]).astype(float)

def projected_electron_velocity(density, velocity, projected_current, ion_skin_depth):
    """Evaluate the paper's projected electron velocity (47) at ordered nodes."""
    import math
    import numpy as np
    rho = float(density)
    di = float(ion_skin_depth)
    velocity = np.asarray(velocity, dtype=float)
    current = np.asarray(projected_current, dtype=float)
    if not math.isfinite(rho) or not math.isfinite(di) or rho <= 0 or di < 0:
        raise ValueError("rho must be positive and ion_skin_depth nonnegative")
    if velocity.ndim != 2 or velocity.shape[1] != 3 or current.shape != velocity.shape:
        raise ValueError("velocity and projected_current must have shape (n,3)")
    if velocity.shape[0] == 0 or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(current)):
        raise ValueError("node data must be nonempty and finite")
    electron = velocity - (di / rho) * current
    return np.column_stack((electron, np.linalg.norm(electron, axis=1))).astype(float)

def rescaled_induction_residual(lumped_mass, domain_measure,
                                        magnetic_field, residual_load):
    """Project and rescale the induction residual using paper Eqs. (50)-(51)."""
    import math
    import numpy as np
    mass = np.asarray(lumped_mass, dtype=float)
    measure = float(domain_measure)
    magnetic = np.asarray(magnetic_field, dtype=float)
    load = np.asarray(residual_load, dtype=float)
    if mass.ndim != 1 or mass.size == 0 or magnetic.shape != (mass.size, 3) or load.shape != magnetic.shape:
        raise ValueError("inconsistent nodal shapes")
    if (not math.isfinite(measure) or measure <= 0 or not np.all(np.isfinite(mass))
            or np.any(mass <= 0) or not np.all(np.isfinite(magnetic)) or not np.all(np.isfinite(load))):
        raise ValueError("inputs must be finite with positive masses and measure")
    projected = load / mass[:, None]
    magnetic_mean = np.sum(mass[:, None] * magnetic, axis=0) / measure
    scale = float(np.max(np.linalg.norm(magnetic - magnetic_mean, axis=1)))
    if scale <= np.finfo(float).eps:
        raise ValueError("magnetic normalization scale is zero")
    return (np.linalg.norm(projected, axis=1) / scale).astype(float)

def blended_artificial_resistivity(mesh_h, physical_resistivity,
                                           electron_speed, rescaled_residual,
                                           node_scale, c_low=0.25, c_res=1.0):
    """Evaluate the low-order, residual, and blended resistivities of Eqs. (48),(52)."""
    import math
    import numpy as np
    h = float(mesh_h)
    r = float(physical_resistivity)
    c_low = float(c_low)
    c_res = float(c_res)
    speed = np.asarray(electron_speed, dtype=float)
    residual = np.asarray(rescaled_residual, dtype=float)
    scale = np.asarray(node_scale, dtype=float)
    if speed.ndim != 1 or speed.size == 0 or residual.shape != speed.shape or scale.shape != speed.shape:
        raise ValueError("node arrays must be nonempty and equal length")
    if (not all(math.isfinite(x) for x in (h, r, c_low, c_res)) or h <= 0 or r < 0
            or c_low < 0 or c_res < 0 or not np.all(np.isfinite(speed))
            or not np.all(np.isfinite(residual)) or not np.all(np.isfinite(scale))
            or np.any(speed < 0) or np.any(residual < 0) or np.any(scale <= 0)):
        raise ValueError("invalid resistivity inputs")
    local_h = h * scale
    low = c_low * local_h * speed
    high = c_res * local_h * local_h * residual
    effective = np.maximum(r, np.minimum(low, high))
    return np.column_stack((low, high, effective)).astype(float)

def resistive_whistler_mode(density, background_field, ion_skin_depth,
                                    resistivity, wavenumber):
    """Evaluate the right-polarized first-order resistive dispersion relation (55)."""
    import math
    import numpy as np
    rho = float(density)
    h0 = float(background_field)
    di = float(ion_skin_depth)
    r = float(resistivity)
    k = float(wavenumber)
    if not all(math.isfinite(x) for x in (rho, h0, di, r, k)):
        raise ValueError("inputs must be finite")
    if rho <= 0 or h0 <= 0 or di < 0 or r < 0 or k <= 0:
        raise ValueError("invalid physical domain")
    hall = di * k * k * h0 / rho
    alfven2 = k * k * h0 * h0 / rho
    root = math.sqrt(hall * hall + 4.0 * alfven2)
    real = 0.5 * (hall + root)
    imag = -0.5 * r * k * k * (1.0 + hall / root)
    period = 2.0 * math.pi / real
    return np.array([real, imag, period, math.exp(imag * period)], dtype=float)

def crank_nicolson_whistler_certificate(omega_real, omega_imag,
                                                time_step, periods=1.0):
    """Compare the paper's Crank--Nicolson source update with one whistler mode."""
    import math
    import numpy as np
    wr = float(omega_real)
    wi = float(omega_imag)
    tau = float(time_step)
    periods = float(periods)
    if not all(math.isfinite(x) for x in (wr, wi, tau, periods)):
        raise ValueError("inputs must be finite")
    if wr <= 0 or wi > 0 or tau <= 0 or periods <= 0:
        raise ValueError("invalid modal or time-step domain")
    omega = complex(wr, wi)
    duration = periods * 2.0 * math.pi / wr
    steps = int(math.ceil(duration / tau))
    elapsed = steps * tau
    factor = (1.0 - 0.5j * omega * tau) / (1.0 + 0.5j * omega * tau)
    numerical = factor ** steps
    exact = np.exp(-1j * omega * elapsed)
    amplitude_error = abs(abs(numerical) / abs(exact) - 1.0)
    phase_error = abs(float(np.angle(numerical / exact)))
    log_damping_error = abs(math.log(abs(numerical)) - math.log(abs(exact)))
    return np.array([amplitude_error, phase_error, log_damping_error,
                     float(steps), elapsed, abs(numerical)], dtype=float)

def newton_coercivity_certificates(time_step, permeability,
                                            mass_constant, density_min,
                                            current_bound, electron_speed_bound,
                                            resistivity_min):
    """Return the normalized Appendix-B.6 and Appendix-B.7 coefficients."""
    import math
    import numpy as np
    tau = float(time_step)
    mu = float(permeability)
    cm = float(mass_constant)
    rho = float(density_min)
    ch = float(current_bound)
    ce = float(electron_speed_bound)
    rmin = float(resistivity_min)
    values = (tau, mu, cm, rho, ch, ce, rmin)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("inputs must be finite")
    if tau <= 0 or mu <= 0 or cm <= 0 or rho <= 0 or ch < 0 or ce < 0 or rmin <= 0:
        raise ValueError("invalid certificate domain")
    base_a = cm * cm * rho
    a = base_a - 0.25 * tau * mu * ch
    b6 = mu * (1.0 - 0.25 * (ch * tau + ce * math.sqrt(tau)))
    c6 = 0.5 * tau * (rmin - 0.5 * mu * ce * math.sqrt(tau))
    b7 = mu * (1.0 - 0.25 * (ch + ce) * tau)
    c7 = 0.5 * tau * (rmin - 0.5 * mu * ce)
    return np.array([
        a / base_a, b6 / mu, c6 / (0.5 * tau * rmin),
        a / base_a, b7 / mu, c7 / (0.5 * tau * rmin),
    ], dtype=float)

def source_energy_update(lumped_mass, density, momentum_old,
                                 momentum_new, total_energy_old, joule_power,
                                 time_step, magnetic_norm_old_sq,
                                 permeability=1.0):
    """Apply the nodal source-energy identity and its magnetic-energy balance."""
    import math
    import numpy as np
    mass = np.asarray(lumped_mass, dtype=float)
    rho = np.asarray(density, dtype=float)
    old_m = np.asarray(momentum_old, dtype=float)
    new_m = np.asarray(momentum_new, dtype=float)
    old_e = np.asarray(total_energy_old, dtype=float)
    joule = np.asarray(joule_power, dtype=float)
    tau = float(time_step)
    h_old_sq = float(magnetic_norm_old_sq)
    mu = float(permeability)
    n = mass.size
    if (mass.ndim != 1 or n == 0 or rho.shape != (n,) or old_m.shape != (n, 3)
            or new_m.shape != old_m.shape or old_e.shape != (n,) or joule.shape != (n,)):
        raise ValueError("inconsistent nodal shapes")
    if (not np.all(np.isfinite(mass)) or not np.all(np.isfinite(rho))
            or not np.all(np.isfinite(old_m)) or not np.all(np.isfinite(new_m))
            or not np.all(np.isfinite(old_e)) or not np.all(np.isfinite(joule))
            or not all(math.isfinite(x) for x in (tau, h_old_sq, mu))):
        raise ValueError("inputs must be finite")
    if np.any(mass <= 0) or np.any(rho <= 0) or np.any(joule < 0) or tau <= 0 or h_old_sq < 0 or mu <= 0:
        raise ValueError("invalid source-energy domain")
    kinetic_old = np.sum(old_m * old_m, axis=1) / (2.0 * rho)
    kinetic_new = np.sum(new_m * new_m, axis=1) / (2.0 * rho)
    new_e = old_e + kinetic_new - kinetic_old + tau * joule
    energy_increment = float(np.sum(mass * (new_e - old_e)))
    h_new_sq = h_old_sq - (2.0 / mu) * energy_increment
    internal = new_e - kinetic_new
    heat = float(np.sum(mass * tau * joule))
    return np.concatenate((new_e, np.array([h_new_sq, np.min(internal), heat])))

def paper_candidate_table(candidates, wave_cube, coercivity_cube,
                                  energy_cube, amplitude_limit=1e-5,
                                  phase_limit=8e-5, log_damping_limit=1e-5,
                                  heat_fraction_limit=1e-4,
                                  b6_margin_min=0.10):
    """Aggregate the paper-native wave, Newton, and energy diagnostics."""
    import math
    import numpy as np
    candidates = np.asarray(candidates, dtype=float)
    wave = np.asarray(wave_cube, dtype=float)
    coercivity = np.asarray(coercivity_cube, dtype=float)
    energy = np.asarray(energy_cube, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.shape[0] == 0:
        raise ValueError("candidates must have shape (C,2)")
    c = candidates.shape[0]
    if wave.ndim != 3 or wave.shape[0] != c or wave.shape[2] != 6 or wave.shape[1] == 0:
        raise ValueError("wave_cube must have shape (C,S,6)")
    if coercivity.shape != (c, wave.shape[1], 6) or energy.shape != (c, wave.shape[1], 2):
        raise ValueError("certificate cubes have inconsistent shapes")
    limits = np.asarray([amplitude_limit, phase_limit, log_damping_limit,
                         heat_fraction_limit], dtype=float)
    margin = float(b6_margin_min)
    if (not np.all(np.isfinite(candidates)) or np.any(candidates <= 0)
            or not np.all(np.isfinite(wave)) or not np.all(np.isfinite(coercivity))
            or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(limits))
            or np.any(limits <= 0) or not math.isfinite(margin) or margin <= 0):
        raise ValueError("all inputs and positive limits must be finite")
    rows = []
    for i, (mesh_h, tau) in enumerate(candidates):
        wave_worst = np.max(wave[i, :, :3], axis=0)
        b6_min = float(np.min(coercivity[i, :, :3]))
        b7_min = float(np.min(coercivity[i, :, 3:]))
        heat_worst = float(np.max(energy[i, :, 1]))
        internal_min = float(np.min(energy[i, :, 0]))
        ratios = np.array([*wave_worst, heat_worst]) / limits
        score = float(np.max(ratios))
        cells = int(math.ceil(1.0 / mesh_h)) ** 2
        work = float(cells * (np.sum(wave[i, :, 3]) + 3.0 * wave.shape[1]))
        feasible = float(score <= 1.0 and b6_min >= margin and internal_min > 0.0)
        rows.append([mesh_h, tau, *wave_worst, b6_min, b7_min, heat_worst,
                     internal_min, score, work, feasible])
    return np.asarray(rows, dtype=float)

def _hall_midpoint_source(curl, mass, density, velocity_old, magnetic_old,
                          resistivity, ion_skin_depth, time_step, permeability):
    import numpy as np
    curl = np.asarray(curl, dtype=float)
    mass = np.asarray(mass, dtype=float)
    v0 = np.asarray(velocity_old, dtype=float)
    h0 = np.asarray(magnetic_old, dtype=float)
    eta = np.asarray(resistivity, dtype=float)
    n = mass.size
    weights = np.repeat(mass, 3)
    adjoint = (curl.T * weights[None, :]) / weights[:, None]
    identity = np.eye(3*n)

    def _cross_matrix(v):
        out = np.zeros((3*n, 3*n))
        for i, (x, y, z) in enumerate(v):
            out[3*i:3*i+3, 3*i:3*i+3] = [[0., -z, y], [z, 0., -x], [-y, x, 0.]]
        return out

    def _residual(midpoint, tau, jacobian=False):
        b = midpoint.reshape(n, 3)
        j = (curl @ midpoint).reshape(n, 3)
        force = np.cross(j, b)
        beta = tau*permeability/(2.0*density)
        vbar = v0 + beta*force
        flux = np.cross(b, vbar) + eta[:, None]*j/permeability + (ion_skin_depth/density)*force
        residual = 2.0*(midpoint-h0.ravel()) + tau*(adjoint @ flux.ravel())
        if not jacobian:
            return residual
        b_cross = _cross_matrix(b)
        force_derivative = -b_cross @ curl + _cross_matrix(j)
        flux_derivative = (-_cross_matrix(vbar) + beta*b_cross @ force_derivative
                           + np.repeat(eta/permeability, 3)[:, None]*curl
                           + (ion_skin_depth/density)*force_derivative)
        return residual, 2.0*identity + tau*adjoint @ flux_derivative

    midpoint = h0.ravel().copy()
    tolerance = 2.0e-13*(1.0 + np.max(np.abs(h0)))
    # Continuation follows the root from tau=0; each stage uses the same old state.
    for fraction in (0.25, 0.5, 0.75, 1.0):
        tau = time_step*fraction
        for _ in range(40):
            residual, jacobian = _residual(midpoint, tau, True)
            norm = float(np.max(np.abs(residual)))
            if norm <= tolerance:
                break
            direction = np.linalg.solve(jacobian, -residual)
            alpha = 1.0
            for _ in range(20):
                trial = midpoint + alpha*direction
                trial_norm = float(np.max(np.abs(_residual(trial, tau))))
                if trial_norm < norm:
                    midpoint = trial
                    break
                alpha *= 0.5
            else:
                raise ValueError('the midpoint source root did not converge')
        else:
            raise ValueError('the midpoint source root did not converge')
    if np.max(np.abs(_residual(midpoint, time_step))) > 1e-11*(1+np.max(np.abs(h0))):
        raise ValueError('the midpoint source residual is unresolved')
    b = midpoint.reshape(n, 3)
    j = (curl @ midpoint).reshape(n, 3)
    magnetic_new = 2.0*b-h0
    velocity_new = v0 + (time_step*permeability/density)*np.cross(j, b)
    return magnetic_new, velocity_new, j


def hall_mhd_paper_design(candidates, density, background_field,
                                  ion_skin_depth, physical_resistivity,
                                  wavenumber, velocity, current_load,
                                  lumped_mass, magnetic_nodes, residual_load,
                                  node_scale, internal_energy,
                                  permeability=1.0, mass_constant=0.92,
                                  c_low=0.25, c_res=1.0,
                                  amplitude_limit=1e-5, phase_limit=8e-5,
                                  log_damping_limit=1e-5,
                                  heat_fraction_limit=2.5e-7,
                                  b6_margin_min=0.10, *, source_curl):
    """Select the least-work candidate using only operators in the main paper."""
    import numpy as np
    candidates = np.asarray(candidates, dtype=float)
    rho = np.asarray(density, dtype=float)
    h0 = np.asarray(background_field, dtype=float)
    di = np.asarray(ion_skin_depth, dtype=float)
    resistive = np.asarray(physical_resistivity, dtype=float)
    k = np.asarray(wavenumber, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    current_load = np.asarray(current_load, dtype=float)
    mass = np.asarray(lumped_mass, dtype=float)
    magnetic = np.asarray(magnetic_nodes, dtype=float)
    residual_load = np.asarray(residual_load, dtype=float)
    scale = np.asarray(node_scale, dtype=float)
    internal = np.asarray(internal_energy, dtype=float)
    scenarios = rho.size
    nodes = mass.size
    if (rho.ndim != 1 or scenarios == 0 or h0.shape != rho.shape or di.shape != rho.shape
            or resistive.shape != rho.shape or k.shape != rho.shape):
        raise ValueError("scenario scalars must be equal-length vectors")
    if (mass.ndim != 1 or nodes == 0 or velocity.shape != (scenarios, nodes, 3)
            or current_load.shape != velocity.shape or magnetic.shape != velocity.shape
            or residual_load.shape != velocity.shape or scale.shape != (nodes,)
            or internal.shape != (scenarios, nodes)):
        raise ValueError("scenario node arrays have inconsistent shapes")
    if not np.all(internal > 0):
        raise ValueError("internal_energy must be positive")
    if source_curl is None:
        raise ValueError("source_curl is required")
    curl_input = np.asarray(source_curl)
    if np.iscomplexobj(curl_input) and np.any(curl_input.imag != 0):
        raise ValueError("source_curl must be real")
    curl_matrix = np.asarray(curl_input.real, dtype=float)
    if curl_matrix.shape != (3*nodes, 3*nodes) or not np.all(np.isfinite(curl_matrix)):
        raise ValueError("source_curl must be a finite (3*n,3*n) matrix")
    predicted = np.stack([(curl_matrix @ magnetic[s].ravel()).reshape(nodes, 3)
                          for s in range(scenarios)]) * mass[None, :, None]
    if np.max(np.abs(predicted-current_load)) > 1e-10*(1+np.max(np.abs(current_load))):
        raise ValueError("source_curl and baseline weak current loads disagree")
    generators = np.array([1, 7, 11, 16, 20, 26, 31], dtype=int)
    stress_count = 37 * scenarios
    wave = np.empty((candidates.shape[0], stress_count, 6), dtype=float)
    coercivity = np.empty((candidates.shape[0], stress_count, 6), dtype=float)
    energy = np.empty((candidates.shape[0], stress_count, 2), dtype=float)
    measure = float(np.sum(mass))
    for i, (mesh_h, tau) in enumerate(candidates):
        p = 0
        for q in range(37):
            xi = 2.0 * ((q * generators) % 37) / 36.0 - 1.0
            for s in range(scenarios):
                rho_q = rho[s] * (1.0 + 0.04 * xi[0])
                h0_q = h0[s] * (1.0 + 0.03 * xi[1])
                di_q = di[s] * (1.0 + 0.05 * xi[2])
                r_q = resistive[s] * (1.0 + 0.06 * xi[3])
                k_q = k[s] * (1.0 + 0.02 * xi[4])
                load_q = current_load[s] * (1.0 + 0.05 * xi[5])
                residual_q = residual_load[s] * (1.0 + 0.08 * xi[6])
                magnetic_q = magnetic[s] * (h0_q / h0[s])
                current = lumped_current_projection(mass, load_q)
                electron = projected_electron_velocity(rho_q, velocity[s], current, di_q)
                residual = rescaled_induction_residual(mass, measure, magnetic_q, residual_q)
                blended = blended_artificial_resistivity(
                    mesh_h, r_q, electron[:, 3], residual, scale, c_low, c_res)
                mode = resistive_whistler_mode(
                    rho_q, h0_q, di_q, float(np.mean(blended[:, 2])), k_q)
                wave[i, p] = crank_nicolson_whistler_certificate(mode[0], mode[1], tau)
                old_momentum = rho_q*velocity[s]
                old_energy = internal[s]+np.sum(old_momentum*old_momentum,axis=1)/(2.0*rho_q)
                curl_q = ((1.0+0.05*xi[5])/(1.0+0.03*xi[1]))*curl_matrix
                magnetic_new, velocity_new, midpoint_current = _hall_midpoint_source(
                    curl_q, mass, rho_q, velocity[s], magnetic_q,
                    blended[:, 2], di_q, tau, permeability)
                new_momentum = rho_q*velocity_new
                joule = blended[:, 2]*np.sum(midpoint_current*midpoint_current, axis=1)
                new_load = mass[:, None]*(curl_q @ magnetic_new.ravel()).reshape(nodes, 3)
                new_current = lumped_current_projection(mass, new_load)
                new_electron = projected_electron_velocity(rho_q, velocity_new, new_current, di_q)
                coercivity[i, p] = newton_coercivity_certificates(
                    tau, permeability, mass_constant, rho_q,
                    max(float(np.max(np.linalg.norm(current, axis=1))),
                        float(np.max(np.linalg.norm(new_current, axis=1)))),
                    max(float(np.max(electron[:, 3])), float(np.max(new_electron[:, 3]))),
                    float(np.min(blended[:, 2])))
                old_h_norm_sq = float(np.sum(mass[:, None] * magnetic_q * magnetic_q))
                updated = source_energy_update(
                    mass, np.full(nodes, rho_q), old_momentum, new_momentum,
                    old_energy, joule, tau, old_h_norm_sq, permeability)
                actual_h_norm_sq = float(np.sum(mass[:,None]*magnetic_new*magnetic_new))
                if abs(updated[-3]-actual_h_norm_sq)>1e-9*(1+old_h_norm_sq+actual_h_norm_sq):
                    raise ValueError("coupled source energy balance is unresolved")
                heat_fraction = updated[-1] / float(np.sum(mass * internal[s]))
                energy[i, p] = [updated[-2], heat_fraction]
                p += 1
    table = paper_candidate_table(
        candidates, wave, coercivity, energy, amplitude_limit, phase_limit,
        log_damping_limit, heat_fraction_limit, b6_margin_min)
    feasible = np.where(table[:, 11] == 1.0)[0]
    if feasible.size == 0:
        raise ValueError("no candidate satisfies every constraint")
    selected = min(feasible.tolist(), key=lambda row: (table[row, 10], row))
    return float(round(table[selected, 9], 6))
SCICODE_GOLD_EOF
