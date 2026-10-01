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


def bar_model_setup(n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4) -> dict:
    """Reference implementation of bar_model_setup."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (length > 0.0 and youngs_modulus > 0.0 and density > 0.0 and area > 0.0):
        raise ValueError("length, youngs_modulus, density, and area must be positive")
    n_e = int(n_e)
    n_nodes = n_e + 1
    h_e = length / n_e
    wave_speed = float(np.sqrt(youngs_modulus / density))
    bar_period = 2.0 * length / wave_speed

    interface_nodes = np.arange(1, n_e, 2)
    n_if = interface_nodes.size

    node_dofs = [None] * n_nodes
    dof = 0
    for k in range(n_nodes):
        if k % 2 == 1 and k != n_e:
            node_dofs[k] = (dof, dof + 1)
            dof += 2
        else:
            node_dofs[k] = (dof,)
            dof += 1
    n_dof = dof

    dof_minus = np.array([node_dofs[k][0] for k in interface_nodes], dtype=int)
    dof_plus = np.array([node_dofs[k][1] for k in interface_nodes], dtype=int)

    element_dofs = np.empty((n_e, 2), dtype=int)
    for e in range(n_e):
        left_node, right_node = e, e + 1
        left_dof = node_dofs[left_node][1] if len(node_dofs[left_node]) == 2 else node_dofs[left_node][0]
        right_dof = node_dofs[right_node][0]
        element_dofs[e] = (left_dof, right_dof)

    mass_regular = density * area * h_e
    mass_face = mass_regular / 2.0
    mass = np.zeros(n_dof)
    for k in range(n_nodes):
        dofs = node_dofs[k]
        if len(dofs) == 2:
            mass[list(dofs)] = mass_face
        elif k == 0 or k == n_e:
            mass[dofs[0]] = mass_face
        else:
            mass[dofs[0]] = mass_regular

    return {
        "n_dof": n_dof,
        "h_e": h_e,
        "wave_speed": wave_speed,
        "bar_period": bar_period,
        "interface_nodes": interface_nodes,
        "dof_minus": dof_minus,
        "dof_plus": dof_plus,
        "element_dofs": element_dofs,
        "mass": mass,
        "mass_regular": mass_regular,
        "mass_face": mass_face,
    }

import numpy as np


def _build_positions_and_mass(n_e, length, density, area):
    """Duplicated-node positions and lumped masses (self-contained; mirrors
    the DOF map of 01_bar_model_setup without importing it)."""
    n_nodes = n_e + 1
    h_e = length / n_e
    node_dofs = [None] * n_nodes
    dof = 0
    for k in range(n_nodes):
        if k % 2 == 1 and k != n_e:
            node_dofs[k] = (dof, dof + 1)
            dof += 2
        else:
            node_dofs[k] = (dof,)
            dof += 1
    n_dof = dof
    mass_regular = density * area * h_e
    mass_face = mass_regular / 2.0
    x = np.zeros(n_dof)
    mass = np.zeros(n_dof)
    for k in range(n_nodes):
        dofs = node_dofs[k]
        for d in dofs:
            x[d] = k * h_e - length / 2.0
        if len(dofs) == 2:
            mass[list(dofs)] = mass_face
        elif k == 0 or k == n_e:
            mass[dofs[0]] = mass_face
        else:
            mass[dofs[0]] = mass_regular
    return x, mass, h_e, mass_regular, mass_face


def initial_state_energy(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, t_star_over_tb: float = 0.42, safety: float = 0.99) -> dict:
    """Reference implementation of initial_state_energy."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (edot > 0.0 and alpha > 0.0 and length > 0.0 and youngs_modulus > 0.0
            and density > 0.0 and area > 0.0 and t_star_over_tb > 0.0 and 0.0 < safety <= 1.0):
        raise ValueError("edot, alpha, length, youngs_modulus, density, area, t_star_over_tb, and safety must be positive")
    n_e = int(n_e)
    x, mass, h_e, mass_regular, mass_face = _build_positions_and_mass(n_e, length, density, area)
    v = edot * x
    E0_discrete = float(np.sum(mass * v * v)) * 0.5
    E0_continuum = 0.5 * density * area * edot ** 2 * length ** 3 / 12.0
    momentum = float(np.sum(mass * v))

    k_e = youngs_modulus * area / h_e
    k_tilde = alpha * youngs_modulus / h_e
    k_tilde_A = k_tilde * area

    ratio_regular = 4.0 * k_e / mass_regular
    ratio_end = 2.0 * k_e / mass_face
    ratio_interface = (2.0 * k_e + 2.0 * k_tilde_A) / mass_face
    omega2_max = max(ratio_regular, ratio_end, ratio_interface)
    omega_max = float(np.sqrt(omega2_max))

    wave_speed = float(np.sqrt(youngs_modulus / density))
    bar_period = 2.0 * length / wave_speed
    t_final = t_star_over_tb * bar_period
    dt_setup = safety * 2.0 / omega_max
    N_steps = int(round(t_final / dt_setup))
    dt_prime = t_final / N_steps
    convexity_margin = dt_prime * omega_max / 2.0

    return {
        "E0_discrete": E0_discrete,
        "E0_continuum": E0_continuum,
        "momentum": momentum,
        "k_tilde": k_tilde,
        "k_tilde_A": k_tilde_A,
        "k_bulk_element": k_e,
        "omega2_max": omega2_max,
        "omega_max": omega_max,
        "dt_setup": dt_setup,
        "N_steps": N_steps,
        "dt_prime": dt_prime,
        "convexity_margin": convexity_margin,
        "bar_period": bar_period,
        "wave_speed": wave_speed,
        "t_final": t_final,
    }

import numpy as np


def cohesive_law_state(delta, dmax_open, n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0) -> dict:
    """Reference implementation of cohesive_law_state."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (length > 0.0 and youngs_modulus > 0.0 and alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0):
        raise ValueError("length, youngs_modulus, alpha, sigma_c, and Gc must be positive")
    delta = np.asarray(delta, dtype=float)
    dmax_open = np.asarray(dmax_open, dtype=float)
    if delta.shape != dmax_open.shape:
        raise ValueError("delta and dmax_open must share the same shape")

    n_e = int(n_e)
    h_e = length / n_e
    delta_c = 2.0 * Gc / sigma_c
    envelope_slope = sigma_c / delta_c
    k_tilde = alpha * youngs_modulus / h_e
    d_tilde = sigma_c / (sigma_c + k_tilde * delta_c)

    def _k_of_d(d):
        d = np.asarray(d, dtype=float)
        safe = np.maximum(d, 1e-250)
        return (1.0 - safe) / safe * envelope_slope

    k_d0 = float(_k_of_d(1e-6))
    cap_ratio = k_d0 / k_tilde

    dmax_new = np.maximum(dmax_open, np.maximum(delta, 0.0))
    damage = np.minimum(dmax_new / delta_c, 1.0)
    cap_regime = damage < d_tilde
    cap_traction = np.where(delta > 0.0, sigma_c * (1.0 - damage), 0.0)
    traction = np.where(cap_regime, cap_traction, _k_of_d(damage) * delta)

    return {
        "delta_c": delta_c,
        "envelope_slope": envelope_slope,
        "d_tilde": d_tilde,
        "k_d0": k_d0,
        "cap_ratio": cap_ratio,
        "damage": damage,
        "dmax_open": dmax_new,
        "traction": traction,
    }

import numpy as np


def _mulK(kdiag, kup, x):
    """Tridiagonal matrix-vector product from a (diagonal, superdiagonal)
    pair, exploiting the assembled stiffness's symmetric banded shape."""
    y = kdiag * x
    y[:-1] += kup * x[1:]
    y[1:] += kup * x[:-1]
    return y


def predictor_forces(u, v, a, damage, dof_minus, dof_plus, element_dofs, mass, k_e: float, d_tilde: float, dt: float, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0, area: float = 6.45e-4) -> dict:
    """Reference implementation of predictor_forces."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    if not (u.shape == v.shape == a.shape):
        raise ValueError("u, v, and a must share the same shape")
    if not (dt > 0.0 and k_e > 0.0):
        raise ValueError("dt and k_e must be positive")
    if not (alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0):
        raise ValueError("alpha, sigma_c, and Gc must be positive")
    if not (0.0 < d_tilde < 1.0):
        raise ValueError("d_tilde must lie in (0, 1)")

    mass = np.asarray(mass, dtype=float)
    damage = np.asarray(damage, dtype=float)
    dof_minus = np.asarray(dof_minus, dtype=int)
    dof_plus = np.asarray(dof_plus, dtype=int)
    element_dofs = np.asarray(element_dofs, dtype=int)
    n_dof = u.shape[0]
    delta_c = 2.0 * Gc / sigma_c

    u_tilde = u + dt * v + 0.5 * dt * dt * a

    kdiag = np.zeros(n_dof)
    kup = np.zeros(n_dof - 1)
    for i, j in element_dofs:
        kdiag[i] += k_e
        kdiag[j] += k_e
        kup[i] += -k_e
    sec = (damage >= d_tilde) & (damage < 1.0)
    if sec.any():
        ks = (1.0 - damage[sec]) / damage[sec] * (sigma_c / delta_c) * area
        np.add.at(kdiag, dof_minus[sec], ks)
        np.add.at(kdiag, dof_plus[sec], ks)
        np.add.at(kup, dof_minus[sec], -ks)

    delta_pred = u_tilde[dof_plus] - u_tilde[dof_minus]
    fcap = np.zeros(n_dof)
    cap_sel = (delta_pred > 0.0) & (damage < d_tilde)
    if cap_sel.any():
        f = area * sigma_c * (1.0 - damage[cap_sel])
        np.add.at(fcap, dof_plus[cap_sel], -f)
        np.add.at(fcap, dof_minus[cap_sel], +f)

    K_ut = _mulK(kdiag, kup, u_tilde)
    v_free = v + 0.5 * dt * a + 0.5 * dt * (fcap - K_ut) / mass

    return {
        "u_tilde": u_tilde,
        "v_free": v_free,
        "fcap": fcap,
        "k_diag": kdiag,
        "k_up": kup,
        "delta_pred": delta_pred,
    }

import numpy as np


def _mulK(kdiag, kup, x):
    """Tridiagonal matrix-vector product from a (diagonal, superdiagonal)
    pair, exploiting the assembled stiffness's symmetric banded shape."""
    y = kdiag * x
    y[:-1] += kup * x[1:]
    y[1:] += kup * x[:-1]
    return y


def _active_set_qp(W, b):
    """Solve the nonnegative QP and return only after validating its KKT conditions."""
    n = len(b)
    p = np.zeros(n)
    if n == 0:
        return p, 0.0
    free = b < -1e-12 * max(1.0, float(np.max(np.abs(b))))
    for _ in range(600):
        fidx = np.flatnonzero(free)
        candidate = np.zeros(n)
        if fidx.size:
            candidate[fidx] = np.linalg.solve(W[np.ix_(fidx, fidx)], -b[fidx])
        scale = max(1.0, float(np.max(np.abs(b))), float(np.max(np.abs(W @ p))))
        tolerance = 1e-12 * scale
        negative = candidate[fidx] < -tolerance
        if negative.any():
            direction = candidate - p
            ratios = p[fidx][negative] / -direction[fidx][negative]
            step = float(np.min(ratios))
            p += step * direction
            p[np.abs(p) <= tolerance] = 0.0
            blocking = fidx[negative][ratios <= step * (1.0 + 1e-12) + 1e-15]
            free[blocking] = False
            continue
        p = np.maximum(candidate, 0.0)
        g = W @ p + b
        scale = max(1.0, float(np.max(np.abs(b))), float(np.max(np.abs(W @ p))))
        tolerance = 1e-12 * scale
        bound = np.flatnonzero(~free)
        primal_ok = float(np.min(p)) >= -tolerance
        dual_ok = float(np.min(g)) >= -tolerance
        complementarity_ok = float(np.max(np.abs(p * g))) <= tolerance * max(
            1.0, float(np.max(np.abs(p)))
        )
        if primal_ok and dual_ok and complementarity_ok:
            return p, (float(np.min(g[bound])) if bound.size else 0.0)
        violated = bound[g[bound] < -tolerance]
        if violated.size:
            free[violated[np.argmin(g[violated])]] = True
            continue
        raise RuntimeError("active-set QP candidate failed KKT validation")
    raise RuntimeError("active-set QP did not converge within 600 iterations")


def contact_qp_solve(dof_minus, dof_plus, mass, k_diag, k_up, u_tilde, v_free, v_n, dt: float, n_dof: int, e_restitution: float = 0.0) -> dict:
    """Reference implementation of contact_qp_solve."""
    if not (dt > 0.0):
        raise ValueError("dt must be positive")
    if e_restitution < 0.0:
        raise ValueError("e_restitution must be nonnegative")
    if not isinstance(n_dof, (int, np.integer)) or int(n_dof) < 2:
        raise ValueError("n_dof must be an integer >= 2")

    dof_minus = np.asarray(dof_minus, dtype=int)
    dof_plus = np.asarray(dof_plus, dtype=int)
    mass = np.asarray(mass, dtype=float)
    k_diag = np.asarray(k_diag, dtype=float)
    k_up = np.asarray(k_up, dtype=float)
    u_tilde = np.asarray(u_tilde, dtype=float)
    v_free = np.asarray(v_free, dtype=float)
    v_n = np.asarray(v_n, dtype=float)
    n_dof = int(n_dof)
    n_if = dof_minus.shape[0]

    H = np.zeros((n_if, n_dof))
    H[np.arange(n_if), dof_minus] = -1.0
    H[np.arange(n_if), dof_plus] = +1.0
    gap = H @ u_tilde
    active = np.flatnonzero(gap <= 0.0)

    p = np.zeros(n_if)
    residual = 0.0
    if active.size:
        HA = H[active]
        rhs_v = v_free + e_restitution * v_n
        b = HA @ rhs_v
        Z = HA / mass[None, :]
        KZ = np.empty_like(Z)
        for r in range(Z.shape[0]):
            KZ[r] = _mulK(k_diag, k_up, Z[r])
        W = Z @ HA.T - (dt * dt / 4.0) * (Z @ KZ.T)
        W = 0.5 * (W + W.T)
        p_active, residual = _active_set_qp(W, b)
        p[active] = p_active

    return {"p": p, "gap": gap, "active": active, "residual": residual}

import numpy as np


def _mulK(kdiag, kup, x):
    """Tridiagonal matrix-vector product from a (diagonal, superdiagonal)
    pair, exploiting the assembled stiffness's symmetric banded shape."""
    y = kdiag * x
    y[:-1] += kup * x[1:]
    y[1:] += kup * x[:-1]
    return y


def nonsmooth_corrector(u_tilde, v, a, p, dof_minus, dof_plus, mass, k_diag, k_up, fcap, dt: float) -> dict:
    """Reference implementation of nonsmooth_corrector."""
    u_tilde = np.asarray(u_tilde, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    if not (u_tilde.shape == v.shape == a.shape):
        raise ValueError("u_tilde, v, and a must share the same shape")
    if not (dt > 0.0):
        raise ValueError("dt must be positive")

    p = np.asarray(p, dtype=float)
    dof_minus = np.asarray(dof_minus, dtype=int)
    dof_plus = np.asarray(dof_plus, dtype=int)
    mass = np.asarray(mass, dtype=float)
    k_diag = np.asarray(k_diag, dtype=float)
    k_up = np.asarray(k_up, dtype=float)
    fcap = np.asarray(fcap, dtype=float)
    n_dof = u_tilde.shape[0]

    HTp = np.zeros(n_dof)
    np.add.at(HTp, dof_minus, -p)
    np.add.at(HTp, dof_plus, +p)
    bv = HTp / mass

    u_next = u_tilde + 0.5 * dt * bv
    a_next = (fcap - _mulK(k_diag, k_up, u_next)) / mass
    v_next = v + 0.5 * dt * (a + a_next) + bv

    return {"u_next": u_next, "v_next": v_next, "a_next": a_next, "bv": bv}

import numpy as np


def _mulK(kdiag, kup, x):
    y = kdiag * x
    y[:-1] += kup * x[1:]
    y[1:] += kup * x[:-1]
    return y


def _build_bar(n_e, length, youngs_modulus, density, area):
    n_nodes = n_e + 1
    h_e = length / n_e
    node_dofs = [None] * n_nodes
    dof = 0
    for k in range(n_nodes):
        if k % 2 == 1 and k != n_e:
            node_dofs[k] = (dof, dof + 1)
            dof += 2
        else:
            node_dofs[k] = (dof,)
            dof += 1
    n_dof = dof
    interface_nodes = np.arange(1, n_e, 2)
    dof_minus = np.array([node_dofs[k][0] for k in interface_nodes], dtype=int)
    dof_plus = np.array([node_dofs[k][1] for k in interface_nodes], dtype=int)
    element_dofs = np.empty((n_e, 2), dtype=int)
    for e in range(n_e):
        left_node, right_node = e, e + 1
        left_dof = node_dofs[left_node][1] if len(node_dofs[left_node]) == 2 else node_dofs[left_node][0]
        right_dof = node_dofs[right_node][0]
        element_dofs[e] = (left_dof, right_dof)
    mass_regular = density * area * h_e
    mass_face = mass_regular / 2.0
    mass = np.zeros(n_dof)
    x = np.zeros(n_dof)
    for k in range(n_nodes):
        dofs = node_dofs[k]
        for d in dofs:
            x[d] = k * h_e - length / 2.0
        if len(dofs) == 2:
            mass[list(dofs)] = mass_face
        elif k == 0 or k == n_e:
            mass[dofs[0]] = mass_face
        else:
            mass[dofs[0]] = mass_regular
    k_e = youngs_modulus * area / h_e
    return dict(n_dof=n_dof, h_e=h_e, x=x, mass=mass, dof_minus=dof_minus, dof_plus=dof_plus,
                element_dofs=element_dofs, k_e=k_e, mass_regular=mass_regular, mass_face=mass_face)


def _assemble_K(n_dof, element_dofs, dof_minus, dof_plus, k_e, damage, d_tilde, sigma_c, delta_c, area):
    kdiag = np.zeros(n_dof)
    kup = np.zeros(n_dof - 1)
    for i, j in element_dofs:
        kdiag[i] += k_e
        kdiag[j] += k_e
        kup[i] += -k_e
    sec = (damage >= d_tilde) & (damage < 1.0)
    if sec.any():
        ks = (1.0 - damage[sec]) / damage[sec] * (sigma_c / delta_c) * area
        np.add.at(kdiag, dof_minus[sec], ks)
        np.add.at(kdiag, dof_plus[sec], ks)
        np.add.at(kup, dof_minus[sec], -ks)
    return kdiag, kup


def _active_set_qp(W, b):
    """Solve the nonnegative QP and return only after validating its KKT conditions."""
    n = len(b)
    p = np.zeros(n)
    if n == 0:
        return p, 0.0
    free = b < -1e-12 * max(1.0, float(np.max(np.abs(b))))
    for _ in range(600):
        fidx = np.flatnonzero(free)
        candidate = np.zeros(n)
        if fidx.size:
            candidate[fidx] = np.linalg.solve(W[np.ix_(fidx, fidx)], -b[fidx])
        scale = max(1.0, float(np.max(np.abs(b))), float(np.max(np.abs(W @ p))))
        tolerance = 1e-12 * scale
        negative = candidate[fidx] < -tolerance
        if negative.any():
            direction = candidate - p
            ratios = p[fidx][negative] / -direction[fidx][negative]
            step = float(np.min(ratios))
            p += step * direction
            p[np.abs(p) <= tolerance] = 0.0
            blocking = fidx[negative][ratios <= step * (1.0 + 1e-12) + 1e-15]
            free[blocking] = False
            continue
        p = np.maximum(candidate, 0.0)
        g = W @ p + b
        scale = max(1.0, float(np.max(np.abs(b))), float(np.max(np.abs(W @ p))))
        tolerance = 1e-12 * scale
        bound = np.flatnonzero(~free)
        primal_ok = float(np.min(p)) >= -tolerance
        dual_ok = float(np.min(g)) >= -tolerance
        complementarity_ok = float(np.max(np.abs(p * g))) <= tolerance * max(
            1.0, float(np.max(np.abs(p)))
        )
        if primal_ok and dual_ok and complementarity_ok:
            return p, (float(np.min(g[bound])) if bound.size else 0.0)
        violated = bound[g[bound] < -tolerance]
        if violated.size:
            free[violated[np.argmin(g[violated])]] = True
            continue
        raise RuntimeError("active-set QP candidate failed KKT validation")
    raise RuntimeError("active-set QP did not converge within 600 iterations")


def nsn_time_march(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> dict:
    """Reference implementation of nsn_time_march."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (edot > 0.0 and alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0 and t_star_over_tb > 0.0
            and length > 0.0 and youngs_modulus > 0.0 and density > 0.0 and area > 0.0
            and 0.0 < safety <= 1.0):
        raise ValueError("edot, alpha, sigma_c, Gc, t_star_over_tb, length, youngs_modulus, density, area, and safety must be positive")
    if not (0.0 < d0 < 1.0):
        raise ValueError("d0 must lie in (0, 1)")

    n_e = int(n_e)
    g = _build_bar(n_e, length, youngs_modulus, density, area)
    n_dof = g["n_dof"]
    dof_minus, dof_plus = g["dof_minus"], g["dof_plus"]
    element_dofs, mass, k_e = g["element_dofs"], g["mass"], g["k_e"]
    n_if = dof_minus.shape[0]

    delta_c = 2.0 * Gc / sigma_c
    k_tilde = alpha * youngs_modulus / g["h_e"]
    k_tilde_A = k_tilde * area
    d_tilde = sigma_c / (sigma_c + k_tilde * delta_c)

    ratio_regular = 4.0 * k_e / g["mass_regular"]
    ratio_end = 2.0 * k_e / g["mass_face"]
    ratio_interface = (2.0 * k_e + 2.0 * k_tilde_A) / g["mass_face"]
    omega_max = float(np.sqrt(max(ratio_regular, ratio_end, ratio_interface)))

    wave_speed = float(np.sqrt(youngs_modulus / density))
    bar_period = 2.0 * length / wave_speed
    t_final = t_star_over_tb * bar_period
    dt0 = safety * 2.0 / omega_max
    N = int(round(t_final / dt0))
    dtp = t_final / N

    dmax_open = np.full(n_if, d0 * delta_c)
    dmg = np.full(n_if, d0)
    u = np.zeros(n_dof)
    v = edot * g["x"].copy()
    a = np.zeros(n_dof)
    an = a.copy()

    H = np.zeros((n_if, n_dof))
    H[np.arange(n_if), dof_minus] = -1.0
    H[np.arange(n_if), dof_plus] = +1.0

    first_spall = -1
    first_contact = -1
    total_impulse = 0.0
    secant_predicted_max_ratio = 0.0

    for n in range(N):
        dt = dtp
        ut = u + dt * v + 0.5 * dt * dt * an
        delta = ut[dof_plus] - ut[dof_minus]
        dmax_open = np.maximum(dmax_open, np.maximum(delta, 0.0))
        dmg = np.minimum(dmax_open / delta_c, 1.0)

        kdiag, kup = _assemble_K(n_dof, element_dofs, dof_minus, dof_plus, k_e, dmg, d_tilde, sigma_c, delta_c, area)

        sec_mask = (dmg >= d_tilde) & (dmg < 1.0) & (delta > 0.0)
        if sec_mask.any():
            ks = (1.0 - dmg[sec_mask]) / dmg[sec_mask] * (sigma_c / delta_c)
            ratio = float(np.max(ks * delta[sec_mask] / sigma_c))
            if ratio > secant_predicted_max_ratio:
                secant_predicted_max_ratio = ratio

        fcap = np.zeros(n_dof)
        cap_sel = (delta > 0.0) & (dmg < d_tilde)
        if cap_sel.any():
            f = area * sigma_c * (1.0 - dmg[cap_sel])
            np.add.at(fcap, dof_plus[cap_sel], -f)
            np.add.at(fcap, dof_minus[cap_sel], +f)

        K_ut = _mulK(kdiag, kup, ut)
        vfree = v + 0.5 * dt * an + 0.5 * dt * (fcap - K_ut) / mass

        gap = H @ ut
        act = np.flatnonzero(gap <= 0.0)
        if act.size:
            if first_contact == -1:
                first_contact = n + 1
            HA = H[act]
            b = HA @ vfree
            Z = HA / mass[None, :]
            KZ = np.empty_like(Z)
            for r in range(Z.shape[0]):
                KZ[r] = _mulK(kdiag, kup, Z[r])
            W = Z @ HA.T - (dt * dt / 4.0) * (Z @ KZ.T)
            W = 0.5 * (W + W.T)
            p_act, _residual = _active_set_qp(W, b)
            p = np.zeros(n_if)
            p[act] = p_act
        else:
            p = np.zeros(n_if)

        total_impulse += float(np.sum(np.abs(p)))

        HTp = np.zeros(n_dof)
        np.add.at(HTp, dof_minus, -p)
        np.add.at(HTp, dof_plus, +p)
        bv = HTp / mass
        u = ut + 0.5 * dt * bv
        a_new = (fcap - _mulK(kdiag, kup, u)) / mass
        v = v + 0.5 * dt * (an + a_new) + bv
        an = a_new

        if first_spall == -1 and int(np.sum(dmg >= 0.999)):
            first_spall = n + 1

    return {
        "u_end_over_L": float(u[-1] / length),
        "damage_sum": float(np.sum(dmg)),
        "total_impulse": total_impulse,
        "secant_predicted_max_ratio": secant_predicted_max_ratio,
        "N": N,
        "dt_prime": dtp,
        "first_spall_step": first_spall,
        "first_contact_step": first_contact,
    }

import numpy as np


def _build_positions_and_mass(n_e, length, density, area):
    """Duplicated-node positions and lumped masses (mirrors the DOF map of
    bar_model_setup, adding the node positions it does not return)."""
    n_nodes = n_e + 1
    h_e = length / n_e
    node_dofs = [None] * n_nodes
    dof = 0
    for k in range(n_nodes):
        if k % 2 == 1 and k != n_e:
            node_dofs[k] = (dof, dof + 1)
            dof += 2
        else:
            node_dofs[k] = (dof,)
            dof += 1
    n_dof = dof
    mass_regular = density * area * h_e
    mass_face = mass_regular / 2.0
    x = np.zeros(n_dof)
    mass = np.zeros(n_dof)
    for k in range(n_nodes):
        dofs = node_dofs[k]
        for d in dofs:
            x[d] = k * h_e - length / 2.0
        if len(dofs) == 2:
            mass[list(dofs)] = mass_face
        elif k == 0 or k == n_e:
            mass[dofs[0]] = mass_face
        else:
            mass[dofs[0]] = mass_regular
    return x, mass, h_e, mass_regular, mass_face


def run_nsn_fragmentation(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> float:
    """Reference implementation of run_nsn_fragmentation: chains
    bar_model_setup, initial_state_energy,
    cohesive_law_state, predictor_forces,
    contact_qp_solve, and nonsmooth_corrector step by step,
    then cross-checks the result against nsn_time_march's fused
    march before returning it."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (edot > 0.0 and alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0 and t_star_over_tb > 0.0
            and length > 0.0 and youngs_modulus > 0.0 and density > 0.0 and area > 0.0
            and 0.0 < safety <= 1.0):
        raise ValueError("edot, alpha, sigma_c, Gc, t_star_over_tb, length, youngs_modulus, density, area, and safety must be positive")
    if not (0.0 < d0 < 1.0):
        raise ValueError("d0 must lie in (0, 1)")

    n_e = int(n_e)
    setup = bar_model_setup(n_e, length, youngs_modulus, density, area)
    state0 = initial_state_energy(n_e, edot, alpha, length, youngs_modulus, density, area, t_star_over_tb, safety)

    n_dof = setup["n_dof"]
    dof_minus, dof_plus = setup["dof_minus"], setup["dof_plus"]
    element_dofs, mass = setup["element_dofs"], setup["mass"]
    k_e = youngs_modulus * area / setup["h_e"]
    n_if = dof_minus.shape[0]

    N = state0["N_steps"]
    dtp = state0["dt_prime"]
    delta_c = 2.0 * Gc / sigma_c

    dmax_open = np.full(n_if, d0 * delta_c)
    u = np.zeros(n_dof)
    x, _mass_check, _h_e_check, _mr, _mf = _build_positions_and_mass(n_e, length, density, area)
    v = edot * x
    a = np.zeros(n_dof)

    for _ in range(N):
        dt = dtp
        ut_probe = u + dt * v + 0.5 * dt * dt * a
        delta_pred = ut_probe[dof_plus] - ut_probe[dof_minus]
        law = cohesive_law_state(delta_pred, dmax_open, n_e, length, youngs_modulus, alpha, sigma_c, Gc)
        damage = law["damage"]
        dmax_open = law["dmax_open"]
        d_tilde = law["d_tilde"]

        r4 = predictor_forces(u, v, a, damage, dof_minus, dof_plus, element_dofs, mass, k_e, d_tilde, dt, alpha, sigma_c, Gc, area)
        r5 = contact_qp_solve(dof_minus, dof_plus, mass, r4["k_diag"], r4["k_up"], r4["u_tilde"], r4["v_free"], v, dt, n_dof)
        r6 = nonsmooth_corrector(r4["u_tilde"], v, a, r5["p"], dof_minus, dof_plus, mass, r4["k_diag"], r4["k_up"], r4["fcap"], dt)
        u, v, a = r6["u_next"], r6["v_next"], r6["a_next"]

    u_end_over_L = float(u[-1] / length)

    fused = nsn_time_march(n_e=n_e, edot=edot, alpha=alpha, d0=d0, sigma_c=sigma_c, Gc=Gc, t_star_over_tb=t_star_over_tb, length=length, youngs_modulus=youngs_modulus, density=density, area=area, safety=safety)
    fused_value = fused["u_end_over_L"]
    if not np.allclose(u_end_over_L, fused_value, rtol=1e-9, atol=1e-12):
        raise ValueError(f"step-by-step chain ({u_end_over_L}) disagrees with the fused march ({fused_value}) beyond tolerance")

    return u_end_over_L
SCICODE_GOLD_EOF
