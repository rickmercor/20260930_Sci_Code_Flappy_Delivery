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


def bar_system_matrices(n_elements, length, area, youngs, density):
    if n_elements < 1:
        raise ValueError("n_elements must be at least 1")
    if min(length, area, youngs, density) <= 0:
        raise ValueError("length, area, youngs and density must be positive")
    n = int(n_elements) + 1
    h = float(length) / int(n_elements)
    k = float(youngs) * float(area) / h
    K = np.zeros((n, n), dtype=float)
    blk = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for i in range(int(n_elements)):
        K[i:i + 2, i:i + 2] += blk
    m = float(density) * float(area) * h
    Md = np.full(n, m, dtype=float)
    Md[0] *= 0.5                       # lumped mass, half at each end node
    Md[-1] *= 0.5
    out = np.zeros((n + 1, n), dtype=float)
    out[:n, :] = K
    out[n, :] = Md
    return out

import numpy as np


def critical_time_step(n_elements, length, youngs, density):
    if n_elements < 1:
        raise ValueError("n_elements must be at least 1")
    if min(length, youngs, density) <= 0:
        raise ValueError("length, youngs and density must be positive")
    c = np.sqrt(float(youngs) / float(density))
    h = float(length) / int(n_elements)
    return float(h / c)

import numpy as np


def smooth_predictor(u, v, a, dt):
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    if not (u.shape == v.shape == a.shape):
        raise ValueError("u, v and a must have the same shape")
    if dt <= 0:
        raise ValueError("dt must be positive")
    dt = float(dt)
    return u + dt * v + 0.5 * dt * dt * a

import numpy as np


def active_contact_set(u, v, a, dt, contact_rows):
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    if H.shape[1] != np.asarray(u, dtype=float).shape[0]:
        raise ValueError("contact_rows columns must match the number of degrees of freedom")
    # CONVENTION (paper, eq. 39-40): the set is decided on the SMOOTH PREDICTION at
    # t_{n+1}, not on the gap at t_n; and the test is NON-STRICT.
    u_pred = smooth_predictor(u, v, a, dt)
    gap = H @ u_pred
    return (gap <= 0.0).astype(float)

import numpy as np


def contact_response_operator(stiffness, lumped_mass, contact_rows, active, dt):
    K = np.asarray(stiffness, dtype=float)
    Md = np.asarray(lumped_mass, dtype=float)
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    act = np.asarray(active, dtype=float) > 0.5
    if dt <= 0:
        raise ValueError("dt must be positive")
    if np.any(Md <= 0):
        raise ValueError("lumped_mass entries must be positive")
    if not act.any():
        return np.zeros((0, 0), dtype=float)
    dt = float(dt)
    HA = H[act.ravel()]
    Minv = 1.0 / Md
    MiHt = Minv[:, None] * HA.T
    # CONVENTION (paper, eq. 46): the MODIFIED Delassus operator. The (dt^2/4) K M^-1
    # correction comes from folding a_{n+1} back through the half-step displacement
    # update. The plain Delassus H M^-1 H^T is the natural wrong answer.
    return HA @ MiHt - (dt * dt / 4.0) * (HA @ (Minv[:, None] * (K @ MiHt)))

import numpy as np


def contact_impulse(response_operator, stiffness, lumped_mass, contact_rows,
                            active, predicted, v, a, dt, restitution):
    K = np.asarray(stiffness, dtype=float)
    Md = np.asarray(lumped_mass, dtype=float)
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    act = np.asarray(active, dtype=float) > 0.5
    if not (0.0 <= float(restitution) <= 1.0):
        raise ValueError("restitution must lie in [0, 1]")
    if not act.any():
        return np.zeros(0, dtype=float)
    v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    dt = float(dt); e = float(restitution)
    W = np.atleast_2d(np.asarray(response_operator, dtype=float))
    if W.shape[0] != int(act.sum()):
        raise ValueError("response_operator must be square of size n_active")
    HA = H[act.ravel()]
    Minv = 1.0 / Md
    u_pred = np.asarray(predicted, dtype=float)
    # CONVENTION (paper, eq. 47): Newton's impact law puts (1 + e) on v_n, and the
    # elastic term enters as -(dt/2) M^-1 K u_pred.
    b = HA @ ((1.0 + e) * v + 0.5 * dt * a - 0.5 * dt * (Minv * (K @ u_pred)))
    p = np.linalg.solve(W, -b)
    return np.maximum(0.0, p)

import numpy as np


def nonsmooth_state_update(stiffness, lumped_mass, contact_rows, active,
                                   predicted, v, a, dt, impulse):
    K = np.asarray(stiffness, dtype=float)
    Md = np.asarray(lumped_mass, dtype=float)
    H = np.atleast_2d(np.asarray(contact_rows, dtype=float))
    act = np.asarray(active, dtype=float) > 0.5
    v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    p = np.asarray(impulse, dtype=float).ravel()
    if dt <= 0:
        raise ValueError("dt must be positive")
    if np.any(Md <= 0):
        raise ValueError("lumped_mass entries must be positive")
    u_pred = np.asarray(predicted, dtype=float)
    if u_pred.shape != v.shape:
        raise ValueError("predicted and v must have the same shape")
    dt = float(dt)
    Minv = 1.0 / Md
    if act.any() and p.size:
        HA = H[act.ravel()]
        v_hat = (Minv[:, None] * HA.T @ p).ravel()
    else:
        v_hat = np.zeros_like(v)
    # CONVENTION (paper, eq. 40): the impulsive velocity enters the displacement
    # through HALF a step, not a full step.
    u_new = u_pred + 0.5 * dt * v_hat
    a_new = Minv * (-(K @ u_new))
    v_new = v + 0.5 * dt * (a + a_new) + v_hat
    out = np.zeros((3, u_new.shape[0]), dtype=float)
    out[0] = u_new; out[1] = v_new; out[2] = a_new
    return out

import numpy as np


def cohesive_effective_opening(delta_n, delta_t, beta):
    dn = np.asarray(delta_n, dtype=float)
    dt = np.atleast_2d(np.asarray(delta_t, dtype=float))
    if float(beta) < 0.0:
        raise ValueError("beta must be non-negative")
    if dt.shape[0] != dn.shape[0]:
        raise ValueError("delta_t must have one row per interface")
    b = float(beta)
    # CONVENTION (paper eq. 17): the mixed-mode effective opening weights the TANGENTIAL
    # part by beta SQUARED under the root. This is NOT the same weighting the traction
    # direction uses; see step 12.
    return np.sqrt(dn * dn + b * b * np.sum(dt * dt, axis=1))

import numpy as np


def cohesive_damage_update(effective_opening, critical_opening, damage_previous):
    d_eff = np.asarray(effective_opening, dtype=float)
    d_prev = np.asarray(damage_previous, dtype=float)
    if float(critical_opening) <= 0.0:
        raise ValueError("critical_opening must be positive")
    if np.any(d_prev < 0.0) or np.any(d_prev > 1.0):
        raise ValueError("damage_previous must lie in [0, 1]")
    if d_eff.shape != d_prev.shape:
        raise ValueError("effective_opening and damage_previous must have the same shape")
    # CONVENTION (paper, loading function f = delta/delta_c - d <= 0 with d_dot >= 0):
    # damage is the running MAXIMUM, never decreasing, and saturates at 1.
    return np.clip(np.maximum(d_prev, d_eff / float(critical_opening)), 0.0, 1.0)

import numpy as np


def cohesive_stiffness_cap(youngs, element_size, alpha, cohesive_strength,
                                   critical_opening):
    if min(float(youngs), float(element_size), float(alpha),
           float(cohesive_strength), float(critical_opening)) <= 0.0:
        raise ValueError("youngs, element_size, alpha, cohesive_strength and "
                         "critical_opening must all be positive")
    # CONVENTION (paper, regularisation): the stiffness cap is tied to the BULK element
    # stiffness through a user parameter alpha, and the damage threshold follows from it.
    k_tilde = float(alpha) * float(youngs) / float(element_size)
    sig = float(cohesive_strength); dc = float(critical_opening)
    d_tilde = sig / (sig + k_tilde * dc)
    return np.array([k_tilde, d_tilde], dtype=float)

import numpy as np


def cohesive_traction(delta_n, delta_t, beta, damage, cohesive_strength,
                              critical_opening, damage_threshold):
    dn = np.asarray(delta_n, dtype=float)
    dt = np.atleast_2d(np.asarray(delta_t, dtype=float))
    d = np.asarray(damage, dtype=float)
    if np.any(d < 0.0) or np.any(d > 1.0):
        raise ValueError("damage must lie in [0, 1]")
    if float(critical_opening) <= 0.0 or float(cohesive_strength) <= 0.0:
        raise ValueError("cohesive_strength and critical_opening must be positive")
    b = float(beta); sig = float(cohesive_strength); dc = float(critical_opening)
    dtil = float(damage_threshold)
    # The vector the traction points along: beta SQUARED multiplies delta_t here, so its
    # norm carries beta to the FOURTH power - deliberately not the effective opening of
    # step 09. Getting these two weightings the same way round is the trap.
    vec = np.concatenate([dn[:, None], (b * b) * dt], axis=1)
    vnorm = np.sqrt(np.sum(vec * vec, axis=1))
    d_eff = cohesive_effective_opening(dn, dt, b)
    out = np.zeros_like(vec)
    safe_v = np.where(vnorm > 0.0, vnorm, 1.0)
    safe_d = np.where(d > 0.0, d, 1.0)
    # CONVENTION (paper, regularised law): below the damage threshold the traction is
    # CONSTANT in magnitude, sigma_c (1 - d), and only its DIRECTION varies. At or above
    # the threshold it is the secant law of eq. 20.
    below = (d < dtil)
    mag_below = sig * (1.0 - d)
    # t = k(d) * (delta_n n + beta^2 delta_t) with the Camacho-Ortiz secant stiffness
    # k(d) = ((1-d)/d)(sigma_c/delta_c). The effective opening cancels in the chain rule,
    # so there is NO 1/delta factor here - checking the units is what catches that.
    scale_above = ((1.0 - d) / safe_d) * (sig / dc)
    for i in range(vec.shape[0]):
        if vnorm[i] <= 0.0:
            continue
        if below[i]:
            out[i] = mag_below[i] * vec[i] / safe_v[i]
        else:
            out[i] = scale_above[i] * vec[i]
    return out

import numpy as np


def nsn_bar_impact(n_elements, length, area, youngs, density,
                           impact_speed, restitution, time_step_fraction,
                           initial_gap_fraction, bounce_cycles,
                           cohesive_strength, critical_opening, beta,
                           alpha, tangential_offset):
    if bounce_cycles <= 0:
        raise ValueError("bounce_cycles must be positive")
    sysmat = bar_system_matrices(n_elements, length, area, youngs, density)
    n = int(n_elements) + 1
    K = sysmat[:n, :]
    Md = sysmat[n, :]
    h_el = float(length) / int(n_elements)
    dtc = critical_time_step(n_elements, length, youngs, density)
    dt = float(time_step_fraction) * dtc
    c = np.sqrt(float(youngs) / float(density))
    t_bounce = 2.0 * float(length) / c
    nstep = int(round(float(bounce_cycles) * t_bounce / dt))
    H = np.zeros((1, n), dtype=float); H[0, 0] = 1.0
    v0 = -abs(float(impact_speed))
    u = np.full(n, float(initial_gap_fraction) * abs(v0) * dt, dtype=float)
    v = np.full(n, v0, dtype=float)
    a = (1.0 / Md) * (-(K @ u))
    total_impulse = 0.0
    onsets = 0
    in_contact = False
    # An extrinsic cohesive interface sits at the midpoint. The element spanning it is
    # released so the two halves can separate; the cohesive traction is the only thing
    # holding them together, and it degrades irreversibly as the interface opens.
    mid = int(n_elements) // 2
    K = K.copy()
    kel = float(youngs) * float(area) / h_el
    K[mid:mid + 2, mid:mid + 2] -= kel * np.array([[1.0, -1.0], [-1.0, 1.0]])
    cap = cohesive_stiffness_cap(youngs, h_el, alpha, cohesive_strength,
                                         critical_opening)
    d_tilde = float(cap[1])
    damage = np.zeros(1, dtype=float)
    dt_off = np.array([[float(tangential_offset)]], dtype=float)
    a = (1.0 / Md) * (-(K @ u))
    for _ in range(nstep):
        u_pred = smooth_predictor(u, v, a, dt)
        active = active_contact_set(u, v, a, dt, H)
        if active.any():
            W = contact_response_operator(K, Md, H, active, dt)
            p = contact_impulse(W, K, Md, H, active, u_pred, v, a, dt, restitution)
            total_impulse += float(np.sum(p))
            if not in_contact:
                onsets += 1
            in_contact = True
        else:
            p = np.zeros(0, dtype=float)
            in_contact = False
        state = nonsmooth_state_update(K, Md, H, active, u_pred, v, a, dt, p)
        u, v, a = state[0], state[1], state[2]
        # cohesive interface: opening -> irreversible damage -> traction -> internal force
        dn = np.array([u[mid + 1] - u[mid]], dtype=float)
        eff = cohesive_effective_opening(dn, dt_off, beta)
        damage = cohesive_damage_update(eff, critical_opening, damage)
        tvec = cohesive_traction(dn, dt_off, beta, damage, cohesive_strength,
                                         critical_opening, d_tilde)
        fcoh = float(tvec[0, 0]) * float(area)
        # The traction evaluated at the end-of-step configuration is part of the
        # end-of-step acceleration a_{n+1} (paper, eq. 41a with K(d)), so it enters the
        # trapezoidal velocity update of this step, not only the next step's predictor.
        # The impulse touches only the constrained node, so u_{n+1} and the smooth
        # prediction coincide at the interface nodes and the force is the same on both.
        acoh = (1.0 / Md) * np.eye(n)[mid] * fcoh - (1.0 / Md) * np.eye(n)[mid + 1] * fcoh
        a = a + acoh
        v = v + 0.5 * dt * acoh
    kinetic = 0.5 * float(np.sum(Md * v * v))
    return np.array([float(v[-1]), total_impulse, kinetic, float(onsets),
                     float(damage[0])], dtype=float)
SCICODE_GOLD_EOF
