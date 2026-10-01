"""
Runs the full semi-explicit nonsmooth Newmark march from the free-expansion initial state to the requested horizon, chaining the smooth predictor, the damage update, the contact QP, and the nonsmooth corrector every step, and reports the end-of-run displacement together with the damage, impulse, and event-census diagnostics that verify the run.

Time march and event census.




Each step runs in a fixed order: the smooth predictor advances displacement using only the previous step's velocity and acceleration; the interface damage is updated from the predicted opening, using the same one-way ratchet history rule that never lets damage shrink; the assembled stiffness and cap force are rebuilt for the updated damage, since an interface can cross from the zero-stiffness cap regime into the secant regime partway through the run; the contact-free velocity, gap, and active set follow from the predicted configuration; the convex quadratic program delivers the normal impulse on every active pair; and the nonsmooth corrector folds that impulse into the end-of-step displacement, velocity, and acceleration. Repeating this fixed order for the whole march and accumulating the impulse magnitude and the interface damage at every step gives the run-level census that verifies the simulation is physical: the accumulated normal impulse reports how much momentum the contact constraint absorbs over the whole run, the summed damage reports how far the interfaces have softened by the horizon, and the largest ratio of secant traction to cohesive strength among interfaces that have crossed into the secant regime at the predicted configuration reports how close the run comes to violating the cohesive strength through the secant law alone.

Returns
-------
dict -- u_end_over_L, damage_sum, total_impulse, secant_predicted_max_ratio, N, dt_prime, first_spall_step, first_contact_step (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nsn_time_march(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> dict:
    """Run the full NSN march and report the end-of-run diagnostics.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2).
    edot : float
        Nominal applied strain rate, in 1/s (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0).
    d0 : float
        Initial damage of every interface (must lie in (0, 1)).
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).
    t_star_over_tb : float
        Requested horizon as a fraction of the bar period (must be > 0).
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    density : float
        Bulk mass density, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).
    safety : float
        Safety factor applied to the Gershgorin critical step, in (0, 1].

    Returns
    -------
    result : dict
        u_end_over_L : float, the right-end displacement at the horizon,
            normalized by length.
        damage_sum : float, the sum of every interface's damage at the
            horizon.
        total_impulse : float, the accumulated normal impulse magnitude
            over the whole march.
        secant_predicted_max_ratio : float, the largest ratio of secant
            traction to sigma_c among predicted-open secant-regime
            interfaces over the whole march.
        N : int, the number of steps taken.
        dt_prime : float, the fixed step used by the march.
        first_spall_step : int, the 1-indexed step at which any interface
            first reaches damage 0.999, or -1 if this never happens.
        first_contact_step : int, the 1-indexed step at which the active
            set first becomes nonempty, or -1 if this never happens.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, or if edot, alpha, sigma_c, Gc,
        t_star_over_tb, length, youngs_modulus, density, area, or safety is
        not a positive real number, or if d0 does not lie in (0, 1).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_nsn_time_march(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> dict:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Tiny meshes (n_e in {8, 40}) keep every case except the last cheap for
    the mutation battery; only the last case runs the full 1407-step
    benchmark march, which is needed to expose the pinned final-answer
    diagnostics at full precision. >= 3 cases: two tiny-mesh scenarios
    (normal, and a boundary case with initial damage close to the crossover
    damage so the secant regime is exercised from the first step) plus the
    one benchmark-config case, and two invalid-input edge cases.
    """
    return [
        {
            # normal: tiny mesh, default (deeply cap-regime) initial damage
            "setup": "import numpy as np",
            "call": (
                "float(nsn_time_march(n_e=8)['u_end_over_L']) "
                "+ float(nsn_time_march(n_e=8)['damage_sum']) "
                "+ float(nsn_time_march(n_e=8)['total_impulse'])"
            ),
            "gold_call": (
                "float(_oracle_nsn_time_march(n_e=8)['u_end_over_L']) "
                "+ float(_oracle_nsn_time_march(n_e=8)['damage_sum']) "
                "+ float(_oracle_nsn_time_march(n_e=8)['total_impulse'])"
            ),
        },
        {
            # boundary: tiny mesh, initial damage near the crossover damage
            # so the secant regime is reached quickly (exercises the
            # secant_predicted_max_ratio diagnostic and the branch that
            # crosses regimes mid-march)
            "setup": "import numpy as np",
            "call": (
                "float(nsn_time_march(n_e=40, d0=3.5e-4)['secant_predicted_max_ratio']) "
                "+ float(nsn_time_march(n_e=40, d0=3.5e-4)['first_contact_step'])"
            ),
            "gold_call": (
                "float(_oracle_nsn_time_march(n_e=40, d0=3.5e-4)['secant_predicted_max_ratio']) "
                "+ float(_oracle_nsn_time_march(n_e=40, d0=3.5e-4)['first_contact_step'])"
            ),
        },
        {
            # benchmark config (the only full-size case in this file):
            # pins the final answer, the damage/impulse census, and the
            # secant-on-predicted-max ratio all at once.
            "setup": "import numpy as np",
            "call": (
                "float(nsn_time_march()['u_end_over_L']) "
                "+ float(nsn_time_march()['damage_sum']) "
                "+ float(nsn_time_march()['total_impulse']) "
                "+ float(nsn_time_march()['secant_predicted_max_ratio'])"
            ),
            "gold_call": (
                "float(_oracle_nsn_time_march()['u_end_over_L']) "
                "+ float(_oracle_nsn_time_march()['damage_sum']) "
                "+ float(_oracle_nsn_time_march()['total_impulse']) "
                "+ float(_oracle_nsn_time_march()['secant_predicted_max_ratio'])"
            ),
        },
        {
            # edge: an odd n_e is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: nsn_time_march(n_e=9))",
            "gold_call": "_guard(lambda: _oracle_nsn_time_march(n_e=9))",
        },
        {
            # edge: an initial damage outside (0, 1) is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: nsn_time_march(n_e=8, d0=0.0))",
            "gold_call": "_guard(lambda: _oracle_nsn_time_march(n_e=8, d0=0.0))",
        },
    ]
