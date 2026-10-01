#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def free_ion_volume_fractions(salt_conc: float, xi_total: float, polymer_conc: float, valence: int,
                                      ions: "np.ndarray") -> "np.ndarray":
    import numpy as np
    if isinstance(valence, bool) or not float(valence).is_integer() or int(valence) < 1:
        raise ValueError("valence must be an integer >= 1")
    z = int(valence)
    ions = np.array(ions, dtype=float)
    if ions.shape != (2, 2):
        raise ValueError("ions must have shape (2, 2)")
    if np.any(ions[:, 0] <= 0.0) or np.any(ions[:, 1] < 0.0):
        raise ValueError("radii must be positive and hydration numbers non-negative")
    if not (salt_conc > 0.0 and polymer_conc > 0.0):
        raise ValueError("concentrations must be positive")
    if not (0.0 <= xi_total < 1.0):
        raise ValueError("xi_total must lie in [0, 1)")
    free = float(salt_conc) - float(polymer_conc) * float(xi_total)
    if free <= 0.0:
        raise ValueError("free salt concentration must be positive")
    v_w, c_w = 29.7, 55.56
    omega = 4.0 / 3.0 * np.pi * ions[:, 0] ** 3 / v_w + ions[:, 1]
    return np.array([omega[0], omega[1], free * omega[0] / c_w, z * free * omega[1] / c_w])

def _water_and_bjerrum(temperature: float, dielectric: float) -> tuple:
    import numpy as np
    e, eps0, k_b, v_w = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23, 29.7
    l = v_w ** (1.0 / 3.0)
    l_b = e ** 2 / (4.0 * np.pi * eps0 * float(dielectric) * k_b * float(temperature)) * 1e10
    return l, l_b


def screening_wavenumber_squared(q: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                                         temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    phi_free = np.array(phi_free, dtype=float)
    omega = np.array(omega, dtype=float)
    if phi_free.shape != (2,) or omega.shape != (2,):
        raise ValueError("phi_free and omega must have shape (2,)")
    if np.any(phi_free < 0.0) or np.any(omega <= 0.0):
        raise ValueError("phi_free must be non-negative and omega positive")
    if isinstance(valence, bool) or not float(valence).is_integer() or int(valence) < 1:
        raise ValueError("valence must be an integer >= 1")
    if not (temperature > 0.0 and dielectric > 0.0):
        raise ValueError("temperature and dielectric must be positive")
    z = int(valence)
    l, l_b = _water_and_bjerrum(temperature, dielectric)
    a = omega ** (1.0 / 3.0)
    q = np.array(q, dtype=float)
    return 4.0 * np.pi * l_b / l * (phi_free[0] / omega[0] * z ** 2 * np.exp(-(q * a[0]) ** 2)
                                    + phi_free[1] / omega[1] * np.exp(-(q * a[1]) ** 2))

def _gauss_legendre_24():
    import numpy as np
    rule = getattr(_gauss_legendre_24, "rule", None)
    if rule is None:
        rule = np.polynomial.legendre.leggauss(24)
        _gauss_legendre_24.rule = rule
    return rule


def log_doping_constants(delta_g: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                                 temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    screening_wavenumber_squared(np.array([1.0]), phi_free, omega, valence, temperature, dielectric)
    z = int(valence)
    delta_g = np.array(delta_g, dtype=float)
    if delta_g.shape != (z,):
        raise ValueError("delta_g must have shape (valence,)")
    phi_free = np.array(phi_free, dtype=float)
    omega = np.array(omega, dtype=float)
    l, l_b = _water_and_bjerrum(temperature, dielectric)
    a = omega ** (1.0 / 3.0)
    # composite Gauss-Legendre rule on panels graded geometrically from far below the screening
    # wavenumber sqrt(k2(0)) up to the point where both Gaussian factors are below exp(-46)
    k2_zero = float(screening_wavenumber_squared(np.array([0.0]), phi_free, omega, z, temperature,
                                                         dielectric)[0])
    q_max = np.sqrt(46.0) / a.min()
    q_small = min(q_max * 1e-8, np.sqrt(k2_zero) * 1e-3) if k2_zero > 0.0 else q_max * 1e-8
    n_panels = int(np.ceil(np.log(q_max / q_small) / np.log(2.0))) + 1
    edges = np.concatenate([[0.0], np.geomspace(q_small, q_max, n_panels)])
    nodes, weights = _gauss_legendre_24()
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    q = (half[:, None] * nodes[None, :] + mid[:, None]).ravel()
    w = (half[:, None] * weights[None, :]).ravel()
    k2 = screening_wavenumber_squared(q, phi_free, omega, z, temperature, dielectric)
    integrand = (z * np.exp(-(q * a[0]) ** 2) + np.exp(-(q * a[1]) ** 2)) * q * q / (q * q + k2)
    integral = float(np.dot(w, integrand))
    k = np.arange(1, z + 1, dtype=float)
    return -delta_g + (z / k) * l_b / (np.pi * l) * integral + 1.0

def doping_at_fixed_swelling(salt_conc: float, polymer_fraction: float, polymer_conc: float, valence: int,
                                     ions: "np.ndarray", delta_g: "np.ndarray", temperature: float,
                                     dielectric: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    if not (0.0 < polymer_fraction < 1.0):
        raise ValueError("polymer_fraction must lie strictly between 0 and 1")
    if not (temperature > 0.0 and dielectric > 0.0):
        raise ValueError("temperature and dielectric must be positive")
    free_ion_volume_fractions(salt_conc, 0.0, polymer_conc, valence, ions)
    z = int(valence)
    delta_g = np.array(delta_g, dtype=float)
    if delta_g.shape != (z,):
        raise ValueError("delta_g must have shape (valence,)")
    k = np.arange(1, z + 1, dtype=float)
    s_max = min(1.0, float(salt_conc) / float(polymer_conc))

    def _log_modes(y):
        # y is the logit of xi_PS, so ln(xi_PS) and ln(1 - xi_PS) both stay accurate
        ln_s, ln_pp = -np.logaddexp(0.0, -y), -np.logaddexp(0.0, y)
        v = free_ion_volume_fractions(salt_conc, np.exp(ln_s), polymer_conc, z, ions)
        ln_k = log_doping_constants(delta_g, v[2:], v[:2], z, temperature, dielectric)
        rhs = ln_k - np.log(polymer_fraction) + ln_pp + np.log(v[2]) / k + (z / k) * np.log(v[3])
        return np.log(k) + k * (rhs - ln_s), ln_s

    def _excess(y):
        lm, ln_s = _log_modes(y)
        top = lm.max()
        return top + np.log(np.sum(np.exp(lm - top))) - ln_s

    s_top = s_max * (1.0 - 1e-13)
    y_high = 700.0 if s_top >= 1.0 else np.log(s_top) - np.log1p(-s_top)
    y = brentq(_excess, -700.0, y_high, xtol=1e-14, rtol=1e-15, maxiter=1000)
    lm, ln_s = _log_modes(y)
    xi = np.exp(lm)
    return xi * (np.exp(ln_s) / np.sum(xi))

def swelling_polymer_fraction(xi_modes: "np.ndarray", chi: float, omega_p: float, n_p: float,
                                      entanglement: float, crosslink: float) -> float:
    import numpy as np
    from scipy.optimize import brentq
    xi = np.array(xi_modes, dtype=float)
    if xi.ndim != 1 or xi.size == 0:
        raise ValueError("xi_modes must be a non-empty one dimensional array")
    if np.any(xi < 0.0) or xi.sum() >= 1.0:
        raise ValueError("site fractions must be non-negative with sum below one")
    if not (omega_p > 0.0 and n_p > 0.0) or entanglement < 0.0 or crosslink < 0.0:
        raise ValueError("omega_p and n_p must be positive, entanglement and crosslink non-negative")
    z = xi.size
    network = (entanglement + 0.5 * crosslink * (1.0 - xi.sum())
               + sum((1.0 - 1.0 / kk) * xi[kk - 1] / 2.0 for kk in range(2, z + 1)))
    if network <= 0.0:
        raise ValueError("the network term must be positive")

    def _condition(p):
        return (np.log1p(-p) + (1.0 - 1.0 / (omega_p * n_p)) * p + chi * p * p
                + network / omega_p * p ** (1.0 / 3.0))

    return float(brentq(_condition, 1e-300, 1.0 - 1e-16, xtol=1e-300, rtol=1e-15, maxiter=2000))

def equilibrium_doping(salt_conc: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                               polymer_conc: float, omega_p: float, n_p: float, entanglement: float, crosslink: float,
                               temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    if not (omega_p > 0.0 and n_p > 0.0) or entanglement < 0.0 or crosslink < 0.0:
        raise ValueError("omega_p and n_p must be positive, entanglement and crosslink non-negative")
    if entanglement == 0.0 and crosslink == 0.0:
        raise ValueError("entanglement and crosslink cannot both be zero")

    def _swelling_of(p):
        xi = doping_at_fixed_swelling(salt_conc, p, polymer_conc, valence, ions, delta_g,
                                              temperature, dielectric)
        return swelling_polymer_fraction(xi, chi, omega_p, n_p, entanglement, crosslink)

    p = _swelling_of(0.5)
    converged = False
    for _ in range(200):
        p_new = _swelling_of(p)
        if abs(p_new - p) <= 1e-15 * p:
            p, converged = p_new, True
            break
        p = p_new
    if not converged:
        p = brentq(lambda x: _swelling_of(x) - x, 1e-9, 1.0 - 1e-9, xtol=1e-16, rtol=1e-15, maxiter=1000)
    xi = doping_at_fixed_swelling(salt_conc, p, polymer_conc, valence, ions, delta_g,
                                          temperature, dielectric)
    return np.append(xi, p)

def dominant_mode_transitions(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                                      polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                                      crosslink: float, temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    if isinstance(valence, bool) or not float(valence).is_integer() or int(valence) < 2:
        raise ValueError("valence must be an integer >= 2")
    if not c_max > 0.0:
        raise ValueError("c_max must be positive")
    z = int(valence)
    dg = np.array(delta_g, dtype=float)
    if dg.shape != (z,):
        raise ValueError("delta_g must have shape (valence,)")
    # From the mass-action laws, ln(xi_k / k) = k (1 - dG_k) + k ln R + (terms common to every mode), with
    # R = (1 - xi_PS) / (xi_PS Phi_P): the correlation term z (l_B / (pi l)) I and the free-ion activities are
    # shared by all modes. The predominant mode is the upper envelope of these lines in y = ln R, and y falls
    # monotonically as salt is added, so the changes are fixed values of ln R located on the equilibrium branch.
    kk = np.arange(1, z + 1)
    offset = kk * (1.0 - dg)
    changes = []
    current = z
    while current > 1:
        y_over = np.array([(offset[j - 1] - offset[current - 1]) / (current - j) for j in range(1, current)])
        y_next = float(y_over.max())
        nxt = int(np.nonzero(y_over == y_next)[0].min()) + 1
        changes.append(y_next)
        current = nxt

    def _ln_r(u):
        state = equilibrium_doping(float(np.exp(u)), z, ions, dg, chi, polymer_conc, omega_p, n_p,
                                           entanglement, crosslink, temperature, dielectric)
        s = float(np.sum(state[:z]))
        return float(np.log1p(-s) - np.log(s) - np.log(state[z]))

    u_max = float(np.log(c_max))
    y_at_max = _ln_r(u_max)
    inside = [y for y in changes if y >= y_at_max]
    found = []
    u_floor = None
    for y_change in inside:
        if u_floor is None:
            # lowest change: walk down from c_max in decades until ln R lies above the change
            u_high = u_max
            u_low = u_max
            while True:
                u_low -= np.log(10.0)
                if u_low < np.log(1e-30):
                    raise ValueError("no change of the predominant mode above 1e-30 mol/L")
                if _ln_r(u_low) - y_change > 0.0:
                    break
                u_high = u_low
        else:
            # later changes lie between the previous change and c_max
            u_low, u_high = u_floor, u_max
        root = brentq(lambda u: _ln_r(u) - y_change, u_low, u_high, xtol=1e-14, rtol=1e-15, maxiter=500)
        u_floor = root
        found.append(float(np.exp(root)))
    return np.array(found, dtype=float)

def mixed_mode_takeover_concentration(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                                              polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                                              crosslink: float, temperature: float, dielectric: float) -> float:
    import numpy as np
    changes = dominant_mode_transitions(c_max, valence, ions, delta_g, chi, polymer_conc, omega_p, n_p,
                                                entanglement, crosslink, temperature, dielectric)
    z = int(valence)
    state = equilibrium_doping(c_max, z, ions, delta_g, chi, polymer_conc, omega_p, n_p, entanglement,
                                       crosslink, temperature, dielectric)
    cations = state[:z] / np.arange(1, z + 1)
    if changes.size == 0 or int(np.argmax(cations)) != 0:
        raise ValueError("the mixed mode is not the predominant mode at c_max")
    return float(changes[-1])
SCICODE_GOLD_EOF
