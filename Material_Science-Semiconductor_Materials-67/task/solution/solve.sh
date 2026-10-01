#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def _sm67_constants():
    return 4.80320471e-10, 9.1093837015e-28, 1.054571817e-27


def _sm67_require_positive(values):
    import numpy as np
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")


def _sm67_screened_eps(q, h, eps_sub, eps_spacer):
    import numpy as np
    return 0.5 * (eps_sub + eps_spacer / np.tanh(q * h))


def _sm67_q_solve(omega, density, h, eps_sub, eps_spacer, m_rel):
    import numpy as np
    e, m0, _ = _sm67_constants()
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    c = 2.0 * np.pi * density * e ** 2 / (m_rel * m0)
    target = 2.0 * np.log(omega) - np.log(c)
    x = np.log(0.5 * (eps_sub + eps_spacer)) + target
    for _ in range(200):
        q = np.exp(x)
        qh = q * h
        ep = _sm67_screened_eps(q, h, eps_sub, eps_spacer)
        with np.errstate(over="ignore"):
            dep = np.where(qh > 300.0, 0.0, -0.5 * eps_spacer * h / np.sinh(np.minimum(qh, 300.0)) ** 2)
        step = (x - np.log(ep) - target) / (1.0 - q * dep / ep)
        x = x - step
        if np.all(np.abs(step) < 1e-14):
            break
    q = np.exp(x)
    resid = np.log(c * q / _sm67_screened_eps(q, h, eps_sub, eps_spacer)) - 2.0 * np.log(omega)
    if not np.all(np.abs(resid) < 1e-10):
        raise ValueError("the local plasma wave number did not converge")
    return q


def local_plasmon_wavenumber(frequency_ghz: float, density: float, gate_nm: float, eps_substrate: float,
                                     eps_spacer: float, mass_ratio: float) -> float:
    import numpy as np
    _sm67_require_positive(dict(frequency_ghz=frequency_ghz, density=density, gate_nm=gate_nm,
                                eps_substrate=eps_substrate, eps_spacer=eps_spacer, mass_ratio=mass_ratio))
    omega = 2.0 * np.pi * frequency_ghz * 1e9
    return float(_sm67_q_solve(omega, density, gate_nm * 1e-7, eps_substrate, eps_spacer, mass_ratio)[0])

def _sm67_check_cell(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio):
    import numpy as np
    _sm67_require_positive(dict(period_um=period_um, density1=density1, density2=density2, gate1_nm=gate1_nm,
                                gate2_nm=gate2_nm, eps_substrate=eps_substrate, eps_spacer1=eps_spacer1,
                                eps_spacer2=eps_spacer2, mass_ratio=mass_ratio))
    if not np.isfinite(fill) or not (0.0 < fill < 1.0):
        raise ValueError("fill must lie strictly between 0 and 1")


def _sm67_cell_phases(omega, period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio):
    L = period_um * 1e-4
    q1 = _sm67_q_solve(omega, density1, gate1_nm * 1e-7, eps_substrate, eps_spacer1, mass_ratio)
    q2 = _sm67_q_solve(omega, density2, gate2_nm * 1e-7, eps_substrate, eps_spacer2, mass_ratio)
    eta = (_sm67_screened_eps(q2, gate2_nm * 1e-7, eps_substrate, eps_spacer2)
           / _sm67_screened_eps(q1, gate1_nm * 1e-7, eps_substrate, eps_spacer1))
    return q1 * fill * L, q2 * (1.0 - fill) * L, eta


def bloch_phase_cosine(frequency_ghz: float, period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float) -> float:
    import numpy as np
    _sm67_check_cell(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    q1 = local_plasmon_wavenumber(frequency_ghz, density1, gate1_nm, eps_substrate, eps_spacer1, mass_ratio)
    q2 = local_plasmon_wavenumber(frequency_ghz, density2, gate2_nm, eps_substrate, eps_spacer2, mass_ratio)
    L = period_um * 1e-4
    eta = (_sm67_screened_eps(q2, gate2_nm * 1e-7, eps_substrate, eps_spacer2)
           / _sm67_screened_eps(q1, gate1_nm * 1e-7, eps_substrate, eps_spacer1))
    a1, a2 = q1 * fill * L, q2 * (1.0 - fill) * L
    z = 0.5 * (eta + 1.0 / eta)
    return float(np.cos(a1) * np.cos(a2) - z * np.sin(a1) * np.sin(a2))

def _sm67_bright_dark(omega, period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio):
    import numpy as np
    a1, a2, eta = _sm67_cell_phases(omega, period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    c1, s1, c2, s2 = np.cos(a1 / 2), np.sin(a1 / 2), np.cos(a2 / 2), np.sin(a2 / 2)
    return c1 * s2 + eta * s1 * c2, c1 * s2 + s1 * c2 / eta


def _sm67_family_value(omega, k, args):
    import numpy as np
    return _sm67_bright_dark(np.array([omega]), *args)[k][0]


def _sm67_zone_roots(args, n_modes):
    import numpy as np
    from scipy.optimize import brentq
    phase_top = 2.0 * np.pi * (n_modes + 1.5)
    top = np.exp(brentq(lambda x: np.sum(_sm67_cell_phases(np.array([np.exp(x)]), *args)[:2]) - phase_top,
                        np.log(1e6), np.log(1e17), xtol=1e-12))
    grid = np.linspace(top * 1e-5, top, 4000 * (n_modes + 2))
    families = _sm67_bright_dark(grid, *args)
    out = []
    for k in (0, 1):
        vals = families[k]
        idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
        if len(idx) < n_modes:
            raise ValueError("fewer zone-centre modes were found than requested")
        out.append([brentq(_sm67_family_value, grid[i], grid[i + 1], args=(k, args), xtol=1e-10, rtol=1e-15)
                    for i in idx[:n_modes]])
    return np.array(out)


def zone_center_modes(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, n_modes: int) -> "np.ndarray":
    import numpy as np
    _sm67_check_cell(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    if isinstance(n_modes, bool) or int(n_modes) != n_modes or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    args = (period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    return _sm67_zone_roots(args, int(n_modes)) / (2.0 * np.pi * 1e9)

def plasmon_effective_mass(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, kind: str) -> float:
    import numpy as np
    if kind not in ("bright", "dark"):
        raise ValueError("kind must be 'bright' or 'dark'")
    modes = zone_center_modes(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio, 1)
    _, m0, hbar = _sm67_constants()
    k = 0 if kind == "bright" else 1
    f_mode = modes[k, 0]
    omega = 2.0 * np.pi * f_mode * 1e9
    if abs(bloch_phase_cosine(f_mode, period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio) - 1.0) > 1e-8:
        raise ValueError("the zone-centre mode does not satisfy the Bloch condition")
    args = (period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    d = omega * 1e-7
    slope = (_sm67_family_value(omega + d, k, args) - _sm67_family_value(omega - d, k, args)) / (2.0 * d)
    partner = _sm67_family_value(omega, 1 - k, args)
    L = period_um * 1e-4
    return float(1e4 * 2.0 * hbar * slope * partner / L ** 2 / (mass_ratio * m0))

def _sm67_fundamental_split(fill, dev):
    modes = zone_center_modes(dev[0], fill, *dev[1:], 1)
    return modes[0, 0] - modes[1, 0]


def degeneracy_point(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    _sm67_require_positive(dict(period_um=period_um, density1=density1, density2=density2, gate1_nm=gate1_nm,
                                gate2_nm=gate2_nm, eps_substrate=eps_substrate, eps_spacer1=eps_spacer1,
                                eps_spacer2=eps_spacer2, mass_ratio=mass_ratio))
    dev = (period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    fills = np.linspace(0.005, 0.995, 199)
    split = np.array([_sm67_fundamental_split(f, dev) for f in fills])
    brackets = [(fills[i], fills[i + 1]) for i in np.where(np.sign(split[:-1]) != np.sign(split[1:]))[0]]
    for i in range(1, len(fills) - 1):
        if abs(split[i]) < abs(split[i - 1]) and abs(split[i]) < abs(split[i + 1]):
            fine = np.linspace(fills[i - 1], fills[i + 1], 201)
            fine_split = np.array([_sm67_fundamental_split(f, dev) for f in fine])
            brackets += [(fine[j], fine[j + 1]) for j in np.where(np.sign(fine_split[:-1]) != np.sign(fine_split[1:]))[0]]
    if not brackets:
        raise ValueError("the fundamental bright and dark frequencies do not coincide for 0 < f < 1")
    lo, hi = max(brackets, key=lambda b: b[0])
    fstar = brentq(_sm67_fundamental_split, lo, hi, args=(dev,), xtol=1e-13, rtol=1e-15)
    modes = zone_center_modes(period_um, fstar, *dev[1:], 1)
    return np.array([fstar, 0.5 * (modes[0, 0] + modes[1, 0])])

def degeneracy_linear_velocity(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    import numpy as np
    fstar, f0 = degeneracy_point(period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    args = (period_um, fstar, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2,
            mass_ratio)
    w0 = 2.0 * np.pi * f0 * 1e9
    d = w0 * 1e-7
    bw = (_sm67_family_value(w0 + d, 0, args) - _sm67_family_value(w0 - d, 0, args)) / (2.0 * d)
    dw = (_sm67_family_value(w0 + d, 1, args) - _sm67_family_value(w0 - d, 1, args)) / (2.0 * d)
    return float(period_um * 1e-4 / (2.0 * np.sqrt(bw * dw)))

def bright_mass_slope_at_degeneracy(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    import numpy as np
    _, m0, hbar = _sm67_constants()
    fstar, f0 = degeneracy_point(period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    s0 = degeneracy_linear_velocity(period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    step = 1e-4
    rest = (density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    above = zone_center_modes(period_um, fstar + step, *rest, 1)
    below = zone_center_modes(period_um, fstar - step, *rest, 1)
    split_rate = 2.0 * np.pi * 1e9 * ((above[0, 0] - above[1, 0]) - (below[0, 0] - below[1, 0])) / (2.0 * step)
    slope_linear = 1e4 * hbar * split_rate / (2.0 * s0 ** 2) / (mass_ratio * m0)
    mass_up = plasmon_effective_mass(period_um, fstar + step, *rest, "bright")
    mass_dn = plasmon_effective_mass(period_um, fstar - step, *rest, "bright")
    slope_mass = (mass_up - mass_dn) / (2.0 * step)
    if not np.isclose(slope_mass, slope_linear, rtol=1e-3, atol=1e-6):
        raise ValueError("the bright mass does not cross zero linearly at f* for these parameters")
    return float(slope_mass)
SCICODE_GOLD_EOF
