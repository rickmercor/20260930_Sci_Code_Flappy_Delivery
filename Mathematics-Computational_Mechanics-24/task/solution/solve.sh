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
def arm_ratio_sum(mode: str, n_cohesive: int, n_total: int, t_ply: float) -> float:
    if not isinstance(mode, str):
        raise ValueError("mode must be a string")
    if mode not in ("opening", "shear"):
        raise ValueError("mode must be either 'opening' or 'shear'")
    for name, val in (("n_cohesive", n_cohesive), ("n_total", n_total)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    n_cohesive = int(n_cohesive)
    n_total = int(n_total)
    if n_cohesive < 1:
        raise ValueError("n_cohesive must be >= 1")
    if n_total < n_cohesive:
        raise ValueError("n_total must be >= n_cohesive")
    if isinstance(t_ply, bool) or not isinstance(t_ply, (int, float, np.integer, np.floating)):
        raise ValueError("t_ply must be a real number")
    t_ply = float(t_ply)
    if not np.isfinite(t_ply) or t_ply <= 0.0:
        raise ValueError("t_ply must be a finite number > 0")

    # One resin-rich layer per represented ply, the first one lying on the
    # cohesive interface; the polynomials are anchored on the traction-free
    # surface of the whole half-stack.
    z_layers = np.arange(n_cohesive, dtype=float) * t_ply
    h = n_total * t_ply

    # Rows of the linear system act on c = [a1, a2, a3, a4] through
    #   tau(z)   = a1*z**2   + a2*z      + a3
    #   sigma(z) = a1*z**3/3 + a2*z**2/2 + a3*z + a4
    tau_0 = np.array([0.0, 0.0, 1.0, 0.0])
    tau_h = np.array([h ** 2, h, 1.0, 0.0])
    sig_0 = np.array([0.0, 0.0, 0.0, 1.0])
    sig_h = np.array([h ** 3 / 3.0, h ** 2 / 2.0, h, 1.0])

    # Both stresses are traction free on the outer surface. At the cohesive
    # interface the mode decides which stress carries the normalised peak and
    # which one vanishes.
    if mode == "opening":
        a_mat = np.vstack([sig_0, tau_0, tau_h, sig_h])
    else:
        a_mat = np.vstack([tau_0, sig_0, tau_h, sig_h])
    rhs = np.array([1.0, 0.0, 0.0, 0.0])
    a1, a2, a3, a4 = np.linalg.solve(a_mat, rhs)

    if mode == "opening":
        row = a1 * z_layers ** 3 / 3.0 + a2 * z_layers ** 2 / 2.0 + a3 * z_layers + a4
    else:
        row = a1 * z_layers ** 2 + a2 * z_layers + a3

    peak = float(row[0])
    if peak == 0.0:
        raise ValueError("the driving stress on the cohesive interface must be non-zero")
    return float(np.sum(row / peak))

import numpy as np
def resin_rich_penalty_stiffness(ratio_sum_total: float, h_rr: float,
                                         modulus: float) -> float:

    values = {"ratio_sum_total": ratio_sum_total, "h_rr": h_rr, "modulus": modulus}
    for name, val in values.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")

    ratio_sum_total = float(ratio_sum_total)
    h_rr = float(h_rr)
    modulus = float(modulus)

    # The compliance of the interface is the stacked resin-layer compliance.
    interface_compliance = ratio_sum_total * h_rr / modulus
    return float(1.0 / interface_compliance)

import numpy as np
def mixed_mode_onset_separation(k_n: float, k_s: float, tau_ic: float,
                                        tau_iic: float, disp_ratio: float) -> float:

    positives = {"k_n": k_n, "k_s": k_s, "tau_ic": tau_ic, "tau_iic": tau_iic}
    for name, val in positives.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if isinstance(disp_ratio, bool) or not isinstance(disp_ratio, (int, float, np.integer, np.floating)):
        raise ValueError("disp_ratio must be a real number")
    disp_ratio = float(disp_ratio)
    if not np.isfinite(disp_ratio) or disp_ratio < 0.0:
        raise ValueError("disp_ratio must be a finite number >= 0")

    k_n = float(k_n)
    k_s = float(k_s)
    tau_ic = float(tau_ic)
    tau_iic = float(tau_iic)

    # Along the proportional path both jumps are fixed fractions of the
    # effective separation, so the quadratic criterion becomes linear in
    # delta**2 and can be inverted directly.
    norm = np.sqrt(1.0 + disp_ratio ** 2)
    normal_term = k_n / (tau_ic * norm)
    shear_term = k_s * disp_ratio / (tau_iic * norm)
    return float(1.0 / np.sqrt(normal_term ** 2 + shear_term ** 2))

import numpy as np
def mixed_mode_energy_ratio(k_n: float, k_s: float, disp_ratio: float) -> float:

    for name, val in (("k_n", k_n), ("k_s", k_s)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if isinstance(disp_ratio, bool) or not isinstance(disp_ratio, (int, float, np.integer, np.floating)):
        raise ValueError("disp_ratio must be a real number")
    disp_ratio = float(disp_ratio)
    if not np.isfinite(disp_ratio) or disp_ratio < 0.0:
        raise ValueError("disp_ratio must be a finite number >= 0")

    k_n = float(k_n)
    k_s = float(k_s)

    # Both energy contributions carry the same delta_n**2, which cancels.
    shear_energy = k_s * disp_ratio ** 2
    return float(shear_energy / (k_n + shear_energy))

import numpy as np
def benzeggagh_kenane_toughness(g_ic: float, g_iic: float, eta: float,
                                        b_ratio: float) -> float:
    
    for name, val in (("g_ic", g_ic), ("g_iic", g_iic), ("eta", eta)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if isinstance(b_ratio, bool) or not isinstance(b_ratio, (int, float, np.integer, np.floating)):
        raise ValueError("b_ratio must be a real number")
    b_ratio = float(b_ratio)
    if not np.isfinite(b_ratio) or b_ratio < 0.0 or b_ratio > 1.0:
        raise ValueError("b_ratio must be a finite number in [0, 1]")

    g_ic = float(g_ic)
    g_iic = float(g_iic)
    eta = float(eta)

    return float(g_ic + (g_iic - g_ic) * b_ratio ** eta)

import numpy as np
def bilinear_failure_separation(k_n: float, k_s: float, disp_ratio: float,
                                        delta_onset: float, g_c: float) -> float:
    positives = {"k_n": k_n, "k_s": k_s, "delta_onset": delta_onset, "g_c": g_c}
    for name, val in positives.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if isinstance(disp_ratio, bool) or not isinstance(disp_ratio, (int, float, np.integer, np.floating)):
        raise ValueError("disp_ratio must be a real number")
    disp_ratio = float(disp_ratio)
    if not np.isfinite(disp_ratio) or disp_ratio < 0.0:
        raise ValueError("disp_ratio must be a finite number >= 0")

    k_n = float(k_n)
    k_s = float(k_s)
    delta_onset = float(delta_onset)
    g_c = float(g_c)

    # Energy-equivalent secant stiffness of the elastic branch.
    k_eff = (k_n + k_s * disp_ratio ** 2) / (1.0 + disp_ratio ** 2)

    # Area of the bilinear triangle equals the mixed-mode toughness.
    delta_failure = 2.0 * g_c / (k_eff * delta_onset)
    if delta_failure < delta_onset:
        raise ValueError("the failure separation must not fall below the onset separation")
    return float(delta_failure)

import numpy as np

def optimal_reinforcement_failure_separation(
        n_cohesive: tuple = (15, 11), n_reinforcement: int = 12,
        t_ply: float = 0.1875, h_rr: float = 0.02286,
        e_rr: float = 4700.0, g_rr: float = 1715.0,
        tau_ic: float = 30.0, tau_iic: float = 60.0,
        g_ic: float = 0.212, g_iic: float = 0.774,
        eta: float = 2.1, disp_ratio: float = 0.75) -> float:
    if not isinstance(n_cohesive, (tuple, list)) or len(n_cohesive) != 2:
        raise ValueError("n_cohesive must be a sequence of exactly two integers")
    for val in n_cohesive:
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError("n_cohesive must contain integers only")
    if (isinstance(n_reinforcement, bool)
            or not isinstance(n_reinforcement, (int, np.integer))):
        raise ValueError("n_reinforcement must be an integer")
    n_reinforcement = int(n_reinforcement)
    if n_reinforcement < 0:
        raise ValueError("n_reinforcement must be >= 0")

    upper_base, lower_base = (int(n_cohesive[0]), int(n_cohesive[1]))
    best = -np.inf
    for upper_extra in range(n_reinforcement + 1):
        lower_extra = n_reinforcement - upper_extra
        n_total = (upper_base + upper_extra, lower_base + lower_extra)

        ratio_sums = {"opening": 0.0, "shear": 0.0}
        for mode in ("opening", "shear"):
            for n_coh, n_tot in zip((upper_base, lower_base), n_total):
                ratio_sums[mode] += arm_ratio_sum(
                    mode, n_coh, n_tot, t_ply)

        k_n = resin_rich_penalty_stiffness(
            ratio_sums["opening"], h_rr, e_rr)
        k_s = resin_rich_penalty_stiffness(
            ratio_sums["shear"], h_rr, g_rr)
        delta_onset = mixed_mode_onset_separation(
            k_n, k_s, tau_ic, tau_iic, disp_ratio)
        b_ratio = mixed_mode_energy_ratio(k_n, k_s, disp_ratio)
        g_c = benzeggagh_kenane_toughness(
            g_ic, g_iic, eta, b_ratio)
        value = bilinear_failure_separation(
            k_n, k_s, disp_ratio, delta_onset, g_c)
        best = max(best, value)

    return float(best)
SCICODE_GOLD_EOF
