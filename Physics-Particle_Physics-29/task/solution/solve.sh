#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_axion_mass_scales(fa: float, Rm: float) -> tuple:
    import numpy as np

    if not (isinstance(fa, (int, float, np.floating)) and np.isfinite(fa) and float(fa) > 0.0):
        raise ValueError("fa must be a finite number > 0")
    if not (isinstance(Rm, (int, float, np.floating)) and np.isfinite(Rm) and float(Rm) > 0.0):
        raise ValueError("Rm must be a finite number > 0")

    fa = float(fa)
    Rm = float(Rm)

    m_pi = 0.135
    f_pi = 0.092
    m_u = 2.16e-3
    m_d = 4.67e-3

    m_a0 = np.sqrt((m_pi**2 * f_pi**2 / fa**2) * (m_u * m_d) / (m_u + m_d) ** 2)
    mS = Rm * m_a0

    return (m_a0, mS)

def compute_axion_mass_squared_at_temperature(T: float, m_a0: float, T_QCD: float, n: float) -> float:
    import numpy as np

    for name, value in (("T", T), ("m_a0", m_a0), ("T_QCD", T_QCD), ("n", n)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    T = float(T)
    m_a0 = float(m_a0)
    T_QCD = float(T_QCD)
    n = float(n)

    m_a_sq = m_a0 ** 2 * min(1.0, (T_QCD / T) ** (2.0 * n))
    return m_a_sq

def diagonalize_axion_mass_matrix(m_a_sq: float, mS: float, Rf: float) -> tuple:
    import numpy as np

    for name, value in (("m_a_sq", m_a_sq), ("mS", mS), ("Rf", Rf)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    m_a_sq = float(m_a_sq)
    mS = float(mS)
    Rf = float(Rf)

    maa2 = m_a_sq + mS ** 2 * Rf ** 2
    mss2 = mS ** 2
    mas2 = mS ** 2 * Rf

    M2 = np.array([[mss2, mas2], [mas2, maa2]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(M2)
    m_L_sq, m_H_sq = float(eigvals[0]), float(eigvals[1])

    v_H = eigvecs[:, 1]
    xi = float(np.arctan2(v_H[1], v_H[0]))

    return (m_H_sq, m_L_sq, xi)

def find_crossing_temperature(m_a0: float, mS: float, Rf: float, T_QCD: float, n: float) -> float:
    import numpy as np
    from scipy.optimize import brentq

    for name, value in (("m_a0", m_a0), ("mS", mS), ("T_QCD", T_QCD), ("n", n)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(Rf, (int, float, np.floating)) and np.isfinite(Rf) and 0.0 < float(Rf) < 1.0):
        raise ValueError("Rf must be a finite number with 0 < Rf < 1")

    m_a0 = float(m_a0)
    mS = float(mS)
    Rf = float(Rf)
    T_QCD = float(T_QCD)
    n = float(n)

    required = mS ** 2 * (1.0 - Rf ** 2)
    if required >= m_a0 ** 2:
        raise ValueError("no crossing exists: mS^2*(1-Rf^2) >= m_a0^2")

    def crossing_eq(T):
        m_a_sq = compute_axion_mass_squared_at_temperature(T, m_a0, T_QCD, n)
        return (m_a_sq + mS ** 2 * Rf ** 2) - mS ** 2

    Tlo, Thi = 1e-8, 1e6
    T_x = brentq(crossing_eq, Tlo, Thi, xtol=1e-14, rtol=1e-13)
    return T_x

def compute_landau_zener_probability(T_x: float, m_a0: float, mS: float, Rf: float,
                                              T_QCD: float, n: float, g_star: float) -> tuple:
    import numpy as np

    M_PL = 2.435e18

    for name, value in (("T_x", T_x), ("m_a0", m_a0), ("mS", mS), ("Rf", Rf),
                        ("T_QCD", T_QCD), ("n", n), ("g_star", g_star)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    T_x = float(T_x)
    m_a0 = float(m_a0)
    mS = float(mS)
    Rf = float(Rf)
    T_QCD = float(T_QCD)
    n = float(n)
    g_star = float(g_star)

    H_x = np.sqrt(np.pi ** 2 * g_star / 90.0) * T_x ** 2 / M_PL

    # d(m_a^2)/dT at T_x in the power-law branch (assumes T_x > T_QCD, the
    # regime in which a crossing exists for this pipeline's parameter choices)
    dma2_dT = m_a0 ** 2 * (-2.0 * n) * T_QCD ** (2.0 * n) * T_x ** (-2.0 * n - 1.0)
    dT_dt = -H_x * T_x
    d_maa2_dt = dma2_dT * dT_dt

    mas2_x = mS ** 2 * Rf
    gamma = abs(4.0 * mas2_x ** 2 / d_maa2_dt) / (2.0 * mS)
    P_LZ = float(np.exp(-np.pi * gamma / 2.0))

    return (float(gamma), P_LZ)

def compute_oscillation_temperatures_and_initial_fields(
    m_a0: float, mS: float, Rf: float, fa: float, T_QCD: float, n: float,
    g_star: float, theta_a: float, theta_s: float,
) -> tuple:
    import numpy as np
    from scipy.optimize import brentq

    M_PL = 2.435e18

    for name, value in (("m_a0", m_a0), ("mS", mS), ("Rf", Rf), ("fa", fa),
                        ("T_QCD", T_QCD), ("n", n), ("g_star", g_star)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("theta_a", theta_a), ("theta_s", theta_s)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    m_a0 = float(m_a0)
    mS = float(mS)
    Rf = float(Rf)
    fa = float(fa)
    T_QCD = float(T_QCD)
    n = float(n)
    g_star = float(g_star)
    theta_a = float(theta_a)
    theta_s = float(theta_s)

    A = np.sqrt(np.pi ** 2 * g_star / 90.0) / M_PL  # H(T) = A * T^2

    x_H = 1.6
    x_L = 1.6 + 0.6 * n

    # T_osc_H solves mS = x_H * A * T^2
    T_osc_H = np.sqrt(mS / (x_H * A))

    # T_osc_L solves m_a0*(T_QCD/T)^n = x_L * A * T^2  (QCD-like branch)
    def f(T):
        return m_a0 * (T_QCD / T) ** n - x_L * A * T ** 2

    T_osc_L = brentq(f, 1e-8, 1e6, xtol=1e-14, rtol=1e-13)

    prefactor = Rf * fa / np.sqrt(1.0 + Rf ** 2)
    a_H = prefactor * (theta_a + theta_s)
    a_L = prefactor * ((1.0 / Rf) * theta_a - Rf * theta_s)

    return (float(T_osc_H), float(T_osc_L), float(a_H), float(a_L))

def assemble_relic_abundance_fraction(
    m_a0: float, mS: float, Rf: float, T_osc_H: float, T_osc_L: float,
    a_H: float, a_L: float, P_LZ: float, T0: float, T_QCD: float, n: float,
) -> float:
    import numpy as np

    for name, value in (("m_a0", m_a0), ("mS", mS), ("Rf", Rf), ("T_osc_H", T_osc_H),
                        ("T_osc_L", T_osc_L), ("T0", T0), ("T_QCD", T_QCD), ("n", n)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("a_H", a_H), ("a_L", a_L)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    if not (isinstance(P_LZ, (int, float, np.floating)) and np.isfinite(P_LZ) and 0.0 <= float(P_LZ) <= 1.0):
        raise ValueError("P_LZ must be a finite number in [0, 1]")
    if not float(T0) < float(T_QCD):
        raise ValueError("T0 must be strictly less than T_QCD")

    m_a0 = float(m_a0); mS = float(mS); Rf = float(Rf)
    T_osc_H = float(T_osc_H); T_osc_L = float(T_osc_L)
    a_H = float(a_H); a_L = float(a_L); P_LZ = float(P_LZ)
    T0 = float(T0); T_QCD = float(T_QCD); n = float(n)

    # WKB number densities (R(T)=1/T convention; overall constant cancels in f_L)
    R_H = 1.0 / T_osc_H
    R_L = 1.0 / T_osc_L
    m_L_sq_osc = compute_axion_mass_squared_at_temperature(T_osc_L, m_a0, T_QCD, n)
    m_L_osc = np.sqrt(m_L_sq_osc)  # QCD-like mass at its own oscillation time
    nH_WKB = 0.5 * mS * a_H ** 2 * R_H ** 3
    nL_WKB = 0.5 * m_L_osc * a_L ** 2 * R_L ** 3

    # post-crossing mixing
    nH_post = (1.0 - P_LZ) * nH_WKB + P_LZ * nL_WKB
    nL_post = (1.0 - P_LZ) * nL_WKB + P_LZ * nH_WKB

    # present-day masses: fresh diagonalization at T0
    m_a_sq_T0 = compute_axion_mass_squared_at_temperature(T0, m_a0, T_QCD, n)
    mH2_T0, mL2_T0, _xi_T0 = diagonalize_axion_mass_matrix(m_a_sq_T0, mS, Rf)
    mL_T0 = np.sqrt(mL2_T0)
    mH_T0 = np.sqrt(mH2_T0)

    rho_H = mH_T0 * nH_post
    rho_L = mL_T0 * nL_post

    f_L = rho_L / (rho_H + rho_L)
    return float(f_L)

def compute_relic_abundance_fraction(
    fa: float = 1.0e12,
    Rf: float = 0.02,
    Rm: float = 0.10,
    theta_a: float = 0.02,
    theta_s: float = 0.8,
    T_QCD: float = 0.100,
    n: float = 3.34,
    g_star: float = 61.75,
    T0: float = 2.348e-13,
) -> float:
    import numpy as np

    # Step 1
    m_a0, mS = compute_axion_mass_scales(fa, Rm)

    # Step 4 (chains Steps 2-3 internally via find_crossing_temperature's own oracle)
    T_x = find_crossing_temperature(m_a0, mS, Rf, T_QCD, n)

    # Step 5
    gamma, P_LZ = compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star)

    # Step 6
    T_osc_H, T_osc_L, a_H, a_L = compute_oscillation_temperatures_and_initial_fields(
        m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s
    )

    # Step 7
    f_L = assemble_relic_abundance_fraction(
        m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n
    )

    return float(f_L)
SCICODE_GOLD_EOF
