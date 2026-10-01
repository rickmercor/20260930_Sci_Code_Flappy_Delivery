"""
Compute the adiabaticity parameter governing the level crossing, and the resulting Landau-Zener probability of conversion between the two mass eigenstates.

The rate at which the mixing angle changes relative to the splitting between the two mass eigenstates determines whether the crossing is traversed adiabatically or not. This is quantified by an adiabaticity parameter gamma evaluated at the crossing time, built from the entries of the squared-mass matrix, their time derivatives, and the physical eigen-masses m_H and m_L of the two fields, which oscillate at frequencies close to their masses. gamma is normalized so that the Landau-Zener probability of conversion between the two mass eigenstates is P_LZ = exp(-pi*gamma/2). The expression for gamma itself is not given here; it follows from the two-state Landau-Zener problem for the coupled fields. For a strongly adiabatic crossing P_LZ can underflow to exactly 0 in floating point, which is a valid result.

Which matrix entries vary with time follows from the matrix itself. Convert any temperature derivative into a time derivative with dT/dt = -H(T)*T (radiation-dominated, entropy-conserving universe with constant g_star). The Hubble rate is H(T) = sqrt(pi^2*g_star/90) * T^2 / M_Pl, where M_Pl = 2.435e18 GeV is the reduced Planck mass (use exactly this value; the tests are evaluated with it). Wherever the sum of the eigen-masses m_H + m_L appears, set it to exactly 2*mS. At the crossing the individual eigen-masses are mS*sqrt(1 + Rf) and mS*sqrt(1 - Rf), so each differs from mS at first order in Rf. Their sum differs from 2*mS only at second order. The tests use exactly 2*mS; using the exact eigenvalue sum does not reproduce them.

Returns
-------
tuple of two floats: (gamma, P_LZ), both dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_landau_zener_probability(T_x: float, m_a0: float, mS: float, Rf: float,
                                      T_QCD: float, n: float, g_star: float) -> tuple:
    """Adiabaticity parameter and Landau-Zener conversion probability at the crossing.

    Parameters
    ----------
    T_x : float
        Crossing temperature (GeV), a finite number > 0.
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.
    g_star : float
        Effective relativistic degrees of freedom at T_x, a finite number > 0.

    Returns
    -------
    result : tuple
        (gamma, P_LZ): the adiabaticity parameter and the Landau-Zener
        conversion probability, both finite, gamma >= 0 and 0 <= P_LZ <= 1
        (P_LZ may underflow to exactly 0 when gamma is large).

    Raises
    ------
    ValueError
        If any of T_x, m_a0, mS, Rf, T_QCD, n, or g_star is not a finite
        number > 0.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_landau_zener_probability(T_x: float, m_a0: float, mS: float, Rf: float,
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark scenario ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
T_x = 0.19926324956246239
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_QCD = 0.100
n = 3.34
g_star = 61.75
""",
            "call": "digest(compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
            "gold_call": "digest(_oracle_compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
        },
        # --- Valid: a different parameter set at its own crossing temperature; Rf = 0.003
        # gives gamma of about 3.7, a mostly adiabatic crossing (P_LZ of about 0.003) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
m_a0 = 1.0e-13
mS = 4.0e-14
Rf = 0.003
T_QCD = 0.120
n = 3.5
g_star = 70.0
T_x = T_QCD * (m_a0**2 / (mS**2 * (1.0 - Rf**2)))**(1.0 / (2.0 * n))
""",
            "call": "digest(compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
            "gold_call": "digest(_oracle_compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
        },
        # --- Boundary: a very small Rf at its own crossing temperature; gamma of about
        # 0.0035, a strongly non-adiabatic crossing (P_LZ close to 1) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
m_a0 = 5.0e-15
mS = 5.0e-16
Rf = 0.001
T_QCD = 0.100
n = 3.34
g_star = 61.75
T_x = T_QCD * (m_a0**2 / (mS**2 * (1.0 - Rf**2)))**(1.0 / (2.0 * n))
""",
            "call": "digest(compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
            "gold_call": "digest(_oracle_compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
        },
        # --- Consistency: P_LZ = exp(-pi*gamma/2) exactly, given the returned gamma ---
        {
            "setup": """import numpy as np
T_x = 0.19926324956246239
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_QCD = 0.100
n = 3.34
g_star = 61.75
def check(fn):
    gamma, P_LZ = fn(T_x, m_a0, mS, Rf, T_QCD, n, g_star)
    return int(abs(P_LZ - np.exp(-np.pi*gamma/2.0)) < 1e-10)
""",
            "call": "check(compute_landau_zener_probability)",
            "gold_call": "check(_oracle_compute_landau_zener_probability)",
        },
        # --- Valid: Rf = 0.01 at its own crossing temperature: gamma of about 0.35 and P_LZ of
        # about 0.57, a largely non-adiabatic crossing (outputs compared separately) ---
        {
            "setup": """import numpy as np
def per_output(out):
    # gamma and P_LZ compared on their own, as sign-preserving logs, so an error in a
    # small P_LZ cannot hide behind gamma
    out = tuple(float(o) for o in out)
    if len(out) != 2:
        return (0.0, 0.0)
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
m_a0 = 5.0e-15
mS = 5.0e-16
Rf = 0.01
T_QCD = 0.100
n = 3.34
g_star = 61.75
T_x = T_QCD * (m_a0**2 / (mS**2 * (1.0 - Rf**2)))**(1.0 / (2.0 * n))
""",
            "call": "per_output(compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
            "gold_call": "per_output(_oracle_compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
        },
        # --- Edge: Rf = 0.05 at its own crossing temperature: gamma of about 8.8 and P_LZ of about
        # 9e-7, a strongly adiabatic crossing. With the previous case: larger Rf gives larger
        # gamma and smaller P_LZ, so less population is converted (outputs compared separately) ---
        {
            "setup": """import numpy as np
def per_output(out):
    # gamma and P_LZ compared on their own, as sign-preserving logs, so an error in a
    # small P_LZ cannot hide behind gamma
    out = tuple(float(o) for o in out)
    if len(out) != 2:
        return (0.0, 0.0)
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
m_a0 = 5.0e-15
mS = 5.0e-16
Rf = 0.05
T_QCD = 0.100
n = 3.34
g_star = 61.75
T_x = T_QCD * (m_a0**2 / (mS**2 * (1.0 - Rf**2)))**(1.0 / (2.0 * n))
""",
            "call": "per_output(compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
            "gold_call": "per_output(_oracle_compute_landau_zener_probability(T_x, m_a0, mS, Rf, T_QCD, n, g_star))",
        },
        # --- Invalid: non-positive g_star ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_landau_zener_probability(0.2, 5e-15, 5e-16, 0.02, 0.1, 3.34, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_landau_zener_probability(0.2, 5e-15, 5e-16, 0.02, 0.1, 3.34, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative crossing temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_landau_zener_probability(-0.2, 5e-15, 5e-16, 0.02, 0.1, 3.34, 61.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_landau_zener_probability(-0.2, 5e-15, 5e-16, 0.02, 0.1, 3.34, 61.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
