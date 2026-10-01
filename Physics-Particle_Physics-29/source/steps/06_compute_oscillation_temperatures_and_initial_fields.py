"""
Determine the temperatures at which each mass eigenstate begins to oscillate, and the initial field values each eigenstate carries once oscillations begin.

A field begins to oscillate once the Hubble expansion rate has dropped enough that it can no longer overdamp the field's motion. The onset temperature of each mass eigenstate is fixed by the refined matching condition m_i(T_osc,i) = x_i * H(T_osc,i), where the O(1) coefficient x_i comes from a fit to numerical single-field solutions. The Hubble rate is H(T) = sqrt(pi^2*g_star/90) * T^2 / M_Pl, where M_Pl = 2.435e18 GeV is the reduced Planck mass (use exactly this value; the tests are evaluated with it).

In the scenario this pipeline is built around, both fields start oscillating well before the level crossing, so each state's mass at its own onset is its unmixed, flavor-basis value. The heavy state is aS-like, with the temperature-independent mass mS. The light state is a-like, with the QCD-like power-law mass from the earlier step, of exponent n; evaluate that state's onset on the high-temperature branch.

The initial field values a_H and a_L of the two mass eigenstates are the components of the initial flavor-field vector (aS, a) = (theta_s*fS, theta_a*fa), with fS = Rf*fa, along the heavy and light eigenvectors of the squared-mass matrix of the mixing potential in its high-temperature limit, m_a(T) -> 0, which holds at both oscillation times here. Use the eigenvector convention of the diagonalization step: heavy (cos xi, sin xi) and light (-sin xi, cos xi) in the (aS, a) basis, with the overall sign chosen so that the heavy eigenvector's aS component and the light eigenvector's a component are positive. The resulting expressions are not given here.

Returns
-------
tuple of four floats: (T_osc_H, T_osc_L, a_H, a_L), with the two temperatures in GeV and the two field values in GeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_oscillation_temperatures_and_initial_fields(
    m_a0: float, mS: float, Rf: float, fa: float, T_QCD: float, n: float,
    g_star: float, theta_a: float, theta_s: float,
) -> tuple:
    """Oscillation temperatures and initial mass-eigenstate field values.

    Parameters
    ----------
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.
    fa : float
        Axion decay constant (GeV), a finite number > 0.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.
    g_star : float
        Effective relativistic degrees of freedom, a finite number > 0.
    theta_a : float
        Initial misalignment angle of the 'a' flavor field, finite.
    theta_s : float
        Initial misalignment angle of the 'aS' flavor field, finite.

    Returns
    -------
    result : tuple
        (T_osc_H, T_osc_L, a_H, a_L): the two oscillation temperatures
        (GeV, > 0) and the two initial mass-eigenstate field values (GeV).

    Raises
    ------
    ValueError
        If m_a0, mS, Rf, fa, T_QCD, n, or g_star is not a finite number
        > 0; or if theta_a or theta_s is not finite.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_oscillation_temperatures_and_initial_fields(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark scenario (the four outputs compared separately with the reference) ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
fa = 1000000000000.0
T_QCD = 0.1
n = 3.34
g_star = 61.75
theta_a = 0.02
theta_s = 0.8
def per_output(out):
    # each of the four outputs compared on its own, as a sign-preserving log, so an error
    # in the two temperatures cannot hide behind the much larger field values
    out = tuple(float(o) for o in out)
    if len(out) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
""",
            "call": "per_output(compute_oscillation_temperatures_and_initial_fields(m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s))",
            "gold_call": "per_output(_oracle_compute_oscillation_temperatures_and_initial_fields(m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s))",
        },
        # --- Valid: different misalignment angles and Rf (four outputs compared separately) ---
        {
            "setup": """import numpy as np
m_a0 = 1e-13
mS = 4e-14
Rf = 0.1
fa = 500000000000.0
T_QCD = 0.12
n = 3.5
g_star = 70.0
theta_a = 2.0
theta_s = -0.5
def per_output(out):
    # each of the four outputs compared on its own, as a sign-preserving log, so an error
    # in the two temperatures cannot hide behind the much larger field values
    out = tuple(float(o) for o in out)
    if len(out) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
""",
            "call": "per_output(compute_oscillation_temperatures_and_initial_fields(m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s))",
            "gold_call": "per_output(_oracle_compute_oscillation_temperatures_and_initial_fields(m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s))",
        },
        # --- Boundary: theta_s = 0, only the 'a' field misaligned (four outputs compared separately) ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
fa = 1000000000000.0
T_QCD = 0.1
n = 3.34
g_star = 61.75
theta_a = 1.5
theta_s = 0.0
def per_output(out):
    # each of the four outputs compared on its own, as a sign-preserving log, so an error
    # in the two temperatures cannot hide behind the much larger field values
    out = tuple(float(o) for o in out)
    if len(out) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(np.sign(o) * np.log(abs(o) + 1e-300)) for o in out)
""",
            "call": "per_output(compute_oscillation_temperatures_and_initial_fields(m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s))",
            "gold_call": "per_output(_oracle_compute_oscillation_temperatures_and_initial_fields(m_a0, mS, Rf, fa, T_QCD, n, g_star, theta_a, theta_s))",
        },
        # --- Consistency: T_osc_H solves its own matching condition to high precision ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
fa = 1.0e12
T_QCD = 0.100
n = 3.34
g_star = 61.75
M_PL = 2.435e18
def check(fn):
    T_osc_H, T_osc_L, aH, aL = fn(m_a0, mS, Rf, fa, T_QCD, n, g_star, 1.0, 0.8)
    A = np.sqrt(np.pi**2*g_star/90.0)/M_PL
    H_H = A*T_osc_H**2
    return int(abs(mS - 1.6*H_H) < 1e-6*mS)
""",
            "call": "check(compute_oscillation_temperatures_and_initial_fields)",
            "gold_call": "check(_oracle_compute_oscillation_temperatures_and_initial_fields)",
        },
        # --- Consistency: T_osc_L solves its own matching condition to high precision ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
fa = 1.0e12
T_QCD = 0.100
n = 3.34
g_star = 61.75
M_PL = 2.435e18
def check(fn):
    T_osc_H, T_osc_L, aH, aL = fn(m_a0, mS, Rf, fa, T_QCD, n, g_star, 1.0, 0.8)
    A = np.sqrt(np.pi**2*g_star/90.0)/M_PL
    x_L = 1.6+0.6*n
    lhs = m_a0*(T_QCD/T_osc_L)**n
    rhs = x_L*A*T_osc_L**2
    return int(abs(lhs-rhs) < 1e-6*rhs)
""",
            "call": "check(compute_oscillation_temperatures_and_initial_fields)",
            "gold_call": "check(_oracle_compute_oscillation_temperatures_and_initial_fields)",
        },
        # --- Invalid: non-positive fa ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_oscillation_temperatures_and_initial_fields(5e-15, 5e-16, 0.02, 0.0, 0.1, 3.34, 61.75, 1.0, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_oscillation_temperatures_and_initial_fields(5e-15, 5e-16, 0.02, 0.0, 0.1, 3.34, 61.75, 1.0, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite misalignment angle ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_oscillation_temperatures_and_initial_fields(5e-15, 5e-16, 0.02, 1e12, 0.1, 3.34, 61.75, float('nan'), 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_oscillation_temperatures_and_initial_fields(5e-15, 5e-16, 0.02, 1e12, 0.1, 3.34, 61.75, float('nan'), 0.8)
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
