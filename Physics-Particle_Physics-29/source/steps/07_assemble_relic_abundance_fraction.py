"""
Assemble the present-day dark matter abundance carried by each mass eigenstate, and report the fraction carried by the light eigenstate.

Before the level crossing, each mass eigenstate's comoving number density is conserved (the adiabatic, or WKB, invariant), given by n_i^WKB = (1/2) * m_i(T_i^osc) * a_i(T_i^osc)^2 * R(T_i^osc)^3, where R(T) is the cosmic scale factor. Here m_i(T_i^osc) is evaluated in the same unmixed, flavor-basis approximation used for the oscillation temperatures, not by re-diagonalizing the mass matrix at the oscillation temperatures: for the heavy eigenstate m_H(T_osc,H) = mS, and for the light eigenstate m_L(T_osc,L) = m_a(T_osc,L), the QCD-coupled field's temperature-dependent mass at T_osc,L (the tests use these flavor-basis masses). For R(T), using the standard entropy-conservation relation with the effective degrees of freedom held fixed across the range of interest, R(T) can be taken proportional to 1/T, and an overall proportionality constant may be fixed arbitrarily (it cancels in the final abundance fraction).

The level crossing, when traversed non-adiabatically, mixes the comoving number densities of the two eigenstates according to the Landau-Zener probability P_LZ:

n_H(T < T_x) = (1 - P_LZ) * n_H^WKB + P_LZ * n_L^WKB,



n_L(T < T_x) = (1 - P_LZ) * n_L^WKB + P_LZ * n_H^WKB.

After the crossing, each of these comoving number densities is again separately conserved all the way to the present day. The present-day energy density carried by each eigenstate is then the product of that eigenstate's present-day mass and its post-crossing comoving number density: rho_i(T0) = m_i(T0) * n_i(T < T_x) (up to the same overall scale-factor convention used above, which cancels in the ratio requested below).

The present-day masses m_H(T0) and m_L(T0) are obtained by evaluating the same mass-matrix construction and diagonalization used in earlier steps, at the present-day temperature T0 (rather than at the crossing temperature or at either oscillation temperature). As the QCD-coupled flavor field's mass grows between the crossing epoch and today, the flavor composition of each mass eigenstate changes. The heavy and light eigenstates remain the branches with the larger and smaller instantaneous eigenvalues, respectively. Evaluate their masses at T0 and pair each with the post-crossing comoving number density of the same branch.

Returns
-------
float, the fraction of the total present-day dark matter energy density carried by the light mass eigenstate, f_L = rho_L(T0) / [rho_H(T0) + rho_L(T0)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_relic_abundance_fraction(
    m_a0: float, mS: float, Rf: float, T_osc_H: float, T_osc_L: float,
    a_H: float, a_L: float, P_LZ: float, T0: float, T_QCD: float, n: float,
) -> float:
    """Present-day relic abundance fraction carried by the light eigenstate.

    Parameters
    ----------
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.
    T_osc_H : float
        Oscillation temperature of the heavy eigenstate (GeV), > 0.
    T_osc_L : float
        Oscillation temperature of the light eigenstate (GeV), > 0.
    a_H : float
        Initial field value of the heavy eigenstate (GeV), finite.
    a_L : float
        Initial field value of the light eigenstate (GeV), finite.
    P_LZ : float
        Landau-Zener conversion probability, 0 <= P_LZ <= 1 (P_LZ = 0 is the
        fully adiabatic limit, with no conversion).
    T0 : float
        Present-day temperature (GeV), a finite number > 0, with T0 < T_QCD.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.

    Returns
    -------
    f_L : float
        The present-day abundance fraction carried by the light eigenstate,
        a finite number in (0, 1).

    Raises
    ------
    ValueError
        If m_a0, mS, Rf, T_osc_H, T_osc_L, T0, or T_QCD is not a finite
        number > 0; if a_H or a_L is not finite; if P_LZ is not in [0, 1];
        or if T0 is not strictly less than T_QCD.
    """
    return f_L  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_relic_abundance_fraction(
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
    m_L_sq_osc = _oracle_compute_axion_mass_squared_at_temperature(T_osc_L, m_a0, T_QCD, n)
    m_L_osc = np.sqrt(m_L_sq_osc)  # QCD-like mass at its own oscillation time
    nH_WKB = 0.5 * mS * a_H ** 2 * R_H ** 3
    nL_WKB = 0.5 * m_L_osc * a_L ** 2 * R_L ** 3

    # post-crossing mixing
    nH_post = (1.0 - P_LZ) * nH_WKB + P_LZ * nL_WKB
    nL_post = (1.0 - P_LZ) * nL_WKB + P_LZ * nH_WKB

    # present-day masses: fresh diagonalization at T0
    m_a_sq_T0 = _oracle_compute_axion_mass_squared_at_temperature(T0, m_a0, T_QCD, n)
    mH2_T0, mL2_T0, _xi_T0 = _oracle_diagonalize_axion_mass_matrix(m_a_sq_T0, mS, Rf)
    mL_T0 = np.sqrt(mL2_T0)
    mH_T0 = np.sqrt(mH2_T0)

    rho_H = mH_T0 * nH_post
    rho_L = mL_T0 * nL_post

    f_L = rho_L / (rho_H + rho_L)
    return float(f_L)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark scenario ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_osc_H = 18.378456977337194
T_osc_L = 0.9317076100503877
a_H = 16396720983.67212
a_L = 19676065180.40654
P_LZ = 0.07719048574146661
T0 = 2.348e-13
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
            "gold_call": "digest(_oracle_assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
        },
        # --- Valid: a different, smaller P_LZ (closer to adiabatic) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
m_a0 = 1.0e-13
mS = 4.0e-14
Rf = 0.1
T_osc_H = 5.0
T_osc_L = 0.5
a_H = 1.0e11
a_L = 2.0e12
P_LZ = 0.05
T0 = 2.348e-13
T_QCD = 0.120
n = 3.5
""",
            "call": "digest(assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
            "gold_call": "digest(_oracle_assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
        },
        # --- Boundary: P_LZ = 1 (complete conversion) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_osc_H = 18.378456977337194
T_osc_L = 0.9317076100503877
a_H = 16396720983.67212
a_L = 19676065180.40654
P_LZ = 1.0
T0 = 2.348e-13
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
            "gold_call": "digest(_oracle_assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
        },
        # --- Consistency: f_L must lie strictly in (0, 1) ---
        {
            "setup": """import numpy as np
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_osc_H = 18.378456977337194
T_osc_L = 0.9317076100503877
a_H = 16396720983.67212
a_L = 19676065180.40654
P_LZ = 0.07719048574146661
T0 = 2.348e-13
T_QCD = 0.100
n = 3.34
def check(fn):
    fL = fn(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n)
    return int(0.0 < fL < 1.0)
""",
            "call": "check(assemble_relic_abundance_fraction)",
            "gold_call": "check(_oracle_assemble_relic_abundance_fraction)",
        },
        # --- Edge: a near-adiabatic crossing, P_LZ = 1e-6: the light state keeps almost all of its own
        # pre-crossing density (f_L of about 0.999628, below the P_LZ = 0 value 0.999630, so the
        # small conversion must still be applied) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
m_a0 = 1.0e-13
mS = 4.0e-14
Rf = 0.1
T_osc_H = 5.0
T_osc_L = 0.5
a_H = 1.0e11
a_L = 2.0e12
P_LZ = 1e-6
T0 = 2.348e-13
T_QCD = 0.120
n = 3.5
""",
            "call": "digest(assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
            "gold_call": "digest(_oracle_assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
        },
        # --- Valid: half conversion, P_LZ = 0.5 (f_L of about 0.285) ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
m_a0 = 1.0e-13
mS = 4.0e-14
Rf = 0.1
T_osc_H = 5.0
T_osc_L = 0.5
a_H = 1.0e11
a_L = 2.0e12
P_LZ = 0.5
T0 = 2.348e-13
T_QCD = 0.120
n = 3.5
""",
            "call": "digest(assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
            "gold_call": "digest(_oracle_assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
        },
        # --- Boundary: P_LZ = 0 (fully adiabatic crossing, no conversion) is a valid input ---
        {
            "setup": """import numpy as np
def digest(x):
    return float(x)
m_a0 = 5.775455048675712e-15
mS = 5.775455048675713e-16
Rf = 0.02
T_osc_H = 18.378456977337194
T_osc_L = 0.9317076100503877
a_H = 16396720983.67212
a_L = 19676065180.40654
P_LZ = 0.0
T0 = 2.348e-13
T_QCD = 0.100
n = 3.34
""",
            "call": "digest(assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
            "gold_call": "digest(_oracle_assemble_relic_abundance_fraction(m_a0, mS, Rf, T_osc_H, T_osc_L, a_H, a_L, P_LZ, T0, T_QCD, n))",
        },
        # --- Invalid: T0 not less than T_QCD ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_relic_abundance_fraction(5e-15, 5e-16, 0.02, 8.0, 0.7, 1e11, 1e12, 0.6, 0.2, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_relic_abundance_fraction(5e-15, 5e-16, 0.02, 8.0, 0.7, 1e11, 1e12, 0.6, 0.2, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: P_LZ outside [0, 1] ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_relic_abundance_fraction(5e-15, 5e-16, 0.02, 8.0, 0.7, 1e11, 1e12, 1.5, 2.3e-13, 0.1, 3.34)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_relic_abundance_fraction(5e-15, 5e-16, 0.02, 8.0, 0.7, 1e11, 1e12, 1.5, 2.3e-13, 0.1, 3.34)
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
