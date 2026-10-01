"""
Speed of the linearly dispersing plasmon branches at the largest bright-dark coincidence of the crystal.

The crystal is a two-dimensional electron layer whose every period L is split into strip 1 of width fL and strip 2 of width (1 - f)L. Strip n carries the equilibrium electron density N_n and sits under its own ideal metal plane at distance h_n; the semiconductor below the layer has permittivity eps_s and the spacer between the layer and the plane above strip n has permittivity kappa_n, so the two strips can differ in density, in gate distance and in spacer permittivity. Damping is neglected throughout.

At the filling factor f* of the previous step the fundamental bright and dark zone-centre modes share the frequency f_0. There the quadratic expansion of step 04 breaks down: the two branches meeting at k = 0 are not parabolic but form a cone, omega(k) = 2 pi f_0 plus or minus s_0 |k| for small k, with the same speed s_0 on both sides. The effective masses vanish at this point for the same reason.

The step returns s_0 in cm/s for the crystal with f = f*, obtained from the Kronig-Penney dispersion of step 02 in the limit k -> 0.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s.

Returns
-------
float: the speed s_0 of the linear branches at f*, in cm/s
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def degeneracy_linear_velocity(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    '''Speed of the linear plasmon branches at the largest bright-dark coincidence.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    speed : float
        s_0 in cm/s.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number, or if no coincidence exists.
    '''
    return speed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_degeneracy_linear_velocity(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    import numpy as np
    fstar, f0 = _oracle_degeneracy_point(period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    args = (period_um, fstar, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2,
            mass_ratio)
    w0 = 2.0 * np.pi * f0 * 1e9
    d = w0 * 1e-7
    bw = (_sm67_family_value(w0 + d, 0, args) - _sm67_family_value(w0 - d, 0, args)) / (2.0 * d)
    dw = (_sm67_family_value(w0 + d, 1, args) - _sm67_family_value(w0 - d, 1, args)) / (2.0 * d)
    return float(period_um * 1e-4 / (2.0 * np.sqrt(bw * dw)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark device at its impedance-matched coincidence ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_linear_velocity(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "gold_call": "_oracle_degeneracy_linear_velocity(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "tol": 1e-05,
        },
        # --- Boundary: equal gate distances and spacers, the gated-gated crystal ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_linear_velocity(8.0, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_degeneracy_linear_velocity(8.0, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "tol": 1e-05,
        },
        # --- Edge: two coincidences only 0.0012 apart in filling factor, which a coarse scan does not resolve ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_linear_velocity(6.0, 8.0e11, 3.0e12, 300.0, 300.0, 12.8, 12.8, 3.97, 0.067)",
            "gold_call": "_oracle_degeneracy_linear_velocity(6.0, 8.0e11, 3.0e12, 300.0, 300.0, 12.8, 12.8, 3.97, 0.067)",
            "tol": 1e-05,
        },
        # --- Edge: GaN-like heavy electrons at high density ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_linear_velocity(4.0, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22)",
            "gold_call": "_oracle_degeneracy_linear_velocity(4.0, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22)",
            "tol": 1e-05,
        },
        # --- Edge: close gates with a low-permittivity spacer on strip 2 ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_linear_velocity(6.0, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "gold_call": "_oracle_degeneracy_linear_velocity(6.0, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "tol": 1e-05,
        },
        # --- Invalid: negative effective mass ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        degeneracy_linear_velocity(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, -0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_degeneracy_linear_velocity(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, -0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
