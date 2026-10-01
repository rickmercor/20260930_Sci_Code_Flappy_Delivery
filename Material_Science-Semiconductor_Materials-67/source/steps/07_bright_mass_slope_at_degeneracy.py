"""
Rate at which the effective mass of the fundamental bright plasmon changes with filling factor at the largest bright-dark coincidence.

The crystal is a two-dimensional electron layer whose every period L is split into strip 1 of width fL and strip 2 of width (1 - f)L. Strip n carries the equilibrium electron density N_n and sits under its own ideal metal plane at distance h_n; the semiconductor below the layer has permittivity eps_s and the spacer between the layer and the plane above strip n has permittivity kappa_n, so the two strips can differ in density, in gate distance and in spacer permittivity. Damping is neglected throughout.

The effective mass M_b of the fundamental bright plasmon (step 04) is a smooth function of the filling factor f with the other device parameters fixed. It passes through zero at the coincidence filling factor f* of step 05, positive on one side and negative on the other, because there the bright and dark zone-centre frequencies cross and the band stops being parabolic. Its slope at the crossing measures how quickly the gratings turn a massless, linearly dispersing plasmon into a heavy one.

The step returns dM_b/df evaluated at f = f*, in units of 10^-4 m* per unit filling factor, where m* = mass_ratio m_0. The derivative is taken along the fundamental bright family, which is continuous through f*. Near a coincidence the partner mode can be only a few MHz away, so the mass on either side must be resolved accurately.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s.

Returns
-------
float: dM_b/df at f*, in units of 1e-4 m*
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bright_mass_slope_at_degeneracy(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    '''Derivative of the fundamental bright plasmon mass with filling factor at the largest coincidence.

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
    slope : float
        dM_b/df at f = f*, in units of 1e-4 m*.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number, if no coincidence exists, or
        if the bright mass does not cross zero linearly at f*.
    '''
    return slope

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_bright_mass_slope_at_degeneracy(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    import numpy as np
    _, m0, hbar = _sm67_constants()
    fstar, f0 = _oracle_degeneracy_point(period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    s0 = _oracle_degeneracy_linear_velocity(period_um, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    step = 1e-4
    rest = (density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    above = _oracle_zone_center_modes(period_um, fstar + step, *rest, 1)
    below = _oracle_zone_center_modes(period_um, fstar - step, *rest, 1)
    split_rate = 2.0 * np.pi * 1e9 * ((above[0, 0] - above[1, 0]) - (below[0, 0] - below[1, 0])) / (2.0 * step)
    slope_linear = 1e4 * hbar * split_rate / (2.0 * s0 ** 2) / (mass_ratio * m0)
    mass_up = _oracle_plasmon_effective_mass(period_um, fstar + step, *rest, "bright")
    mass_dn = _oracle_plasmon_effective_mass(period_um, fstar - step, *rest, "bright")
    slope_mass = (mass_up - mass_dn) / (2.0 * step)
    if not np.isclose(slope_mass, slope_linear, rtol=1e-3, atol=1e-6):
        raise ValueError("the bright mass does not cross zero linearly at f* for these parameters")
    return float(slope_mass)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark device at its impedance-matched coincidence ---
        {
            "setup": """import numpy as np
""",
            "call": "bright_mass_slope_at_degeneracy(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "gold_call": "_oracle_bright_mass_slope_at_degeneracy(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "tol": 0.001,
        },
        # --- Boundary: equal gate distances and spacers, the gated-gated crystal ---
        {
            "setup": """import numpy as np
""",
            "call": "bright_mass_slope_at_degeneracy(8.0, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_bright_mass_slope_at_degeneracy(8.0, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "tol": 0.001,
        },
        # --- Edge: two coincidences only 0.0012 apart in filling factor, which a coarse scan does not resolve ---
        {
            "setup": """import numpy as np
""",
            "call": "bright_mass_slope_at_degeneracy(6.0, 8.0e11, 3.0e12, 300.0, 300.0, 12.8, 12.8, 3.97, 0.067)",
            "gold_call": "_oracle_bright_mass_slope_at_degeneracy(6.0, 8.0e11, 3.0e12, 300.0, 300.0, 12.8, 12.8, 3.97, 0.067)",
            "tol": 0.001,
        },
        # --- Edge: GaN-like heavy electrons at high density ---
        {
            "setup": """import numpy as np
""",
            "call": "bright_mass_slope_at_degeneracy(4.0, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22)",
            "gold_call": "_oracle_bright_mass_slope_at_degeneracy(4.0, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22)",
            "tol": 0.001,
        },
        # --- Edge: close gates with a low-permittivity spacer on strip 2 ---
        {
            "setup": """import numpy as np
""",
            "call": "bright_mass_slope_at_degeneracy(6.0, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "gold_call": "_oracle_bright_mass_slope_at_degeneracy(6.0, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "tol": 0.001,
        },
        # --- Invalid: undefined gate distance ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bright_mass_slope_at_degeneracy(6.0, 3.0e11, 3.0e12, 300.0, float('nan'), 12.8, 12.8, 7.0, 0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_bright_mass_slope_at_degeneracy(6.0, 3.0e11, 3.0e12, 300.0, float('nan'), 12.8, 12.8, 7.0, 0.067)
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
