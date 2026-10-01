"""
Effective mass of the fundamental bright or dark plasmon of the two-strip plasmonic crystal.

The crystal is a two-dimensional electron layer whose every period L is split into strip 1 of width fL and strip 2 of width (1 - f)L. Strip n carries the equilibrium electron density N_n and sits under its own ideal metal plane at distance h_n; the semiconductor below the layer has permittivity eps_s and the spacer between the layer and the plane above strip n has permittivity kappa_n, so the two strips can differ in density, in gate distance and in spacer permittivity. Damping is neglected throughout.

Near the zone centre the band of a fundamental mode, the lowest bright or the lowest dark zone-centre mode of the previous step, is quadratic in the quasimomentum k. Writing hbar omega(k) = hbar omega(0) + hbar^2 k^2 / (2 M) + O(k^4) defines the effective plasmon mass M of that mode, by analogy with the effective mass of a band electron. M is positive when the band curves upwards away from k = 0 and negative when it curves downwards.

The band omega(k) is the branch of the Kronig-Penney dispersion of step 02 that passes through the chosen zone-centre frequency. Close to a filling factor where the fundamental bright and dark modes coincide, the partner mode can lie only a few MHz away, and the mass must still be resolved. The mass is returned in units of 10^-4 m*, where m* = mass_ratio m_0 is the electron effective mass, so a returned value of 0.5 means M = 0.5e-4 m*.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s.

Returns
-------
float: the effective plasmon mass M of the fundamental mode of the requested family, in units of 1e-4 m*
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def plasmon_effective_mass(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, kind: str) -> float:
    '''Zone-centre effective mass of the fundamental bright or dark plasmon.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    fill : float
        Filling factor f, the fraction of each period occupied by strip 1; 0 < f < 1.
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
    kind : str
        Either "bright" or "dark".

    Returns
    -------
    mass : float
        Signed effective plasmon mass M in units of 1e-4 m*.

    Raises
    ------
    ValueError
        If kind is not "bright" or "dark", if any physical argument is not a positive
        finite number, or if fill is not strictly between 0 and 1.
    '''
    return mass

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_plasmon_effective_mass(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, kind: str) -> float:
    import numpy as np
    if kind not in ("bright", "dark"):
        raise ValueError("kind must be 'bright' or 'dark'")
    modes = _oracle_zone_center_modes(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio, 1)
    _, m0, hbar = _sm67_constants()
    k = 0 if kind == "bright" else 1
    f_mode = modes[k, 0]
    omega = 2.0 * np.pi * f_mode * 1e9
    if abs(_oracle_bloch_phase_cosine(f_mode, period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio) - 1.0) > 1e-8:
        raise ValueError("the zone-centre mode does not satisfy the Bloch condition")
    args = (period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    d = omega * 1e-7
    slope = (_sm67_family_value(omega + d, k, args) - _sm67_family_value(omega - d, k, args)) / (2.0 * d)
    partner = _sm67_family_value(omega, 1 - k, args)
    L = period_um * 1e-4
    return float(1e4 * 2.0 * hbar * slope * partner / L ** 2 / (mass_ratio * m0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: bright mass of the benchmark crystal between its degeneracies ---
        {
            "setup": """import numpy as np
""",
            "call": "plasmon_effective_mass(6.0, 0.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'bright')",
            "gold_call": "_oracle_plasmon_effective_mass(6.0, 0.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'bright')",
            "tol": 1e-05,
        },
        # --- Normal: dark mass of the benchmark crystal above the impedance-matched degeneracy ---
        {
            "setup": """import numpy as np
""",
            "call": "plasmon_effective_mass(6.0, 0.6, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'dark')",
            "gold_call": "_oracle_plasmon_effective_mass(6.0, 0.6, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'dark')",
            "tol": 1e-05,
        },
        # --- Boundary: gated-gated crystal at equal strip widths ---
        {
            "setup": """import numpy as np
""",
            "call": "plasmon_effective_mass(8.0, 0.5, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067, 'bright')",
            "gold_call": "_oracle_plasmon_effective_mass(8.0, 0.5, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067, 'bright')",
            "tol": 1e-05,
        },
        # --- Edge: nearly ungated strip 2 with strong impedance mismatch ---
        {
            "setup": """import numpy as np
""",
            "call": "plasmon_effective_mass(8.0, 0.6, 1.0e11, 1.6e12, 320.0, 50000.0, 12.8, 12.8, 12.8, 0.067, 'bright')",
            "gold_call": "_oracle_plasmon_effective_mass(8.0, 0.6, 1.0e11, 1.6e12, 320.0, 50000.0, 12.8, 12.8, 12.8, 0.067, 'bright')",
            "tol": 1e-05,
        },
        # --- Edge: GaN-like heavy electrons, high density and a very close first gate ---
        {
            "setup": """import numpy as np
""",
            "call": "plasmon_effective_mass(4.0, 0.35, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22, 'dark')",
            "gold_call": "_oracle_plasmon_effective_mass(4.0, 0.35, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22, 'dark')",
            "tol": 1e-05,
        },
        # --- Edge: 1e-3 above the impedance-matched degeneracy, partner mode a few MHz away ---
        {
            "setup": """import numpy as np
""",
            "call": "plasmon_effective_mass(6.0, 0.4416, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'bright')",
            "gold_call": "_oracle_plasmon_effective_mass(6.0, 0.4416, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'bright')",
            "tol": 1e-06,
        },
        # --- Invalid: unknown mode family ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        plasmon_effective_mass(6.0, 0.3, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'grey')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_plasmon_effective_mass(6.0, 0.3, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 'grey')
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
