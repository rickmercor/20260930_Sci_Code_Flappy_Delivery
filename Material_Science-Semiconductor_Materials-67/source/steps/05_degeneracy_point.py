"""
Largest filling factor, and the common frequency there, at which the fundamental bright and dark zone-centre plasmons of the crystal coincide.

The crystal is a two-dimensional electron layer whose every period L is split into strip 1 of width fL and strip 2 of width (1 - f)L. Strip n carries the equilibrium electron density N_n and sits under its own ideal metal plane at distance h_n; the semiconductor below the layer has permittivity eps_s and the spacer between the layer and the plane above strip n has permittivity kappa_n, so the two strips can differ in density, in gate distance and in spacer permittivity. Damping is neglected throughout.

The frequencies of the fundamental bright and fundamental dark zone-centre modes depend on the filling factor f with all other device parameters fixed. At the trivial ends f = 0 and f = 1 the crystal is homogeneous and every bright mode coincides with a dark one. Between the ends the two fundamental frequencies can coincide at one or at several filling factors, and at each such point the order of the two frequencies is reversed on its two sides and the zone-centre effective masses of step 04 change sign.

The step returns the largest filling factor f* strictly between 0 and 1 at which the fundamental bright and dark frequencies coincide, together with the common frequency f_0 there.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s.

Returns
-------
numpy.ndarray of shape (2,): the largest coincidence filling factor f* and the common frequency f_0 in GHz
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def degeneracy_point(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> "np.ndarray":
    '''Largest filling factor at which the fundamental bright and dark modes coincide.

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
    point : numpy.ndarray
        Array [f_star, f0_ghz] with 0 < f_star < 1 and f0_ghz in GHz.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number, or if the fundamental bright
        and dark frequencies do not coincide anywhere in 0 < f < 1.
    '''
    return point

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _sm67_fundamental_split(fill, dev):
    modes = _oracle_zone_center_modes(dev[0], fill, *dev[1:], 1)
    return modes[0, 0] - modes[1, 0]


def _oracle_degeneracy_point(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
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
    modes = _oracle_zone_center_modes(period_um, fstar, *dev[1:], 1)
    return np.array([fstar, 0.5 * (modes[0, 0] + modes[1, 0])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark device, two coincidences of different origin ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_point(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "gold_call": "_oracle_degeneracy_point(6.0, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "tol": 1e-05,
        },
        # --- Boundary: equal gate distances and spacers, the gated-gated crystal ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_point(8.0, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_degeneracy_point(8.0, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "tol": 1e-05,
        },
        # --- Edge: two coincidences only 0.0012 apart in filling factor, which a coarse scan does not resolve ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_point(6.0, 8.0e11, 3.0e12, 300.0, 300.0, 12.8, 12.8, 3.97, 0.067)",
            "gold_call": "_oracle_degeneracy_point(6.0, 8.0e11, 3.0e12, 300.0, 300.0, 12.8, 12.8, 3.97, 0.067)",
            "tol": 1e-05,
        },
        # --- Edge: GaN-like heavy electrons at high density ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_point(4.0, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22)",
            "gold_call": "_oracle_degeneracy_point(4.0, 6.0e12, 9.0e12, 25.0, 250.0, 9.5, 9.5, 9.5, 0.22)",
            "tol": 1e-05,
        },
        # --- Edge: close gates with a low-permittivity spacer on strip 2 ---
        {
            "setup": """import numpy as np
""",
            "call": "degeneracy_point(6.0, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "gold_call": "_oracle_degeneracy_point(6.0, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "tol": 1e-05,
        },
        # --- Invalid: zero density in strip 2 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        degeneracy_point(6.0, 3.0e11, 0.0, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_degeneracy_point(6.0, 3.0e11, 0.0, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)
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
