"""
Frequencies of the lowest bright and dark zone-centre plasma modes of the two-strip plasmonic crystal.

The crystal is a two-dimensional electron layer whose every period L is split into strip 1 of width fL and strip 2 of width (1 - f)L. Strip n carries the equilibrium electron density N_n and sits under its own ideal metal plane at distance h_n; the semiconductor below the layer has permittivity eps_s and the spacer between the layer and the plane above strip n has permittivity kappa_n, so the two strips can differ in density, in gate distance and in spacer permittivity. Damping is neglected throughout.

With the Kronig-Penney matching of the previous step, zone-centre modes (k = 0) are the frequencies at which a wave reproduces itself exactly after one period. They fall into two families that are told apart by the ac current. In a bright mode the particle current averaged over one period is nonzero, so the mode carries a net dipole and couples to radiation incident normally on the crystal. In a dark mode that average vanishes and the mode is optically inactive. The frequency zero, where the whole electron layer is at rest, is not a mode.

The step returns the n_modes lowest positive frequencies of each family, bright ones in the first row and dark ones in the second, each row in increasing order, in GHz. The two families interleave and can come arbitrarily close to each other, so they must be separated by their current symmetry and not by their order.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s.

Returns
-------
numpy.ndarray of shape (2, n_modes): row 0 the lowest bright zone-centre frequencies, row 1 the lowest dark ones, in GHz
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def zone_center_modes(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, n_modes: int) -> "np.ndarray":
    '''Lowest bright and dark zone-centre plasma frequencies of the crystal.

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
    n_modes : int
        Number of modes of each family to return; at least 1.

    Returns
    -------
    frequencies : numpy.ndarray
        Array of shape (2, n_modes); row 0 bright, row 1 dark, in GHz, each row increasing.

    Raises
    ------
    ValueError
        If any physical argument is not a positive finite number, if fill is not
        strictly between 0 and 1, or if n_modes is not a positive integer.
    '''
    return frequencies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_zone_center_modes(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, n_modes: int) -> "np.ndarray":
    import numpy as np
    _sm67_check_cell(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    if isinstance(n_modes, bool) or int(n_modes) != n_modes or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    args = (period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    return _sm67_zone_roots(args, int(n_modes)) / (2.0 * np.pi * 1e9)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: benchmark crystal between its two degeneracies ---
        {
            "setup": """import numpy as np
""",
            "call": "zone_center_modes(6.0, 0.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 2)",
            "gold_call": "_oracle_zone_center_modes(6.0, 0.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 2)",
            "tol": 1e-07,
        },
        # --- Normal: benchmark crystal above its impedance-matched degeneracy ---
        {
            "setup": """import numpy as np
""",
            "call": "zone_center_modes(6.0, 0.6, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 2)",
            "gold_call": "_oracle_zone_center_modes(6.0, 0.6, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 2)",
            "tol": 1e-07,
        },
        # --- Boundary: gated-gated crystal with three modes of each family ---
        {
            "setup": """import numpy as np
""",
            "call": "zone_center_modes(8.0, 0.5, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067, 3)",
            "gold_call": "_oracle_zone_center_modes(8.0, 0.5, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067, 3)",
            "tol": 1e-07,
        },
        # --- Edge: nearly ungated strip 2 at small filling with sixteenfold density contrast ---
        {
            "setup": """import numpy as np
""",
            "call": "zone_center_modes(8.0, 0.15, 1.0e11, 1.6e12, 320.0, 50000.0, 12.8, 12.8, 12.8, 0.067, 2)",
            "gold_call": "_oracle_zone_center_modes(8.0, 0.15, 1.0e11, 1.6e12, 320.0, 50000.0, 12.8, 12.8, 12.8, 0.067, 2)",
            "tol": 1e-07,
        },
        # --- Edge: next to the impedance-matched degeneracy, fundamental bright and dark modes a few MHz apart ---
        {
            "setup": """import numpy as np
""",
            "call": "zone_center_modes(6.0, 0.4406, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 1)",
            "gold_call": "_oracle_zone_center_modes(6.0, 0.4406, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 1)",
            "tol": 1e-07,
        },
        # --- Invalid: zero modes requested ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        zone_center_modes(6.0, 0.3, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_zone_center_modes(6.0, 0.3, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067, 0)
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
