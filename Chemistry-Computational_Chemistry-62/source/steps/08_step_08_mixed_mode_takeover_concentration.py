"""
Salt concentration above which the mixed binding mode holds more cations than every other mode; for a divalent cation, where the mixed and bridged modes hold equal numbers of cations.

Salt concentration at which mixed binding becomes the predominant binding mode.

For a z-valent cation (z >= 2) bound to the complex in modes k = 1..z, mode k holds xi_k / k cations per
repeat unit and the predominant mode is the one holding the most cations. In dilute salt the fully
bridged mode is predominant; in concentrated salt the mixed mode (k = 1) can take over, and for the
complexes considered here, once the mixed mode is predominant it stays predominant at every higher salt
concentration. The takeover concentration C_mix is the salt concentration above which the mixed mode
holds more cations than every other binding mode. For a divalent cation it is the concentration at which
the mixed and the bridged mode hold equal numbers of cations, xi_1 = xi_2 / 2. For z >= 3 the mode the
mixed mode takes over from need not be the fully bridged mode.

C_mix is the last change of the predominant mode at or below c_max, provided the mixed mode is the
predominant mode at c_max. If the mixed mode is not predominant at c_max, there is no takeover up to
c_max. Return C_mix with a relative accuracy of about 1e-7.

Returns
-------
float: salt concentration C_mix in mol/L above which the mixed mode holds the most cations (xi_1 = xi_2 / 2 for z = 2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixed_mode_takeover_concentration(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                                      polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                                      crosslink: float, temperature: float, dielectric: float) -> float:
    '''Salt concentration above which the mixed mode holds more cations than every other binding mode.

    Parameters
    ----------
    c_max : float
        Largest salt concentration considered, mol/L.
    valence : int
        Cation charge number z (at least 2).
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    chi : float
        Polymer-water Flory-Huggins parameter.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    omega_p : float
        Size of a repeat unit relative to water.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : float
        The takeover salt concentration C_mix in mol/L.

    Raises
    ------
    ValueError
        If valence is not an integer >= 2, c_max is not positive, the mixed mode is not the predominant
        mode at c_max (no takeover up to c_max), or any input is invalid for the equilibrium doping of the
        complex.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mixed_mode_takeover_concentration(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                                              polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                                              crosslink: float, temperature: float, dielectric: float) -> float:
    import numpy as np
    changes = _oracle_dominant_mode_transitions(c_max, valence, ions, delta_g, chi, polymer_conc, omega_p, n_p,
                                                entanglement, crosslink, temperature, dielectric)
    z = int(valence)
    state = _oracle_equilibrium_doping(c_max, z, ions, delta_g, chi, polymer_conc, omega_p, n_p, entanglement,
                                       crosslink, temperature, dielectric)
    cations = state[:z] / np.arange(1, z + 1)
    if changes.size == 0 or int(np.argmax(cations)) != 0:
        raise ValueError("the mixed mode is not the predominant mode at c_max")
    return float(changes[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent calcium with a strongly favoured mixed mode
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-4.2, -0.1])
""",
            "call": "mixed_mode_takeover_concentration(1.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(1.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1.4e-8,
        },
        # boundary: strongly hydrated divalent cation that bridges up to high salt
        {
            "setup": """import numpy as np
ions = np.array([[0.72, 10.0], [1.81, 2.0]])
dg = np.array([-2.6, -0.7])
""",
            "call": "mixed_mode_takeover_concentration(3.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(3.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-7,
        },
        # normal: trivalent cation, the mixed mode takes over from the intermediate mode
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.5, -0.5])
""",
            "call": "mixed_mode_takeover_concentration(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 1e-7,
        },
        # edge: trivalent cation close to a triple point, the intermediate mode is predominant just before takeover
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.1255, -0.5])
""",
            "call": "mixed_mode_takeover_concentration(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 1e-7,
        },
        # edge: tetravalent cation passing through all four modes
        {
            "setup": """import numpy as np
ions = np.array([[0.94, 12.0], [1.81, 2.0]])
dg = np.array([-6.0, -3.3, -2.1003, -1.5])
""",
            "call": "mixed_mode_takeover_concentration(3.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(3.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "tol": 1e-7,
        },
        # boundary: strongly bound bromide salt, takeover below one millimolar
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.96, 1.5]])
dg = np.array([-13.0, -2.5])
""",
            "call": "mixed_mode_takeover_concentration(2.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(2.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 4e-11,
        },
        # edge: trivalent bromide with strongly bound modes in a cold medium
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.96, 1.5]])
dg = np.array([-16.0, -6.0, -3.5])
""",
            "call": "mixed_mode_takeover_concentration(0.5, 3, ions.copy(), dg.copy(), 0.4, 0.03, 5.5, 1200.0, 0.8, 1.2, 288.15, 81.0)",
            "gold_call": "_oracle_mixed_mode_takeover_concentration(0.5, 3, ions.copy(), dg.copy(), 0.4, 0.03, 5.5, 1200.0, 0.8, 1.2, 288.15, 81.0)",
            "tol": 2.5e-9,
        },
        # invalid: trivalent cation whose intermediate mode is still predominant at c_max
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.5, -0.5])
def run_model():
    try:
        mixed_mode_takeover_concentration(1.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_mixed_mode_takeover_concentration(1.0, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # invalid: weakly binding short-chain complex whose modes do not change up to 3 M
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([1.0, -0.5])
def run_model():
    try:
        mixed_mode_takeover_concentration(3.0, 2, ions.copy(), dg.copy(), 0.1, 0.05, 3.5, 120.0, 1.0, 0.1, 298.15, 79.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_mixed_mode_takeover_concentration(3.0, 2, ions.copy(), dg.copy(), 0.1, 0.05, 3.5, 120.0, 1.0, 0.1, 298.15, 79.0)
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
