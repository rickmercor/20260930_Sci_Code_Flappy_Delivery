"""
Site fractions of all binding modes from their coupled mass-action laws at a prescribed polymer volume fraction, including free-salt depletion.

Mode-resolved doping of a symmetric complex at a fixed polymer volume fraction.

For a symmetric complex (identical polycation and polyanion, all charged sites occupied), let xi_k be
the fraction of polyion repeat units bound in mode k (k = 1..z; k = 1 mixed, k = z fully bridged),
xi_PS = sum_k xi_k the total doped fraction and Phi_P the polymer volume fraction of the complex. The
mass-action law of mode k, with its correlation-corrected doping constant K_k, reads

    K_k = xi_PS * (xi_k / k)^(1/k) * Phi_P / [ (1 - xi_PS) * phi_M^(1/k) * phi_A^(z/k) ]

where phi_M and phi_A are the free cation and anion volume fractions of the bathing solution. These
depend on xi_PS through the salt taken up by the complex (free salt C_S - C_P0 xi_PS, one cation and
z anions per salt unit, effective ion volumes from bare radius and hydration number), and ln K_k in
turn depends on them through the correlation integral.

For 0 < Phi_P < 1 the physical solution (every xi_k > 0, xi_PS < 1 and C_S - C_P0 xi_PS > 0) is
unique. Return it with a relative accuracy of about 1e-7 on every xi_k.

Returns
-------
np.ndarray of shape (z,): site fractions xi_k of the binding modes k = 1..z
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def doping_at_fixed_swelling(salt_conc: float, polymer_fraction: float, polymer_conc: float, valence: int,
                             ions: "np.ndarray", delta_g: "np.ndarray", temperature: float,
                             dielectric: float) -> "np.ndarray":
    '''Site fractions of the z binding modes at a prescribed polymer volume fraction.

    Parameters
    ----------
    salt_conc : float
        Salt concentration C_S in mol/L.
    polymer_fraction : float
        Polymer volume fraction Phi_P of the complex.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    valence : int
        Cation charge number z.
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (z,): the site fractions xi_k.

    Raises
    ------
    ValueError
        If polymer_fraction is not strictly between 0 and 1, delta_g does not have shape (valence,),
        valence is not an integer >= 1, ions is not a (2, 2) array with positive radii and non-negative
        hydration numbers, salt_conc, polymer_conc, temperature or dielectric is not positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_doping_at_fixed_swelling(salt_conc: float, polymer_fraction: float, polymer_conc: float, valence: int,
                                     ions: "np.ndarray", delta_g: "np.ndarray", temperature: float,
                                     dielectric: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    if not (0.0 < polymer_fraction < 1.0):
        raise ValueError("polymer_fraction must lie strictly between 0 and 1")
    if not (temperature > 0.0 and dielectric > 0.0):
        raise ValueError("temperature and dielectric must be positive")
    _oracle_free_ion_volume_fractions(salt_conc, 0.0, polymer_conc, valence, ions)
    z = int(valence)
    delta_g = np.array(delta_g, dtype=float)
    if delta_g.shape != (z,):
        raise ValueError("delta_g must have shape (valence,)")
    k = np.arange(1, z + 1, dtype=float)
    s_max = min(1.0, float(salt_conc) / float(polymer_conc))

    def _log_modes(y):
        # y is the logit of xi_PS, so ln(xi_PS) and ln(1 - xi_PS) both stay accurate
        ln_s, ln_pp = -np.logaddexp(0.0, -y), -np.logaddexp(0.0, y)
        v = _oracle_free_ion_volume_fractions(salt_conc, np.exp(ln_s), polymer_conc, z, ions)
        ln_k = _oracle_log_doping_constants(delta_g, v[2:], v[:2], z, temperature, dielectric)
        rhs = ln_k - np.log(polymer_fraction) + ln_pp + np.log(v[2]) / k + (z / k) * np.log(v[3])
        return np.log(k) + k * (rhs - ln_s), ln_s

    def _excess(y):
        lm, ln_s = _log_modes(y)
        top = lm.max()
        return top + np.log(np.sum(np.exp(lm - top))) - ln_s

    s_top = s_max * (1.0 - 1e-13)
    y_high = 700.0 if s_top >= 1.0 else np.log(s_top) - np.log1p(-s_top)
    y = brentq(_excess, -700.0, y_high, xtol=1e-14, rtol=1e-15, maxiter=1000)
    lm, ln_s = _log_modes(y)
    xi = np.exp(lm)
    return xi * (np.exp(ln_s) / np.sum(xi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent chloride at 0.10 M
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.10, 0.6, 0.05, 2, ions.copy(), np.array([-4.2, -0.1]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.10, 0.6, 0.05, 2, ions.copy(), np.array([-4.2, -0.1]), 298.15, 79.0)",
            "tol": 1e-5,
        },
        # boundary: very dilute salt with weak binding, doping far below one percent
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.004, 0.55, 0.05, 2, ions.copy(), np.array([3.5, 1.2]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.004, 0.55, 0.05, 2, ions.copy(), np.array([3.5, 1.2]), 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: strong binding where doping is limited by the salt available (C_S barely above C_P0)
        {
            "setup": """import numpy as np
ions = np.array([[0.72, 10.0], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.0505, 0.35, 0.05, 2, ions.copy(), np.array([-12.0, -10.0]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.0505, 0.35, 0.05, 2, ions.copy(), np.array([-12.0, -10.0]), 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: concentrated salt close to saturation of the sites, monovalent
        {
            "setup": """import numpy as np
ions = np.array([[1.02, 3.5], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(3.0, 0.9, 0.05, 1, ions.copy(), np.array([-3.0]), 310.0, 67.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(3.0, 0.9, 0.05, 1, ions.copy(), np.array([-3.0]), 310.0, 67.0)",
            "tol": 1e-5,
        },
        # normal: trivalent cation with three modes
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.2, 0.58, 0.05, 3, ions.copy(), np.array([-3.0, -1.5, -0.5]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.2, 0.58, 0.05, 3, ions.copy(), np.array([-3.0, -1.5, -0.5]), 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: nearly saturated sites in concentrated salt, bridging fraction of order 1e-5
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(2.0, 0.6, 0.05, 2, ions.copy(), np.array([-10.0, -2.0]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(2.0, 0.6, 0.05, 2, ions.copy(), np.array([-10.0, -2.0]), 298.15, 79.0)",
            "tol": 1e-5,
        },
        # boundary: millimolar salt bath
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.001, 0.6, 0.05, 2, ions.copy(), np.array([-4.2, -0.1]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.001, 0.6, 0.05, 2, ions.copy(), np.array([-4.2, -0.1]), 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: trivalent cation in millimolar salt over a concentrated polyelectrolyte, doping capped by the salt available
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.003, 0.62, 0.15, 3, ions.copy(), np.array([-11.0, -6.0, -4.0]), 298.15, 79.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.003, 0.62, 0.15, 3, ions.copy(), np.array([-11.0, -6.0, -4.0]), 298.15, 79.0)",
            "tol": 1e-7,
        },
        # normal: tetravalent cation with four binding modes
        {
            "setup": """import numpy as np
ions = np.array([[0.94, 12.0], [1.81, 2.0]])
""",
            "call": "doping_at_fixed_swelling(0.5, 0.6, 0.05, 4, ions.copy(), np.array([-5.0, -3.0, -2.0, -1.5]), 298.15, 78.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.5, 0.6, 0.05, 4, ions.copy(), np.array([-5.0, -3.0, -2.0, -1.5]), 298.15, 78.0)",
            "tol": 1e-5,
        },
        # edge: iodide salt in a warm, less polar medium, mixed mode close to saturating the sites
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [2.20, 1.0]])
""",
            "call": "doping_at_fixed_swelling(0.3, 0.51, 0.08, 2, ions.copy(), np.array([-15.0, -3.0]), 318.15, 72.0)",
            "gold_call": "_oracle_doping_at_fixed_swelling(0.3, 0.51, 0.08, 2, ions.copy(), np.array([-15.0, -3.0]), 318.15, 72.0)",
            "tol": 1e-5,
        },
        # invalid: polymer fraction must be inside (0, 1)
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
def run_model():
    try:
        doping_at_fixed_swelling(0.10, 1.0, 0.05, 2, ions.copy(), np.array([-4.2, -0.1]), 298.15, 79.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_doping_at_fixed_swelling(0.10, 1.0, 0.05, 2, ions.copy(), np.array([-4.2, -0.1]), 298.15, 79.0)
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
