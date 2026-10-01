"""
Self-consistent mode site fractions and polymer volume fraction of the complex swollen to equilibrium in the salt solution.

Self-consistent doping and swelling of a complex immersed in a salt solution.

The mode site fractions xi_k depend on the polymer volume fraction Phi_P through the mass-action laws
of the binding modes (with the free-salt depletion and the correlation-corrected doping constants),
while Phi_P depends on the xi_k through the swelling condition, whose network term counts the
intrinsic pairs and the cations bridging two or more polyanion sites. The equilibrium state of a
complex that is dry and salt-free before immersion is the simultaneous solution of both conditions.
It is unique for the parameters used here and must be returned with a relative accuracy of about 1e-7
on every component.

Returns
-------
np.ndarray of shape (z + 1,): [xi_1, ..., xi_z, Phi_P] at the self-consistent equilibrium
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_doping(salt_conc: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                       polymer_conc: float, omega_p: float, n_p: float, entanglement: float, crosslink: float,
                       temperature: float, dielectric: float) -> "np.ndarray":
    '''Equilibrium mode site fractions and polymer volume fraction of the swollen, doped complex.

    Parameters
    ----------
    salt_conc : float
        Salt concentration C_S in mol/L.
    valence : int
        Cation charge number z.
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
    result : np.ndarray
        Shape (z + 1,): [xi_1, ..., xi_z, Phi_P].

    Raises
    ------
    ValueError
        If any input violates the conditions of the mode doping at fixed swelling (salt_conc,
        polymer_conc, temperature or dielectric not positive, valence not an integer >= 1, ions not a
        (2, 2) array with positive radii and non-negative hydration numbers, delta_g not of shape
        (valence,)), or omega_p or n_p is not positive, or entanglement or crosslink is negative, or
        entanglement and crosslink are both zero.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_equilibrium_doping(salt_conc: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                               polymer_conc: float, omega_p: float, n_p: float, entanglement: float, crosslink: float,
                               temperature: float, dielectric: float) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import brentq
    if not (omega_p > 0.0 and n_p > 0.0) or entanglement < 0.0 or crosslink < 0.0:
        raise ValueError("omega_p and n_p must be positive, entanglement and crosslink non-negative")
    if entanglement == 0.0 and crosslink == 0.0:
        raise ValueError("entanglement and crosslink cannot both be zero")

    def _swelling_of(p):
        xi = _oracle_doping_at_fixed_swelling(salt_conc, p, polymer_conc, valence, ions, delta_g,
                                              temperature, dielectric)
        return _oracle_swelling_polymer_fraction(xi, chi, omega_p, n_p, entanglement, crosslink)

    p = _swelling_of(0.5)
    converged = False
    for _ in range(200):
        p_new = _swelling_of(p)
        if abs(p_new - p) <= 1e-15 * p:
            p, converged = p_new, True
            break
        p = p_new
    if not converged:
        p = brentq(lambda x: _swelling_of(x) - x, 1e-9, 1.0 - 1e-9, xtol=1e-16, rtol=1e-15, maxiter=1000)
    xi = _oracle_doping_at_fixed_swelling(salt_conc, p, polymer_conc, valence, ions, delta_g,
                                          temperature, dielectric)
    return np.append(xi, p)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent chloride of a long-chain complex
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-4.2, -0.1])
""",
            "call": "equilibrium_doping(0.07, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(0.07, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # boundary: weakly binding divalent salt on a short-chain complex, bridging-only regime
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([5.1, 1.5])
""",
            "call": "equilibrium_doping(0.3, 2, ions.copy(), dg.copy(), 0.1, 0.05, 3.5, 120.0, 1.0, 0.1, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(0.3, 2, ions.copy(), dg.copy(), 0.1, 0.05, 3.5, 120.0, 1.0, 0.1, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: concentrated salt, mixed mode dominating the sites
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-4.2, -0.1])
""",
            "call": "equilibrium_doping(2.2, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(2.2, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: trivalent cation, three modes, different medium
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.81, 2.0]])
dg = np.array([-3.0, -1.5, -0.5])
""",
            "call": "equilibrium_doping(0.15, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "gold_call": "_oracle_equilibrium_doping(0.15, 3, ions.copy(), dg.copy(), 0.45, 0.02, 5.0, 800.0, 0.5, 1.5, 310.0, 70.0)",
            "tol": 1e-5,
        },
        # normal: monovalent salt at high concentration
        {
            "setup": """import numpy as np
ions = np.array([[1.02, 3.5], [1.81, 2.0]])
dg = np.array([-1.5])
""",
            "call": "equilibrium_doping(1.2, 1, ions.copy(), dg.copy(), 0.32, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(1.2, 1, ions.copy(), dg.copy(), 0.32, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # edge: nearly saturated complex in concentrated salt, bridging fraction of order 1e-5
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-10.0, -2.0])
""",
            "call": "equilibrium_doping(2.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(2.0, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-5,
        },
        # boundary: sub-millimolar MgCl2, mixed-mode fraction of order 1e-7
        {
            "setup": """import numpy as np
ions = np.array([[0.72, 10.0], [1.81, 2.0]])
dg = np.array([-2.6, -0.7])
""",
            "call": "equilibrium_doping(0.0008, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(0.0008, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-7,
        },
        # edge: concentrated polyelectrolyte in millimolar salt, most of the salt taken up by the complex
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-9.0, -1.0])
""",
            "call": "equilibrium_doping(0.004, 2, ions.copy(), dg.copy(), 0.35, 0.2, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "gold_call": "_oracle_equilibrium_doping(0.004, 2, ions.copy(), dg.copy(), 0.35, 0.2, 6.5, 2500.0, 1.0, 1.0, 298.15, 79.0)",
            "tol": 1e-7,
        },
        # normal: tetravalent cation with four binding modes at 1 M
        {
            "setup": """import numpy as np
ions = np.array([[0.94, 12.0], [1.81, 2.0]])
dg = np.array([-5.0, -3.0, -2.0, -1.5])
""",
            "call": "equilibrium_doping(1.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "gold_call": "_oracle_equilibrium_doping(1.0, 4, ions.copy(), dg.copy(), 0.4, 0.05, 6.5, 1500.0, 1.0, 1.0, 298.15, 78.0)",
            "tol": 1e-5,
        },
        # edge: trivalent bromide in concentrated salt, sites nearly saturated by the mixed mode
        {
            "setup": """import numpy as np
ions = np.array([[1.03, 9.0], [1.96, 1.5]])
dg = np.array([-8.0, -4.0, -2.0])
""",
            "call": "equilibrium_doping(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.03, 5.5, 1200.0, 0.8, 1.2, 288.15, 81.0)",
            "gold_call": "_oracle_equilibrium_doping(3.0, 3, ions.copy(), dg.copy(), 0.45, 0.03, 5.5, 1200.0, 0.8, 1.2, 288.15, 81.0)",
            "tol": 1e-5,
        },
        # edge: iodide salt in a warm, less polar medium with a very strongly bound mixed mode
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [2.20, 1.0]])
dg = np.array([-15.0, -3.0])
""",
            "call": "equilibrium_doping(0.3, 2, ions.copy(), dg.copy(), 0.3, 0.08, 6.5, 2500.0, 1.0, 1.0, 318.15, 72.0)",
            "gold_call": "_oracle_equilibrium_doping(0.3, 2, ions.copy(), dg.copy(), 0.3, 0.08, 6.5, 2500.0, 1.0, 1.0, 318.15, 72.0)",
            "tol": 1e-5,
        },
        # invalid: no network at all
        {
            "setup": """import numpy as np
ions = np.array([[1.00, 7.2], [1.81, 2.0]])
dg = np.array([-4.2, -0.1])
def run_model():
    try:
        equilibrium_doping(0.1, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 0.0, 0.0, 298.15, 79.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_equilibrium_doping(0.1, 2, ions.copy(), dg.copy(), 0.3, 0.05, 6.5, 2500.0, 0.0, 0.0, 298.15, 79.0)
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
