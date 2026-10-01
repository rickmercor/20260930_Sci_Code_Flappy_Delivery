"""
Step 11 - ORCHESTRATOR - crystallization driving force out of the dense liquid phase.

This final step chains the whole model. Protein crystals in these systems nucleate preferentially inside the protein-rich droplets produced by liquid-liquid phase separation, so the quantity that controls crystallisation yield is the excess chemical potential of a protein in that dense liquid relative to a protein in the crystal:

delta_mu = mu(phi_II, beta) - mu(phi_S, beta)

phi_II is the protein-rich coexisting volume fraction of the liquid-liquid binodal at the working temperature (Step 09) and phi_S is the crystal solubility at the same temperature (Step 10), so that mu(phi_S, beta) is by construction the crystal chemical potential of cell theory. A positive delta_mu means the dense liquid is supersaturated with respect to the crystal and crystallisation from the droplet phase is downhill. The quantity is only defined inside the demixing dome, so the orchestrator first locates the critical point (Step 08) and refuses any working temperature at or above the critical temperature, where no protein-rich phase exists to crystallise out of. Both volume fractions come from the same fluid free energy, so every earlier step feeds this one: the Pade map and the well-neighbour count set the perturbation terms, the degeneracy factor converts the bare well depth into the effective one, the restored well factor fixes the position of the dense branch, and the critical point bounds the temperature range over which the answer exists.

Returns
-------
float, the reduced driving force delta_mu as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def crystallization_driving_force(lam: float, eps_sw: float, alpha: float,
                                  temperature: float, n_s: float, eps_s: float,
                                  ln_omega_s: float) -> float:
    '''Reduced chemical-potential driving force for crystallisation from the dense phase.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.
    temperature : float
        Absolute temperature in kelvin, must be > 0 and below the critical
        temperature of the system.
    n_s : float
        Number of contact regions per protein in the crystal, must be > 0.
    eps_s : float
        Average attraction energy of one crystal contact in kelvin, must be > 0.
    ln_omega_s : float
        Natural logarithm of the residual phase volume Omega_S of the crystal.

    Returns
    -------
    delta_mu : float
        mu(phi_II) - mu(phi_S) at the working temperature, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside the domain described above or is not finite,
        if the working temperature is at or above the critical temperature, if
        the state point admits no liquid-liquid coexistence resolvable on the
        volume-fraction window (0, 0.74), or if no crystal solubility root
        exists in (0, 0.70).
    '''
    return delta_mu  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_crystallization_driving_force(lam: float, eps_sw: float, alpha: float,
                                          temperature: float, n_s: float, eps_s: float,
                                          ln_omega_s: float) -> float:
    import numpy as np
    phi_c, t_c = _oracle_critical_point(lam, eps_sw, alpha)
    if float(temperature) >= t_c:
        raise ValueError("temperature is at or above the critical temperature; "
                         "no protein-rich coexisting phase exists")
    binodal = _oracle_llps_binodal(lam, eps_sw, alpha, temperature)
    phi_dense = float(binodal[1])
    phi_sol = _oracle_crystal_solubility(lam, eps_sw, alpha, temperature,
                                         n_s, eps_s, ln_omega_s)
    x = float(eps_sw) / float(temperature)
    mu_dense = _oracle_reduced_potentials(phi_dense, lam, x, alpha)[1]
    mu_sol = _oracle_reduced_potentials(phi_sol, lam, x, alpha)[1]
    return float(mu_dense - mu_sol)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: additive system at the working temperature of the task ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1732.0, 0.0293
temperature = 250.0
n_s, eps_s, ln_omega_s = 6.0, 1732.0, np.log(2.6e-6)
""",
            "call": "crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Normal: reference system at the same temperature ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1687.0, 0.0372
temperature = 250.0
n_s, eps_s, ln_omega_s = 6.0, 1687.0, np.log(2.6e-6)
""",
            "call": "crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Boundary: deep quench widens the binodal and raises the driving force ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1687.0, 0.0372
temperature = 240.0
n_s, eps_s, ln_omega_s = 6.0, 1687.0, np.log(2.6e-6)
""",
            "call": "crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Boundary: a weaker crystal lattice lowers the driving force while
        #     retaining a stable dilute crystal-solubility root ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1732.0, 0.0293
temperature = 250.0
n_s, eps_s, ln_omega_s = 5.0, 1732.0, np.log(2.6e-6)
""",
            "call": "crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystallization_driving_force(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Edge: above the critical temperature there is no dense phase ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        crystallization_driving_force(1.3, 1732.0, 0.0293, 400.0, 6.0, 1732.0, -12.86)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystallization_driving_force(1.3, 1732.0, 0.0293, 400.0, 6.0, 1732.0, -12.86)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: degeneracy factor outside (0, 1] must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        crystallization_driving_force(1.3, 1732.0, 0.0, 250.0, 6.0, 1732.0, -12.86)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystallization_driving_force(1.3, 1732.0, 0.0, 250.0, 6.0, 1732.0, -12.86)
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
