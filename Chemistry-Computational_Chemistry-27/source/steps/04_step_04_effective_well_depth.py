"""
Step 04 - Anisotropy-corrected effective well depth.

Barker-Henderson perturbation theory treats particle-particle attraction as isotropic, but protein-protein contacts are strongly directional: only a small fraction of the protein surface carries binding regions. Two proteins therefore gain the full well energy only when their mutual orientation is right, and free rotation in the fluid phase costs entropy that a crystal lattice does not pay. Anisotropy is folded into the isotropic machinery by replacing the bare depth eps_SW with a temperature-dependent effective depth eps_eff, defined so that the pair Boltzmann weight of the effective isotropic well equals the orientational average of the pair Boltzmann weight of the patchy one. Use a two-state surface model: a fraction alpha of each particle surface carries binding sites of energy eps_SW and the remaining fraction (1 - alpha) carries none, orientations are equally likely, and one contact is shared between two proteins, so each of the two surfaces contributes one independently averaged patch. Derive eps_eff from that average. It is fixed by the logarithm of the averaged Boltzmann weight, not by any linear average of the depths, and it depends on temperature even though eps_SW does not. Two checks the result must satisfy: at alpha = 1 it must return eps_eff = eps_SW exactly, and as alpha falls at fixed eps_SW the effective attraction must weaken sharply. Work with the dimensionless products throughout: the input is beta*eps_SW and the output is beta*eps_eff. Evaluate the expression in a way that does not overflow when beta*eps_SW is large, which happens routinely at protein well depths of order 10**3 K and temperatures of order 250 K.

Returns
-------
float, the dimensionless effective well depth beta * eps_eff as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_well_depth(beta_eps_sw: float, alpha: float) -> float:
    '''Dimensionless anisotropy-corrected well depth beta * eps_eff.

    Parameters
    ----------
    beta_eps_sw : float
        Dimensionless bare square-well depth beta * eps_SW, must be >= 0.
    alpha : float
        Degeneracy factor, the fraction of particle surface carrying binding
        sites, 0 < alpha <= 1.

    Returns
    -------
    beta_eps_eff : float
        Dimensionless effective well depth beta * eps_eff as a native Python float.

    Raises
    ------
    ValueError
        If beta_eps_sw is negative or not finite, or if alpha is outside (0, 1],
        or if alpha is not finite.
    '''
    return beta_eps_eff  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_well_depth(beta_eps_sw: float, alpha: float) -> float:
    import numpy as np
    if not np.isfinite(beta_eps_sw) or not np.isfinite(alpha):
        raise ValueError("beta_eps_sw and alpha must be finite")
    x = float(beta_eps_sw)
    a = float(alpha)
    if x < 0.0:
        raise ValueError("beta_eps_sw must be >= 0")
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    h = 0.5 * x
    # log(alpha*e^h + 1-alpha) = h + log(alpha + (1-alpha)*e^-h) avoids overflow
    return float(2.0 * (h + np.log(a + (1.0 - a) * np.exp(-h))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: additive system at 298.15 K (eps_SW = 1732 K, alpha = 0.0293) ---
        {
            "setup": """import numpy as np
beta_eps_sw = 1732.0 / 298.15
alpha = 0.0293
""",
            "call": "effective_well_depth(beta_eps_sw, alpha)",
            "gold_call": "_oracle_effective_well_depth(beta_eps_sw, alpha)",
        },
        # --- Normal: reference system at 250 K (eps_SW = 1687 K, alpha = 0.0372) ---
        {
            "setup": """import numpy as np
beta_eps_sw = 1687.0 / 250.0
alpha = 0.0372
""",
            "call": "effective_well_depth(beta_eps_sw, alpha)",
            "gold_call": "_oracle_effective_well_depth(beta_eps_sw, alpha)",
        },
        # --- Boundary: alpha = 1 must return the bare depth exactly ---
        {
            "setup": """import numpy as np
beta_eps_sw = 6.928
alpha = 1.0
""",
            "call": "effective_well_depth(beta_eps_sw, alpha)",
            "gold_call": "_oracle_effective_well_depth(beta_eps_sw, alpha)",
        },
        # --- Boundary: very large argument, the overflow-prone regime ---
        {
            "setup": """import numpy as np
beta_eps_sw = 1500.0
alpha = 0.001
""",
            "call": "effective_well_depth(beta_eps_sw, alpha)",
            "gold_call": "_oracle_effective_well_depth(beta_eps_sw, alpha)",
        },
        # --- Boundary: zero attraction returns zero ---
        {
            "setup": """import numpy as np
beta_eps_sw = 0.0
alpha = 0.05
""",
            "call": "effective_well_depth(beta_eps_sw, alpha)",
            "gold_call": "_oracle_effective_well_depth(beta_eps_sw, alpha)",
        },
        # --- Edge: alpha out of range must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        effective_well_depth(6.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_effective_well_depth(6.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: negative dimensionless depth must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        effective_well_depth(-1.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_effective_well_depth(-1.0, 0.5)
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
