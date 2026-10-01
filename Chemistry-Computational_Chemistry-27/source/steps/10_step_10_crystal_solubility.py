"""
Step 10 - Crystal solubility from cell theory.

The crystalline phase sits at a protein volume fraction above one half and is treated as incompressible, so the protein chemical potential in the crystal equals the Helmholtz free energy of one particle and depends on temperature but not on the amount of solid present. Cell theory writes it as

mu_S(beta) = mu0 - n_S * beta * eps_S / 2 - ln(Omega_S)

where n_S is the number of contact regions a protein makes with its lattice neighbours, eps_S is the average attraction energy of one contact, the factor 2 prevents double counting because a contact is shared between two proteins, and Omega_S is the residual phase volume (in units of the particle volume) left to the translation and rotation of a protein inside its cage. Take mu0 = 0, matching Step 03. The solubility boundary phi_S(beta) is the volume fraction of the fluid that is in equilibrium with the crystal,

mu(phi_S, beta) = mu_S(beta)

Solve this numerically with the full fluid chemical potential of Step 07 rather than the ideal-dilute approximation. Below the critical temperature mu(phi) is non-monotonic, so the equation can have more than one root; the physically meaningful solubility lies on the locally stable dilute branch. Select the smallest root in (0, 0.70), verify that d(mu)/d(phi) is positive there and, when a van der Waals loop is present, that the root lies below the low-density spinodal. Raise ValueError if no root satisfies those conditions.

Returns
-------
float, the crystal solubility volume fraction phi_S as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def crystal_solubility(lam: float, eps_sw: float, alpha: float, temperature: float,
                       n_s: float, eps_s: float, ln_omega_s: float) -> float:
    '''Protein volume fraction of the fluid in equilibrium with the crystal.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth of the fluid model in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.
    temperature : float
        Absolute temperature in kelvin, must be > 0.
    n_s : float
        Number of contact regions per protein in the crystal, must be > 0.
    eps_s : float
        Average attraction energy of one crystal contact in kelvin, must be > 0.
    ln_omega_s : float
        Natural logarithm of the residual phase volume Omega_S of the crystal.

    Returns
    -------
    phi_s : float
        Crystal solubility volume fraction, the smallest root on the stable
        dilute branch, as a native Python float.

    Raises
    ------
    ValueError
        If lam <= 1, or eps_sw <= 0, or alpha is outside (0, 1], or temperature
        <= 0, or n_s <= 0, or eps_s <= 0, or any argument is not finite, or no
        locally stable dilute solubility root exists in (0, 0.70).
    '''
    return phi_s  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _crystal_chemical_potential(temperature, n_s, eps_s, ln_omega_s):
    """Reduced chemical potential of the incompressible crystal (mu0 = 0)."""
    return -float(n_s) * (float(eps_s) / float(temperature)) / 2.0 - float(ln_omega_s)


def _oracle_crystal_solubility(lam: float, eps_sw: float, alpha: float, temperature: float,
                               n_s: float, eps_s: float, ln_omega_s: float) -> float:
    import numpy as np
    _validate_state(lam, eps_sw, alpha, temperature)
    for v in (n_s, eps_s, ln_omega_s):
        if not np.isfinite(v):
            raise ValueError("crystal parameters must be finite")
    if float(n_s) <= 0.0:
        raise ValueError("n_s must be > 0")
    if float(eps_s) <= 0.0:
        raise ValueError("eps_s must be > 0")

    x = float(eps_sw) / float(temperature)
    mu_s = _crystal_chemical_potential(temperature, n_s, eps_s, ln_omega_s)
    resid = lambda p: _oracle_reduced_potentials(p, lam, x, alpha)[1] - mu_s

    grid = np.linspace(1e-12, 0.70, 4001)
    f = np.array([resid(p) for p in grid], dtype=float)
    idx = np.where(np.diff(np.sign(f)) != 0)[0]
    if idx.size == 0:
        raise ValueError("no solubility root in (0, 0.70)")
    i = int(idx[0])
    phi_s = float(_bisect_root(resid, float(grid[i]), float(grid[i + 1])))
    state = _oracle_reduced_potentials(phi_s, lam, x, alpha)
    if not np.isfinite(state[3]) or float(state[3]) <= 0.0:
        raise ValueError("no locally stable solubility root in (0, 0.70)")
    spinodal_grid = np.linspace(1e-6, 0.70, 2001)
    spinodals = _spinodal_roots(temperature, lam, eps_sw, alpha, spinodal_grid)
    if spinodals is not None and phi_s >= float(spinodals[0]):
        raise ValueError("no solubility root on the stable dilute branch")
    return phi_s

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: additive system at 298.15 K ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1732.0, 0.0293
temperature = 298.15
n_s, eps_s, ln_omega_s = 6.0, 1732.0, np.log(2.6e-6)
""",
            "call": "crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Normal: reference system at 298.15 K ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1687.0, 0.0372
temperature = 298.15
n_s, eps_s, ln_omega_s = 6.0, 1687.0, np.log(2.6e-6)
""",
            "call": "crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Boundary: below the critical temperature, where the fluid chemical potential
        # carries a van der Waals loop while the saturation root stays on the dilute branch
        #     and only the smallest root is the stable solubility ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1732.0, 0.0293
temperature = 250.0
n_s, eps_s, ln_omega_s = 6.0, 1732.0, np.log(2.6e-6)
""",
            "call": "crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Edge: a weakened lattice with no dilute root must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        crystal_solubility(1.3, 1732.0, 0.0293, 250.0, 4.0, 1732.0, np.log(2.6e-6))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystal_solubility(1.3, 1732.0, 0.0293, 250.0, 4.0, 1732.0, np.log(2.6e-6))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Boundary: higher temperature raises solubility into the concentrated regime ---
        {
            "setup": """import numpy as np
lam, eps_sw, alpha = 1.3, 1687.0, 0.0372
temperature = 330.0
n_s, eps_s, ln_omega_s = 6.0, 1687.0, np.log(2.6e-6)
""",
            "call": "crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
            "gold_call": "_oracle_crystal_solubility(lam, eps_sw, alpha, temperature, n_s, eps_s, ln_omega_s)",
        },
        # --- Edge: n_s <= 0 must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        crystal_solubility(1.3, 1732.0, 0.0293, 298.15, 0.0, 1732.0, -12.86)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystal_solubility(1.3, 1732.0, 0.0293, 298.15, 0.0, 1732.0, -12.86)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Edge: non-positive contact energy must raise ValueError ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        crystal_solubility(1.3, 1732.0, 0.0293, 298.15, 6.0, -5.0, -12.86)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_crystal_solubility(1.3, 1732.0, 0.0293, 298.15, 6.0, -5.0, -12.86)
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
