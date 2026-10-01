"""
Polymer volume fraction at which the free energy of the doped, crosslinked complex is stationary with respect to its water content.

Swelling equilibrium of a doped polyelectrolyte complex.

A complex that is initially dry and salt-free swells in the salt solution until the chemical potential
of the solvent inside it matches the bath. Take n_P chains of N_P repeat units, each repeat unit
occupying omega_P water volumes, together with n_SW water molecules, so that the polymer volume
fraction is

    Phi_P = n_P N_P omega_P / (n_SW + n_P N_P omega_P)

and the water volume fraction is Phi_SW = 1 - Phi_P (the bound ions add no volume). Swelling is
isotropic: the stretch ratio in each of the three principal directions is lambda = Phi_P^(-1/3), the
dry complex being unstretched. Relative to the dry, salt-free complex, the free energy in units of RT is

    F / RT = n_SW ln Phi_SW + n_P ln(Phi_P / 2) + chi n_SW Phi_P
             + (n_P N_P / 2) [ E + (1/2) X xi_PP + sum_{k=2..z} (1 - 1/k) xi_k / 2 ] (3 lambda^2 - 3)
             + F_dop

The first line is the Flory-Huggins mixing free energy with the polymer-water parameter chi. The
bracket collects the tube-model entanglement group E = v_P N_e alpha_tube (b/a_0)^2 and the
phantom-network crosslinks: the intrinsic polycation-polyanion pairs through the crosslink group
X = v_P N_+- and the cations that bridge k >= 2 polyanion sites. Here xi_k is the site fraction of
binding mode k (k = 1 mixed, k = z fully bridged) and xi_PP = 1 - sum_k xi_k is the fraction of
intrinsic pairs. F_dop, the configurational and reaction free energy of doping, is proportional to
n_P and does not depend on n_SW.

The complex is at swelling equilibrium when the derivative of F with respect to n_SW, taken at fixed
n_P and fixed site fractions, vanishes. When chi <= 1/2 and the bracketed network term is positive,
this condition has exactly one root with 0 < Phi_P < 1; the poorer-solvent cases used here also have a
single root. Return it with a relative accuracy of about 1e-12.

Returns
-------
float: polymer volume fraction Phi_P of the complex at swelling equilibrium, strictly between 0 and 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def swelling_polymer_fraction(xi_modes: "np.ndarray", chi: float, omega_p: float, n_p: float, entanglement: float,
                              crosslink: float) -> float:
    '''Polymer volume fraction of the complex at swelling equilibrium for given mode site fractions.

    Parameters
    ----------
    xi_modes : np.ndarray
        Shape (z,): site fractions of the binding modes k = 1..z.
    chi : float
        Polymer-water Flory-Huggins parameter.
    omega_p : float
        Size of a polyion repeat unit relative to a water molecule.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.

    Returns
    -------
    result : float
        Phi_P, strictly between 0 and 1.

    Raises
    ------
    ValueError
        If xi_modes is empty or not one dimensional, has a negative entry or entries summing to 1 or
        more, omega_p or n_p is not positive, entanglement or crosslink is negative, or the network term
        E + (1/2) X xi_PP + sum_{k=2..z} (1 - 1/k) xi_k / 2 is not positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_swelling_polymer_fraction(xi_modes: "np.ndarray", chi: float, omega_p: float, n_p: float,
                                      entanglement: float, crosslink: float) -> float:
    import numpy as np
    from scipy.optimize import brentq
    xi = np.array(xi_modes, dtype=float)
    if xi.ndim != 1 or xi.size == 0:
        raise ValueError("xi_modes must be a non-empty one dimensional array")
    if np.any(xi < 0.0) or xi.sum() >= 1.0:
        raise ValueError("site fractions must be non-negative with sum below one")
    if not (omega_p > 0.0 and n_p > 0.0) or entanglement < 0.0 or crosslink < 0.0:
        raise ValueError("omega_p and n_p must be positive, entanglement and crosslink non-negative")
    z = xi.size
    network = (entanglement + 0.5 * crosslink * (1.0 - xi.sum())
               + sum((1.0 - 1.0 / kk) * xi[kk - 1] / 2.0 for kk in range(2, z + 1)))
    if network <= 0.0:
        raise ValueError("the network term must be positive")

    def _condition(p):
        return (np.log1p(-p) + (1.0 - 1.0 / (omega_p * n_p)) * p + chi * p * p
                + network / omega_p * p ** (1.0 / 3.0))

    return float(brentq(_condition, 1e-300, 1.0 - 1e-16, xtol=1e-300, rtol=1e-15, maxiter=2000))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: divalent doping of a long-chain complex
        {
            "setup": """import numpy as np
xi = np.array([0.03, 0.05])
""",
            "call": "swelling_polymer_fraction(xi.copy(), 0.3, 6.5, 2500.0, 1.0, 1.0)",
            "gold_call": "_oracle_swelling_polymer_fraction(xi.copy(), 0.3, 6.5, 2500.0, 1.0, 1.0)",
            "tol": 1e-9,
        },
        # boundary: undoped monovalent case with a vanishingly weak network
        {
            "setup": """import numpy as np
xi = np.array([0.0])
""",
            "call": "swelling_polymer_fraction(xi.copy(), 0.1, 3.5, 120.0, 1e-4, 0.0)",
            "gold_call": "_oracle_swelling_polymer_fraction(xi.copy(), 0.1, 3.5, 120.0, 1e-4, 0.0)",
            "tol": 1e-9,
        },
        # edge: good solvent (negative chi) and weak network, strongly swollen complex
        {
            "setup": """import numpy as np
xi = np.array([0.1, 0.25, 0.3])
""",
            "call": "swelling_polymer_fraction(xi.copy(), -2.0, 6.5, 2500.0, 0.02, 0.05)",
            "gold_call": "_oracle_swelling_polymer_fraction(xi.copy(), -2.0, 6.5, 2500.0, 0.02, 0.05)",
            "tol": 1e-9,
        },
        # edge: poor solvent and stiff network, nearly dry complex
        {
            "setup": """import numpy as np
xi = np.array([0.2, 0.4])
""",
            "call": "swelling_polymer_fraction(xi.copy(), 2.5, 1.2, 50.0, 3.0, 2.0)",
            "gold_call": "_oracle_swelling_polymer_fraction(xi.copy(), 2.5, 1.2, 50.0, 3.0, 2.0)",
            "tol": 1e-9,
        },
        # edge: stiff network in a poor solvent, the complex barely swells
        {
            "setup": """import numpy as np
xi = np.array([0.1, 0.2])
""",
            "call": "swelling_polymer_fraction(xi.copy(), 1.0, 1.0, 100.0, 8.0, 2.0)",
            "gold_call": "_oracle_swelling_polymer_fraction(xi.copy(), 1.0, 1.0, 100.0, 8.0, 2.0)",
            "tol": 1e-9,
        },
        # edge: good solvent with an extremely weak network, highly swollen complex
        {
            "setup": """import numpy as np
xi = np.array([0.01])
""",
            "call": "swelling_polymer_fraction(xi.copy(), -3.0, 20.0, 5000.0, 1e-06, 0.0)",
            "gold_call": "_oracle_swelling_polymer_fraction(xi.copy(), -3.0, 20.0, 5000.0, 1e-06, 0.0)",
            "tol": 1e-9,
        },
        # invalid: site fractions summing to one leave no intrinsic pairs
        {
            "setup": """import numpy as np
def run_model():
    try:
        swelling_polymer_fraction(np.array([0.6, 0.4]), 0.3, 6.5, 2500.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_swelling_polymer_fraction(np.array([0.6, 0.4]), 0.3, 6.5, 2500.0, 1.0, 1.0)
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
