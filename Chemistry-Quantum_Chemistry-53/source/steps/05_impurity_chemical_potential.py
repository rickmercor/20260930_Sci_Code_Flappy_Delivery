"""
Projecting the Hamiltonian onto an embedding cluster is exact only for a non-interacting or Hartree-Fock state.

Projecting the Hamiltonian onto an embedding cluster is exact only for a non-interacting or Hartree-Fock state. For a correlated cluster, density matrix embedding therefore complements the projected Hamiltonian with a chemical potential term, -mu n_I, acting on the impurity orbital. In density matrix embedding theory (and in its density-only variant, DET) the value of mu is global: one Lagrange multiplier, shared by all clusters, that enforces the correct total electron number.

Local potential functional embedding theory (LPFET) replaced that global multiplier by an impurity chemical potential that is specific to each embedded orbital and that is an explicit functional of the local Hartree-exchange-correlation potential of the lattice Kohn-Sham reference. The generalized theory (gLPFET) used in this task moves the reference to a generalized Kohn-Sham determinant with the Hartree-Fock exchange potential and keeps only the local correlation potential as the basic variable. Its rationale, derived in the strongly correlated limit (Appendix A of the source), gives the impurity chemical potential of the cluster built on site i as an expression in the correlation potential v_c and the bath orbital b^(i) of that site, in which the Hartree-exchange part no longer appears. This step implements that gLPFET expression. Its value is what distinguishes gLPFET from gDET (global mu) and from LPFET (mu built from the full Hartree-exchange-correlation potential).

Returns
-------
float: the gLPFET impurity chemical potential of the cluster built on the bath orbital b for the local correlation potential v_c.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def impurity_chemical_potential(b, v_c):
    '''gLPFET impurity chemical potential of one embedding cluster.

    Parameters
    ----------
    b : array_like of float, shape (L,)
        Normalized bath orbital of the embedded site in the site basis
        (zero on the embedded site itself).
    v_c : array_like of float, shape (L,)
        Local correlation potential of the gKS reference, one value per site.

    Returns
    -------
    mu : float
        Impurity chemical potential mu_imp of the cluster according to the
        gLPFET ansatz (the quantity subtracted, multiplied by the impurity
        occupation operator, from the projected cluster Hamiltonian).
        Raises ValueError if b and v_c do not have the same length or are
        not finite.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_impurity_chemical_potential(b, v_c):
    b = np.asarray(b, dtype=float)
    v_c = np.asarray(v_c, dtype=float)
    if b.ndim != 1 or v_c.ndim != 1 or b.shape != v_c.shape or b.shape[0] < 1:
        raise ValueError("b and v_c must be vectors of the same length")
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(v_c)):
        raise ValueError("b and v_c must be finite")
    return float(np.sum(b * b * v_c))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
b = np.array([0.3, 0.5, 0.0, -0.6, 0.2, -0.4])
b = b / np.linalg.norm(b)
vc = np.array([0.75, -1.27, 1.31, -1.56, 1.55, -0.78])
"""
    return [
        # --- Normal: generic bath and correlation potential ---
        {
            "setup": setup,
            "call": "impurity_chemical_potential(b, vc)",
            "gold_call": "_oracle_impurity_chemical_potential(b, vc)",
        },
        # --- Normal: bath with negative coefficients, different potential ---
        {
            "setup": setup + "b2 = np.array([0.0, -0.8, 0.0, 0.6, 0.0, 0.0])\nvc2 = np.array([2.0, -1.0, 0.5, 3.0, -2.0, 1.0])\n",
            "call": "impurity_chemical_potential(b2, vc2)",
            "gold_call": "_oracle_impurity_chemical_potential(b2, vc2)",
        },
        # --- Boundary: a site-independent correlation potential ---
        {
            "setup": setup,
            "call": "impurity_chemical_potential(b, 0.37 * np.ones(6))",
            "gold_call": "_oracle_impurity_chemical_potential(b, 0.37 * np.ones(6))",
        },
        # --- Edge: vanishing potential gives zero, and a constant shift of the
        # potential shifts the result by exactly that constant ---
        {
            "setup": setup,
            "call": "np.round([impurity_chemical_potential(b, np.zeros(6)), impurity_chemical_potential(b, vc + 1.25) - impurity_chemical_potential(b, vc)], 8)",
            "gold_call": "np.round([_oracle_impurity_chemical_potential(b, np.zeros(6)), _oracle_impurity_chemical_potential(b, vc + 1.25) - _oracle_impurity_chemical_potential(b, vc)], 8)",
        },
        # --- Invalid: mismatched lengths ---
        {
            "setup": setup + """
def run_model():
    try:
        impurity_chemical_potential(b, vc[:5])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_impurity_chemical_potential(b, vc[:5])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite potential entry ---
        {
            "setup": setup + """
vbad = vc.copy(); vbad[1] = np.nan
def run_model():
    try:
        impurity_chemical_potential(b, vbad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_impurity_chemical_potential(b, vbad)
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
