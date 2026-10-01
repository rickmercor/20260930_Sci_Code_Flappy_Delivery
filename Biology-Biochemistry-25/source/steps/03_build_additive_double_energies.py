"""
Construct the microscopic free-energy configurations of all specified double mutants by combining the free-energy perturbations of two single mutants under the additive microscopic null model. Preserve the wild-type reference energies and generate one deterministic configuration for every requested mutant pair.

The paper separates microscopic additivity from macroscopic epistasis. Under the microscopic null assumption, two mutations do not introduce an explicit energetic interaction: their effects on a given catalytic-cycle state are summed relative to the wild-type state. Thus, for a state with wild-type free energy Gwt and single-mutant perturbations ΔGi and ΔGj, the corresponding double-mutant state is represented by Gwt + ΔGi + ΔGj. Apparent epistasis can nevertheless emerge after these additive state energies are transformed into microscopic rates and subsequently into nonlinear kinetic parameters. This construction therefore provides the mechanistic null against which macroscopic interaction factors are evaluated.

Returns
-------
np.ndarray, a two-dimensional floating-point array of shape (number_of_pairs, number_of_states), containing the additive double-mutant free-energy configurations in the same state ordering as the supplied wild-type energies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_additive_double_energies(
    wild_type_energies: "np.ndarray",
    mutation_deltas: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    """
    Construct microscopic double-mutant state energies from the wild-type
    reference and the supplied mutation perturbations.

    Parameters
    ----------
    wild_type_energies : np.ndarray
        Wild-type microscopic state energies.

    mutation_deltas : np.ndarray
        Mutation perturbation vectors.

    pair_indices : np.ndarray
        Integer pairs identifying the two mutations to combine.

    Returns
    -------
    np.ndarray
        Microscopic energies for the requested double-mutant states.

    Raises
    ------
    ValueError
        If the wild-type energies, perturbation matrix, or pair indices
        have invalid shapes or contain invalid values.
    """
    return double_energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_additive_double_energies(
    wild_type_energies: "np.ndarray",
    mutation_deltas: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    wt = np.asarray(wild_type_energies, dtype=float)
    d = np.asarray(mutation_deltas, dtype=float)
    p = np.asarray(pair_indices)
    if wt.ndim != 1 or d.ndim != 2 or d.shape[1] != wt.size or wt.size not in (4, 6) or not np.all(np.isfinite(wt)) or not np.all(np.isfinite(d)):
        raise ValueError("invalid energies")
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 1 or not np.issubdtype(p.dtype, np.integer) or np.any(p < 0) or np.any(p >= d.shape[0]):
        raise ValueError("invalid pairs")
    out = wt[None, :] + d[p[:, 0]] + d[p[:, 1]]
    return out.astype(float, copy=False)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nwt=np.array([0.,10.,-5.,11.])\nd=np.array([[1.,2.,3.,4.],[-1.,0.,2.,-2.],[.5,.5,.5,.5]])\np=np.array([[0,1],[1,2]],int)\n", "call": "build_additive_double_energies(wt,d,p)", "gold_call": "_oracle_build_additive_double_energies(wt,d,p)"},
        {"setup": "import numpy as np\nwt=np.array([0.,10.,-5.,11.,-9.,9.])\nd=np.zeros((2,6))\np=np.array([[0,0]],int)\n", "call": "build_additive_double_energies(wt,d,p)", "gold_call": "_oracle_build_additive_double_energies(wt,d,p)"},
        {"setup": "import numpy as np\nwt=np.array([0.,10.,-5.,11.])\nd=np.array([[1e-9]*4,[-1e-9]*4])\np=np.array([[0,1]],int)\n", "call": "build_additive_double_energies(wt,d,p)", "gold_call": "_oracle_build_additive_double_energies(wt,d,p)"},
        {"setup": "import numpy as np\nwt=np.zeros(4); d=np.zeros((2,4)); p=np.array([[0,2]],int)\ndef run_model():\n    try: build_additive_double_energies(wt,d,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_build_additive_double_energies(wt,d,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "run_model()", "gold_call": "run_gold()"},
    ]
