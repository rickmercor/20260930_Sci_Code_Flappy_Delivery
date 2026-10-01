"""
Run the whole pipeline and return the reference energy plus the complete second order correction plus nuclear repulsion.

The final step drives the whole calculation. It builds the primitive overlap,

core Hamiltonian and repulsion integrals, orthogonalises symmetrically and

forms the bond space orbitals, transforms the integrals into that basis,

iterates the stationarity condition for the pair amplitudes, builds the

generalised Fock matrix that the single excitation couplings need, evaluates the

reference energy, adds the second order contributions of the single, double and

pair transfer families, and finally adds the nuclear repulsion of the fixed

nuclei. The result is a total energy that should sit below both the restricted

Hartree-Fock energy of the same basis and the pair reference alone, while

staying above the full configuration interaction limit.

Returns
-------
float, the corrected total energy in hartree including nuclear repulsion, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def total_pp_en2_energy(centres: np.ndarray = (0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6),
                        exponent: float = 0.30, charges: np.ndarray = (1.0,) * 8,
                        n_units: int = 4) -> float:
    '''Corrected total energy of the pair reference.

    Parameters
    ----------
    centres : np.ndarray
        Positions of the nuclei along a line, in bohr, an even number of them.
    exponent : float
        Gaussian exponent shared by every primitive, strictly positive.
    charges : np.ndarray
        Nuclear charge of each centre.
    n_units : int
        Number of valence bond spaces, half the number of centres.

    Returns
    -------
    energy : float
        Reference energy plus the complete second order correction plus
        nuclear repulsion, in hartree.
    
    Raises
    ------
    ValueError
        If `centres` does not hold an even number of distinct positions, if
        `exponent` is not positive, if `charges` does not hold one entry per
        centre, or if `n_units` is not half the number of centres.
'''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_total_pp_en2_energy(centres=(0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6),
                                exponent=0.30, charges=(1.0,) * 8, n_units=4):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must hold an even number of positions")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    if np.size(charges) != R.size:
        raise ValueError("charges must hold one entry per centre")
    if 2 * int(n_units) != R.size:
        raise ValueError("n_units must be half the number of centres")
    stack = _oracle_ao_overlap_and_core(centres, exponent, charges)
    overlap, core = stack[0], stack[1]
    eri = _oracle_ao_two_electron_integrals(centres, exponent)
    C = _oracle_vbs_orbital_coefficients(overlap, n_units)
    fam = _oracle_pair_integral_families(C, core, eri)
    gaps = _oracle_optimal_pair_gaps(fam, n_units)
    fock = _oracle_generalized_fock_matrix(C, core, eri, gaps)
    reference = _oracle_pp_reference_energy(fam, gaps)
    singles = _oracle_single_excitation_en2(C, core, eri, gaps, fock)
    doubles = _oracle_double_excitation_en2(C, core, eri, gaps)
    pairs = _oracle_pair_transfer_en2(C, core, eri, gaps)
    correction = singles + doubles + pairs
    R = np.asarray(centres, dtype=float)
    Z = np.asarray(charges, dtype=float)
    repulsion = 0.0
    for i in range(R.size):
        for j in range(i + 1, R.size):
            repulsion += Z[i] * Z[j] / abs(R[i] - R[j])
    return float(reference + correction + repulsion)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6])
Z = np.ones(8)
""",
            "call": "round(total_pp_en2_energy(X, 0.30, Z, 4), 9)",
            "gold_call": "round(_oracle_total_pp_en2_energy(X, 0.30, Z, 4), 9)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4])
Z = np.ones(6)
""",
            "call": "round(total_pp_en2_energy(X, 0.30, Z, 3), 9)",
            "gold_call": "round(_oracle_total_pp_en2_energy(X, 0.30, Z, 3), 9)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 2.8, 6.2, 8.9])
Z = np.ones(4)
""",
            "call": "round(total_pp_en2_energy(X, 0.85, Z, 2), 9)",
            "gold_call": "round(_oracle_total_pp_en2_energy(X, 0.85, Z, 2), 9)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 6.0, 14.0, 20.0])
Z = np.ones(4)
""",
            "call": "round(total_pp_en2_energy(X, 0.30, Z, 2), 9)",
            "gold_call": "round(_oracle_total_pp_en2_energy(X, 0.30, Z, 2), 9)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4])
Z = np.ones(6)
""",
            "call": "round(total_pp_en2_energy(X, 0.26, Z, 3), 9)",
            "gold_call": "round(_oracle_total_pp_en2_energy(X, 0.26, Z, 3), 9)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4])
Z = np.ones(6)

def run_model():
    try:
        total_pp_en2_energy(X, 0.30, Z, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_total_pp_en2_energy(X, 0.30, Z, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 3.0, 9.0])
Z = np.ones(4)

def run_model():
    try:
        total_pp_en2_energy(X, 0.30, Z, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_total_pp_en2_energy(X, 0.30, Z, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4])
Z = np.ones(6)

def run_model():
    try:
        total_pp_en2_energy(X, 0.0, Z, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_total_pp_en2_energy(X, 0.0, Z, 3)
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
