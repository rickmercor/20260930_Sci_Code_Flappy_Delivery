"""
Compute the microscopic transition-state-theory rate constants associated with a specified enzyme kinetic mechanism from its state free energies. The calculation must use the thermodynamic barrier between the initial state and transition state for each elementary transition and the common transition-state-theory prefactor. Support both the four-state mechanism containing k1, k−1, and k2 and the extended mechanism containing the additional k−2 and k3 transitions.

The kinetic models represent an enzyme reaction as a sequence of thermodynamic states connected by elementary transitions. The paper assigns a relative Gibbs free energy to each state and converts the free-energy barrier of an elementary transition into a microscopic rate constant using transition-state theory. The common pre-exponential factor is A = kBT/h, while each rate is exponentially controlled by the corresponding activation free-energy difference. Because forward and reverse transitions have different initial states, their activation barriers must be evaluated relative to the state from which the transition originates. The extended catalytic cycle introduces two additional microscopic transitions after formation of the ES intermediate, so its rate representation contains more elementary rate constants than the simple cycle.

Returns
-------
np.ndarray, a two-dimensional floating-point array containing the microscopic rate constants for each input energy-state configuration, with columns [k1, k_minus1, k2] for the simple mechanism or [k1, k_minus1, k2, k_minus2, k3] for the extended mechanism.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_tst_rates(
    energies: "np.ndarray",
    T: float,
    R: float,
    kB: float,
    h: float,
    complex_model: bool,
) -> "np.ndarray":
    """
    Convert microscopic state energies into the source-defined transition
    rates for the selected kinetic representation.

    Parameters
    ----------
    energies : np.ndarray
        Microscopic state-energy array.

    T : float
        Temperature used by the source-defined rate construction.

    R : float
        Gas constant in the units required by the supplied energies.

    kB : float
        Boltzmann constant in the units required by the rate construction.

    h : float
        Planck constant in the units required by the rate construction.

    complex_model : bool
        Selects the reduced or complete kinetic representation.

    Returns
    -------
    np.ndarray
        Microscopic rate constants in the representation selected by
        ``complex_model``.

    Raises
    ------
    ValueError
        If the energy array, thermodynamic constants, or representation
        selector is invalid.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_tst_rates(
    energies: "np.ndarray",
    T: float,
    R: float,
    kB: float,
    h: float,
    complex_model: bool,
) -> "np.ndarray":
    E = np.asarray(energies, dtype=float)

    nstates = 6 if complex_model else 4
    if E.ndim != 2 or E.shape[1] != nstates or E.shape[0] < 1 or not np.all(np.isfinite(E)):
        raise ValueError("energies have invalid shape or values")
    if not all(np.isfinite(v) and v > 0 for v in (T, R, kB, h)):
        raise ValueError("physical constants must be finite and positive")
    A = kB * T / h
    if complex_model:
        barriers = np.column_stack((E[:, 1] - E[:, 0], E[:, 1] - E[:, 2], E[:, 3] - E[:, 2], E[:, 3] - E[:, 4], E[:, 5] - E[:, 4]))
    else:
        barriers = np.column_stack((E[:, 1] - E[:, 0], E[:, 1] - E[:, 2], E[:, 3] - E[:, 2]))
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        out = A * np.exp(-barriers / (R * T))
    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("rates are not finite positive")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nenergies=np.array([[0.,10.,-5.,11.]],float)\nargs=(energies,298.15,1.98720425864083e-3,1.380649e-23,6.62607015e-34,False)\n", "call": "compute_tst_rates(*args)", "gold_call": "_oracle_compute_tst_rates(*args)"},
        {"setup": "import numpy as np\nenergies=np.array([[0.,10.,-5.,11.,-9.,9.],[1.,9.,-4.,10.,-8.,8.]],float)\nargs=(energies,298.15,1.98720425864083e-3,1.380649e-23,6.62607015e-34,True)\n", "call": "compute_tst_rates(*args)", "gold_call": "_oracle_compute_tst_rates(*args)"},
        {"setup": "import numpy as np\nenergies=np.zeros((2,4),float)\nargs=(energies,300.,1e-3,1e-23,1e-34,False)\n", "call": "compute_tst_rates(*args)", "gold_call": "_oracle_compute_tst_rates(*args)"},
        {"setup": "import numpy as np\nenergies=np.zeros((1,5),float)\nargs=(energies,298.15,1.98720425864083e-3,1.380649e-23,6.62607015e-34,False)\ndef run_model():\n    try: compute_tst_rates(*args); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_compute_tst_rates(*args); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "run_model()", "gold_call": "run_gold()"},
    ]
