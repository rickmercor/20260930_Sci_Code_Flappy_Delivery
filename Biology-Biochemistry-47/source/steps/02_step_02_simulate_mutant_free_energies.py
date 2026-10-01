"""
Build the deterministic library of single-mutant reaction coordinates by perturbing every sub-state free energy of the wild type independently.

A substitution in an enzyme is represented here in the only way that keeps the model free of assumed interactions: it shifts the free energy of every state in the catalytic cycle by its own amount, and nothing else. Substitutions are known empirically to touch several states at once rather than one, so the perturbation applied to a variant is a full vector over the reaction coordinate and not a single scalar. Sampling those shifts independently and symmetrically about zero, over a window of a couple of kilocalories per mole, spans the range within which the overwhelming majority of measured single substitution effects fall while keeping the library free of any built in correlation between what a substitution does to a ground state and what it does to a transition state.




Determinism matters as much as the distribution. The library is generated once from a stated seed with numpy's default bit generator, drawn as a single block of independent uniform variates with one row per variant and one column per sub state in the order the reaction coordinate lists them. Any change to the draw order, the block shape or the generator would produce a different library, so the convention is part of the specification rather than an implementation detail.

Returns
-------
np.ndarray of shape (n_mutants, 6), float: the perturbed sub-state free energies of every single mutant in the library, in kcal/mol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def simulate_mutant_free_energies(wt_free_energies, n_mutants: int = 1000,
                                  amplitude: float = 2.0,
                                  seed: int = 314159) -> np.ndarray:
    """Generate the single-mutant reaction coordinates of the reference library.

    Parameters
    ----------
    wt_free_energies : array_like
        Six wild-type sub-state free energies in kcal/mol, in reaction
        coordinate order.
    n_mutants : int
        Number of single mutants in the library (n_mutants >= 1).
    amplitude : float
        Half-width in kcal/mol of the symmetric uniform window from which
        each sub-state perturbation is drawn (amplitude > 0).
    seed : int
        Seed of numpy's default bit generator (seed >= 0).

    Returns
    -------
    mutant_free_energies : np.ndarray
        Array of shape (n_mutants, 6) holding the perturbed sub-state free
        energies of each single mutant in kcal/mol.

    Raises
    ------
    ValueError
        If wt_free_energies does not hold six finite entries, or if
        n_mutants, amplitude or seed violates the constraints stated above.
    """
    return mutant_free_energies  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np

def _oracle_simulate_mutant_free_energies(wt_free_energies, n_mutants: int = 1000,
                                          amplitude: float = 2.0,
                                          seed: int = 314159) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    energies = np.asarray(wt_free_energies, dtype=float)
    if energies.shape != (6,):
        raise ValueError("wt_free_energies must have shape (6,)")
    if not np.all(np.isfinite(energies)):
        raise ValueError("wt_free_energies must be finite")
    if not (isinstance(n_mutants, (int, np.integer)) and not isinstance(n_mutants, bool)
            and int(n_mutants) >= 1):
        raise ValueError("n_mutants must be an integer >= 1")
    if not (isinstance(amplitude, (int, float, np.floating, np.integer))
            and not isinstance(amplitude, bool)
            and np.isfinite(amplitude) and float(amplitude) > 0.0):
        raise ValueError("amplitude must be a finite number > 0")
    if not (isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
            and int(seed) >= 0):
        raise ValueError("seed must be an integer >= 0")

    n_mutants = int(n_mutants)
    amplitude = float(amplitude)

    # One block draw, rows indexed by variant and columns by sub state in
    # reaction coordinate order, so the library is reproducible verbatim.
    generator = np.random.default_rng(int(seed))
    perturbations = generator.uniform(-amplitude, amplitude, size=(n_mutants, 6))

    return energies[None, :] + perturbations

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: F401  (keeps this field self-contained)
def test_cases():
    """Return list of test case specifications."""
    import numpy as np  # noqa: F401  (keeps this field self-contained)

    return [
        # --- Valid: a small slice of the reference library (normal scenario) ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
n_mutants = 25
""",
            "call": "simulate_mutant_free_energies(wt_free_energies, n_mutants)",
            "gold_call": "_oracle_simulate_mutant_free_energies(wt_free_energies, n_mutants)",
        },
        # --- Valid: a different seed and a wider perturbation window ---
        {
            "setup": """import numpy as np
wt_free_energies = np.array([1.0, 12.0, -6.0, 13.0, -8.0, 10.0])
n_mutants = 40
amplitude = 3.5
seed = 7
""",
            "call": "simulate_mutant_free_energies(wt_free_energies, n_mutants, amplitude, seed)",
            "gold_call": "_oracle_simulate_mutant_free_energies(wt_free_energies, n_mutants, amplitude, seed)",
        },
        # --- Boundary: a library of one variant drawn with seed zero ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
""",
            "call": "simulate_mutant_free_energies(wt_free_energies, 1, 2.0, 0)",
            "gold_call": "_oracle_simulate_mutant_free_energies(wt_free_energies, 1, 2.0, 0)",
        },
        # --- Edge: a vanishingly narrow window, so every variant is near wild type ---
        {
            "setup": """import numpy as np
wt_free_energies = [0.0, 10.0, -5.0, 11.0, -9.0, 9.0]
""",
            "call": "simulate_mutant_free_energies(wt_free_energies, 12, 1.0e-9, 2024)",
            "gold_call": "_oracle_simulate_mutant_free_energies(wt_free_energies, 12, 1.0e-9, 2024)",
        },
        # --- Invalid: wild type given with the wrong number of sub-states ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        simulate_mutant_free_energies([0.0, 10.0, -5.0], 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_simulate_mutant_free_energies([0.0, 10.0, -5.0], 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive perturbation amplitude ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        simulate_mutant_free_energies([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], 10, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_simulate_mutant_free_energies([0.0, 10.0, -5.0, 11.0, -9.0, 9.0], 10, 0.0)
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
