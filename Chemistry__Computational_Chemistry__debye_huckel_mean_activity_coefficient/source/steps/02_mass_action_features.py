"""
Build mass-action monomial features from reactant stoichiometry.

Each candidate reaction contributes a rate feature defined by its reactant complex.

Returns
-------
Feature matrix with shape (n_times, n_reactions).
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mass_action_features(concentrations: np.ndarray, reactions: np.ndarray) -> np.ndarray:
    """Build mass-action monomial features from reactant stoichiometry.

    Parameters
    ----------
    concentrations : np.ndarray
        Nonnegative array with shape (n_times, n_species).
    reactions : np.ndarray
        Integer rows in [reactants | products] form.

    Returns
    -------
    np.ndarray
        Feature matrix with shape (n_times, n_reactions).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mass_action_features(concentrations: np.ndarray, reactions: np.ndarray) -> np.ndarray:
    """Reference mass-action feature construction."""
    c = np.asarray(concentrations, dtype=float)
    r = np.asarray(reactions)

    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
        raise ValueError("concentrations must be a nonempty 2D array")
    if r.ndim != 2 or r.shape[0] < 1 or r.shape[1] != 2 * c.shape[1]:
        raise ValueError("reactions must have shape (n_reactions, 2*n_species)")
    if not np.all(np.isfinite(c)) or np.any(c < 0.0):
        raise ValueError("concentrations must be finite and nonnegative")
    if not np.all(np.isfinite(r)) or np.any(r < 0) or not np.all(r == np.floor(r)):
        raise ValueError("reaction coefficients must be nonnegative integers")

    reactants = r[:, : c.shape[1]].astype(int)
    return np.prod(c[:, None, :] ** reactants[None, :, :], axis=2).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
concentrations = np.array([[2.0, 3.0], [4.0, 0.5]])
reactions = np.array([[1, 1, 0, 2], [2, 0, 0, 1]])""",
            "call": "mass_action_features(concentrations, reactions)",
            "gold_call": "_oracle_mass_action_features(concentrations, reactions)",
        },
        {
            "setup": """import numpy as np
concentrations = np.array([[0.0, 1.5, 2.0], [0.3, 0.0, 4.0]])
reactions = np.array([[0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0]])""",
            "call": "mass_action_features(concentrations, reactions)",
            "gold_call": "_oracle_mass_action_features(concentrations, reactions)",
        },
        {
            "setup": """import numpy as np
concentrations = np.array([[0.2, 0.4], [0.8, 0.1], [1.0, 1.0]])
reactions = np.array([[0, 0, 1, 0], [0, 2, 1, 0], [3, 0, 0, 1]])""",
            "call": "mass_action_features(concentrations, reactions)",
            "gold_call": "_oracle_mass_action_features(concentrations, reactions)",
        },
    ]
