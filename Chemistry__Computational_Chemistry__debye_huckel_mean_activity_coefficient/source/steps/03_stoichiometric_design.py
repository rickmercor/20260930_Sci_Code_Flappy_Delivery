"""
Convert reaction features into species-balance design columns.

The stoichiometric matrix links each reaction rate to the derivative of every species.

Returns
-------
Time-major design matrix with shape (n_times*n_species, n_reactions).
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stoichiometric_design(
    features: np.ndarray,
    reactions: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Convert reaction features into species-balance design columns.

    Parameters
    ----------
    features : np.ndarray
        Mass-action features with shape (n_times, n_reactions).
    reactions : np.ndarray
        Integer rows in [reactants | products] form.
    derivative_scales : np.ndarray
        Positive maximum derivative magnitude for each species.

    Returns
    -------
    np.ndarray
        Time-major design matrix with shape (n_times*n_species, n_reactions).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_stoichiometric_design(
    features: np.ndarray,
    reactions: np.ndarray,
    derivative_scales: np.ndarray,
) -> np.ndarray:
    """Reference normalized stoichiometric design construction."""
    phi = np.asarray(features, dtype=float)
    r = np.asarray(reactions, dtype=float)
    scales = np.asarray(derivative_scales, dtype=float)

    if phi.ndim != 2 or phi.shape[0] < 1 or phi.shape[1] < 1:
        raise ValueError("features must be a nonempty 2D array")
    if r.ndim != 2 or r.shape[0] != phi.shape[1] or r.shape[1] % 2 != 0:
        raise ValueError("reaction dimensions do not match features")

    n_species = r.shape[1] // 2
    if scales.shape != (n_species,) or np.any(scales <= 0.0):
        raise ValueError("derivative_scales must contain one positive value per species")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(scales)):
        raise ValueError("inputs must be finite")

    stoichiometry = r[:, n_species:] - r[:, :n_species]
    tensor = np.einsum("tr,rs->tsr", phi, stoichiometry)
    tensor = tensor / scales[None, :, None]
    return tensor.reshape(phi.shape[0] * n_species, phi.shape[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
features = np.array([[2.0, 1.0], [3.0, 4.0]])
reactions = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
derivative_scales = np.array([2.0, 4.0])""",
            "call": "stoichiometric_design(features, reactions, derivative_scales)",
            "gold_call": "_oracle_stoichiometric_design(features, reactions, derivative_scales)",
        },
        {
            "setup": """import numpy as np
features = np.array([[0.0], [1.5], [2.5]])
reactions = np.array([[2, 0, 0, 1]])
derivative_scales = np.array([3.0, 1.0])""",
            "call": "stoichiometric_design(features, reactions, derivative_scales)",
            "gold_call": "_oracle_stoichiometric_design(features, reactions, derivative_scales)",
        },
        {
            "setup": """import numpy as np
features = np.array([[1.0, 2.0, 0.5]])
reactions = np.array([[1,0,0, 0,1,0], [0,1,0, 0,0,1], [0,0,1, 1,0,0]])
derivative_scales = np.array([1.0, 2.0, 5.0])""",
            "call": "stoichiometric_design(features, reactions, derivative_scales)",
            "gold_call": "_oracle_stoichiometric_design(features, reactions, derivative_scales)",
        },
    ]
