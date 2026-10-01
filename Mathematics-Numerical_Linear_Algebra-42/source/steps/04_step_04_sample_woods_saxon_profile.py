"""
Sample the positive Woods--Saxon weight on the reduced half-line grid.



The generalized Sturm--Liouville pencil uses the decaying profile

`$q(x) = 1 / (1 + exp((x - radius) / diffuseness))$`.  A stable logistic

evaluation is required because large Laguerre nodes can make the raw

exponential overflow even though the mathematical profile remains finite.

Returns
-------
one finite float array with shape (m,) and every entry strictly between zero and one
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_woods_saxon_profile(
    physical_nodes: np.ndarray, radius: float, diffuseness: float
) -> np.ndarray:
    """Return Woods--Saxon samples at positive physical nodes.

    ``physical_nodes`` must be a nonempty, finite, strictly increasing vector
    of positive values.  ``radius`` and ``diffuseness`` must be finite positive
    scalars.  Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    physical_nodes : np.ndarray
        Positive physical nodes, shape ``(m,)``.
    radius : float
        Positive profile radius.
    diffuseness : float
        Positive surface thickness.

    Returns
    -------
    np.ndarray
        Finite profile values in ``(0, 1)``, shape ``(m,)``.
    """
    return potential  # noqa: F821 - model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special

def _oracle_sample_woods_saxon_profile(
    physical_nodes: np.ndarray, radius: float, diffuseness: float
) -> np.ndarray:
    """Reference stable Woods--Saxon sampling."""
    x = np.asarray(physical_nodes, dtype=float)
    if (
        x.ndim != 1
        or x.size == 0
        or not np.all(np.isfinite(x))
        or np.any(x <= 0.0)
        or np.any(np.diff(x) <= 0.0)
    ):
        raise ValueError("physical_nodes must be finite, positive, and increasing")
    if not np.isscalar(radius) or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if (
        not np.isscalar(diffuseness)
        or not np.isfinite(diffuseness)
        or diffuseness <= 0.0
    ):
        raise ValueError("diffuseness must be positive and finite")

    potential = special.expit(-(x - float(radius)) / float(diffuseness))
    if (
        not np.all(np.isfinite(potential))
        or np.any(potential <= 0.0)
        or np.any(potential >= 1.0)
    ):
        raise ValueError("profile samples must lie strictly between zero and one")
    return potential

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, extreme-tail, altered-profile, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
physical_nodes = np.array([0.05, 1.5, 5.08685476, 12.0])
radius = 5.08685476
diffuseness = 0.929852862
""",
            "call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(sample_woods_saxon_profile(physical_nodes, radius, diffuseness))",
            "gold_call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(_oracle_sample_woods_saxon_profile(physical_nodes, radius, diffuseness))",
        },
        {
            "setup": """import numpy as np
physical_nodes = np.array([1e-12, 1000.0])
radius = 4.0
diffuseness = 25.0
""",
            "call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(sample_woods_saxon_profile(physical_nodes, radius, diffuseness))",
            "gold_call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(_oracle_sample_woods_saxon_profile(physical_nodes, radius, diffuseness))",
        },
        {
            "setup": """import numpy as np
physical_nodes = np.array([0.2, 0.7, 2.5])
radius = 0.35
diffuseness = 0.07
""",
            "call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(sample_woods_saxon_profile(physical_nodes, radius, diffuseness))",
            "gold_call": "(lambda value: float(np.dot(value, np.arange(1, value.size + 1, dtype=float))))(_oracle_sample_woods_saxon_profile(physical_nodes, radius, diffuseness))",
        },
        {
            "setup": """import numpy as np
physical_nodes = np.array([0.2, 0.7, 2.5])
radius = 1.0
diffuseness = 0.0
def run_model():
    try:
        sample_woods_saxon_profile(physical_nodes, radius, diffuseness)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_woods_saxon_profile(physical_nodes, radius, diffuseness)
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
