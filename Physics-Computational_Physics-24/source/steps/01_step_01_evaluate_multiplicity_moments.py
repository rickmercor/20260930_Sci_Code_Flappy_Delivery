"""
Reduce a tabulated prompt-fission multiplicity distribution to the two moments that the point-kinetic and stochastic descriptions of the neutron balance both need.

A fission emits a random number of prompt neutrons while removing the incident one, so the mean multiplicity sets the deterministic production rate while the second moment of the net gain sets the variance the neutron population inherits from fission. Tabulating the multiplicity by abundance rather than by its mean alone is what allows the two to be separated.

Returns
-------
np.ndarray of shape (2,), float: the mean prompt multiplicity and the mean square net prompt gain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_multiplicity_moments(multiplicity: np.ndarray,
                                  abundance: np.ndarray) -> np.ndarray:
    """Reduce a prompt-fission multiplicity table to its two required moments.

    Parameters
    ----------
    multiplicity : np.ndarray
        One-dimensional array of the distinct prompt-neutron multiplicities of a
        fission event, each a non-negative integer value.
    abundance : np.ndarray
        One-dimensional array of the same length holding the probability of each
        multiplicity; the entries are non-negative and sum to one.

    Returns
    -------
    moments : np.ndarray
        Array of shape (2,), holding in order the mean prompt multiplicity and
        the mean square of the net prompt-neutron gain of a fission, both
        dimensionless.

    Raises
    ------
    ValueError
        If either input is not a one-dimensional, non-empty array of finite
        entries, if the two arrays differ in length, if any ``multiplicity``
        entry is negative or is not an integer value, if any ``abundance`` entry
        is negative, or if the ``abundance`` entries do not sum to one within
        1e-10.
    """
    return moments  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_multiplicity_moments(multiplicity: np.ndarray,
                                          abundance: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    arrays = [np.asarray(a, dtype=float) for a in (multiplicity, abundance)]
    for name, array in zip(("multiplicity", "abundance"), arrays):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    counts, weights = arrays
    if counts.size != weights.size:
        raise ValueError("multiplicity and abundance must have the same length")
    if np.any(counts < 0.0):
        raise ValueError("multiplicity entries must be non-negative")
    if not np.allclose(counts, np.round(counts), rtol=0.0, atol=1e-12):
        raise ValueError("multiplicity entries must be integer valued")
    if np.any(weights < 0.0):
        raise ValueError("abundance entries must be non-negative")
    if abs(float(weights.sum()) - 1.0) > 1e-10:
        raise ValueError("abundance entries must sum to one")

    # Mean multiplicity fixes the deterministic prompt production per fission.
    mean_multiplicity = float(weights @ counts)
    # A fission consumes the incident neutron, so the net gain is nu_p - 1 and
    # its mean square is what the neutron noise amplitude is built from.
    mean_square_net_gain = float(weights @ (counts - 1.0) ** 2)

    return np.array([mean_multiplicity, mean_square_net_gain], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the six-point testbed multiplicity table (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
multiplicity = np.arange(6.0)
abundance = np.array([0.027, 0.158, 0.339, 0.305, 0.133, 0.038])
""",
            "call": "sig(evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
            "gold_call": "sig(_oracle_evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
        },
        # --- Valid: a heavier, wider table with a non-contiguous support ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
multiplicity = np.array([0.0, 2.0, 3.0, 5.0, 8.0])
abundance = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
""",
            "call": "sig(evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
            "gold_call": "sig(_oracle_evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
        },
        # --- Boundary: a deterministic multiplicity of exactly one, for which the
        #     net gain and hence its mean square vanish identically ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
multiplicity = np.array([1.0])
abundance = np.array([1.0])
""",
            "call": "sig(evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
            "gold_call": "sig(_oracle_evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
        },
        # --- Edge: a table carrying a zero-probability entry at a large
        #     multiplicity, which must not shift either moment ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
multiplicity = np.array([0.0, 1.0, 2.0, 40.0])
abundance = np.array([0.1, 0.3, 0.6, 0.0])
""",
            "call": "sig(evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
            "gold_call": "sig(_oracle_evaluate_multiplicity_moments(multiplicity, abundance), 1.0)",
        },
        # --- Invalid: abundances that do not form a probability distribution ---
        {
            "setup": """import numpy as np
multiplicity = np.arange(4.0)
abundance = np.array([0.2, 0.2, 0.2, 0.2])
def run_model():
    try:
        evaluate_multiplicity_moments(multiplicity, abundance)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_multiplicity_moments(multiplicity, abundance)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a fractional multiplicity, which no fission event can have ---
        {
            "setup": """import numpy as np
multiplicity = np.array([0.0, 1.5, 3.0])
abundance = np.array([0.25, 0.5, 0.25])
def run_model():
    try:
        evaluate_multiplicity_moments(multiplicity, abundance)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_multiplicity_moments(multiplicity, abundance)
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
