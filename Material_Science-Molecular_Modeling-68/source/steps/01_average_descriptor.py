"""
Implement average_descriptor, which averages per-atom descriptor vectors into a single system-level descriptor. ERBS treats the machine-learned interatomic potential's own atomic descriptor as a general-purpose reaction coordinate. Rather than biasing individual atoms, the method averages the per-atom descriptor vectors over the whole system to obtain a single system-level descriptor, which is later reduced via PCA into a low-dimensional collective variable.

ERBS builds collective variables from the same descriptor a machine-learned interatomic potential would use for energy/force prediction, rather than hand-picked physical coordinates. The first step in this pipeline is reducing a per-atom description of the system down to a single system-level vector by simple averaging, so that later PCA and kernel-density steps operate on one descriptor per configuration rather than one per atom.

Returns
-------
np.ndarray of shape (D, ), the system-level descriptor averaged over atoms, as a NumPy float array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def average_descriptor(G: np.ndarray) -> np.ndarray:
    '''Average per-atom descriptors into a single system-level descriptor.

    Parameters
    ----------
    G : np.ndarray
        Array of shape (N_atoms, D) containing one D-dimensional descriptor
        vector per atom.

    Returns
    -------
    s_prime : np.ndarray
        Array of shape (D,), the descriptor averaged over all atoms.

    Raises
    ------
    ValueError
        If G is not a 2D array, or if G has zero rows (N_atoms == 0), or if
        G contains any non-finite values (NaN or Inf).
    '''
    return s_prime  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_average_descriptor(G: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    G = np.asarray(G, dtype=float)
    if G.ndim != 2:
        raise ValueError("G must be a 2D array of shape (N_atoms, D)")
    if G.shape[0] == 0:
        raise ValueError("G must contain at least one atom (N_atoms >= 1)")
    if not np.all(np.isfinite(G)):
        raise ValueError("G must contain only finite values")
    return G.mean(axis=0).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal case: the actual task input (seed=11, 5 atoms x 12 dims)
            "setup": (
                "import numpy as np\n"
                "rng = np.random.default_rng(11)\n"
                "G = rng.standard_normal((5, 12))"
            ),
            "call": "average_descriptor(G)",
            "gold_call": "_oracle_average_descriptor(G)",
        },
        {
            # Boundary case: a single atom -- average equals that atom's descriptor
            "setup": (
                "import numpy as np\n"
                "G = np.array([[1.0, 2.0, 3.0]])"
            ),
            "call": "average_descriptor(G)",
            "gold_call": "_oracle_average_descriptor(G)",
        },
        {
            # Edge case: many atoms, some negative/zero values
            "setup": (
                "import numpy as np\n"
                "G = np.array([[0.0, -1.0], [0.0, 1.0], [0.0, 0.0]])"
            ),
            "call": "average_descriptor(G)",
            "gold_call": "_oracle_average_descriptor(G)",
        },
        {
            # Invalid-input case: 1D array should raise ValueError
           "setup": (
        "import numpy as np\n"
        "G = np.array([1.0, 2.0, 3.0])\n"
        "def run_model():\n"
        "    try:\n"
        "        average_descriptor(G)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_average_descriptor(G)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2"
    ),
    "call": "run_model()",
    "gold_call": "run_gold()",
},
    ]
