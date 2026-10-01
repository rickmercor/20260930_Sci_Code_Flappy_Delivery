"""
Return the zone-averaged quantum metric of the occupied states along the axis the transition list was built for, from that list and the linear size of the grid it was built on. The source fixes a convention here that the natural reading does not; follow the source.

The quantum metric measures how quickly the occupied states change as the wavevector moves, and it is assembled from the same interband matrix elements that carry the optical weight, each divided by the square of its transition energy. What remains is to turn the sum over the sampled zone into an average over it.

Returns
-------
float, the zone-averaged quantum metric in Angstrom^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quantum_metric(transitions: "np.ndarray", n_grid: int) -> float:
    """Return the zone-averaged quantum metric of the occupied states along the axis the transition list was built for, from that list and the linear size of the grid it was built on. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    float, the zone-averaged quantum metric in Angstrom^2.

    Raises
    ------
    ValueError: if transitions is not a non-empty array of shape (N, 2), if n_grid is not positive, or if any transition energy is not positive.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_quantum_metric(transitions: "np.ndarray", n_grid: int) -> float:
    t = np.asarray(transitions, dtype=float)
    if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] == 0:
        raise ValueError("transitions must be a non-empty array of shape (N, 2)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if np.any(t[:, 0] <= 0.0):
        raise ValueError("every transition energy must be positive")
    # eq. (16): the sum over every listed transition divided by the NUMBER OF K POINTS --
    # not by the number of transitions, and not by the zone area.
    return float((t[:, 1] / t[:, 0] ** 2).sum() / (n_grid * n_grid))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])",
         "call": "quantum_metric(T, 3)",
         "gold_call": "_oracle_quantum_metric(T, 3)"},   # normal
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])",
         "call": "quantum_metric(T[:1], 1)",
         "gold_call": "_oracle_quantum_metric(T[:1], 1)"},   # boundary
        {"setup": "import numpy as np\nT=np.array([[1.5,0.83],[2.14,0.31],[3.77,1.92],[2.14,0.07],[5.6,0.44]])",
         "call": "quantum_metric(T*1e-4, 12)",
         "gold_call": "_oracle_quantum_metric(T*1e-4, 12)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        quantum_metric(np.array([[0.0,1.0]]), 3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_quantum_metric(np.array([[0.0,1.0]]), 3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
