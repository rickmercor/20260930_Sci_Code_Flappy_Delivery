"""
Select one local likelihood maximum per injected trajectory.

The candidate stage keeps one representative of each local stack: exclude the outer pixel border, locate the first row-major maximum in the remaining $23\times23$ interior, and record its column, row, and significance.  This fixed local suppression prevents a border artifact or neighboring pixels of the same peak from entering the recovery list as separate candidates.

Returns
-------
`np.ndarray` of shape `(M,3)` with columns `(x_pix, y_pix, significance)` and one finite candidate per input map.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Select one local likelihood maximum per injected trajectory."""

import numpy as np


def deduplicated_candidates(significance_maps: np.ndarray) -> np.ndarray:
    """Return one ``(x, y, significance)`` candidate per coadded map."""
    return np.empty((0, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deduplicated_candidates(significance_maps: np.ndarray) -> np.ndarray:
    """Keep the highest peak after deterministic one-pixel local suppression."""
    import numpy as np

    maps = np.asarray(significance_maps, dtype=float)
    if maps.ndim != 3 or maps.shape[1] < 3 or maps.shape[2] < 3 or not np.all(np.isfinite(maps)):
        raise ValueError("invalid significance-map cube")
    selected = np.empty((maps.shape[0], 3), dtype=float)
    for source, image in enumerate(maps):
        interior = image[1:-1, 1:-1]
        row, col = np.unravel_index(np.argmax(interior), interior.shape)
        row, col = row + 1, col + 1
        selected[source] = [float(col), float(row), float(image[row, col])]
    return selected

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, tied-edge-excluded, and multi-map cases."""
    return [
        {"setup": "import numpy as np\nm=np.array([[[0.,0.,0.],[0.,3.,1.],[0.,1.,0.]]])", "call": "deduplicated_candidates(m)", "gold_call": "_oracle_deduplicated_candidates(m)"},
        {"setup": "import numpy as np\nm=np.array([[[9.,0.,0.,0.],[0.,2.,1.,0.],[0.,1.,2.,0.],[0.,0.,0.,8.]]])", "call": "deduplicated_candidates(m)", "gold_call": "_oracle_deduplicated_candidates(m)"},
        {"setup": "import numpy as np\nm=np.array([[[0.,0.,0.,0.],[0.,1.,4.,0.],[0.,2.,3.,0.],[0.,0.,0.,0.]],[[0.,0.,0.,0.],[0.,5.,1.,0.],[0.,1.,2.,0.],[0.,0.,0.,0.]]])", "call": "deduplicated_candidates(m)", "gold_call": "_oracle_deduplicated_candidates(m)"},
    ]
