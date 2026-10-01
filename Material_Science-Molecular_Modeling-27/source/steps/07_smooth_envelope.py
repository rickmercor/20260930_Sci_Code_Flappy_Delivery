"""
Regularise the half-width profile along the progress coordinate into the continuous envelope the restraint reads, by a centred moving average over an odd number of nodes. Replace any node that carries no value by the mean of the nodes that do before averaging; extend the profile beyond each end by its mirror image about the end node with the end node included, so that the point one step outside an end carries the end value itself (numpy's symmetric padding, not the reflect padding that omits the end value), and the averaged profile keeps the same number of nodes; return the profile unchanged when the window is a single node.

The source states that the update is regularised by smoothing along the progress coordinate, because the restraint needs an envelope with bounded gradients rather than a node-by-node estimate. The smoothing is bookkeeping, not physics, and it is written down here so that the envelope the audit integrates is reproducible rather than left to taste.

Returns
-------
A (n_stations,) float64 array, the smoothed envelope.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smooth_envelope(profile: "np.ndarray", window: int) -> "np.ndarray":
    """Regularise the half-width profile along the progress coordinate into the continuous envelope the
    restraint reads, by a centred moving average over an odd number of nodes. Replace any node that
    carries no value by the mean of the nodes that do before averaging; extend the profile beyond
    each end by its mirror image about the end node with the end node included, so that the point
    one step outside an end carries the end value itself (numpy's symmetric padding, not the reflect
    padding that omits the end value), and the averaged profile keeps the same number of nodes;
    return the profile unchanged when the window is a single node.

    Args:
        profile: array-like of shape (n_stations,), the half-width profile; entries may be NaN where
            no value was found.
        window: int, a positive odd number of nodes for the centred moving average.

    Returns:
        A (n_stations,) float64 array, the smoothed envelope.

    Raises:
        ValueError: if window is not a positive odd integer.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_smooth_envelope(profile: "np.ndarray", window: int) -> "np.ndarray":
    profile = np.asarray(profile, dtype=float).ravel()
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1:
        return profile.copy()
    filled = np.nan_to_num(profile, nan=float(np.nanmean(profile)))
    h = window//2
    padded = np.concatenate([filled[:h][::-1], filled, filled[-h:][::-1]])
    return np.convolve(padded, np.ones(window)/window, mode="valid")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nprofile = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])\nwindow = 3\n',
         'call': 'smooth_envelope(profile, window)',
         'gold_call': '_oracle_smooth_envelope(profile, window)'},
        {'setup': 'import numpy as np\nprofile = np.array([0.5, np.nan, 0.9, 1.4, 0.2, 0.3, 0.8, 1.1, 0.4])\nwindow = 5\n',
         'call': 'smooth_envelope(profile, window)',
         'gold_call': '_oracle_smooth_envelope(profile, window)'},
        {'setup': 'import numpy as np\nprofile = np.linspace(0.01, 0.2, 41)**2\nwindow = 1\n',
         'call': 'smooth_envelope(profile, window)',
         'gold_call': '_oracle_smooth_envelope(profile, window)'},
    ]
