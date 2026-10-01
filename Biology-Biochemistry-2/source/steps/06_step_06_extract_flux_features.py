"""
Peak and steady flux summarizers of one ensemble-averaged time course.

The treatment evaluates emergent quantities from the ensemble average, after averaging - not from individual runs. On the pinned 0.02 ms grid over [0, 100] ms, the emergent orders are: the peak flux, the maximum of the averaged course, with no smoothing; and the steady flux, the mean of the trace's last 5 ms (the final 250 grid points). This step returns the pair as a float array. The conventions are pinned here so every downstream comparison is mechanical; nothing about the model itself is decided in this step.

Returns
-------
features : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def extract_flux_features(flux_time_course: np.ndarray) -> np.ndarray:
    """Peak and steady flux of one ensemble-averaged trace.

    Parameters
    ----------
    flux_time_course : np.ndarray
        Shape (n,), the ensemble-averaged flux on the uniform 0.02 ms grid.

    Returns
    -------
    np.ndarray
        Shape (2,), native floats: [peak, steady], where peak is the trace maximum (no
        smoothing) and steady is the mean of the last 5 ms of the trace.

    Raises
    ------
    ValueError
        If the input is not a one-dimensional finite array with at least one full steady
        window of points.
    """
    features = np.empty(2, dtype=float)
    return features  # placeholder to complete

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_extract_flux_features(flux_time_course):
    """Reference implementation for extract_flux_features."""
    import numpy as np

    DT = 0.02
    STEADY_TAIL_MS = 5.0
    f = np.asarray(flux_time_course, dtype=float)
    if f.ndim != 1:
        raise ValueError("flux_time_course must be one-dimensional")
    if not np.all(np.isfinite(f)):
        raise ValueError("flux_time_course must be finite")
    tail = int(round(STEADY_TAIL_MS / DT))
    if f.size < tail:
        raise ValueError("trace must contain at least one steady window")
    return np.array([float(f.max()), float(f[-tail:].mean())])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    invalid = setup + (
        "def run_model(x):\n"
        "    try:\n"
        "        extract_flux_features(x)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(x):\n"
        "    try:\n"
        "        _oracle_extract_flux_features(x)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    traces = [
        # Normal: a synthetic peak-then-decay course on the grid.
        "t = np.arange(5001) * 0.02; x = 20.0 * np.exp(-t / 40.0) * (1.0 - np.exp(-t / 5.0))\n",
        # Boundary: a strictly flat course (peak == steady == value).
        "x = np.full(5001, 3.25)\n",
        # Edge: a course whose maximum lives in one spike cell.
        "x = np.zeros(5001); x[2501] = 7.5; x[-250:].fill(2.0)\n",
    ]
    return [
        # Normal: a synthetic peak-then-decay course on the grid.
        {"setup": setup + traces[0],
         "call": "extract_flux_features(x)",
         "gold_call": "_oracle_extract_flux_features(x)"},
        # Boundary: a strictly flat course (peak == steady == value).
        {"setup": setup + traces[1],
         "call": "extract_flux_features(x)",
         "gold_call": "_oracle_extract_flux_features(x)"},
        # Edge: a course whose maximum lives in a single spike cell.
        {"setup": setup + traces[2],
         "call": "extract_flux_features(x)",
         "gold_call": "_oracle_extract_flux_features(x)"},
    ] + [
        # Invalid: NaN in the trace.
        {"setup": invalid + "x = np.zeros(5001); x[10] = np.nan\n",
         "call": "run_model(x)",
         "gold_call": "run_gold(x)"},
        # Invalid: trace shorter than a steady window.
        {"setup": invalid + "x = np.ones(100)\n",
         "call": "run_model(x)",
         "gold_call": "run_gold(x)"},
        # Invalid: two-dimensional input must not be flattened implicitly.
        {"setup": invalid + "x = np.ones((25, 10))\n",
         "call": "run_model(x)",
         "gold_call": "run_gold(x)"},
    ]
