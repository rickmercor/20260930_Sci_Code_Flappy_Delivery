"""
Step 6 - the level histogram of the analyzed amplitudes.

The study instrument for finding the fluorescence steps is a histogram of the normalized amplitudes taken across the analyzed window (the activation phase, through the peak; see steps 01-02). 

Two conventions are fixed: the bin width (the study adopted 0.015 F/F0, justified from the background noise in step 05) and the portion of the record grouped in histogram (converged measurements only). 

The bin edges of this step are anchored at the resting-baseline level so the alignment is noise-independent, and the histogram spans exactly the sampled range.

Returns
-------
counts, centers : tuple of np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def level_histogram(
    amp_act: "numpy.ndarray",
    bin_width: float = 0.015,
    anchor: float = 1.0,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Bin the activation-phase amplitudes at the adopted width.

    Parameters
    ----------
    amp_act : numpy.ndarray
        Analyzed amplitudes in F/F0: the converged samples of the activation phase,
        through the peak (see steps 01-02).
    bin_width : float, optional
        Histogram bin width in F/F0 (the study's adopted value).
    anchor : float, optional
        Level the bin edges are anchored to (the resting baseline), so that bin
        alignment is independent of the record's noise.

    Returns
    -------
    tuple of numpy.ndarray
        ``(counts, centers)``: integer bin counts and the bin centers in F/F0.

    Raises
    ------
    ValueError
        If the input is empty or non-finite, or `bin_width` is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_level_histogram(
    amp_act: "numpy.ndarray",
    bin_width: float = 0.015,
    anchor: float = 1.0,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Reference implementation for level_histogram."""
    import numpy as np

    a = np.asarray(amp_act, dtype=float)
    if a.size == 0:
        raise ValueError("the activation phase must not be empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("the activation phase must be finite")
    for name, value in (("bin_width", bin_width), ("anchor", anchor)):
        if not np.isfinite(value) or (name == "bin_width" and value <= 0):
            raise ValueError(f"{name} must be finite" + (" and positive" if name == "bin_width" else ""))

    # Edges anchored at the baseline so the alignment is noise-independent.
    lo = float(anchor) + np.floor((float(np.min(a)) - float(anchor)) / bin_width) * bin_width
    hi = float(anchor) + np.ceil((float(np.max(a)) - float(anchor)) / bin_width) * bin_width
    edges = lo + bin_width * np.arange(int(round((hi - lo) / bin_width)) + 1)
    counts, edges = np.histogram(a, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, centers

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "amp_act = np.array([1.01, 1.06, 1.07, 1.072, 1.068, 1.14, 1.142, 1.141, 1.139,\n"
        "                    1.21, 1.213, 1.212, 1.36, 1.355, 1.356])\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        level_histogram(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_level_histogram(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: five resolved levels on a short record.
        {"setup": sized,
         "call": "level_histogram(amp_act)",
         "gold_call": "_oracle_level_histogram(amp_act)"},
        # Boundary: a single amplitude (one occupied bin).
        {"setup": "import numpy as np",
         "call": "level_histogram(np.array([1.07]))",
         "gold_call": "_oracle_level_histogram(np.array([1.07]))"},
        # Edge: symmetric levels around the anchor with a coarser bin.
        {"setup": "import numpy as np\nvals = np.array([0.98, 1.0, 1.02, 1.04])\n",
         "call": "level_histogram(vals, bin_width=0.03)",
         "gold_call": "_oracle_level_histogram(vals, bin_width=0.03)"},
        # Invalid: empty input.
        {"setup": invalid,
         "call": "run_model(amp_act=np.array([]))",
         "gold_call": "run_gold(amp_act=np.array([]))"},
        # Invalid: non-positive bin width.
        {"setup": invalid,
         "call": "run_model(amp_act=np.array([1.0, 1.1]), bin_width=0.0)",
         "gold_call": "run_gold(amp_act=np.array([1.0, 1.1]), bin_width=0.0)"},
    ]
