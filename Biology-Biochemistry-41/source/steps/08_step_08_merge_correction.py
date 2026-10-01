"""
Step 8 - the double-quantum (merge) correction of the step count.

Two release-channel groups can open nearly simultaneously. In that case a step in the record is twice as large as the channel-turnover quantum, and the study corrects for this: when two consecutive resolved step levels are separated by a gap that exceeds the typical inter-level spacing markedly (a double-quantum gap), that gap counts as two steps rather than one.

This step applies that counting rule and reports the per-step gap (step size) of the corrected staircase.

Returns
-------
step_count, step_size : tuple (int, float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_merge_correction(
    levels: "numpy.ndarray",
    factor: float = 1.5,
) -> "tuple[int, float]":
    """Apply the double-quantum merge correction to the resolved step levels.

    Parameters
    ----------
    levels : numpy.ndarray
        Sorted step-level positions in F/F0 (see step 07).
    factor : float, optional
        A consecutive-level gap strictly greater than `factor` times the median
        gap is read as two steps.

    Returns
    -------
    tuple
        ``(step_count, step_size)``: the merge-corrected number of release steps
        (int) and the mean corrected inter-level gap in F/F0 (float; 0.0 when a
        single level leaves no gap).

    Raises
    ------
    ValueError
        If `levels` is empty or non-finite, or `factor` is out of range.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_apply_merge_correction(
    levels: "numpy.ndarray",
    factor: float = 1.5,
) -> "tuple[int, float]":
    """Reference implementation for apply_merge_correction."""
    import numpy as np

    lv = np.asarray(levels, dtype=float)
    if lv.size == 0:
        raise ValueError("at least one level is required")
    if not np.all(np.isfinite(lv)):
        raise ValueError("levels must be finite")
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError("factor must exceed 1")

    if lv.size == 1:
        return int(lv.size), 0.0
    gaps = np.diff(np.sort(lv))
    m = float(np.median(gaps))
    corrected = []
    for g in gaps:
        if g > factor * m:
            corrected.extend([g / 2.0, g / 2.0])
        else:
            corrected.append(float(g))
    corrected = np.asarray(corrected)
    # count = levels + one extra per split gap
    step_count = int(lv.size + sum(1 for g in gaps if g > factor * m))
    step_size = float(np.mean(corrected))
    return step_count, step_size

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "lv6 = np.array([1.0300, 1.1020, 1.1750, 1.3230, 1.3960, 1.4700])\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        apply_merge_correction(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_apply_merge_correction(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a six-level record with a double-quantum third gap (0.148 > 1.5 x
        # the 0.073 median -> split -> 7).
        {"setup": sized,
         "call": "apply_merge_correction(lv6)",
         "gold_call": "_oracle_apply_merge_correction(lv6)"},
        # Boundary: a single level (no gap to split; step size is 0.0).
        {"setup": "import numpy as np",
         "call": "apply_merge_correction(np.array([1.07]))",
         "gold_call": "_oracle_apply_merge_correction(np.array([1.07]))"},
        # Edge: a clean four-level staircase with no merged gap.
        {"setup": "import numpy as np",
         "call": "apply_merge_correction(np.array([1.071, 1.142, 1.213, 1.284]))",
         "gold_call": "_oracle_apply_merge_correction(np.array([1.071, 1.142, 1.213, 1.284]))"},
        # Invalid: empty levels.
        {"setup": invalid,
         "call": "run_model(levels=np.array([]))",
         "gold_call": "run_gold(levels=np.array([]))"},
        # Invalid: factor at or below one (the rule degenerates).
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(levels=lv6, factor=1.0)",
         "gold_call": "run_gold(levels=lv6, factor=1.0)"},
    ]
