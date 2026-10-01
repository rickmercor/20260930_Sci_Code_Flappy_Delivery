"""
Step 5 - the histogram scale: Scott's rule at the study's background noise.

The bin width of the level histogram is not a free choice: the study fixes it from the noise of its own raw, spark-free background through Scott normal-reference rule, 3.49 * sigma * n^(-1/3), evaluated at the number of samples actually analyzed, and then adopts the member of its tested grid that the rule lands on. 

The raw background noise is a property of the acquisition, not of the record handed to an analysis: a record that has been spatially fitted carries a much smaller residual scatter, and running the rule on that scatter instead would produce a far finer bin than the study's. 

This step evaluates the rule at a supplied background noise and sample count and reports both the raw value and the grid member it selects.

Returns
-------
scott_raw, adopted : tuple of float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reference_bin_width(
    sigma_background: float,
    n_active: int,
    tested: "numpy.typing.ArrayLike" = (0.015, 0.020, 0.025),
) -> "tuple[float, float]":
    """Evaluate Scott's normal-reference rule and select the tested bin it lands on.

    Parameters
    ----------
    sigma_background : float
        Standard deviation of the acquisition's raw spark-free background, in F/F0.
    n_active : int
        Number of samples in the analyzed segment.
    tested : sequence of float, optional
        The candidate bin widths the study tested, in F/F0.

    Returns
    -------
    tuple of float
        ``(scott_raw, adopted)``: Scott's normal-reference bin width
        ``3.49 * sigma_background * n_active ** (-1/3)``, and the member of `tested`
        closest to it. Distance is absolute; an exact tie selects the smaller member.

    Raises
    ------
    ValueError
        If `sigma_background` is not positive and finite, `n_active` is not a positive
        integer, or `tested` is empty or holds a value that is not positive and finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reference_bin_width(
    sigma_background: float,
    n_active: int,
    tested: "numpy.typing.ArrayLike" = (0.015, 0.020, 0.025),
) -> "tuple[float, float]":
    """Reference implementation for reference_bin_width."""
    import numpy as np

    if not np.isfinite(sigma_background) or sigma_background <= 0:
        raise ValueError("sigma_background must be positive and finite")
    if int(n_active) != n_active or n_active <= 0:
        raise ValueError("n_active must be a positive integer")
    grid = np.asarray(tested, dtype=float)
    if grid.size == 0:
        raise ValueError("tested must not be empty")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("tested bin widths must be positive and finite")

    scott_raw = float(3.49 * float(sigma_background)
                      * float(n_active) ** (-1.0 / 3.0))
    order = np.lexsort((grid, np.abs(grid - scott_raw)))
    adopted = float(grid[order[0]])
    return scott_raw, adopted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        reference_bin_width(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_reference_bin_width(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a raw acquisition background over a few hundred analyzed samples.
        {"setup": "import numpy as np",
         "call": "reference_bin_width(0.0346, 400)",
         "gold_call": "_oracle_reference_bin_width(0.0346, 400)"},
        # Normal: a longer segment lowers the rule and can change the grid member.
        {"setup": "import numpy as np",
         "call": "reference_bin_width(0.0346, 1600)",
         "gold_call": "_oracle_reference_bin_width(0.0346, 1600)"},
        # Edge: a noisier background selects the coarsest tested member.
        {"setup": "import numpy as np",
         "call": "reference_bin_width(0.060, 800)",
         "gold_call": "_oracle_reference_bin_width(0.060, 800)"},
        # Edge: the residual scatter of an already-fitted record drives the rule far
        # below the tested grid, and the nearest member is then the finest one.
        {"setup": "import numpy as np",
         "call": "reference_bin_width(0.0051, 360)",
         "gold_call": "_oracle_reference_bin_width(0.0051, 360)"},
        # Boundary: a single-member grid is always the adopted one.
        {"setup": "import numpy as np",
         "call": "reference_bin_width(0.0346, 400, tested=(0.02,))",
         "gold_call": "_oracle_reference_bin_width(0.0346, 400, tested=(0.02,))"},
        # Invalid: non-positive background noise.
        {"setup": invalid,
         "call": "run_model(sigma_background=0.0, n_active=400)",
         "gold_call": "run_gold(sigma_background=0.0, n_active=400)"},
        # Invalid: non-integer sample count.
        {"setup": invalid,
         "call": "run_model(sigma_background=0.0346, n_active=400.5)",
         "gold_call": "run_gold(sigma_background=0.0346, n_active=400.5)"},
    ]
