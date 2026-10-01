"""
Step 7 - resolve the step levels: histogram modes refined by a Gaussian fit.

The study resolves the staircase step levels by fitting a sum of Gaussian functions to the activation-phase amplitude histogram: the resolved peaks are the step levels. 

Two safeguards keep that count honest on a noisy record: modes must tower over the between-level noise floor (the prominence filter), and the fitted positions are the Gaussian means, seeded at the detected peak.

Returns
-------
levels : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_step_levels(
    counts: "numpy.ndarray",
    centers: "numpy.ndarray",
    bin_width: float = 0.015,
    prominence: float = 5.0,
) -> "numpy.ndarray":
    """Resolve the fluorescence step levels from a level histogram.

    Parameters
    ----------
    counts : numpy.ndarray
        Integer bin counts of the level histogram (see step 06).
    centers : numpy.ndarray
        Bin centers in F/F0, same length as `counts`.
    bin_width : float, optional
        Histogram bin width (the study's adopted value).
    prominence : float, optional
        Minimum prominence (in counts) a histogram mode must have to count as a
        level.

    Returns
    -------
    numpy.ndarray
        The sorted step-level positions in F/F0 (Gaussian-refined when the fit
        converges; otherwise the raw mode centers).

    Raises
    ------
    ValueError
        If the histogram inputs are misaligned, empty, or non-finite, or if
        `bin_width` or `prominence` is out of range.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resolve_step_levels(
    counts: "numpy.ndarray",
    centers: "numpy.ndarray",
    bin_width: float = 0.015,
    prominence: float = 5.0,
) -> "numpy.ndarray":
    """Reference implementation for resolve_step_levels."""
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    counts = np.asarray(counts, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if counts.shape != centers.shape or counts.size == 0:
        raise ValueError("counts and centers must share a non-empty shape")
    if not (np.all(np.isfinite(counts)) and np.all(np.isfinite(centers))):
        raise ValueError("counts and centers must be finite")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be positive and finite")
    if not np.isfinite(prominence) or prominence < 0:
        raise ValueError("prominence must be non-negative and finite")

    def _gauss_sum(x, *p):
        n = len(p) // 3
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            a, mu, s = p[3 * i: 3 * i + 3]
            y = y + a * np.exp(-0.5 * ((x - mu) / s) ** 2)
        return y

    pks, _ = find_peaks(counts, prominence=prominence)
    if pks.size == 0:
        pks = np.array([int(np.argmax(counts))])
    pks = np.sort(pks)
    k = pks.size
    p0 = []
    lo = []
    hi = []
    for pk in pks:
        p0.extend([float(counts[pk]), float(centers[pk]), float(bin_width)])
        lo.extend([0.0, float(centers[pk]) - bin_width, bin_width * 0.3])
        hi.extend([float(counts.max()) * 5.0, float(centers[pk]) + bin_width,
                   bin_width * 4.0])
    try:
        popt, _ = curve_fit(_gauss_sum, centers, counts, p0=p0,
                            bounds=(lo, hi), maxfev=60000)
        return np.sort(np.array([popt[3 * i + 1] for i in range(k)], dtype=float))
    except Exception:
        return np.sort(np.array([float(centers[i]) for i in pks], dtype=float))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "centers = 1.0 + 0.015 * np.arange(30)\n"
        "counts = np.array([2, 3, 8, 25, 26, 12, 4, 2, 1, 5, 20, 22, 10, 3, 1, 1, 2,\n"
        "                   9, 24, 24, 11, 3, 1, 1, 0, 2, 8, 30, 28, 14])\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        resolve_step_levels(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_resolve_step_levels(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: four modes above the noise floor.
        {"setup": sized,
         "call": "resolve_step_levels(counts, centers)",
         "gold_call": "_oracle_resolve_step_levels(counts, centers)"},
        # Boundary: a single occupied bin.
        {"setup": "import numpy as np\ncounts = np.array([0, 0, 5.0, 0.0])\ncenters = np.array([1.0, 1.015, 1.03, 1.045])",
         "call": "resolve_step_levels(counts, centers)",
         "gold_call": "_oracle_resolve_step_levels(counts, centers)"},
        # Edge: one tall mode below the prominence floor -> no resolved level (falls
        # back to the tallest bin); a second tall one passes.
        {"setup": "import numpy as np\n"
                  "counts = np.array([0, 3, 6, 9, 3])\ncenters = 1.0 + 0.015 * np.arange(5)",
         "call": "resolve_step_levels(counts, centers, prominence=3.0)",
         "gold_call": "_oracle_resolve_step_levels(counts, centers, prominence=3.0)"},
        # Edge: an all-zero histogram (degenerate activation phase). The Gaussian
        # amplitude bounds collapse to [0, 0], curve_fit rejects that, and the
        # implementation must fall back to the raw mode center instead of raising.
        {"setup": "import numpy as np\ncounts = np.array([0.0, 0.0, 0.0, 0.0, 0.0])\ncenters = 1.0 + 0.015 * np.arange(5)",
         "call": "resolve_step_levels(counts, centers)",
         "gold_call": "_oracle_resolve_step_levels(counts, centers)"},
        # Invalid: misaligned inputs.
        {"setup": invalid,
         "call": "run_model(counts=np.array([1.0, 2.0]), centers=np.array([1.0]))",
         "gold_call": "run_gold(counts=np.array([1.0, 2.0]), centers=np.array([1.0]))"},
        # Invalid: non-positive bin width.
        {"setup": invalid,
         "call": "run_model(counts=np.array([5.0]), centers=np.array([1.0]), bin_width=-0.01)",
         "gold_call": "run_gold(counts=np.array([5.0]), centers=np.array([1.0]), bin_width=-0.01)"},
    ]
