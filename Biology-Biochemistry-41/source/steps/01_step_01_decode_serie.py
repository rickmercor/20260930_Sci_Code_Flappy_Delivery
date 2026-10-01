"""
Step 1 - decode the two digitized series.

Line-scan fluorescence is digitized, and the study analysis consumes two Gaussian-fitted series per event: the amplitude of the spatial fluorescence profile and its full width at half maximum (FWHM). 

When the least-squares spatial fit failed to converge at a sample, the study records a zero in both series instead of a measurement. The amplitude is normalized fluorescence (F/F0), so the resting level sits at 1.0 and the FWHM is in micrometres. 

This step decodes the integer counts of both supplied series onto those scales, builds the acquisition's time axis, and marks which samples are converged measurements rather than failure markers.

Returns
-------
t, amp, fwhm, valid : tuple of np.ndarray, each shape (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def decode_series(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    amp_unit: float = 1.0e-4,
    fwhm_unit: float = 1.0e-3,
) -> "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]":
    """Decode the amplitude and FWHM count series to normalized units.

    Parameters
    ----------
    amp_counts : sequence of int
        Amplitude in integer multiples of `amp_unit` (F/F0, so 10000 is the resting
        level 1.0000), in time order. A zero marks a sample whose spatial fit did
        not converge.
    fwhm_counts : sequence of int
        Full width at half maximum in integer multiples of `fwhm_unit` (um), same
        samples and same failure-marker convention.
    dt_ms : float, optional
        Sampling interval in milliseconds.
    amp_unit : float, optional
        Size of one amplitude count in F/F0.
    fwhm_unit : float, optional
        Size of one FWHM count in um.

    Returns
    -------
    tuple of numpy.ndarray
        ``(t, amp, fwhm, valid)``: time in milliseconds starting at 0.0 for the first
        sample; amplitude in F/F0; FWHM in um; and a boolean array, True at samples
        where both counts are positive, i.e. where the spatial fit converged and the
        sample is a measurement.

    Raises
    ------
    ValueError
        If the two count series do not share a non-empty 1-D shape, if any count is
        non-finite, non-integer or negative, or if `dt_ms`, `amp_unit` or `fwhm_unit`
        is not positive and finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_decode_series(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    amp_unit: float = 1.0e-4,
    fwhm_unit: float = 1.0e-3,
) -> "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]":
    """Reference implementation for decode_series."""
    import numpy as np

    for name, value in (("dt_ms", dt_ms), ("amp_unit", amp_unit),
                        ("fwhm_unit", fwhm_unit)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    a = np.asarray(amp_counts, dtype=float)
    f = np.asarray(fwhm_counts, dtype=float)
    if a.ndim != 1 or a.shape != f.shape or a.size == 0:
        raise ValueError("amp_counts and fwhm_counts must share a non-empty 1-D shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f))):
        raise ValueError("the counts must be finite")
    if np.any(a != np.round(a)) or np.any(f != np.round(f)):
        raise ValueError("the counts must be integer-valued")
    if np.any(a < 0) or np.any(f < 0):
        raise ValueError("the counts must be non-negative; a zero marks a failed fit")

    amp = np.round(a * float(amp_unit), 4)
    fwhm = np.round(f * float(fwhm_unit), 3)
    t = np.arange(a.size, dtype=float) * float(dt_ms)
    valid = (a > 0) & (f > 0)
    return t, amp, fwhm, valid

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "amp_c = np.array([10020, 10075, 10130, 0, 0, 10640, 10705, 10712], dtype=int)\n"
        "fwhm_c = np.array([1510, 1580, 1650, 0, 0, 1910, 1980, 2040], dtype=int)\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        decode_series(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_decode_series(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: two aligned series with a short non-convergence run in the middle.
        {"setup": sized,
         "call": "decode_series(amp_c, fwhm_c)",
         "gold_call": "_oracle_decode_series(amp_c, fwhm_c)"},
        # Normal: a coarser sampling interval only moves the time axis.
        {"setup": sized,
         "call": "decode_series(amp_c, fwhm_c, dt_ms=0.0474)",
         "gold_call": "_oracle_decode_series(amp_c, fwhm_c, dt_ms=0.0474)"},
        # Boundary: a single sample, and a single failure marker.
        {"setup": "import numpy as np",
         "call": "decode_series(np.array([10010], dtype=int), np.array([1505], dtype=int))",
         "gold_call": "_oracle_decode_series(np.array([10010], dtype=int), np.array([1505], dtype=int))"},
        # Edge: a record with no failure markers decodes with every sample valid.
        {"setup": "import numpy as np",
         "call": ("decode_series(np.array([9990, 10040, 10110], dtype=int),\n"
                  "              np.array([1495, 1540, 1610], dtype=int))"),
         "gold_call": ("_oracle_decode_series(np.array([9990, 10040, 10110], dtype=int),\n"
                       "                       np.array([1495, 1540, 1610], dtype=int))")},
        # Invalid: misaligned series.
        {"setup": invalid,
         "call": "run_model(amp_counts=np.array([10020, 10075]), fwhm_counts=np.array([1510]))",
         "gold_call": "run_gold(amp_counts=np.array([10020, 10075]), fwhm_counts=np.array([1510]))"},
        # Invalid: non-integer counts.
        {"setup": invalid,
         "call": "run_model(amp_counts=np.array([10020.25]), fwhm_counts=np.array([1510.0]))",
         "gold_call": "run_gold(amp_counts=np.array([10020.25]), fwhm_counts=np.array([1510.0]))"},
        # Invalid: negative counts.
        {"setup": invalid,
         "call": "run_model(amp_counts=np.array([-3, 4]), fwhm_counts=np.array([1500, 1500]))",
         "gold_call": "run_gold(amp_counts=np.array([-3, 4]), fwhm_counts=np.array([1500, 1500]))"},
        # Invalid: non-positive sampling interval.
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp_counts=amp_c, fwhm_counts=fwhm_c, dt_ms=-0.01)",
         "gold_call": "run_gold(amp_counts=amp_c, fwhm_counts=fwhm_c, dt_ms=-0.01)"},
    ]
