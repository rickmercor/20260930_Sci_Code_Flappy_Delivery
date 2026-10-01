"""
Step 4 - the interval statistics behind the per-step increment.

The signal-mass interval resolved in step 02 enters the graded computation through three of its statistics: its length in samples, the mean FWHM over it, and the signal mass at its start; its maximum is reported as a cross-check. 

The mean FWHM is taken over the interval converged samples only  (a failure marker is not a measurement) and the interval start is itself a converged sample, so the signal mass there is always defined. This step collects those four quantities.

Returns
-------
n_samples, mean_fwhm, sm_at_start, sm_max : tuple (int, float, float, float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interval_statistics(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[int, float, float, float]":
    """Collect the signal-mass interval's summary statistics.

    Parameters
    ----------
    amp : numpy.ndarray
        Amplitude in F/F0 (see step 01).
    fwhm : numpy.ndarray
        FWHM in um, same shape as `amp`.
    valid : numpy.ndarray of bool
        Converged-sample mask, same shape as `amp`.
    ts1, ts2 : int
        First and last sample indices of the interval, inclusive (see step 02).
    baseline : float, optional
        Normalized resting level, subtracted from the amplitude before the signal
        mass is formed (see step 03).
    k : float, optional
        The study's signal-mass coefficient.

    Returns
    -------
    tuple
        ``(n_samples, mean_fwhm, sm_at_start, sm_max)``: the interval's length in
        samples (int); the mean FWHM over its converged samples, in um (float); the
        signal mass at its first sample, in F/F0 x um^3 (float); and the largest
        signal mass over the interval (float).

    Raises
    ------
    ValueError
        If the inputs fail the validation of steps 01-03, if the interval holds no
        converged sample, if its first sample is not converged, or if its indices
        lie outside the record or are reversed.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_interval_statistics(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[int, float, float, float]":
    """Reference implementation for interval_statistics."""
    import numpy as np

    curves = _oracle_signal_mass_curve(amp, fwhm, valid, ts1, ts2, baseline, k)
    sm, _continuous = curves
    a = np.asarray(amp, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    v = np.asarray(valid, dtype=bool)
    ts1, ts2 = int(ts1), int(ts2)

    keep = v[ts1: ts2 + 1]
    if not keep.any():
        raise ValueError("the interval holds no converged sample")
    if not bool(keep[0]):
        raise ValueError("the interval's first sample is not a converged measurement")

    n_samples = ts2 - ts1 + 1
    mean_fwhm = float(np.mean(f[ts1: ts2 + 1][keep]))
    sm_at_start = float(sm[0])
    sm_max = float(np.nanmax(sm))
    return n_samples, mean_fwhm, sm_at_start, sm_max

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "amp = np.array([1.05, 1.11, 1.17, 1.23, 0.0, 1.33, 1.36, 1.36])\n"
        "fwhm = np.array([1.6, 1.8, 2.0, 2.1, 0.0, 2.25, 2.3, 2.3])\n"
        "valid = np.array([True, True, True, True, False, True, True, True])\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        interval_statistics(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_interval_statistics(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a fully converged interval.
        {"setup": sized,
         "call": "interval_statistics(amp, fwhm, valid, 2, 7)",
         "gold_call": "_oracle_interval_statistics(amp, fwhm, valid, 2, 7)"},
        # Edge: a failure marker inside the interval is excluded from the FWHM mean
        # and from the signal-mass maximum.
        {"setup": sized,
         "call": "interval_statistics(amp, fwhm, valid, 1, 6)",
         "gold_call": "_oracle_interval_statistics(amp, fwhm, valid, 1, 6)"},
        # Boundary: a single-sample interval.
        {"setup": sized,
         "call": "interval_statistics(amp, fwhm, valid, 6, 6)",
         "gold_call": "_oracle_interval_statistics(amp, fwhm, valid, 6, 6)"},
        # Invalid: the interval starts on a failure marker.
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid, ts1=4, ts2=6)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid, ts1=4, ts2=6)"},
        # Invalid: the interval holds no converged sample.
        {"setup": ("import numpy as np\n"
                   "amp = np.array([1.0, 0.0, 0.0, 1.2])\n"
                   "fwhm = np.array([2.0, 0.0, 0.0, 2.0])\n"
                   "valid = np.array([True, False, False, True])\n" +
                   invalid.split("import numpy as np\n", 1)[1]),
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid, ts1=1, ts2=2)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid, ts1=1, ts2=2)"},
    ]
