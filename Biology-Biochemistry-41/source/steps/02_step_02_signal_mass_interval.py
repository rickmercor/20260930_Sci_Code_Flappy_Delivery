"""
Step 2 - the signal-mass interval.

The study computes its signal-mass parameters over an interval that is not the window its step analysis uses. The interval starts at ts1, the earliest sample at which two criteria hold together: (a) the amplitude above the resting baseline has reached at least one fifth (20 %) of its maximum, and (b) both the amplitude and the FWHM are available at every sample from there through the signal peak. 

Criterion (b) is non-local: a later convergence failure disqualifies every earlier candidate, so ts1 sits just after the last failure marker before the peak. The interval ends at ts2, one millisecond after the peak sample. 

This step resolves both indices.

Returns
-------
ts1, ts2 : tuple (int)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def signal_mass_interval(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    baseline: float = 1.0,
    peak_fraction: float = 0.20,
    tail_ms: float = 1.0,
    dt_ms: float = 0.0154,
) -> "tuple[int, int]":
    """Resolve the start and end indices of the signal-mass interval.

    Parameters
    ----------
    amp : numpy.ndarray
        Amplitude in F/F0 (see step 01).
    fwhm : numpy.ndarray
        FWHM in um, same shape as `amp`.
    valid : numpy.ndarray of bool
        Converged-sample mask, same shape as `amp` (see step 01).
    baseline : float, optional
        Normalized resting level the amplitude threshold is measured above.
    peak_fraction : float, optional
        Fraction of the peak amplitude above baseline that criterion (a) requires.
    tail_ms : float, optional
        Milliseconds the interval extends past the peak sample.
    dt_ms : float, optional
        Sampling interval in milliseconds.

    Returns
    -------
    tuple of int
        ``(ts1, ts2)``: the first and last sample indices of the interval, inclusive.
        ts1 is the earliest sample satisfying criterion (a) that has no later failure
        marker before the peak, and ts2 is the sample `tail_ms` after the peak.

    Raises
    ------
    ValueError
        If the three arrays do not share a non-empty 1-D shape, if any amplitude or
        FWHM entry is non-finite, if the parameters are out of range, if the record
        has no converged sample, if the peak is not a converged sample or does not
        rise above the baseline, or if the record does not extend `tail_ms` past the
        peak.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_signal_mass_interval(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    baseline: float = 1.0,
    peak_fraction: float = 0.20,
    tail_ms: float = 1.0,
    dt_ms: float = 0.0154,
) -> "tuple[int, int]":
    """Reference implementation for signal_mass_interval."""
    import numpy as np

    a = np.asarray(amp, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    v = np.asarray(valid, dtype=bool)
    if a.ndim != 1 or a.size == 0 or a.shape != f.shape or a.shape != v.shape:
        raise ValueError("amp, fwhm and valid must share a non-empty 1-D shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f))):
        raise ValueError("amp and fwhm must be finite")
    if not np.isfinite(baseline):
        raise ValueError("baseline must be finite")
    if not np.isfinite(peak_fraction) or not (0.0 < peak_fraction < 1.0):
        raise ValueError("peak_fraction must lie strictly between 0 and 1")
    if not np.isfinite(tail_ms) or tail_ms <= 0:
        raise ValueError("tail_ms must be positive and finite")
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")
    if not v.any():
        raise ValueError("the record has no converged samples")

    signal = a - float(baseline)
    peak_idx = int(np.argmax(signal))
    if not v[peak_idx]:
        raise ValueError("the peak sample is not a converged measurement")
    peak_signal = float(signal[peak_idx])
    if peak_signal <= 0.0:
        raise ValueError("the peak amplitude must rise above the baseline")

    threshold = float(peak_fraction) * peak_signal
    candidates = np.flatnonzero(v & (signal >= threshold))
    if candidates.size == 0:
        raise ValueError("no converged sample reaches the criterion (a) threshold")
    ts1_a = int(candidates[0])

    failures = np.flatnonzero(~v[: peak_idx + 1])
    ts1 = ts1_a if failures.size == 0 else max(ts1_a, int(failures[-1]) + 1)
    if ts1 > peak_idx:
        raise ValueError("criterion (b) leaves no converged interval before the peak")

    ts2 = peak_idx + int(round(float(tail_ms) / float(dt_ms)))
    if ts2 >= a.size:
        raise ValueError("the record does not extend tail_ms past the peak sample")
    return ts1, ts2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "r = np.random.default_rng(5)\n"
        "sig = np.concatenate([np.linspace(0.0, 0.36, 60), np.full(40, 0.36),\n"
        "                      0.36 * np.exp(-np.arange(70) / 20.0)])\n"
        "amp = 1.0 + sig + r.normal(0.0, 0.004, sig.size)\n"
        "fwhm = np.concatenate([np.linspace(1.5, 2.3, 60), np.full(110, 2.3)])\n"
        "fwhm = fwhm + r.normal(0.0, 0.02, fwhm.size)\n"
        "valid = np.ones(amp.size, dtype=bool)\n"
    )
    marked = (
        "drop = np.arange(70, 78)\n"
        "amp[drop] = 0.0\nfwhm[drop] = 0.0\nvalid[drop] = False\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        signal_mass_interval(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_signal_mass_interval(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the last non-convergence run before the peak binds criterion (b),
        # so the interval starts after it, not at the 20% crossing.
        {"setup": sized + marked,
         "call": "signal_mass_interval(amp, fwhm, valid)",
         "gold_call": "_oracle_signal_mass_interval(amp, fwhm, valid)"},
        # Normal: a fully converged record starts the interval at the 20% crossing.
        {"setup": sized,
         "call": "signal_mass_interval(amp, fwhm, valid)",
         "gold_call": "_oracle_signal_mass_interval(amp, fwhm, valid)"},
        # Edge: a failure run before the criterion (a) crossing does not move ts1.
        {"setup": sized + "drop = np.arange(5, 10)\n"
                  "amp[drop] = 0.0\nfwhm[drop] = 0.0\nvalid[drop] = False\n",
         "call": "signal_mass_interval(amp, fwhm, valid)",
         "gold_call": "_oracle_signal_mass_interval(amp, fwhm, valid)"},
        # Edge: a shorter tail moves ts2.
        {"setup": sized,
         "call": "signal_mass_interval(amp, fwhm, valid, tail_ms=0.25)",
         "gold_call": "_oracle_signal_mass_interval(amp, fwhm, valid, tail_ms=0.25)"},
        # Invalid: the record does not extend one millisecond past the peak.
        {"setup": sized + marked + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp[:120], fwhm=fwhm[:120], valid=valid[:120])",
         "gold_call": "run_gold(amp=amp[:120], fwhm=fwhm[:120], valid=valid[:120])"},
        # Invalid: the peak sample is a failure marker.
        {"setup": sized + "amp[99] = 0.0\nfwhm[99] = 0.0\nvalid[99] = False\n"
                  + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid)"},
        # Invalid: a peak fraction outside (0, 1).
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid, peak_fraction=1.5)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid, peak_fraction=1.5)"},
        # Invalid: misaligned series.
        {"setup": invalid,
         "call": "run_model(amp=np.arange(5.0), fwhm=np.arange(4.0), valid=np.ones(4, bool))",
         "gold_call": "run_gold(amp=np.arange(5.0), fwhm=np.arange(4.0), valid=np.ones(4, bool))"},
    ]
