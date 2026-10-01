"""
Step 3 - the signal-mass curve over the interval.

The study signal mass is the integrated optical signal of the event: SM(t) = 1.206 x Amp(t) x FWHM(t)^3, with the FWHM in um and Amp the spark amplitude (the height of the spatial Gaussian above the resting fluorescence, not the normalized level itself). 

The supplied series is normalized fluorescence, so the resting level is subtracted before the product is formed. The coefficient is the study published geometry factor. The curve is defined over the signal-mass interval resolved in step 02, and only where both series are available: at a sample whose spatial fit failed, neither series carries a measurement. When both series are available throughout the interval, SM is a continuous curve and its first time derivative, proportional to the net release flux, is defined there.

Returns
-------
sm, continuous : tuple of (np.ndarray shape (n,), bool)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def signal_mass_curve(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[numpy.ndarray, bool]":
    """Evaluate the signal-mass curve over an inclusive interval.

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
        Normalized resting level. The signal-mass amplitude is the spark's height
        above this level, so it is subtracted from the supplied series.
    k : float, optional
        The study's signal-mass coefficient, in F/F0^-1 um^-3.

    Returns
    -------
    tuple
        ``(sm, continuous)``: `sm` is a float array of length ``ts2 - ts1 + 1``
        holding SM in F/F0 x um^3 at each sample of the interval, with NaN at
        samples where either series is unavailable; `continuous` is True exactly
        when every sample of the interval is a converged measurement.

    Raises
    ------
    ValueError
        If the arrays do not share a non-empty 1-D shape, if any entry is
        non-finite, if the interval indices are outside the record or reversed, or
        if `baseline` or `k` is not finite (`k` must also be positive).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_signal_mass_curve(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[numpy.ndarray, bool]":
    """Reference implementation for signal_mass_curve."""
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
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive and finite")
    if int(ts1) != ts1 or int(ts2) != ts2:
        raise ValueError("ts1 and ts2 must be integer sample indices")
    ts1, ts2 = int(ts1), int(ts2)
    if ts1 < 0 or ts2 >= a.size or ts1 > ts2:
        raise ValueError("the interval must lie inside the record and not be reversed")

    sm = float(k) * (a[ts1: ts2 + 1] - float(baseline)) * f[ts1: ts2 + 1] ** 3
    keep = v[ts1: ts2 + 1]
    sm = np.where(keep, sm, np.nan)
    continuous = bool(np.all(keep))
    return sm, continuous

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "amp = np.array([1.05, 1.11, 1.17, 0.0, 1.29, 1.36, 1.36])\n"
        "fwhm = np.array([1.6, 1.8, 2.0, 0.0, 2.2, 2.3, 2.3])\n"
        "valid = np.array([True, True, True, False, True, True, True])\n"
        "def _pin(v):\n"
        "    sm, continuous = v\n"
        "    a = np.asarray(sm, dtype=float)\n"
        "    q = np.where(np.isfinite(a), a, -700.0)\n"
        "    r = np.arange(1.0, q.size + 1.0)\n"
        "    return float(np.sum(np.sin(0.31*r)*q) + np.sum(np.cos(0.13*r)*q*q)) + (1000.0 if continuous else 0.0)\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        signal_mass_curve(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_signal_mass_curve(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {"setup": sized,
         "call": "_pin(signal_mass_curve(amp, fwhm, valid, 4, 6))",
         "gold_call": "_pin(_oracle_signal_mass_curve(amp, fwhm, valid, 4, 6))"},
        {"setup": sized,
         "call": "_pin(signal_mass_curve(amp, fwhm, valid, 1, 4))",
         "gold_call": "_pin(_oracle_signal_mass_curve(amp, fwhm, valid, 1, 4))"},
        {"setup": sized,
         "call": "_pin(signal_mass_curve(amp, fwhm, valid, 5, 5))",
         "gold_call": "_pin(_oracle_signal_mass_curve(amp, fwhm, valid, 5, 5))"},
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid, ts1=5, ts2=2)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid, ts1=5, ts2=2)"},
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid, ts1=0, ts2=7)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid, ts1=0, ts2=7)"},
        {"setup": sized + invalid.split("import numpy as np\n", 1)[1],
         "call": "run_model(amp=amp, fwhm=fwhm, valid=valid, ts1=0, ts2=2, k=0.0)",
         "gold_call": "run_gold(amp=amp, fwhm=fwhm, valid=valid, ts1=0, ts2=2, k=0.0)"},
    ]
