"""
Step 9 -  The signal-mass increment per release step.

The graded pipeline chains steps 01-08 end to end on one event: 

* decode the two Gaussian-fitted series and mark the converged samples: resolve the signal-mass interval under the study two criteria evaluate the signal-mass curve and its interval statistics
*  fix the histogram bin from the acquisition's raw background noise through Scott rule and the study tested grid histogram the analyzed amplitudes at that width with edges anchored at the resting baseline
*  resolve the step levels as the Gaussian-refined histogram modes and apply the double-quantum merge correction, which returns both the corrected count and this event quantal step size. 

The returned scalar is the signal-mass increment corresponding to one release step, dSM_step = 1.206 * q * <FWHM>^3, where q is the event' merge-corrected quantal step size and <FWHM> is the mean FWHM over the signal-mass interval, in F/F0 x um^3.

Returns
-------
increment : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def signal_mass_increment(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    sigma_background: float = 0.0346,
) -> float:
    """Per-release-step signal-mass increment of one event.

    Parameters
    ----------
    amp_counts : sequence of int
        Amplitude in integer multiples of 1e-4 F/F0, in time order (see step 01).
    fwhm_counts : sequence of int
        FWHM in integer multiples of 1e-3 um, same samples (see step 01).
    dt_ms : float, optional
        Sampling interval in milliseconds.
    sigma_background : float, optional
        Standard deviation of the acquisition's raw spark-free background, in F/F0,
        from which the histogram bin width is fixed (see step 05).

    Returns
    -------
    float
        The signal-mass increment per release step, in F/F0 x um^3.

    Raises
    ------
    ValueError
        If the record fails the input validation of the pipeline steps.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_signal_mass_increment(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    sigma_background: float = 0.0346,
) -> float:
    """Reference implementation for signal_mass_increment."""
    import numpy as np

    _t, amp, fwhm, valid = _oracle_decode_series(amp_counts, fwhm_counts, dt_ms)
    ts1, ts2 = _oracle_signal_mass_interval(amp, fwhm, valid, dt_ms=dt_ms)
    _n, mean_fwhm, _sm_start, _sm_max = _oracle_interval_statistics(
        amp, fwhm, valid, ts1, ts2)

    peak_idx = int(np.argmax(amp))
    n_analyzed = int(valid[: peak_idx + 1].sum())
    _raw, bin_width = _oracle_reference_bin_width(sigma_background, n_analyzed)
    hist, centers = _oracle_level_histogram(
        amp[: peak_idx + 1][valid[: peak_idx + 1]], bin_width)
    levels = _oracle_resolve_step_levels(hist, centers, bin_width, 5.0)
    _step_count, q = _oracle_apply_merge_correction(levels, 1.5)
    return float(1.206 * q * mean_fwhm ** 3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    sized = (
        "import numpy as np\n"
        "def rec(seed=0, drop=None):\n"
        "    r = np.random.default_rng(seed)\n"
        "    levels = [0.062, 0.124, 0.186, 0.310, 0.372]\n"
        "    parts = []\n"
        "    prev = 0.0\n"
        "    for lev in levels:\n"
        "        parts.append(np.linspace(prev, lev, 4, endpoint=False))\n"
        "        parts.append(np.full(70, lev))\n"
        "        prev = lev\n"
        "    sig = np.concatenate(parts)\n"
        "    tail = levels[-1] * np.exp(-np.arange(80) / 20.0)\n"
        "    amp = np.concatenate([1.0 + sig, 1.0 + tail])\n"
        "    amp = amp + r.normal(0.0, 0.004, amp.size)\n"
        "    fwhm = np.concatenate([np.linspace(1.5, 2.3, sig.size),\n"
        "                           np.full(80, 2.3)]) + r.normal(0.0, 0.02, sig.size + 80)\n"
        "    if drop is not None:\n"
        "        amp[drop] = 0.0\n"
        "        fwhm[drop] = 0.0\n"
        "    return np.round(amp * 1e4).astype(int), np.round(fwhm * 1e3).astype(int)\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        signal_mass_increment(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_signal_mass_increment(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a fully converged event through the whole chain.
        {"setup": sized + "amp_c, fwhm_c = rec()\n",
         "call": "signal_mass_increment(amp_c, fwhm_c)",
         "gold_call": "_oracle_signal_mass_increment(amp_c, fwhm_c)"},
        # Edge: a non-convergence run inside the top plateau binds criterion (b),
        # which moves the interval and therefore the increment.
        {"setup": sized + "amp_c, fwhm_c = rec(drop=np.arange(300, 308))\n",
         "call": "signal_mass_increment(amp_c, fwhm_c)",
         "gold_call": "_oracle_signal_mass_increment(amp_c, fwhm_c)"},
        # Boundary: the failure run sits after the peak, inside the interval but
        # outside criterion (b)'s "through the peak" clause, so the interval's start
        # does not move while those samples stay out of the mean FWHM.
        {"setup": sized + "amp_c, fwhm_c = rec(drop=np.arange(372, 380))\n",
         "call": "signal_mass_increment(amp_c, fwhm_c)",
         "gold_call": "_oracle_signal_mass_increment(amp_c, fwhm_c)"},
        # Invalid: an empty record.
        {"setup": invalid,
         "call": "run_model(amp_counts=np.array([], dtype=int), fwhm_counts=np.array([], dtype=int))",
         "gold_call": "run_gold(amp_counts=np.array([], dtype=int), fwhm_counts=np.array([], dtype=int))"},
    ]
