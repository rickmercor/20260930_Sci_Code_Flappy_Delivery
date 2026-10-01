"""
Step 4: the developmental window of a nest, given the end of incubation.

The developmental window is the stretch of incubation during which temperature determines the sex of the embryo. It is placed by size, not by date. Each of the two branch ends of the task, pipping and emergence, supplies its own end of incubation, and the size the growth model reaches at that end sets the scale of the window.

The window runs from one stage fraction of that modelled size to a second fraction. The two fractions come from a species stage table of embryo size against developmental stage. Both limits are crossing times of the growth model, so they fall inside logging intervals and carry the held temperature of those intervals.

Returns
-------
limits: np.ndarray, shape (3,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def window_limits(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    end_time: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
) -> "numpy.ndarray":
    """Place the developmental window from a supplied end of incubation.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive.
    end_time : float
        Time in minutes from laying at which this branch's incubation ends, from
        the first to the last reading time. The window is sized from the modelled
        size at this time, not from the measured hatchling length.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    frac_begin : float, optional
        Lower stage fraction of the modelled size at `end_time`.
    frac_end : float, optional
        Upper stage fraction of the modelled size at `end_time`.

    Returns
    -------
    numpy.ndarray
        Three floats. Entry 0 is the modelled size in mm at `end_time`, the size
        that scales the window. Entries 1 and 2 are the window start and window
        end in minutes from laying, the crossing times of ``frac_begin`` and
        ``frac_end`` times that size, located as in step 3, so a target below the
        modelled size at laying gives the first reading time.

    Raises
    ------
    ValueError
        If the growth inputs fail the validation of step 2, if `end_time` is not
        finite or lies outside the record, or if the two fractions are not
        positive and strictly increasing.
    """
    return limits

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_window_limits(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    end_time: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
) -> "numpy.ndarray":
    """Reference implementation for window_limits."""
    t = np.asarray(times, dtype=float)
    if not np.isfinite(float(end_time)):
        raise ValueError("end_time must be finite")
    if not (float(t[0]) <= float(end_time) <= float(t[-1])):
        raise ValueError("end_time must lie inside the record")
    if not (0.0 < float(frac_begin) < float(frac_end)):
        raise ValueError("the stage fractions must be positive and strictly increasing")
    size_end = _oracle_embryo_growth(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                                     start_size, asymptotic_ratio, query_times=[float(end_time)])[0]
    begin = _oracle_pipping_time(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                                 start_size, asymptotic_ratio, target_size=float(frac_begin) * size_end)
    end = _oracle_pipping_time(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                               start_size, asymptotic_ratio, target_size=float(frac_end) * size_end)
    return np.array([float(size_end), begin, end], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return list of test case specifications."""
    record = (
        "import numpy as np\n"
        "def make_record(days=48.0, step=720.0, base=30.0, amp=1.0, trend=2.0):\n"
        "    n = int(days * 1440.0 / step) + 1\n"
        "    times = np.arange(n) * step\n"
        "    temps = base + amp * np.sin(np.arange(n) * 0.8) + trend * (times / times[-1])\n"
        "    return times, temps\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        window_limits(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_window_limits(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the pipping branch, whose end of incubation is the modelled
        # crossing of the hatchling length.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "window_limits(TIMES, TEMPS, 50.0, 65400.0)",
         "gold_call": "_oracle_window_limits(TIMES, TEMPS, 50.0, 65400.0)"},
        # Boundary: the emergence branch, whose end of incubation is the record end,
        # so the window is sized from a modelled size above the hatchling length.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "window_limits(TIMES, TEMPS, 50.0, 69120.0)",
         "gold_call": "_oracle_window_limits(TIMES, TEMPS, 50.0, 69120.0)"},
        # Edge: the lower fraction of the end size falls below the modelled size at
        # laying, so the window starts at the first reading.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([0.0, 20000.0], dtype=float)\n"
                  "TEMPS = np.array([31.0, 31.0], dtype=float)\n",
         "call": "window_limits(TIMES, TEMPS, 0.45, 20000.0)",
         "gold_call": "_oracle_window_limits(TIMES, TEMPS, 0.45, 20000.0)"},
        # Edge: an end of incubation inside the first logging interval, where the two
        # fractions of a barely grown embryo sit close to the start size.
        {"setup": record + "TIMES, TEMPS = make_record(days=10.0)\n",
         "call": "window_limits(TIMES, TEMPS, 50.0, 5000.0)",
         "gold_call": "_oracle_window_limits(TIMES, TEMPS, 50.0, 5000.0)"},
        # Invalid: an end of incubation beyond the record.
        {"setup": record + "TIMES, TEMPS = make_record()\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0, end_time=80000.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0, end_time=80000.0)"},
        # Invalid: stage fractions that are not increasing.
        {"setup": record + "TIMES, TEMPS = make_record()\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0, end_time=65400.0, "
                 "frac_begin=0.8, frac_end=0.3)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0, end_time=65400.0, "
                      "frac_begin=0.8, frac_end=0.3)"},
    ]
