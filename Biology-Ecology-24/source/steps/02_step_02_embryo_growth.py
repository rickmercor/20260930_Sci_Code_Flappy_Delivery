"""
Step 2: the modelled embryo size over a held temperature record.

The growth follows the Gompertz law, driven by the development rate of step 1, and the equation is an asymptotic ratio times the hatchling length of the nest. The temperature is held at the value of each reading until the next reading, so the rate is piecewise constant across the record.

The returned sizes are the modelled equivalent of the straight carapace length of the embryo. They are the basis of the developmental window, whose limits are the sizes at which the record reaches the two stage fractions of the size at the end of incubation.

Returns
-------
size : np.ndarray, shape of n_readings
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def embryo_growth(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    query_times: "numpy.typing.ArrayLike | None" = None,
) -> "numpy.ndarray":
    """Modelled embryo size at the reading times, or at supplied query times.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite. The temperature
        of a reading holds until the next reading.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive. The Gompertz
        asymptote is ``asymptotic_ratio * hatchling_length``.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    start_size : float, optional
        Modelled size at laying in mm (0.347089). Must lie between zero and the
        asymptote.
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    query_times : array-like, optional
        Times in minutes at which the size is returned. A query time inside a
        logging interval is evaluated with that interval's held temperature and
        rate. By default the reading times are used.

    Returns
    -------
    size : numpy.ndarray
        Modelled size in mm. At the reading times it has one entry per reading,
        with `start_size` at the first reading. For `query_times` it has their
        shape. Each entry is the Gompertz trajectory under the held temperature
        of its interval.

    Raises
    ------
    ValueError
        If the two arrays differ in length or hold fewer than two readings, if a
        time is not strictly increasing, if a temperature is not finite or is at
        or below -273.15 degrees Celsius, if a size parameter is not positive or
        does not place the start size below the asymptote, or if a query time is
        not finite.
    """
    return size

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_embryo_growth(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    query_times: "numpy.typing.ArrayLike | None" = None,
) -> "numpy.ndarray":
    """Reference implementation for embryo_growth."""
    t = np.asarray(times, dtype=float)
    T = np.asarray(temperature, dtype=float)
    if t.ndim != 1 or T.ndim != 1 or t.size != T.size or t.size < 2:
        raise ValueError("times and temperature must be 1-D arrays of at least two readings")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(T))):
        raise ValueError("times and temperature must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must strictly increase")
    if not (float(hatchling_length) > 0.0):
        raise ValueError("hatchling_length must be positive")
    if not (float(asymptotic_ratio) > 0.0) or not (0.0 < float(start_size)):
        raise ValueError("start_size and asymptotic_ratio must be positive")
    K = float(asymptotic_ratio) * float(hatchling_length)
    if not (float(start_size) < K):
        raise ValueError("start_size must lie below the asymptote")
    rate = _oracle_development_rate(T, DHA, DHH, T12H, Rho25)
    u = np.empty(t.size, dtype=float)
    u[0] = np.log(K / float(start_size))
    u[1:] = u[0] * np.exp(-np.cumsum(rate[:-1] * np.diff(t)))
    if query_times is None:
        return K * np.exp(-u)
    q = np.asarray(query_times, dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError("query_times must be finite")
    flat = np.atleast_1d(q)
    k = np.clip(np.searchsorted(t, flat, side="right") - 1, 0, t.size - 2)
    sizes = K * np.exp(-u[k] * np.exp(-rate[k] * (flat - t[k])))
    return sizes.reshape(q.shape) if q.ndim else sizes

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
        "        embryo_growth(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_embryo_growth(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the sizes over a 48-day record with a diurnal cycle and a warming trend.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "embryo_growth(TIMES, TEMPS, 50.0)",
         "gold_call": "_oracle_embryo_growth(TIMES, TEMPS, 50.0)"},
        # Boundary: one long interval, where the exact per-interval product is the whole record.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([0.0, 70000.0], dtype=float)\n"
                  "TEMPS = np.array([30.0, 30.0], dtype=float)\n",
         "call": "embryo_growth(TIMES, TEMPS, 50.0)",
         "gold_call": "_oracle_embryo_growth(TIMES, TEMPS, 50.0)"},
        # Edge: query times inside logging intervals and at the record end.
        {"setup": record + "TIMES, TEMPS = make_record()\n"
                  "QUERY = np.array([1000.0, 20000.0, 33120.0, 65000.0, 69120.0], dtype=float)\n",
         "call": "embryo_growth(TIMES, TEMPS, 50.0, query_times=QUERY)",
         "gold_call": "_oracle_embryo_growth(TIMES, TEMPS, 50.0, query_times=QUERY)"},
        # Edge: a second parameter block, whose DHA is negative and whose rate is a
        # temperature of half high-temperature inactivation far above the record.
        {"setup": record + "TIMES, TEMPS = make_record(days=30.0, base=28.0)\n",
         "call": "embryo_growth(TIMES, TEMPS, 48.0, DHA=-719.576256, DHH=685.203574, "
                 "T12H=552.204465, Rho25=100.0, start_size=0.347089, asymptotic_ratio=1.208968)",
         "gold_call": "_oracle_embryo_growth(TIMES, TEMPS, 48.0, DHA=-719.576256, DHH=685.203574, "
                      "T12H=552.204465, Rho25=100.0, start_size=0.347089, asymptotic_ratio=1.208968)"},
        # Edge: strongly unequal intervals and alternating temperatures, where a
        # rate averaged over the whole record or a one-step Euler update departs
        # from the exact piecewise trajectory.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([0.0, 500.0, 30000.0, 80000.0], dtype=float)\n"
                  "TEMPS = np.array([22.0, 33.0, 26.0, 34.0], dtype=float)\n",
         "call": "embryo_growth(TIMES, TEMPS, 50.0)",
         "gold_call": "_oracle_embryo_growth(TIMES, TEMPS, 50.0)"},
        # Invalid: a record of one reading.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([0.0], dtype=float)\n"
                  "TEMPS = np.array([30.0], dtype=float)\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0)"},
        # Invalid: repeated reading times.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([0.0, 720.0, 720.0], dtype=float)\n"
                  "TEMPS = np.array([30.0, 30.0, 30.5], dtype=float)\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0)"},
    ]
