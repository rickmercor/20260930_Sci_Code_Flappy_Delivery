"""
Step 3: the time at which the modelled embryo reaches a size.

Pipping is defined by size rather than by a clock: it is the moment the modelled embryo reaches its hatchling length. Field temperature records end at emergence, so the moment of pipping has to be derived from the growth model of step 2 and the record itself.

Growth is monotone, so the crossing lies in one logging interval, the first interval whose end size exceeds the target. Inside that interval the size follows the held temperature. A record that ends before the modelled embryo reaches the target carries no crossing, and the record end is returned instead, which is the same end of incubation for both conventions of the task.

Returns
-------
time: float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pipping_time(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    target_size: "float | None" = None,
) -> float:
    """Time at which the modelled embryo size reaches a target size.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite. The temperature
        of a reading holds until the next reading.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive. It sets the Gompertz
        asymptote of the growth model.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    target_size : float, optional
        Size in mm whose crossing time is returned. The default is the hatchling
        length, which is the definition of pipping.

    Returns
    -------
    float
        Time in minutes from laying at which the modelled size reaches
        `target_size`. The crossing lies inside the first logging interval whose
        end size exceeds the target. If the modelled size never reaches the
        target inside the record, the last reading time is returned. If the
        modelled size already exceeds the target at the first reading, the first
        reading time is returned.

    Raises
    ------
    ValueError
        If the growth inputs fail the validation of step 2, or if `target_size`
        is not positive.
    """
    return time

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pipping_time(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    target_size: "float | None" = None,
) -> float:
    """Reference implementation for pipping_time."""
    t = np.asarray(times, dtype=float)
    target = float(hatchling_length) if target_size is None else float(target_size)
    if not (target > 0.0):
        raise ValueError("target_size must be positive")
    sizes = _oracle_embryo_growth(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                                  start_size, asymptotic_ratio)
    if np.all(sizes <= target):
        return float(t[-1])
    if sizes[0] > target:
        return float(t[0])
    kb = int(np.argmax(sizes > target)) - 1
    K = float(asymptotic_ratio) * float(hatchling_length)
    rate = _oracle_development_rate(np.asarray(temperature, dtype=float)[kb], DHA, DHH, T12H, Rho25)
    u_kb = np.log(K / sizes[kb])
    return float(t[kb] + np.log(u_kb / np.log(K / target)) / rate)

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
        "        pipping_time(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_pipping_time(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a 48-day record whose modelled embryo reaches its hatchling length.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "pipping_time(TIMES, TEMPS, 50.0)",
         "gold_call": "_oracle_pipping_time(TIMES, TEMPS, 50.0)"},
        # Boundary: the crossing sits inside the first logging interval of the record.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([0.0, 20000.0], dtype=float)\n"
                  "TEMPS = np.array([31.0, 31.0], dtype=float)\n",
         "call": "pipping_time(TIMES, TEMPS, 0.45)",
         "gold_call": "_oracle_pipping_time(TIMES, TEMPS, 0.45)"},
        # Boundary: a target that is one of the two window fractions of the hatchling
        # length, the general crossing the window step needs.
        {"setup": record + "TIMES, TEMPS = make_record()\n",
         "call": "pipping_time(TIMES, TEMPS, 50.0, target_size=0.250554 * 50.0)",
         "gold_call": "_oracle_pipping_time(TIMES, TEMPS, 50.0, target_size=0.250554 * 50.0)"},
        # Edge: a record that ends before the modelled embryo reaches its hatchling
        # length, so the record end is returned.
        {"setup": "import numpy as np\n"
                  "TIMES = np.arange(0.0, 7200.0 + 1.0, 720.0)\n"
                  "TEMPS = np.full(TIMES.size, 29.0)\n",
         "call": "pipping_time(TIMES, TEMPS, 50.0)",
         "gold_call": "_oracle_pipping_time(TIMES, TEMPS, 50.0)"},
        # Invalid: a non-positive target.
        {"setup": record + "TIMES, TEMPS = make_record()\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0, target_size=0.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0, target_size=0.0)"},
        # Invalid: the growth inputs are rejected by step 2.
        {"setup": "import numpy as np\n"
                  "TIMES = np.array([720.0, 0.0], dtype=float)\n"
                  "TEMPS = np.array([30.0, 30.0], dtype=float)\n" + invalid,
         "call": "run_model(times=TIMES, temperature=TEMPS, hatchling_length=50.0)",
         "gold_call": "run_gold(times=TIMES, temperature=TEMPS, hatchling_length=50.0)"},
    ]
