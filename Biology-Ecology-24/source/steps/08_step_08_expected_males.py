"""
Step 8: the expected number of male hatchlings over a set of nests.

For each nest the supplied end of incubation places the developmental window, the window yields its equivalent temperature, and the sex-ratio curve turns that temperature into a male fraction. The expected number of males of the nest is its number of sexed hatchlings times that fraction, and the step returns the sum over the nests.

The end of incubation is an input here, so the same nests can be summed under two conventions. Under pipping the end is the modelled crossing of the hatchling length, under emergence it is the last reading of the record. The two sums are the two sides of the change that the orchestrator reports.

Returns
-------
total : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expected_males(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
    end_times: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
    Rho25_s: float = 100.0,
) -> float:
    """Sum of the expected male hatchlings over the nests, for supplied ends.

    Parameters
    ----------
    records : sequence of array-like
        One record per nest. Each is a 2-D array with two columns, the times in
        minutes from laying and the temperature in degrees Celsius at each
        reading, with strictly increasing times and at least two readings.
    hatchling_lengths : array-like
        One hatchling straight carapace length in mm per nest, positive.
    sexed_counts : array-like
        One number of sexed hatchlings per nest, non-negative.
    end_times : array-like
        One end of incubation in minutes from laying per nest, inside its record.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    P, SL, SH : float, optional
        Parameters of the male fraction curve of step 7.
    DHA_s, DHH_s, T12H_s, shape1, shape2 : float, optional
        Parameters of the sexualization norm and the fitted beta density.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    frac_begin, frac_end : float, optional
        Stage fractions of the modelled size at each nest's end of incubation.
    Rho25_s : float, optional
        Rate of the sexualization norm at the reference temperature (100.0).

    Returns
    -------
    float
        The sum over the nests of the number of sexed hatchlings times the male
        fraction of that nest. Each male fraction is the sex-ratio curve of step 7
        at the equivalent temperature of the window placed by step 4 and weighted
        by step 6 for that nest's supplied end of incubation.

    Raises
    ------
    ValueError
        If the four per-nest inputs do not have the same length, if a record is
        not a 2-D array with two columns and at least two readings, if a
        hatchling length is not positive, if a sexed count is negative or not
        finite, or if any per-nest computation rejects its inputs.
    """
    return total

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_expected_males(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
    end_times: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
    Rho25_s: float = 100.0,
) -> float:
    """Reference implementation for expected_males."""
    recs = list(records)
    hatchlings = np.asarray(hatchling_lengths, dtype=float)
    sexed = np.asarray(sexed_counts, dtype=float)
    ends = np.asarray(end_times, dtype=float)
    if not recs:
        raise ValueError("records must not be empty")
    if not (hatchlings.ndim == 1 and sexed.ndim == 1 and ends.ndim == 1
            and hatchlings.size == sexed.size == ends.size == len(recs)):
        raise ValueError("the per-nest inputs must be one-dimensional and equally long")
    if not (np.all(np.isfinite(hatchlings)) and np.all(np.isfinite(sexed)) and np.all(np.isfinite(ends))):
        raise ValueError("the per-nest inputs must be finite")
    if np.any(hatchlings <= 0.0):
        raise ValueError("hatchling lengths must be positive")
    if np.any(sexed < 0.0):
        raise ValueError("sexed counts must be non-negative")
    total = 0.0
    for record, hatchling, count, end in zip(recs, hatchlings, sexed, ends):
        r = np.asarray(record, dtype=float)
        if r.ndim != 2 or r.shape[1] != 2 or r.shape[0] < 2:
            raise ValueError("each record must be a 2-D array with two columns and at least two readings")
        times, temperature = r[:, 0], r[:, 1]
        limits = _oracle_window_limits(times, temperature, float(hatchling), float(end), DHA, DHH,
                                       T12H, Rho25, start_size, asymptotic_ratio, frac_begin, frac_end)
        equivalent = _oracle_constant_temperature_equivalent(
            times, temperature, float(hatchling), float(limits[1]), float(limits[2]), DHA, DHH,
            T12H, Rho25, DHA_s, DHH_s, T12H_s, shape1, shape2, start_size, asymptotic_ratio, Rho25_s)
        total += float(count) * _oracle_male_fraction(equivalent, P, SL, SH)
    return float(total)

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
        "    return np.column_stack([times, temps])\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        expected_males(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_expected_males(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: three nests of different lengths and sexed counts, each summed under
        # the emergence convention, so every end of incubation is a record end.
        {"setup": record +
                  "RECORDS = [make_record(48.0), make_record(44.0, 360.0, 29.5), make_record(52.0, 1440.0, 30.5)]\n"
                  "HATCH = np.array([50.0, 49.0, 52.0], dtype=float)\n"
                  "SEXED = np.array([10.0, 8.0, 10.0], dtype=float)\n"
                  "ENDS = np.array([69120.0, 63360.0, 74880.0], dtype=float)\n",
         "call": "expected_males(RECORDS, HATCH, SEXED, ENDS)",
         "gold_call": "_oracle_expected_males(RECORDS, HATCH, SEXED, ENDS)"},
        # Normal: the same nests under the pipping convention, where the ends are the
        # modelled crossings of the hatchling lengths.
        {"setup": record +
                  "RECORDS = [make_record(48.0), make_record(44.0, 360.0, 29.5), make_record(52.0, 1440.0, 30.5)]\n"
                  "HATCH = np.array([50.0, 49.0, 52.0], dtype=float)\n"
                  "SEXED = np.array([10.0, 8.0, 10.0], dtype=float)\n"
                  "ENDS = np.array([65400.0, 59800.0, 71000.0], dtype=float)\n",
         "call": "expected_males(RECORDS, HATCH, SEXED, ENDS)",
         "gold_call": "_oracle_expected_males(RECORDS, HATCH, SEXED, ENDS)"},
        # Boundary: one nest only.
        {"setup": record +
                  "RECORDS = [make_record(46.0)]\n"
                  "HATCH = np.array([50.5], dtype=float)\n"
                  "SEXED = np.array([12.0], dtype=float)\n"
                  "ENDS = np.array([66240.0], dtype=float)\n",
         "call": "expected_males(RECORDS, HATCH, SEXED, ENDS)",
         "gold_call": "_oracle_expected_males(RECORDS, HATCH, SEXED, ENDS)"},
        # Edge: a nest whose sexed count is zero contributes nothing.
        {"setup": record +
                  "RECORDS = [make_record(46.0)]\n"
                  "HATCH = np.array([50.5], dtype=float)\n"
                  "SEXED = np.array([0.0], dtype=float)\n"
                  "ENDS = np.array([66240.0], dtype=float)\n",
         "call": "expected_males(RECORDS, HATCH, SEXED, ENDS)",
         "gold_call": "_oracle_expected_males(RECORDS, HATCH, SEXED, ENDS)"},
        # Invalid: the per-nest inputs differ in length.
        {"setup": record +
                  "RECORDS = [make_record(46.0)]\n"
                  "HATCH = np.array([50.5], dtype=float)\n"
                  "SEXED = np.array([10.0, 8.0], dtype=float)\n"
                  "ENDS = np.array([66240.0], dtype=float)\n" + invalid,
         "call": "run_model(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED, end_times=ENDS)",
         "gold_call": "run_gold(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED, end_times=ENDS)"},
        # Invalid: a record with one column.
        {"setup": record +
                  "RECORDS = [np.arange(10.0).reshape(-1, 1)]\n"
                  "HATCH = np.array([50.5], dtype=float)\n"
                  "SEXED = np.array([10.0], dtype=float)\n"
                  "ENDS = np.array([1000.0], dtype=float)\n" + invalid,
         "call": "run_model(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED, end_times=ENDS)",
         "gold_call": "run_gold(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED, end_times=ENDS)"},
    ]
