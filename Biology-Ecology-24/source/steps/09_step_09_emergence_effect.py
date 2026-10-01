"""
Step 9: orchestrator: the emergence effect on the expected male production.

The graded pipeline chains steps 1 through 8 on the nest records. It derives the pipping time of every nest from the growth model of step 2, because the records end at emergence while the developmental window must be placed against the two conventions. It then places both windows, weights them, and converts the two equivalent temperatures into male fractions. It sums the expected males over the nests once with pipping as the end of incubation and once with emergence, and returns the difference in the stated order, pipping minus emergence.

The returned scalar is the change in the expected number of male hatchlings across the nests when emergence is treated as the end of incubation. The graded endpoint is that change over the nine supplied nests, reported to two decimal places.

Returns
-------
effect : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def emergence_effect(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
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
    """Change in the expected male hatchlings from the emergence convention.

    Parameters
    ----------
    records : sequence of array-like
        One record per nest. Each is a 2-D array with two columns, the times in
        minutes from laying and the temperature in degrees Celsius at each
        reading, with strictly increasing times and at least two readings. The
        record ends at emergence.
    hatchling_lengths : array-like
        One hatchling straight carapace length in mm per nest, positive.
    sexed_counts : array-like
        One number of sexed hatchlings per nest, non-negative.
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
        The pipping sum minus the emergence sum of the expected male hatchlings
        over the nests. Pipping is derived per nest as the modelled crossing of
        that nest's hatchling length. Emergence is the last reading of that
        nest's record, including a final reading whose temperature repeats the
        one before. A nest whose modelled embryo never reaches its hatchling
        length has coincident ends and contributes zero, and it does not cancel
        the other nests. A window limit whose target lies below the modelled
        size at laying is the first reading time.

    Raises
    ------
    ValueError
        If the three per-nest inputs do not have the same length, if a record is
        not a 2-D array with two columns and at least two readings, if a hatchling
        length is not positive, if a sexed count is negative or not finite, or if
        any per-nest computation rejects its inputs.
    """
    return effect

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_emergence_effect(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
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
    """Reference implementation for emergence_effect."""
    recs = list(records)
    hatchlings = np.asarray(hatchling_lengths, dtype=float)
    sexed = np.asarray(sexed_counts, dtype=float)
    if not recs:
        raise ValueError("records must not be empty")
    if not (hatchlings.ndim == 1 and sexed.ndim == 1
            and hatchlings.size == sexed.size == len(recs)):
        raise ValueError("the per-nest inputs must be one-dimensional and equally long")
    if not (np.all(np.isfinite(hatchlings)) and np.all(np.isfinite(sexed))):
        raise ValueError("the per-nest inputs must be finite")
    if np.any(hatchlings <= 0.0):
        raise ValueError("hatchling lengths must be positive")
    if np.any(sexed < 0.0):
        raise ValueError("sexed counts must be non-negative")
    pipping = np.empty(len(recs), dtype=float)
    emergence = np.empty(len(recs), dtype=float)
    for i, (record, hatchling) in enumerate(zip(recs, hatchlings)):
        r = np.asarray(record, dtype=float)
        if r.ndim != 2 or r.shape[1] != 2 or r.shape[0] < 2:
            raise ValueError("each record must be a 2-D array with two columns and at least two readings")
        pipping[i] = _oracle_pipping_time(r[:, 0], r[:, 1], float(hatchling), DHA, DHH, T12H,
                                          Rho25, start_size, asymptotic_ratio)
        emergence[i] = float(r[-1, 0])
    under_pipping = _oracle_expected_males(recs, hatchlings, sexed, pipping, DHA, DHH, T12H, Rho25,
                                           P, SL, SH, DHA_s, DHH_s, T12H_s, shape1, shape2,
                                           start_size, asymptotic_ratio, frac_begin, frac_end, Rho25_s)
    under_emergence = _oracle_expected_males(recs, hatchlings, sexed, emergence, DHA, DHH, T12H, Rho25,
                                             P, SL, SH, DHA_s, DHH_s, T12H_s, shape1, shape2,
                                             start_size, asymptotic_ratio, frac_begin, frac_end, Rho25_s)
    return float(under_pipping - under_emergence)

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
        "        emergence_effect(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_emergence_effect(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: one nest whose record reaches the modelled hatchling length, so the
        # two conventions place different windows.
        {"setup": record +
                  "RECORDS = [make_record(48.0)]\n"
                  "HATCH = np.array([50.0], dtype=float)\n"
                  "SEXED = np.array([10.0], dtype=float)\n",
         "call": "emergence_effect(RECORDS, HATCH, SEXED)",
         "gold_call": "_oracle_emergence_effect(RECORDS, HATCH, SEXED)"},
        # Normal: three nests of different lengths, step sizes and sexed counts, so the
        # pipeline runs the whole chain several times.
        {"setup": record +
                  "RECORDS = [make_record(48.0), make_record(44.0, 360.0, 29.5), make_record(52.0, 1440.0, 30.5)]\n"
                  "HATCH = np.array([50.0, 49.0, 52.0], dtype=float)\n"
                  "SEXED = np.array([10.0, 8.0, 10.0], dtype=float)\n",
         "call": "emergence_effect(RECORDS, HATCH, SEXED)",
         "gold_call": "_oracle_emergence_effect(RECORDS, HATCH, SEXED)"},
        # Boundary: a record that ends before the modelled embryo reaches its hatchling
        # length, so both conventions share the same end of incubation and the change
        # is exactly zero.
        {"setup": "import numpy as np\n"
                  "TIMES = np.arange(0.0, 7200.0 + 1.0, 720.0)\n"
                  "RECORDS = [np.column_stack([TIMES, np.full(TIMES.size, 29.0)])]\n"
                  "HATCH = np.array([50.0], dtype=float)\n"
                  "SEXED = np.array([10.0], dtype=float)\n",
         "call": "emergence_effect(RECORDS, HATCH, SEXED)",
         "gold_call": "_oracle_emergence_effect(RECORDS, HATCH, SEXED)"},
        # Edge: a two-reading record whose modelled embryo crosses its hatchling length
        # inside the first logging interval, so both window starts fall at the first
        # reading time.
        {"setup": "import numpy as np\n"
                  "RECORDS = [np.array([[0.0, 31.0], [20000.0, 31.0]])]\n"
                  "HATCH = np.array([0.45], dtype=float)\n"
                  "SEXED = np.array([10.0], dtype=float)\n",
         "call": "emergence_effect(RECORDS, HATCH, SEXED)",
         "gold_call": "_oracle_emergence_effect(RECORDS, HATCH, SEXED)"},
        # Edge: one crossing nest and one nest that never reaches its hatchling
        # length, so the noncrossing nest contributes zero without cancelling the
        # other nest's contribution.
        {"setup": record +
                  "SHORT = np.column_stack([np.arange(0.0, 7200.0 + 1.0, 720.0), np.full(11, 29.0)])\n"
                  "RECORDS = [make_record(48.0), SHORT]\n"
                  "HATCH = np.array([50.0, 50.0], dtype=float)\n"
                  "SEXED = np.array([10.0, 10.0], dtype=float)\n",
         "call": "emergence_effect(RECORDS, HATCH, SEXED)",
         "gold_call": "_oracle_emergence_effect(RECORDS, HATCH, SEXED)"},
        # Edge: a record whose last temperature repeats the one before, where
        # dropping the end marker would move the emergence end earlier and remove
        # the crossing.
        {"setup": "import numpy as np\n"
                  "RECORDS = [np.array([[0.0, 29.0], [20000.0, 29.5], [40000.0, 30.0], [60000.0, 30.5], [80000.0, 30.5]])]\n"
                  "HATCH = np.array([50.0], dtype=float)\n"
                  "SEXED = np.array([10.0], dtype=float)\n",
         "call": "emergence_effect(RECORDS, HATCH, SEXED)",
         "gold_call": "_oracle_emergence_effect(RECORDS, HATCH, SEXED)"},
        # Invalid: the per-nest inputs differ in length.
        {"setup": record +
                  "RECORDS = [make_record(48.0)]\n"
                  "HATCH = np.array([50.0], dtype=float)\n"
                  "SEXED = np.array([10.0, 8.0], dtype=float)\n" + invalid,
         "call": "run_model(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED)",
         "gold_call": "run_gold(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED)"},
        # Invalid: a non-positive hatchling length.
        {"setup": record +
                  "RECORDS = [make_record(48.0)]\n"
                  "HATCH = np.array([0.0], dtype=float)\n"
                  "SEXED = np.array([10.0], dtype=float)\n" + invalid,
         "call": "run_model(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED)",
         "gold_call": "run_gold(records=RECORDS, hatchling_lengths=HATCH, sexed_counts=SEXED)"},
    ]
