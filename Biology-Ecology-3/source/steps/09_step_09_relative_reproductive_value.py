"""
Step 9: orchestrator: the relative reproductive value of one class.

The graded pipeline chains steps 01 through 08 end to end on the monitoring records. It estimates the age-specific survival from the release records (with the detection probability as an auxiliary maximum-likelihood quantity) and the fecundity from the reproduction records.

 It then builds the transition matrix and the birth-pulse operator. Next it separates the transition matrix into survival and ageing and splits the survival operator around the birth pulse for the census month. 

After, it assembles the intermediate projection matrix S^(M/12) (I + R) T S^(1 - M/12). Finally it computes the growth rate and stable age distribution of that matrix and from them its reproductive values, expressed relative to the youngest class. 

The returned scalar is that relative reproductive value for one named class. The graded endpoint is the class aged 8 years under a census taken 11 months after the birth pulse.

Returns
-------
float(rv[int(age)]) : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relative_reproductive_value(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    age: int = 8,
    months: float = 11,
) -> float:
    """Reproductive value of one age class, relative to the youngest class.

    Parameters
    ----------
    release_records : array-like of shape (n_records, 4 + K)
        Release records of step 01: one row per release group of the April census
        programme, ``[release year, birth year, females released, females later found
        dead, first recaptures at the 1st, 2nd, ..., K-th following census]``.
    reproduction_records : array-like of shape (n_records, 5)
        Reproduction records of step 01: one row per cohort and May birth pulse,
        ``[pulse year, birth year, females present at the pulse, females that gave
        birth, fawns born]``. The female share of the fawns is one half.
    age : int, optional
        Age class whose value is returned, from 0 to n-1 (the last class is the pooled
        terminal class). The graded endpoint uses the default, age 8.
    months : float, optional
        Months elapsed since the birth pulse at the census, from 0 to 12 inclusive.
        The task's graded census uses the default, 11 months.

    Returns
    -------
    float
        The reproductive value of that class relative to the youngest class (class 0,
        which has value 1), for a census taken `months` after the birth pulse.

    Raises
    ------
    ValueError
        If the release records or the reproduction records fail the validation of
        step 01, if a later pipeline step rejects its inputs (including `months`
        outside 0 to 12), or if `age` is outside 0 to n-1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_relative_reproductive_value(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    age: int = 8,
    months: float = 11,
) -> float:
    """Reference implementation for relative_reproductive_value."""
    import numpy as np

    survival, fecundity = _oracle_estimate_rates(release_records, reproduction_records)
    projection = _oracle_intermediate_projection(survival, fecundity, months)
    rv = _oracle_reproductive_values(projection)
    if not isinstance(age, (int, np.integer)) or isinstance(age, bool):
        raise ValueError("age must be an integer")
    if age < 0 or age > rv.size - 1:
        raise ValueError("age must be from 0 to the number of classes minus one")
    return float(rv[int(age)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    build = (
        "import numpy as np\n"
        "def build_releases(n_events=6, released=200, survival=0.55, detection=0.9, k=4):\n"
        "    rows = []\n"
        "    for a in range(14):\n"
        "        for j in range(n_events):\n"
        "            year = 2000 + a + j\n"
        "            cells = []\n"
        "            for t in range(k):\n"
        "                q = survival ** (t + 1) * detection * (1 - detection) ** t\n"
        "                cells.append(int(round(released * q)))\n"
        "            never = released - sum(cells)\n"
        "            rows.append([year, year - a - 1, released, int(round(never * 0.2))] + cells)\n"
        "    return np.array(rows, dtype=int)\n"
        "def build_reproduction():\n"
        "    return np.array([[2000 + age, 2000, 50, 35, 60] for age in range(2, 14)], dtype=int)\n"
    )
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        relative_reproductive_value(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_relative_reproductive_value(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the full chain on a fourteen-class release table.
        {"setup": build + "REL = build_releases()\nREC = build_reproduction()\n",
         "call": "relative_reproductive_value(REL, REC)",
         "gold_call": "_oracle_relative_reproductive_value(REL, REC)"},
        # Normal: a nine-month census, the paper's example configuration.
        {"setup": build + "REL = build_releases()\nREC = build_reproduction()\n",
         "call": "relative_reproductive_value(REL, REC, 8, 9)",
         "gold_call": "_oracle_relative_reproductive_value(REL, REC, 8, 9)"},
        # Boundary: the youngest class is the anchor and has value one.
        {"setup": build + "REL = build_releases()\nREC = build_reproduction()\n",
         "call": "relative_reproductive_value(REL, REC, 0)",
         "gold_call": "_oracle_relative_reproductive_value(REL, REC, 0)"},
        # Boundary: a post-breeding census, where the newborn class is not observable.
        {"setup": build + "REL = build_releases()\nREC = build_reproduction()\n",
         "call": "relative_reproductive_value(REL, REC, 8, 0)",
         "gold_call": "_oracle_relative_reproductive_value(REL, REC, 8, 0)"},
        # Invalid: an age outside the class range.
        {"setup": build + "REL = build_releases()\nREC = build_reproduction()\n" + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, age=14)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, age=14)"},
        # Invalid: a census month outside the year.
        {"setup": build + "REL = build_releases()\nREC = build_reproduction()\n" + invalid,
         "call": "run_model(release_records=REL, reproduction_records=REC, months=13)",
         "gold_call": "run_gold(release_records=REL, reproduction_records=REC, months=13)"},
    ]
