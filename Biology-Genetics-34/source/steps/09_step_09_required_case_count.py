"""
Convert an equivalent quantitative-trait sample size on the liability scale into the number of cases an ascertained case-control study of a disease must collect.

A case-control study carries only as much information about a liability threshold trait as a quantitative-trait study of size N_eq = i**2 * v * (1 - v) * N01 / (1 - K)**2, where K is the population prevalence, v the case fraction of the sample, N01 the total of cases and controls and i = phi(z_K) / K the mean liability of cases, with z_K the standard normal quantile at 1 - K. Inverting for N01 and taking the case fraction of it gives the number of cases the study needs.

Returns
-------
float: the number of cases the case-control study must collect, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def required_case_count(n_equivalent: float, prevalence: float,
                        case_fraction: float) -> float:
    """Cases needed to match a given equivalent quantitative-trait sample size.

    Parameters
    ----------
    n_equivalent : float
        Equivalent quantitative-trait sample size on the liability scale
        (n_equivalent > 0).
    prevalence : float
        Lifetime prevalence of the disease in the population
        (0 < prevalence < 1).
    case_fraction : float
        Fraction of the ascertained sample that are cases
        (0 < case_fraction < 1).

    Returns
    -------
    n_cases : float
        Number of cases the study must collect, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is not a finite real number; if ``n_equivalent`` is
        not greater than zero; if ``prevalence`` is not strictly between 0
        and 1; or if ``case_fraction`` is not strictly between 0 and 1, a
        study of all cases or all controls being uninformative.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_required_case_count(n_equivalent: float, prevalence: float,
                                case_fraction: float) -> float:
    import numpy as np
    from math import erf, exp, sqrt, pi

    for name, val in (("n_equivalent", n_equivalent), ("prevalence", prevalence),
                      ("case_fraction", case_fraction)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    n_equivalent = float(n_equivalent)
    prevalence = float(prevalence)
    case_fraction = float(case_fraction)
    if n_equivalent <= 0.0:
        raise ValueError("n_equivalent must be > 0")
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise ValueError("prevalence must lie strictly between 0 and 1")
    if case_fraction <= 0.0 or case_fraction >= 1.0:
        raise ValueError("case_fraction must lie strictly between 0 and 1")

    # Standard normal quantile at 1 - K, obtained by bisecting the error
    # function so that no external special-function package is needed.
    def _standard_normal_quantile(p):
        lo, hi = -40.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    truncation = _standard_normal_quantile(1.0 - prevalence)
    density = exp(-0.5 * truncation * truncation) / sqrt(2.0 * pi)

    # Mean liability of cases: the taller this is, the more a single case
    # tells us, so a rarer disease needs fewer cases per unit of information.
    mean_liability_cases = density / prevalence

    # Invert the equivalent-sample-size relation for the total sample, then
    # take the case fraction of it.
    total = (n_equivalent * (1.0 - prevalence) ** 2
             / (mean_liability_cases ** 2 * case_fraction * (1.0 - case_fraction)))
    return float(case_fraction * total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the two quantiles below are standard constants, so
        # the whole conversion can be written down without the oracle. A
        # response that returns the total sample rather than the cases, or that
        # omits the squared complement of the prevalence, misses both.
        {
            "setup": """import numpy as np
n_eq = 400000.0
q05 = 1.6448536269514722
i05 = (np.exp(-0.5 * q05 ** 2) / np.sqrt(2.0 * np.pi)) / 0.05
half = (np.exp(0.0) / np.sqrt(2.0 * np.pi)) / 0.5
cases_05 = n_eq * (1.0 - 0.05) ** 2 / (i05 ** 2 * (1.0 - 0.5))
cases_50 = n_eq * (1.0 - 0.5) ** 2 / (half ** 2 * (1.0 - 0.5))
EXPECTED = float(cases_05 + 1000.0 * cases_50)
""",
            "call": ("float(required_case_count(n_eq, 0.05, 0.5)"
                     " + 1000.0 * required_case_count(n_eq, 0.5, 0.5))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: an unbalanced study of a moderately common disease ---
        {
            "setup": """import numpy as np
""",
            "call": "required_case_count(750000.0, 0.08, 0.25)",
            "gold_call": "_oracle_required_case_count(750000.0, 0.08, 0.25)",
        },
        # --- Boundary: a very rare disease, where each case is highly
        # informative but the population complement barely moves ---
        {
            "setup": """import numpy as np
""",
            "call": "required_case_count(1.2e6, 0.0005, 0.5)",
            "gold_call": "_oracle_required_case_count(1.2e6, 0.0005, 0.5)",
        },
        # --- Edge: a case-poor design, where nearly the whole sample is
        # controls ---
        {
            "setup": """import numpy as np
""",
            "call": "required_case_count(3.0e5, 0.12, 0.02)",
            "gold_call": "_oracle_required_case_count(3.0e5, 0.12, 0.02)",
        },
        # --- Invalid: a prevalence of zero ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        required_case_count(1.0e6, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_required_case_count(1.0e6, 0.0, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a case fraction of one, which leaves no controls ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        required_case_count(1.0e6, 0.01, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_required_case_count(1.0e6, 0.01, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
