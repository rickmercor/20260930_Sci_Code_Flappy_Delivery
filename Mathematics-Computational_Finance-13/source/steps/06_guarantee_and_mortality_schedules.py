"""
Build the deterministic contractual schedules for the contract. Two guarantee levels accumulate the fund contributions at two different rates: one governs the maturity and death benefits, the other the guarantee component of the surrender payoff. Both are indexed by policy anniversary and account only for contributions credited at strictly earlier anniversaries, so both vanish at inception. Alongside them, build the probability of surviving from issue to each anniversary under a Gompertz-Makeham hazard. Return the three schedules stacked, each of length one more than the number of policy years.

The contract credits a fixed amount D to the fund at each policy anniversary at which it remains in force, and guarantees a minimum benefit defined by accumulating those contributions at a contractual rate. For an accumulation rate gamma, the guarantee level at anniversary i is

    G_i = sum over j from 0 to i-1 of D * exp(gamma * (i - j))

so the contribution credited at anniversary j has been accumulating for i - j years by the time anniversary i is reached. The sum runs to i-1 rather than i, so G_0 is zero and each level reflects only contributions already credited.

Two such levels are required because the contract uses different rates for different benefits. One rate governs the benefit paid at maturity and the benefit paid on death, the other governs the guarantee component of the surrender payoff. These are separate contractual parameters and are not interchangeable: using a single rate for both misprices the surrender right, since the two schedules enter different terms of the backward recursion and diverge as the contract runs.

Mortality is assumed independent of the financial factors, so it enters the valuation only through deterministic weights. Under a Gompertz-Makeham specification the one-year hazard at attained age x is a constant term plus a geometrically increasing term,

    hazard(x) = A + B * c^x

and the one-year probability of death between ages x and x+1, conditional on survival to x, is one minus the exponential of the negative hazard. The probability of surviving from issue age a0 to anniversary i is then the product of the one-year survival probabilities over the intervening ages, equivalently the exponential of minus the accumulated hazard. This survival schedule is what weights the premium stream, since premiums are payable only while the policyholder is alive, and the one-year death probabilities needed by the backward recursion are recoverable from consecutive entries.

Returns
-------
np.ndarray of shape (3, n_years + 1): row 0 the maturity guarantee level by anniversary, row 1 the surrender guarantee level, row 2 the survival probability from issue to that anniversary
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def guarantee_and_mortality_schedules(contribution: float, g_maturity: float, g_surrender: float,
                                      n_years: int, issue_age: int, mk_a: float, mk_b: float,
                                      mk_c: float) -> np.ndarray:
    '''Guarantee accumulation schedules and survival probabilities by anniversary.

    Parameters
    ----------
    contribution : float
        Strictly positive amount credited to the fund at each anniversary.
    g_maturity : float
        Accumulation rate for the maturity and death guarantee.
    g_surrender : float
        Accumulation rate for the surrender guarantee.
    n_years : int
        Number of policy years, at least one.
    issue_age : int
        Non-negative age of the policyholder at issue.
    mk_a : float
        Non-negative constant term of the Gompertz-Makeham hazard.
    mk_b : float
        Strictly positive scale of the age-dependent term.
    mk_c : float
        Shape of the age-dependent term, strictly greater than one.

    Returns
    -------
    schedules : np.ndarray
        Shape (3, n_years + 1). Row 0 holds the maturity guarantee level at each
        anniversary, row 1 the surrender guarantee level, and row 2 the
        probability of surviving from issue to that anniversary.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid contract and mortality basis.
    '''
    return schedules

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_guarantee_and_mortality_schedules(contribution: float, g_maturity: float,
                                              g_surrender: float, n_years: int, issue_age: int,
                                              mk_a: float, mk_b: float, mk_c: float) -> np.ndarray:
    if not isinstance(contribution, (int, float, np.integer, np.floating)) or not float(contribution) > 0.0:
        raise ValueError("contribution must be a positive real number")
    for name, val in (("g_maturity", g_maturity), ("g_surrender", g_surrender)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
    if not isinstance(n_years, (int, np.integer)) or int(n_years) < 1:
        raise ValueError("n_years must be an integer of at least one")
    if not isinstance(issue_age, (int, np.integer)) or int(issue_age) < 0:
        raise ValueError("issue_age must be a non-negative integer")
    if not isinstance(mk_a, (int, float, np.integer, np.floating)) or float(mk_a) < 0.0:
        raise ValueError("mk_a must be non-negative")
    if not isinstance(mk_b, (int, float, np.integer, np.floating)) or not float(mk_b) > 0.0:
        raise ValueError("mk_b must be positive")
    if not isinstance(mk_c, (int, float, np.integer, np.floating)) or not float(mk_c) > 1.0:
        raise ValueError("mk_c must exceed one")

    D = float(contribution)
    gm, gs = float(g_maturity), float(g_surrender)
    T, a0 = int(n_years), int(issue_age)
    A, B, C = float(mk_a), float(mk_b), float(mk_c)

    out = np.zeros((3, T + 1), dtype=float)
    for i in range(T + 1):
        if i == 0:
            continue
        j = np.arange(0, i)
        out[0, i] = float((D * np.exp(gm * (i - j))).sum())
        out[1, i] = float((D * np.exp(gs * (i - j))).sum())

    out[2, 0] = 1.0
    for i in range(T):
        hazard = A + B * C ** (a0 + i)
        if not np.isfinite(hazard):
            raise ValueError("the mortality parameters produce a non-finite hazard")
        out[2, i + 1] = out[2, i] * np.exp(-hazard)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = "import numpy as np\n"
    err = """import numpy as np
def run_model(**kw):
    a = dict(contribution=100.0, g_maturity=0.02, g_surrender=0.01, n_years=3,
             issue_age=50, mk_a=5e-4, mk_b=1e-5, mk_c=1.10)
    a.update(kw)
    try:
        guarantee_and_mortality_schedules(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(contribution=100.0, g_maturity=0.02, g_surrender=0.01, n_years=3,
             issue_age=50, mk_a=5e-4, mk_b=1e-5, mk_c=1.10)
    a.update(kw)
    try:
        _oracle_guarantee_and_mortality_schedules(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": pre,
         "call": "guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 3, 50, 5e-4, 1e-5, 1.10)",
         "gold_call": "_oracle_guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 3, 50, 5e-4, 1e-5, 1.10)"},
        {"setup": pre,
         "call": "guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 10, 65, 5e-4, 1e-5, 1.10)",
         "gold_call": "_oracle_guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 10, 65, 5e-4, 1e-5, 1.10)"},
        {"setup": pre,
         "call": "guarantee_and_mortality_schedules(250.0, 0.015, 0.035, 5, 40, 3e-4, 2e-5, 1.08)",
         "gold_call": "_oracle_guarantee_and_mortality_schedules(250.0, 0.015, 0.035, 5, 40, 3e-4, 2e-5, 1.08)"},
        {"setup": pre,
         "call": "guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 1, 50, 5e-4, 1e-5, 1.10)",
         "gold_call": "_oracle_guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 1, 50, 5e-4, 1e-5, 1.10)"},
        {"setup": pre,
         "call": "guarantee_and_mortality_schedules(100.0, 0.0, 0.0, 3, 50, 5e-4, 1e-5, 1.10)",
         "gold_call": "_oracle_guarantee_and_mortality_schedules(100.0, 0.0, 0.0, 3, 50, 5e-4, 1e-5, 1.10)"},
        {"setup": pre,
         "call": "guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 4, 50, 2e-3, 1e-12, 1.0001)",
         "gold_call": "_oracle_guarantee_and_mortality_schedules(100.0, 0.02, 0.01, 4, 50, 2e-3, 1e-12, 1.0001)"},
        {"setup": err, "call": "run_model(contribution=0.0)", "gold_call": "run_gold(contribution=0.0)"},
        {"setup": err, "call": "run_model(n_years=0)", "gold_call": "run_gold(n_years=0)"},
        {"setup": err, "call": "run_model(mk_c=1.0)", "gold_call": "run_gold(mk_c=1.0)"},
    ]
