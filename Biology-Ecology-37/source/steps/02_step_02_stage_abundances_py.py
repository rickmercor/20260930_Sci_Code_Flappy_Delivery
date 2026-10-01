"""
Adult and juvenile female abundance in each requested year under quasi-equilibrium stage-structured dynamics.

Close-kin and self-recapture probabilities are reciprocals of the number of animals

that compete to be the kin, or to be the same animal, so they need abundance by year

and by stage. A stage-structured female model tracks adults directly, with abundance

changing at a constant annual rate over the modelled years. Juveniles are not tracked

separately: if the age composition within each stage is the quasi-equilibrium one

implied by the survival rates and the rate of change, juvenile abundance follows from

adult abundance in closed form.

Returns
-------
np.ndarray of shape (2, Y), adult (row 0) and juvenile (row 1) female abundance in each requested year
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stage_abundances(theta: "np.ndarray", ref_year: int, years: "np.ndarray") -> "np.ndarray":
    '''Adult and juvenile female abundance in each requested year under quasi-equilibrium stage-structured dynamics.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3):
    N_ref is the number of adult females (age 6 and older, with no maximum
    age) in ref_year, adult female abundance changes by the factor exp(r)
    per year, phi_A is the annual survival of adults and phi_J that of
    juveniles (ages 1 to 5; the year from age 5 to age 6 is a juvenile
    year). The last two entries are breeding probabilities and do not enter
    this step. Juvenile female abundance is the number of females aged 1 to
    5 when the age composition within each stage is the quasi-equilibrium
    (stable-age) one implied by the survival rates and the rate of change
    exp(r).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,): ln N_ref from 0 to 30, r from -0.5 to 0.5, and each of
        the four logits from -15 to 15, with exp(r) > phi_A so that the
        adult age distribution is summable. Every later step that takes
        theta uses this domain.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    years : np.ndarray
        Shape (Y,), Y >= 1; integers from 1900 to 2200, in any order.

    Returns
    -------
    abundance : np.ndarray
        Shape (2, Y). Row 0 is adult female abundance and row 1 juvenile
        female abundance in years[i].

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain above
        (exp(r) <= phi_A included), if ref_year is not an integer from 1900
        to 2200, or if years is not a non-empty one-dimensional array of
        integers from 1900 to 2200.
    '''
    return abundance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sa_year(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1900 <= value <= 2200:
        raise ValueError(f"{name} must be an integer from 1900 to 2200")
    return int(value)


def _oracle_stage_abundances(theta: "np.ndarray", ref_year: int, years: "np.ndarray") -> "np.ndarray":
    th = np.asarray(theta)
    if th.shape != (6,) or not np.issubdtype(th.dtype, np.number) or not np.all(np.isfinite(th)):
        raise ValueError("theta must be a finite array of shape (6,)")
    re_th = np.real(th)
    if not (0.0 <= re_th[0] <= 30.0 and -0.5 <= re_th[1] <= 0.5 and np.all(np.abs(re_th[2:]) <= 15.0)):
        raise ValueError("theta must lie in the stated domain")
    y0 = _sa_year("ref_year", ref_year)
    y = np.asarray(years)
    if y.ndim != 1 or y.size == 0 or not np.issubdtype(y.dtype, np.integer) or y.min() < 1900 or y.max() > 2200:
        raise ValueError("years must be a non-empty 1-D array of integers from 1900 to 2200")
    log_n, r = th[0], th[1]
    phi_j = 1.0 / (1.0 + np.exp(-th[3]))
    lam = np.exp(r)
    lam_minus_phi_a = np.expm1(r) + 1.0 / (1.0 + np.exp(th[2]))   # exp(r) - phi_A without cancellation near phi_A = 1
    if not float(np.real(lam_minus_phi_a)) > 0.0:
        raise ValueError("exp(r) must exceed adult survival")
    adult = np.exp(log_n + r * (y - y0))
    # juveniles aged 1..5 per adult: sum_a lam^-a phi_J^(a-1) over the adult sum lam^-6 phi_J^5 / (1 - phi_A/lam)
    juv_sum = sum(lam ** (-a) * phi_j ** (a - 1) for a in range(1, 6))
    ratio = juv_sum * lam_minus_phi_a * lam ** 5 / phi_j ** 5
    return np.array([adult, adult * ratio])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a slowly declining population over the modelled years ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(48000.0), -0.012, np.log(0.955 / 0.045), np.log(0.85 / 0.15), np.log(0.16 / 0.84), np.log(0.38 / 0.62)])\n",
            "call": "stage_abundances(theta, 2016, np.arange(2002, 2028))",
            "gold_call": "_oracle_stage_abundances(theta, 2016, np.arange(2002, 2028))",
        },
        # --- boundary: stationary population (r = 0), years in reverse order around the reference year ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(1000.0), 0.0, np.log(0.9 / 0.1), np.log(0.7 / 0.3), 0.0, 0.0])\n",
            "call": "stage_abundances(theta, 2000, np.array([2003, 2000, 1998]))",
            "gold_call": "_oracle_stage_abundances(theta, 2000, np.array([2003, 2000, 1998]))",
        },
        # --- edge: fast decline barely above adult survival, so adults live long and juveniles are scarce ---
        {
            "setup": "import numpy as np\ntheta = np.array([np.log(250000.0), -0.08, np.log(0.92 / 0.08), np.log(0.6 / 0.4), -1.0, 1.0])\n",
            "call": "stage_abundances(theta, 2010, np.array([1990, 2010, 2050]))",
            "gold_call": "_oracle_stage_abundances(theta, 2010, np.array([1990, 2010, 2050]))",
        },
    ]
