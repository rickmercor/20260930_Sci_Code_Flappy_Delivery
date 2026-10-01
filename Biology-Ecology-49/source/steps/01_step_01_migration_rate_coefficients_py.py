"""
Drift source and squared diffusion coefficient of the unit-time migrant count.

The number of migrants passing a fixed point of a river per unit time rises and

falls over a season and is highly variable between days, with a coefficient of

variation of order one even where the seasonal shape is well known. A convenient

way to carry both features is a nonnegative diffusion that is pinned to zero at

both ends of the season, with square-root noise so that the count cannot go

negative, and with a drift that is chosen after the fact: rather than positing

coefficients and deducing the moments, one posits the seasonal mean and spread

that field counts support and asks which coefficients reproduce them.

Returns
-------
np.ndarray of shape (2, n): row 0 the drift source in fish per day squared, row 1 the squared diffusion coefficient in fish per day squared, in the order of times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def migration_rate_coefficients(times: "np.ndarray", run: "np.ndarray") -> "np.ndarray":
    '''Drift source and squared diffusion coefficient of the unit-time migrant count.

    The unit-time count Z_t, in fish per day, is the nonnegative diffusion

        dZ = (a(t) - pull * Z / (season - t)) dt + sqrt(D(t) * Z) dB,

    on 0 < t < season, with Z = 0 at t = 0 and at t = season. Its mean follows the
    seasonal profile

        mean(t) = peak * (th/th_pk)**shape_up * ((1-th)/(1-th_pk))**shape_down,
        th = t/season,  th_pk = shape_up/(shape_up + shape_down),

    and its coefficient of variation equals cv at every time strictly inside the
    season. Exactly one pair of continuous functions (a, D) gives the process that
    mean and that coefficient of variation; return that pair.

    Parameters
    ----------
    times : np.ndarray
        Shape (n,), times in days at which the coefficients are wanted, each a
        finite value with 0 < t < season.
    run : np.ndarray
        Shape (6,), the run description in the order
        [season length in days, peak of the mean count in fish per day, shape_up,
        shape_down, coefficient of variation, pull]. The season length, the peak,
        the coefficient of variation and the pull are positive; shape_up and
        shape_down are greater than 1.

    Returns
    -------
    coefficients : np.ndarray
        Shape (2, n) of floats. Row 0 holds a(t) in fish per day squared, row 1
        holds D(t) in fish per day squared, in the order of times.

    Raises
    ------
    ValueError
        If times is not a one-dimensional array of finite values strictly inside
        the season, or if run is not a shape (6,) array of finite values with
        positive season length, peak, coefficient of variation and pull and with
        shape_up and shape_down greater than 1.
    '''
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_run(run):
    arr = np.asarray(run, dtype=float)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError("run must be a shape (6,) array of finite values")
    season, peak, up, down, cv, pull = arr
    if season <= 0.0 or peak <= 0.0 or cv <= 0.0 or pull <= 0.0:
        raise ValueError("season, peak, cv and pull must be positive")
    if up <= 1.0 or down <= 1.0:
        raise ValueError("shape_up and shape_down must be greater than 1")
    return season, peak, up, down, cv, pull


def _count_scale(run):
    """Amplitude p0 with mean(t) = p0 * th**up * (1-th)**down."""
    season, peak, up, down, cv, pull = _check_run(run)
    th_pk = up / (up + down)
    return peak / (th_pk ** up * (1.0 - th_pk) ** down)


def _count_mean(times, run):
    season, peak, up, down, cv, pull = _check_run(run)
    p0 = _count_scale(run)
    th = np.clip(np.asarray(times, dtype=float) / season, 0.0, 1.0)
    inside = (np.asarray(times, dtype=float) > 0.0) & (np.asarray(times, dtype=float) < season)
    return np.where(inside, p0 * th ** up * (1.0 - th) ** down, 0.0)


def _oracle_migration_rate_coefficients(times: "np.ndarray", run: "np.ndarray") -> "np.ndarray":
    season, peak, up, down, cv, pull = _check_run(run)
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a non-empty one-dimensional array of finite values")
    if np.any(t <= 0.0) or np.any(t >= season):
        raise ValueError("times must lie strictly inside the season")
    p0 = _count_scale(run)
    q0 = (cv * p0) ** 2
    q_up, q_down = 2.0 * up, 2.0 * down
    th = t / season
    one = 1.0 - th
    # mean(t) = p0 th^up (1-th)^down solves  d mean/dt = a - pull*mean/(season-t)
    a = (p0 / season) * th ** (up - 1.0) * one ** (down - 1.0) * ((pull - up - down) * th + up)
    # var(t) = q0 th^q_up (1-th)^q_down solves  d var/dt = -2 pull var/(season-t) + D*mean
    d = (q0 / (p0 * season)) * th ** (q_up - up - 1.0) * one ** (q_down - down - 1.0) \
        * ((2.0 * pull - q_up - q_down) * th + q_up)
    return np.vstack([a, d])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = (
        "import numpy as np\n"
        "run_a = np.array([80.0, 1200.0, 6.0, 8.0, 1.8, 30.0])\n"
        "run_b = np.array([45.0, 260.0, 2.5, 2.5, 0.8, 4.0])\n"
        "run_c = np.array([210.0, 9.0e4, 11.0, 4.0, 3.2, 95.0])\n"
        "t_a = np.array([4.0, 20.0, 34.2857142857, 52.0, 76.0])\n"
        "t_b = np.array([0.05, 22.5, 44.9])\n"
        "t_c = np.linspace(2.0, 208.0, 9)\n"
    )
    return [
        # --- normal: the shipped run, sampled across the season including its mean peak ---
        {
            "setup": setup,
            "call": "migration_rate_coefficients(t_a, run_a)",
            "gold_call": "_oracle_migration_rate_coefficients(t_a, run_a)",
        },
        # --- boundary: a short symmetric run with weak pull, evaluated hard against both ends ---
        {
            "setup": setup,
            "call": "migration_rate_coefficients(t_b, run_b)",
            "gold_call": "_oracle_migration_rate_coefficients(t_b, run_b)",
        },
        # --- edge: a long, strongly skewed, highly overdispersed run with strong pull ---
        {
            "setup": setup,
            "call": "migration_rate_coefficients(t_c, run_c)",
            "gold_call": "_oracle_migration_rate_coefficients(t_c, run_c)",
        },
    ]
