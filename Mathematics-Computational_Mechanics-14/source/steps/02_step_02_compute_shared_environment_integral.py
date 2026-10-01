"""
Each cell obeys $\dot P_i=r_ics(1-P_i/K_i)u(t)$ while the shared environment obeys $\dot u=-\frac{1}{N}\sum_i\frac{r_ics}{V}(1-P_i/K_i)u(t)$, so summing the two gives $u(t)=u_0-\frac{1}{N}\sum_iP_i(t)/V$: the mean free-particle density at a node is the pilot average of $u_0-P_q(t)/V$. Evaluate each pilot on its own, as a population in which every cell shares that pilot's rate and capacity, so that $P_q(t)$ is the exact trajectory of that homogeneous population. At each fine-grid node, including $t=0$, draw the $N_{\mathrm{prep}}$ pilot association rates as one vector and then the $N_{\mathrm{prep}}$ pilot carrying capacities as one vector, in that order, without reseeding. Integrate the resulting $\bar u$ by the trapezoidal rule and return $I(t)$ at the requested times.

Pilot cells are sampled from replicate-specific log-normal distributions. Their closed particle-association trajectories determine the mean free-particle density, which is integrated on a fine time grid by the trapezoidal rule.

Returns
-------
np.ndarray, the integrated free-particle density at each time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_shared_environment_integral(
    log_params: np.ndarray,
    times: np.ndarray,
    n_prep: int,
    fine_step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Compute $I(t)=\int_0^t \bar u(\tau)d\tau$ for one replicate.

    Parameters
    ----------
    log_params : np.ndarray
        Length-4 vector $(m_r,s_r,m_K,s_K)$ for one replicate.
    times : np.ndarray
        Observation times in seconds, all lying on the fine grid.
    n_prep : int
        Number of pilot cells sampled at each fine-grid time.
    fine_step : float
        Fine-grid spacing in seconds.
    rng : np.random.Generator
        Shared PCG64 random-number stream. The function draws pilot pairs at
        every fine-grid time, including zero, without reseeding.

    Returns
    -------
    result : np.ndarray
        Integrated mean free-particle density at each requested time.

    Raises
    ------
    ValueError
        If log_params does not have shape (4,), any log-normal scale is
        negative, times is empty, negative or off the fine grid, or n_prep or
        fine_step is not positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Helper functions used by the oracle

def _solve_particle_association(time, association_rate, carrying_capacity):
    t, r, k = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(association_rate, dtype=float),
        np.asarray(carrying_capacity, dtype=float),
    )
    if np.any(k <= 0.0):
        raise ValueError("carrying capacity must be positive")
    volume = 1.01e-6
    u0 = 9.95e7
    coverage = 1.0
    surface_area = 5.31e-5
    vu0 = volume * u0
    out = np.empty_like(t, dtype=float)
    singular = np.isclose(k, vu0, rtol=1.0e-12, atol=1.0e-14)
    regular = ~singular
    if np.any(regular):
        kr = k[regular]
        rr = r[regular]
        tr = t[regular]
        expo = np.exp(
            -rr * coverage * surface_area * (vu0 - kr) * tr / (kr * volume)
        )
        out[regular] = vu0 * (1.0 - (vu0 - kr) / (vu0 - kr * expo))
    if np.any(singular):
        ks = k[singular]
        rs = r[singular]
        ts = t[singular]
        out[singular] = (
            rs * coverage * surface_area * u0**2 * volume * ts
            / (ks + rs * coverage * surface_area * u0 * ts)
        )
    return out


def _compute_shared_environment(log_params, times, n_prep, fine_step, rng):
    log_params = np.asarray(log_params, dtype=float)
    times = np.asarray(times, dtype=float)
    if log_params.shape != (4,):
        raise ValueError("log_params must have shape (4,)")
    if times.ndim != 1 or times.size == 0 or np.any(times < 0.0):
        raise ValueError("times must be a non-empty non-negative one-dimensional array")
    if int(n_prep) < 1 or float(fine_step) <= 0.0:
        raise ValueError("n_prep and fine_step must be positive")

    mr, sr, mk, sk = log_params
    if sr < 0.0 or sk < 0.0:
        raise ValueError("log-normal scales must be non-negative")

    end = float(np.max(times))
    count = int(round(end / float(fine_step)))
    fine = float(fine_step) * np.arange(count + 1, dtype=float)
    if not np.all(np.isclose(times[:, None], fine[None, :], rtol=0.0, atol=1.0e-9).any(axis=1)):
        raise ValueError("all observation times must lie on the fine grid")

    volume = 1.01e-6
    u0 = 9.95e7
    ubar = np.empty(fine.size, dtype=float)
    for i, t in enumerate(fine):
        r = rng.lognormal(mean=mr, sigma=sr, size=int(n_prep))
        k = rng.lognormal(mean=mk, sigma=sk, size=int(n_prep))
        p = _solve_particle_association(t, r, k)
        ubar[i] = np.mean(u0 - p / volume)

    integral = np.zeros_like(ubar)
    if fine.size > 1:
        increments = 0.5 * (ubar[1:] + ubar[:-1]) * np.diff(fine)
        integral[1:] = np.cumsum(increments)

    indices = [int(np.flatnonzero(np.isclose(fine, t, rtol=0.0, atol=1.0e-9))[0]) for t in times]
    return integral[np.asarray(indices, dtype=int)]


# Oracle

def _oracle_compute_shared_environment_integral(
    log_params: np.ndarray,
    times: np.ndarray,
    n_prep: int,
    fine_step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.asarray(log_params).shape != (4,):
        raise ValueError("log_params must have shape (4,)")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return _compute_shared_environment(log_params, times, n_prep, fine_step, rng).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic tests for the coupled environmental integral."""
    return [
        {
            "setup": """import numpy as np
log_params = np.array([-14.035551324356856, 0.4789159649348846, 2.282974736417405, 0.19804220043536505])
times = np.array([3600.0, 7200.0, 14400.0])
n_prep = 30
fine_step = 1800.0
rng_call = np.random.default_rng(1701)
rng_gold = np.random.default_rng(1701)
""",
            "call": "compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_call)",
            "gold_call": "_oracle_compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_gold)",
        },
        {
            "setup": """import numpy as np
vu0 = 1.01e-6 * 9.95e7
log_params = np.array([np.log(4.0e-7), 0.0, np.log(vu0), 0.0])
times = np.array([1800.0, 3600.0])
n_prep = 5
fine_step = 1800.0
rng_call = np.random.default_rng(11)
rng_gold = np.random.default_rng(11)
""",
            "call": "compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_call)",
            "gold_call": "_oracle_compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_gold)",
        },
        {
            "setup": """import numpy as np
log_params = np.array([np.log(2.0e-7), 0.25, np.log(7.0), 0.4])
times = np.array([900.0, 2700.0, 4500.0])
n_prep = 17
fine_step = 900.0
rng_call = np.random.default_rng(2026)
rng_gold = np.random.default_rng(2026)
""",
            "call": "compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_call)",
            "gold_call": "_oracle_compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_gold)",
        },
        {
            "setup": """import numpy as np
def value_error_code(fn):
    try:
        fn()
    except ValueError:
        return 1.0
    return 0.0
log_params = np.array([np.log(3.0e-7), 0.2, np.log(9.0), 0.1])
times = np.array([1000.0])
n_prep = 5
fine_step = 900.0
rng_call = np.random.default_rng(8)
rng_gold = np.random.default_rng(8)
""",
            "call": "value_error_code(lambda: compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_call))",
            "gold_call": "value_error_code(lambda: _oracle_compute_shared_environment_integral(log_params, times, n_prep, fine_step, rng_gold))",
        },
    ]
