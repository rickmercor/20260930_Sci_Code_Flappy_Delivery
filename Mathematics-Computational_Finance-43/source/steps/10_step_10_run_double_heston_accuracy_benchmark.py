"""
Compare fixed-stream early-exercise prices from the long-step and truncated-Euler schemes.

An absolute-error ratio measures the accuracy amplification of the long-step construction against the study's Euler comparator.

Returns
-------
float, the positive Euler-to-long-step absolute pricing-error ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_double_heston_accuracy_benchmark(
    n_paths: int = 16384, n_steps: int = 12, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389, reference_price: float = 9.504,
) -> float:
    r"""Return the Euler-to-long-step absolute pricing-error ratio.
    Use all preceding public functions to cross-check the long-step price, price the Euler paths at strike 61.9, and form the error ratio.

    Parameters
    ----------
    n_paths, n_steps : int
        Positive counts; seeds are nonnegative integers.
    reference_price : float
        Finite positive benchmark; other parameters are deterministic seeds.
    Returns
    -------
    ratio : float
        Euler absolute error divided by long-step absolute error.
    Raises
    ------
    ValueError
        For invalid inputs, inconsistent long-step results, or a zero denominator.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_double_heston_accuracy_benchmark(
    n_paths: int = 16384, n_steps: int = 12, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389, reference_price: float = 9.504,
) -> float:
    if (any(isinstance(x, bool) or not isinstance(x, (int, np.integer))
            for x in (n_paths, n_steps, seed_v1, seed_v2, seed_price))
            or n_paths < 1 or n_steps < 1 or min(seed_v1, seed_v2, seed_price) < 0):
        raise ValueError("counts must be positive integers and seeds nonnegative")
    if not np.isfinite(reference_price) or reference_price <= 0.0:
        raise ValueError("reference_price must be finite and positive")
    kappa = np.array([0.9, 1.2]); theta = np.array([0.1, 0.15])
    gamma = np.array([0.1, 0.2]); rho = np.array([-0.5, -0.5])
    initial = np.array([0.2, 0.49]); dt = 0.25 / n_steps
    coeff = _oracle_compute_double_heston_aes_coefficients(dt, 0.03, kappa, theta, gamma, rho)
    trans1 = _oracle_compute_cir_transition_parameters(kappa[0], theta[0], gamma[0], dt)
    trans2 = _oracle_compute_cir_transition_parameters(kappa[1], theta[1], gamma[1], dt)
    v1 = _oracle_simulate_cir_variance_paths(initial[0], trans1, n_steps, n_paths, seed_v1)
    v2 = _oracle_simulate_cir_variance_paths(initial[1], trans2, n_steps, n_paths, seed_v2)
    spot = _oracle_simulate_double_heston_aes_paths(61.9, coeff, v1, v2, seed_price)
    terminal = np.maximum(61.9 - spot[-1], 0.0)
    cashflows = terminal.copy(); exercise_steps = np.full(n_paths, n_steps, dtype=float)
    for step in range(n_steps - 1, 0, -1):
        state = _oracle_apply_double_heston_lsm_step(
            spot[step], v1[step], v2[step], 61.9, step,
            cashflows, exercise_steps, 0.03, dt)
        cashflows, exercise_steps = state
    direct = _oracle_price_discounted_cashflows(cashflows, exercise_steps, 0.03, dt)
    diagnostic = _oracle_summarize_bermudan_exercise_effect(
        direct, terminal, exercise_steps, 0.03, dt, n_steps)
    reconstructed = diagnostic[0] + diagnostic[2] * diagnostic[1] / n_paths
    aes_price = _oracle_run_double_heston_aes_bermudan(
        n_paths, n_steps, seed_v1, seed_v2, seed_price)
    if not np.isclose(aes_price, reconstructed, rtol=0.0, atol=1e-12):
        raise ValueError("long-step public-chain results are inconsistent")
    euler = _oracle_simulate_double_heston_truncated_euler_paths(
        n_paths, n_steps, seed_v1, seed_v2, seed_price)
    strike, rate = 61.9, 0.03
    cashflows = np.maximum(strike - euler[0, -1], 0.0)
    exercise_steps = np.full(n_paths, n_steps, dtype=float)
    for step in range(n_steps - 1, 0, -1):
        state = _oracle_apply_double_heston_lsm_step(
            euler[0, step], euler[1, step], euler[2, step], strike, step,
            cashflows, exercise_steps, rate, dt)
        cashflows, exercise_steps = state
    euler_price = _oracle_price_discounted_cashflows(
        cashflows, exercise_steps, rate, dt)
    aes_error = abs(aes_price - reference_price)
    if aes_error <= np.finfo(float).eps * max(1.0, abs(reference_price)):
        raise ValueError("long-step reference error is numerically zero")
    return float(abs(euler_price - reference_price) / aes_error)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three deterministic whole-benchmark configurations."""
    return [
        {"setup": "import numpy as np", "call": "run_double_heston_accuracy_benchmark(512,3,11,23,37,9.504)", "gold_call": "_oracle_run_double_heston_accuracy_benchmark(512,3,11,23,37,9.504)"},
        {"setup": "import numpy as np", "call": "run_double_heston_accuracy_benchmark(768,1,5,7,13,9.504)", "gold_call": "_oracle_run_double_heston_accuracy_benchmark(768,1,5,7,13,9.504)"},
        {"setup": "import numpy as np", "call": "run_double_heston_accuracy_benchmark(1024,6,41,43,47,9.504)", "gold_call": "_oracle_run_double_heston_accuracy_benchmark(1024,6,41,43,47,9.504)"},
    ]
