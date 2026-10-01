"""
Run the complete seeded double-Heston long-step Bermudan pricing pipeline.

Finite-interval variance transitions feed the correlated price construction before regression-based stopping and valuation.

Returns
-------
float, the seeded time-zero Bermudan put value as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_double_heston_aes_bermudan(
    n_paths: int = 4096, n_steps: int = 6, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389,
) -> float:
    r"""Return the seeded Bermudan put value for the fixed task parameters.

    Parameters
    ----------
    n_paths, n_steps : int
        Positive path and interval counts.
    seed_v1, seed_v2, seed_price : int
        Dedicated deterministic random seeds.

    Returns
    -------
    price : float
        Time-zero option value.

    Raises
    ------
    ValueError
        If counts are not positive integers or seeds are not integers.

    Notes
    -----
    Call the seven preceding public functions in order.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_double_heston_aes_bermudan(
    n_paths: int = 4096, n_steps: int = 6, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389,
) -> float:
    import numpy as np
    for value in (n_paths, n_steps, seed_v1, seed_v2, seed_price):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts and seeds must be integers")
    if n_paths < 1 or n_steps < 1 or min(seed_v1, seed_v2, seed_price) < 0:
        raise ValueError("counts must be positive and seeds nonnegative")
    spot0 = strike = 61.9
    maturity, rate = 0.25, 0.03
    kappa = np.array([0.9, 1.2]); theta = np.array([0.1, 0.15])
    gamma = np.array([0.1, 0.2]); rho = np.array([-0.5, -0.5])
    initial = np.array([0.2, 0.49]); dt = maturity / n_steps
    coeff = _oracle_compute_double_heston_aes_coefficients(dt, rate, kappa, theta, gamma, rho)
    trans1 = _oracle_compute_cir_transition_parameters(kappa[0], theta[0], gamma[0], dt)
    trans2 = _oracle_compute_cir_transition_parameters(kappa[1], theta[1], gamma[1], dt)
    v1 = _oracle_simulate_cir_variance_paths(initial[0], trans1, n_steps, n_paths, seed_v1)
    v2 = _oracle_simulate_cir_variance_paths(initial[1], trans2, n_steps, n_paths, seed_v2)
    spot = _oracle_simulate_double_heston_aes_paths(spot0, coeff, v1, v2, seed_price)
    terminal_payoffs = np.maximum(strike - spot[-1], 0.0)
    cashflows = terminal_payoffs.copy()
    exercise_steps = np.full(n_paths, n_steps, dtype=float)
    for step in range(n_steps - 1, 0, -1):
        state = _oracle_apply_double_heston_lsm_step(
            spot[step], v1[step], v2[step], strike, step, cashflows, exercise_steps, rate, dt)
        cashflows, exercise_steps = state[0], state[1]
    bermudan_price = _oracle_price_discounted_cashflows(cashflows, exercise_steps, rate, dt)
    diagnostics = _oracle_summarize_bermudan_exercise_effect(
        bermudan_price, terminal_payoffs, exercise_steps, rate, dt, n_steps)
    maturity_value, early_count, mean_increment = diagnostics
    return float(maturity_value + mean_increment * early_count / n_paths)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and alternate-stream specifications."""
    return [
        {"setup": "import numpy as np", "call": "run_double_heston_aes_bermudan()",
         "gold_call": "_oracle_run_double_heston_aes_bermudan()"},
        {"setup": "import numpy as np", "call": "run_double_heston_aes_bermudan(1024,4,11,23,37)",
         "gold_call": "_oracle_run_double_heston_aes_bermudan(1024,4,11,23,37)"},
        {"setup": "import numpy as np", "call": "run_double_heston_aes_bermudan(512,1,5,7,13)",
         "gold_call": "_oracle_run_double_heston_aes_bermudan(512,1,5,7,13)"},
        {"setup": "import numpy as np", "call": "run_double_heston_aes_bermudan(384,3,41,43,47)",
         "gold_call": "_oracle_run_double_heston_aes_bermudan(384,3,41,43,47)"},
    ]
