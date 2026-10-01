"""
Run the complete weak-form sparse identification pipeline on a noisy synthetic age-structured population and return the prediction error of the learned model.

The benchmark learns the mortality $d(a)$ and fecundity $\beta(a)$ of an age-structured population from noisy histogram data. A noise-free density $n^\star(t,a)$ is generated from a known model, corrupted with seeded multiplicative log-normal noise, and restricted to a training window $[0,T_{test}]$. The known unit aging speed is moved into the data vector $b$ of the weak form; mortality is selected from exponential library terms $e^{c_m a}\,n$ and fecundity from Gaussian library terms $e^{-(a-\mu_m)^2/(2s^2)}$ by stacking the weak-form PDE with the weak form of the total-population ODE. The sparse coefficients are obtained with MSTLS and the boundary cross-validation that re-selects birth terms from the ODE block. The learned model is solved exactly beyond the training window and compared with the truth through the prediction error $E_p$.

Returns
-------
float, the prediction error E_p of the model learned by the full pipeline
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wsindy_prediction_pipeline(noise_to_signal_ratio: float, seed: int, age_step: float, n_time_tests: int, n_age_tests: int) -> float:
    '''Return the prediction error E_p of the model learned from noisy age-structured data.
 
    Benchmark (all fixed except the arguments):
      * Grids: age-class midpoints a_j = (j + 1/2) h (j = 0..25/h - 1) on [0, 25] and
        times t_i = i h on [0, 10] with h = age_step (so 25/h and 5/h must be integers);
        training window [0, 5], testing window [5, 10].
      * Truth: dn/dt + dn/da = -0.1 exp(0.08 a) n, n(t, 0) = integral beta*(a) n(t, a) da
        with beta*(a) = exp(-(a - 10)^2 / 50), and n(0, a) = 1 - cos(2 pi a / 15) for
        0 <= a <= 15, 0 for a > 15 (no individual reaches age 25 before t = 10).
        The noise-free density is the exact solution of this continuous model at the
        grid points (renewal equation for the birth flux solved to 1e-10 accuracy).
      * Library: source terms exp(c a) n with c = (0.08, 0.40, 0.72, 1.04, 1.36); birth
        terms exp(-(a - mu)^2 / 50) with mu = (5, 10, 15). True coefficients
        w* = (-0.1, 0, 0, 0, 0, 0, 1, 0).
      * Noise: sigma solves E[(exp(z) - 1)^2] = noise_to_signal_ratio, z ~ N(0, sigma^2);
        the training rows t_i <= 5 of the noise-free density are multiplied by
        exp(sigma Z) with Z = np.random.default_rng(seed).standard_normal(shape of the
        training array), drawn in one call.
      * Weak form (known aging speed 1): sup-norm-normalized test functions
        C (x - x1)^14 (x1 + ell - x)^14 with support ratio 0.5 in time and age
        (ell_t = 2.5, ell_a = 12.5); n_time_tests temporal and n_age_tests age test
        functions with left endpoints evenly spaced from 0 to 2.5 and from 0 to 12.5;
        trapezoidal rule in time and midpoint rule over age classes; PDE rows b = -<d_t Phi, n> - <d_a Phi, n>,
        G = <Phi, f_m n> (time index major), zero birth columns; ODE rows from the
        de-biased density n / exp(sigma^2/2) with b = -integral phi_k' N dt and columns
        <phi_k, f_m n>, <phi_k, beta_m n>.
      * Regression: MSTLS with thresholds 10^(-4 + 4 j / 49), j = 0..49, and the
        boundary cross-validation of the birth support (re-selection from the ODE block,
        intersection/union pruning and refit, or residual comparison if supports agree).
      * Prediction: solve the learned model exactly from n(0, a) on [0, 10] at the same
        points and return
        E_p = ||n_pred - n*||_{L2((5,10) x (0,25))} / ||n*||_{L2((5,10) x (0,25))}
        with the trapezoidal rule in time and the midpoint rule in age.
 
    Parameters
    ----------
    noise_to_signal_ratio : float
        Expected mean-square noise-to-signal ratio sigma_NR >= 0.
    seed : int
        Noise seed.
    age_step : float
        Grid spacing h (age and time), h > 0 with 25/h and 5/h integers.
    n_time_tests : int
        Number of temporal test functions (>= 2).
    n_age_tests : int
        Number of age test functions (>= 2).
 
    Returns
    -------
    E_p : float
        Relative L2 prediction error on the testing window.
 
    Raises
    ------
    ValueError
        If age_step does not divide 25 and 5 into integers, or any argument is invalid
        for the pipeline steps (negative noise ratio, fewer than two test functions).
    '''
    return E_p

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_wsindy_prediction_pipeline(noise_to_signal_ratio: float, seed: int, age_step: float, n_time_tests: int, n_age_tests: int) -> float:
    h = float(age_step)
    if not h > 0:
        raise ValueError("age_step must be positive")
    n_age, n_train = 25.0 / h, 5.0 / h
    if abs(n_age - round(n_age)) > 1e-9 or abs(n_train - round(n_train)) > 1e-9:
        raise ValueError("age_step must divide 25 and 5")
    n_age, n_train = int(round(n_age)), int(round(n_train))
    n_total = 2 * n_train
    rates = np.array([0.08, 0.40, 0.72, 1.04, 1.36])
    means = np.array([5.0, 10.0, 15.0])
    width = 5.0
    truth = _oracle_simulate_age_structured(15.0, h, n_age, n_total, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([10.0]), width)
    sigma = _lognormal_sigma(noise_to_signal_ratio)
    noisy = _oracle_add_lognormal_noise(truth[:n_train + 1], noise_to_signal_ratio, seed)
    system = _oracle_assemble_weak_system(noisy, h, h, sigma, 1.0, rates, means, width, n_time_tests, n_age_tests, 0.5, 0.5, 14)
    G, b = system[:, :-1], system[:, -1]
    lambdas = 10.0 ** (-4.0 + 4.0 * np.arange(50) / 49.0)
    w = _oracle_boundary_bagging_regression(G, b, int(n_time_tests) * int(n_age_tests), rates.size, lambdas)
    return _oracle_prediction_error(w, 15.0, h, n_train, rates, means, width, truth)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
""",
            "call": 'wsindy_prediction_pipeline(0.1, 5, 0.05, 26, 126)',
            "gold_call": '_oracle_wsindy_prediction_pipeline(0.1, 5, 0.05, 26, 126)',
        },
        {
            "setup": """
""",
            "call": 'wsindy_prediction_pipeline(0.05, 3, 0.05, 26, 126)',
            "gold_call": '_oracle_wsindy_prediction_pipeline(0.05, 3, 0.05, 26, 126)',
        },
        {
            "setup": """
""",
            "call": 'wsindy_prediction_pipeline(0.2, 1, 0.05, 11, 51)',
            "gold_call": '_oracle_wsindy_prediction_pipeline(0.2, 1, 0.05, 11, 51)',
        },
        {
            "setup": """
""",
            "call": 'wsindy_prediction_pipeline(0.0, 0, 0.1, 11, 51)',
            "gold_call": '_oracle_wsindy_prediction_pipeline(0.0, 0, 0.1, 11, 51)',
        },
        {
            "setup": """def run_model():
    try:
        wsindy_prediction_pipeline(0.1, 0, 0.3, 11, 26)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
