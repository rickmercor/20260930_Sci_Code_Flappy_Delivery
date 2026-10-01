"""
Measure the out-of-sample prediction error of a learned age-structured model by solving it exactly and comparing with the true density over the testing interval.

A learned model can fit the training data even when it selects analytically incorrect library terms, so its quality is also judged by prediction. The learned mortality and fecundity (a sparse combination of the library functions) define a new age-structured model, whose exact solution from the true initial density is evaluated at the same age-class midpoints and times as the data. The prediction error is the relative $L^2$ norm of the difference between the predicted density $\tilde n$ and the true (noise-free) density $n^\star$ over the testing window,

 

$$E_p=\frac{\|\tilde n-n^\star\|_{L^2((T_{test},T)\times\Omega)}}{\|n^\star\|_{L^2((T_{test},T)\times\Omega)}},$$

 

where $\Omega$ is the age domain, evaluated with the trapezoidal rule in time and the midpoint rule over the age classes.

Returns
-------
float, the relative L2 prediction error E_p of the exactly solved learned model on the testing window
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prediction_error(learned_weights: "np.ndarray", initial_support: float, age_step: float, n_train_steps: int, source_rates: "np.ndarray", birth_means: "np.ndarray", birth_width: float, true_density: "np.ndarray") -> float:
    '''Return the relative L2 prediction error E_p of a learned linear age-structured model.
 
    learned_weights = (w_f, w_beta) with M_f = len(source_rates) source coefficients
    followed by M_beta = len(birth_means) birth coefficients. With h = age_step, the
    true density is given at times t_i = i h (i = 0..I) and age-class midpoints
    a_j = (j + 1/2) h (j = 0..J-1), where (I+1, J) is the shape of true_density. The
    predicted density n_pred is the exact solution, at the same points, of
 
        dn/dt + dn/da = sum_m w_f[m] exp(c_m a) n,
        n(t, 0) = integral_0^{J h} sum_m w_beta[m] exp(-(a - mu_m)^2 / (2 s^2)) n(t, a) da,
        n(0, a) = 1 - cos(2 pi a / L0) for 0 <= a <= L0 and 0 otherwise (L0 = initial_support),
 
    each value accurate to within 1e-10 * (1 + |n_pred|). With rows
    i >= n_train_steps (times t_i >= T_test = n_train_steps h),
 
        E_p = sqrt( sum_{i >= n_train} sum_j wt_i h (n_pred - n_true)^2
                    / sum_{i >= n_train} sum_j wt_i h n_true^2 ),
 
    where wt are trapezoidal weights over the time nodes i = n_train..I (h/2 at both
    ends of that window, h inside) and each age class has midpoint weight h.
 
    Parameters
    ----------
    learned_weights : np.ndarray
        Coefficients of shape (M_f + M_beta,).
    initial_support : float
        L0 > 0; the initial density vanishes beyond it, and L0 + I h <= J h is required.
    age_step : float
        Width h > 0 of the age classes (also the time step).
    n_train_steps : int
        Index of T_test; 0 <= n_train_steps < I.
    source_rates : np.ndarray
        Exponential rates c_m of the source library.
    birth_means : np.ndarray
        Gaussian centres mu_m of the birth library.
    birth_width : float
        Gaussian standard deviation s > 0.
    true_density : np.ndarray
        Noise-free density on the same points, shape (I+1, J).
 
    Returns
    -------
    E_p : float
        Relative L2 prediction error on the testing window.
 
    Raises
    ------
    ValueError
        If shapes are inconsistent, n_train_steps is out of range, the true density
        vanishes on the testing window, or L0 + I h exceeds J h.
    '''
    return E_p

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_prediction_error(learned_weights: "np.ndarray", initial_support: float, age_step: float, n_train_steps: int, source_rates: "np.ndarray", birth_means: "np.ndarray", birth_width: float, true_density: "np.ndarray") -> float:
    w = np.asarray(learned_weights, dtype=float).ravel()
    rates = np.asarray(source_rates, dtype=float).ravel()
    means = np.asarray(birth_means, dtype=float).ravel()
    truth = np.asarray(true_density, dtype=float)
    if w.size != rates.size + means.size or truth.ndim != 2:
        raise ValueError("inconsistent shapes")
    n_rows, n_ages = truth.shape
    k = int(n_train_steps)
    if k != n_train_steps or not (0 <= k < n_rows - 1):
        raise ValueError("n_train_steps out of range")
    pred = _oracle_simulate_age_structured(initial_support, age_step, n_ages, n_rows - 1, w[:rates.size], rates, w[rates.size:], means, birth_width)
    wt = _trapezoid_weights(n_rows - k, age_step)
    weight = wt[:, None] * float(age_step)
    denom = np.sum(weight * truth[k:] ** 2)
    if not denom > 0:
        raise ValueError("true density vanishes on the testing window")
    return float(np.sqrt(np.sum(weight * (pred[k:] - truth[k:]) ** 2) / denom))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def exact_truth(h, n_ages, n_steps):
    # exact benchmark density: death 0.1 exp(0.08 a), birth exp(-(a-10)^2/50), n0 = 1 - cos(2 pi a/15) on [0, 15]
    ls = lambda x: -0.1 * np.expm1(0.08 * x) / 0.08
    bet = lambda x: np.exp(-(x - 10.0) ** 2 / 50.0)
    n0 = lambda u: np.where((u >= 0) & (u <= 15.0), 1.0 - np.cos(2.0 * np.pi * u / 15.0), 0.0)
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 7.5 * (x + 1.0); uw = 7.5 * gw
    flux_levels = []
    for lev in range(3):
        k = h / 4.0 / 2 ** lev
        N = int(round(n_steps * h / k))
        tau = np.arange(N + 1) * k
        F = (bet(u[None, :] + tau[:, None]) * np.exp(ls(u[None, :] + tau[:, None]) - ls(u)[None, :]) * n0(u)[None, :]) @ uw
        K = bet(tau) * np.exp(ls(tau))
        Bf = np.empty(N + 1); Bf[0] = F[0]
        for n in range(1, N + 1):
            Bf[n] = (F[n] + k * (np.dot(K[1:n], Bf[n - 1:0:-1]) + 0.5 * K[n] * Bf[0])) / (1.0 - 0.5 * k * K[0])
        flux_levels.append(Bf[::2 ** lev])
    r1 = [(4.0 * flux_levels[1] - flux_levels[0]) / 3.0, (4.0 * flux_levels[2] - flux_levels[1]) / 3.0]
    flux = (16.0 * r1[1] - r1[0]) / 15.0
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    old = A > T
    out = np.empty_like(T)
    out[old] = n0(A[old] - T[old]) * np.exp(ls(A[old]) - ls(A[old] - T[old]))
    out[~old] = flux[np.rint((T[~old] - A[~old]) / (h / 4.0)).astype(int)] * np.exp(ls(A[~old]))
    return out
truth = exact_truth(0.05, 500, 200)
rates = np.array([0.08, 0.4, 0.72, 1.04, 1.36])
means = np.array([5.0, 10.0, 15.0])
w = np.array([-0.0998, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0036, 0.0])
""",
            "call": 'prediction_error(w.copy(), 15.0, 0.05, 100, rates.copy(), means.copy(), 5.0, truth.copy())',
            "gold_call": '_oracle_prediction_error(w.copy(), 15.0, 0.05, 100, rates.copy(), means.copy(), 5.0, truth.copy())',
        },
        {
            "setup": """import numpy as np
def exact_truth(h, n_ages, n_steps):
    # exact benchmark density: death 0.1 exp(0.08 a), birth exp(-(a-10)^2/50), n0 = 1 - cos(2 pi a/15) on [0, 15]
    ls = lambda x: -0.1 * np.expm1(0.08 * x) / 0.08
    bet = lambda x: np.exp(-(x - 10.0) ** 2 / 50.0)
    n0 = lambda u: np.where((u >= 0) & (u <= 15.0), 1.0 - np.cos(2.0 * np.pi * u / 15.0), 0.0)
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 7.5 * (x + 1.0); uw = 7.5 * gw
    flux_levels = []
    for lev in range(3):
        k = h / 4.0 / 2 ** lev
        N = int(round(n_steps * h / k))
        tau = np.arange(N + 1) * k
        F = (bet(u[None, :] + tau[:, None]) * np.exp(ls(u[None, :] + tau[:, None]) - ls(u)[None, :]) * n0(u)[None, :]) @ uw
        K = bet(tau) * np.exp(ls(tau))
        Bf = np.empty(N + 1); Bf[0] = F[0]
        for n in range(1, N + 1):
            Bf[n] = (F[n] + k * (np.dot(K[1:n], Bf[n - 1:0:-1]) + 0.5 * K[n] * Bf[0])) / (1.0 - 0.5 * k * K[0])
        flux_levels.append(Bf[::2 ** lev])
    r1 = [(4.0 * flux_levels[1] - flux_levels[0]) / 3.0, (4.0 * flux_levels[2] - flux_levels[1]) / 3.0]
    flux = (16.0 * r1[1] - r1[0]) / 15.0
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    old = A > T
    out = np.empty_like(T)
    out[old] = n0(A[old] - T[old]) * np.exp(ls(A[old]) - ls(A[old] - T[old]))
    out[~old] = flux[np.rint((T[~old] - A[~old]) / (h / 4.0)).astype(int)] * np.exp(ls(A[~old]))
    return out
truth = exact_truth(0.05, 500, 200)
rates = np.array([0.08, 0.4, 0.72, 1.04, 1.36])
means = np.array([5.0, 10.0, 15.0])
w = np.array([-0.0999, 0.0, 0.0, 0.0, 0.0, 0.64, -1.90, 4.14])
""",
            "call": 'prediction_error(w.copy(), 15.0, 0.05, 100, rates.copy(), means.copy(), 5.0, truth.copy())',
            "gold_call": '_oracle_prediction_error(w.copy(), 15.0, 0.05, 100, rates.copy(), means.copy(), 5.0, truth.copy())',
        },
        {
            "setup": """import numpy as np
def exact_truth(h, n_ages, n_steps):
    # exact benchmark density: death 0.1 exp(0.08 a), birth exp(-(a-10)^2/50), n0 = 1 - cos(2 pi a/15) on [0, 15]
    ls = lambda x: -0.1 * np.expm1(0.08 * x) / 0.08
    bet = lambda x: np.exp(-(x - 10.0) ** 2 / 50.0)
    n0 = lambda u: np.where((u >= 0) & (u <= 15.0), 1.0 - np.cos(2.0 * np.pi * u / 15.0), 0.0)
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 7.5 * (x + 1.0); uw = 7.5 * gw
    flux_levels = []
    for lev in range(3):
        k = h / 4.0 / 2 ** lev
        N = int(round(n_steps * h / k))
        tau = np.arange(N + 1) * k
        F = (bet(u[None, :] + tau[:, None]) * np.exp(ls(u[None, :] + tau[:, None]) - ls(u)[None, :]) * n0(u)[None, :]) @ uw
        K = bet(tau) * np.exp(ls(tau))
        Bf = np.empty(N + 1); Bf[0] = F[0]
        for n in range(1, N + 1):
            Bf[n] = (F[n] + k * (np.dot(K[1:n], Bf[n - 1:0:-1]) + 0.5 * K[n] * Bf[0])) / (1.0 - 0.5 * k * K[0])
        flux_levels.append(Bf[::2 ** lev])
    r1 = [(4.0 * flux_levels[1] - flux_levels[0]) / 3.0, (4.0 * flux_levels[2] - flux_levels[1]) / 3.0]
    flux = (16.0 * r1[1] - r1[0]) / 15.0
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    old = A > T
    out = np.empty_like(T)
    out[old] = n0(A[old] - T[old]) * np.exp(ls(A[old]) - ls(A[old] - T[old]))
    out[~old] = flux[np.rint((T[~old] - A[~old]) / (h / 4.0)).astype(int)] * np.exp(ls(A[~old]))
    return out
truth = exact_truth(0.05, 500, 200)
rates = np.array([0.08, 0.4, 0.72, 1.04, 1.36])
means = np.array([5.0, 10.0, 15.0])
w = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
""",
            "call": 'prediction_error(w.copy(), 15.0, 0.05, 100, rates.copy(), means.copy(), 5.0, truth.copy())',
            "gold_call": '_oracle_prediction_error(w.copy(), 15.0, 0.05, 100, rates.copy(), means.copy(), 5.0, truth.copy())',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
def exact_truth(h, n_ages, n_steps):
    # exact benchmark density: death 0.1 exp(0.08 a), birth exp(-(a-10)^2/50), n0 = 1 - cos(2 pi a/15) on [0, 15]
    ls = lambda x: -0.1 * np.expm1(0.08 * x) / 0.08
    bet = lambda x: np.exp(-(x - 10.0) ** 2 / 50.0)
    n0 = lambda u: np.where((u >= 0) & (u <= 15.0), 1.0 - np.cos(2.0 * np.pi * u / 15.0), 0.0)
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 7.5 * (x + 1.0); uw = 7.5 * gw
    flux_levels = []
    for lev in range(3):
        k = h / 4.0 / 2 ** lev
        N = int(round(n_steps * h / k))
        tau = np.arange(N + 1) * k
        F = (bet(u[None, :] + tau[:, None]) * np.exp(ls(u[None, :] + tau[:, None]) - ls(u)[None, :]) * n0(u)[None, :]) @ uw
        K = bet(tau) * np.exp(ls(tau))
        Bf = np.empty(N + 1); Bf[0] = F[0]
        for n in range(1, N + 1):
            Bf[n] = (F[n] + k * (np.dot(K[1:n], Bf[n - 1:0:-1]) + 0.5 * K[n] * Bf[0])) / (1.0 - 0.5 * k * K[0])
        flux_levels.append(Bf[::2 ** lev])
    r1 = [(4.0 * flux_levels[1] - flux_levels[0]) / 3.0, (4.0 * flux_levels[2] - flux_levels[1]) / 3.0]
    flux = (16.0 * r1[1] - r1[0]) / 15.0
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    old = A > T
    out = np.empty_like(T)
    out[old] = n0(A[old] - T[old]) * np.exp(ls(A[old]) - ls(A[old] - T[old]))
    out[~old] = flux[np.rint((T[~old] - A[~old]) / (h / 4.0)).astype(int)] * np.exp(ls(A[~old]))
    return out
truth = exact_truth(0.05, 500, 200)
rates = np.array([0.08, 0.4, 0.72, 1.04, 1.36])
means = np.array([5.0, 10.0, 15.0])
w = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0])
""",
            "call": 'prediction_error(w.copy(), 15.0, 0.05, 199, rates.copy(), means.copy(), 5.0, truth.copy())',
            "gold_call": '_oracle_prediction_error(w.copy(), 15.0, 0.05, 199, rates.copy(), means.copy(), 5.0, truth.copy())',
        },
        {
            "setup": """import numpy as np
def exact_truth(h, n_ages, n_steps):
    # exact benchmark density: death 0.1 exp(0.08 a), birth exp(-(a-10)^2/50), n0 = 1 - cos(2 pi a/15) on [0, 15]
    ls = lambda x: -0.1 * np.expm1(0.08 * x) / 0.08
    bet = lambda x: np.exp(-(x - 10.0) ** 2 / 50.0)
    n0 = lambda u: np.where((u >= 0) & (u <= 15.0), 1.0 - np.cos(2.0 * np.pi * u / 15.0), 0.0)
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 7.5 * (x + 1.0); uw = 7.5 * gw
    flux_levels = []
    for lev in range(3):
        k = h / 4.0 / 2 ** lev
        N = int(round(n_steps * h / k))
        tau = np.arange(N + 1) * k
        F = (bet(u[None, :] + tau[:, None]) * np.exp(ls(u[None, :] + tau[:, None]) - ls(u)[None, :]) * n0(u)[None, :]) @ uw
        K = bet(tau) * np.exp(ls(tau))
        Bf = np.empty(N + 1); Bf[0] = F[0]
        for n in range(1, N + 1):
            Bf[n] = (F[n] + k * (np.dot(K[1:n], Bf[n - 1:0:-1]) + 0.5 * K[n] * Bf[0])) / (1.0 - 0.5 * k * K[0])
        flux_levels.append(Bf[::2 ** lev])
    r1 = [(4.0 * flux_levels[1] - flux_levels[0]) / 3.0, (4.0 * flux_levels[2] - flux_levels[1]) / 3.0]
    flux = (16.0 * r1[1] - r1[0]) / 15.0
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    old = A > T
    out = np.empty_like(T)
    out[old] = n0(A[old] - T[old]) * np.exp(ls(A[old]) - ls(A[old] - T[old]))
    out[~old] = flux[np.rint((T[~old] - A[~old]) / (h / 4.0)).astype(int)] * np.exp(ls(A[~old]))
    return out
truth = exact_truth(0.05, 500, 200)
rates = np.array([0.08, 0.4, 0.72, 1.04, 1.36])
means = np.array([5.0, 10.0, 15.0])
w = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
def run_model():
    try:
        prediction_error(w.copy(), 15.0, 0.05, 200, rates.copy(), means.copy(), 5.0, truth.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
