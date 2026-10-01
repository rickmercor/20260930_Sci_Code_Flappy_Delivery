"""
Assemble the stacked weak-form linear system that couples the transport PDE with the total-population ODE for learning mortality and fecundity terms of an age-structured model with known aging speed.

For the age-structured model $\partial_t n+\alpha\,\partial_a n=f(a,n)$ with renewal condition $n(t,0)=\int\beta(a)\,n(t,a)\,da$, the unknown source and birth terms are expanded in libraries, $f=\sum_m w_{f,m}\,f_m(a)\,n$ and $\beta=\sum_m w_{\beta,m}\,\beta_m(a)$. Multiplying the PDE by a compactly supported test function $\phi(t,a)$ and integrating by parts gives

 

$$-\langle\partial_t\phi,n\rangle-\alpha\,\langle\partial_a\phi,n\rangle=\sum_m w_{f,m}\,\langle\phi,f_m n\rangle,$$

 

in which no derivative of the data appears; when the aging speed $\alpha$ is known, the transport term is moved into the data vector $b$ on the left-hand side. Because $\phi$ vanishes at $a=0$, the birth process is invisible to this weak form. It is recovered by also using the total population $N(t)=\int n(t,a)\,da$, which satisfies (when no individual leaves the age domain) $\dot N=\int\beta\,n\,da+\int f\,da$. Its weak form with a temporal test function $\phi_k(t)$ is

 

$$-\int\phi_k'(t)\,N(t)\,dt=\sum_m w_{f,m}\,\langle\phi_k,f_m n\rangle+\sum_m w_{\beta,m}\,\langle\phi_k,\beta_m n\rangle.$$

 

Stacking the two sets of equations yields one linear system $G\,w=b$ for the combined coefficient vector $w=(w_f,w_\beta)$, where the PDE rows have zeros in the birth columns. With multiplicative log-normal noise the total population is biased upward by the mean noise factor $e^{\sigma^2/2}$, so the ODE block is built from the de-biased density $n/e^{\sigma^2/2}$.

Returns
-------
np.ndarray of shape (K_t K_a + K_t, M_f + M_beta + 1): stacked weak-form matrix G with right-hand side b as last column
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_weak_system(noisy_density: "np.ndarray", time_step: float, age_step: float, sigma: float, aging_speed: float, source_rates: "np.ndarray", birth_means: "np.ndarray", birth_width: float, n_time_tests: int, n_age_tests: int, support_ratio_time: float, support_ratio_age: float, power: int) -> "np.ndarray":
    '''Return the stacked weak-form system [G | b] for mortality and birth learning.
 
    Grids: t_i = i * time_step (i = 0..I) and age-class midpoints
    a_j = (j + 1/2) * age_step (j = 0..J-1), where noisy_density[i, j] = n(t_i, a_j);
    write T_end = t_I and A_end = J * age_step (the age domain is [0, A_end]).
 
    Test functions. Temporal bumps phi_k (k = 0..K_t-1) and age bumps psi_l
    (l = 0..K_a-1) are sup-norm-normalized piecewise polynomials
    C (x - x1)^p (x1 + ell - x)^p on (x1, x1 + ell) and 0 elsewhere, with p = power,
    support lengths ell_t = support_ratio_time * T_end and ell_a = support_ratio_age * A_end,
    and left endpoints evenly spaced from the domain start to the last position that
    keeps the support inside the domain:
        t1_k = k (T_end - ell_t) / (K_t - 1),   a1_l = l (A_end - ell_a) / (K_a - 1).
    Their derivatives are exact.
 
    Quadrature. <u, v> = sum_i sum_j wt_i wa_j u(t_i, a_j) v(t_i, a_j) with composite
    trapezoidal weights wt in time (time_step/2 at both ends, time_step inside) and
    midpoint-rule weights wa_j = age_step over the age classes. One-dimensional
    integrals use the same weights.
 
    Library. Source columns f_m(a) n with f_m(a) = exp(source_rates[m] a); birth columns
    beta_m(a) n with beta_m(a) = exp(-(a - birth_means[m])^2 / (2 birth_width^2)).
 
    PDE rows r = k * K_a + l (time index major) use Phi_r(t, a) = phi_k(t) psi_l(a):
        b[r]         = -<d_t Phi_r, n> - aging_speed * <d_a Phi_r, n>,
        G[r, m]      = <Phi_r, f_m n>           (source columns),
        G[r, M_f+m]  = 0                        (birth columns).
    ODE rows r = K_t K_a + k use the de-biased density nt = n / exp(sigma^2 / 2) and
    N_i = sum_j wa_j nt[i, j]:
        b[r]         = -sum_i wt_i phi_k'(t_i) N_i,
        G[r, m]      = sum_i wt_i phi_k(t_i) sum_j wa_j f_m(a_j) nt[i, j],
        G[r, M_f+m]  = sum_i wt_i phi_k(t_i) sum_j wa_j beta_m(a_j) nt[i, j].
 
    Parameters
    ----------
    noisy_density : np.ndarray
        Observed density, shape (I+1, J) with I >= 1 and J >= 2.
    time_step, age_step : float
        Positive grid spacings.
    sigma : float
        Log-normal noise level (>= 0) used for the ODE-block de-biasing.
    aging_speed : float
        Known aging speed alpha.
    source_rates : np.ndarray
        1-D array of M_f exponential rates.
    birth_means : np.ndarray
        1-D array of M_beta Gaussian centres.
    birth_width : float
        Gaussian standard deviation (> 0).
    n_time_tests, n_age_tests : int
        Numbers K_t >= 2 and K_a >= 2 of temporal and age test functions.
    support_ratio_time, support_ratio_age : float
        Support fractions in (0, 1].
    power : int
        Test-function exponent p >= 2.
 
    Returns
    -------
    system : np.ndarray
        Array of shape (K_t K_a + K_t, M_f + M_beta + 1): columns 0..M_f+M_beta-1 are G
        (source columns first, then birth columns) and the last column is b.
 
    Raises
    ------
    ValueError
        If noisy_density is not 2-D with at least 2 rows and 2 columns, a step or
        birth_width is not positive, sigma < 0, K_t or K_a < 2, a support ratio is
        outside (0, 1], or power is not an integer >= 2.
    '''
    return system

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _trapezoid_weights(n_points: int, step: float) -> "np.ndarray":
    w = np.full(n_points, float(step))
    w[0] = w[-1] = 0.5 * float(step)
    return w
 
def _oracle_assemble_weak_system(noisy_density: "np.ndarray", time_step: float, age_step: float, sigma: float, aging_speed: float, source_rates: "np.ndarray", birth_means: "np.ndarray", birth_width: float, n_time_tests: int, n_age_tests: int, support_ratio_time: float, support_ratio_age: float, power: int) -> "np.ndarray":
    n = np.asarray(noisy_density, dtype=float)
    if n.ndim != 2 or n.shape[0] < 2 or n.shape[1] < 2:
        raise ValueError("noisy_density must be 2-D with at least 2 rows and 2 columns")
    if not (time_step > 0 and age_step > 0 and birth_width > 0):
        raise ValueError("steps and birth_width must be positive")
    if not sigma >= 0:
        raise ValueError("sigma must be nonnegative")
    if int(n_time_tests) != n_time_tests or int(n_age_tests) != n_age_tests or n_time_tests < 2 or n_age_tests < 2:
        raise ValueError("need at least two test functions in each direction")
    if not (0 < support_ratio_time <= 1 and 0 < support_ratio_age <= 1):
        raise ValueError("support ratios must lie in (0, 1]")
    rates = np.asarray(source_rates, dtype=float).ravel()
    means = np.asarray(birth_means, dtype=float).ravel()
    t = np.arange(n.shape[0]) * float(time_step)
    a = (np.arange(n.shape[1]) + 0.5) * float(age_step)
    a_end = n.shape[1] * float(age_step)
    ell_t = support_ratio_time * t[-1]
    ell_a = support_ratio_age * a_end
    t1 = np.arange(int(n_time_tests)) * (t[-1] - ell_t) / (int(n_time_tests) - 1)
    a1 = np.arange(int(n_age_tests)) * (a_end - ell_a) / (int(n_age_tests) - 1)
    tf = _oracle_polynomial_test_functions(t, t1, ell_t, power)
    af = _oracle_polynomial_test_functions(a, a1, ell_a, power)
    phi, dphi = tf[0], tf[1]
    psi, dpsi = af[0], af[1]
    wt = _trapezoid_weights(t.size, time_step)
    wa = np.full(a.size, float(age_step))
 
    def _inner(time_part, age_part, field):
        return (time_part @ (field * wt[:, None] * wa[None, :]) @ age_part.T).ravel()
 
    source_fields = [np.exp(c * a)[None, :] * n for c in rates]
    birth_profiles = [np.exp(-(a - mu) ** 2 / (2.0 * birth_width ** 2)) for mu in means]
    b_pde = -_inner(dphi, psi, n) - float(aging_speed) * _inner(phi, dpsi, n)
    G_pde = np.zeros((b_pde.size, rates.size + means.size))
    for m, field in enumerate(source_fields):
        G_pde[:, m] = _inner(phi, psi, field)
    debias = np.exp(0.5 * float(sigma) ** 2)
    nt = n / debias
    total = nt @ wa
    b_ode = -(dphi * wt[None, :]) @ total
    G_ode = np.zeros((phi.shape[0], rates.size + means.size))
    for m, c in enumerate(rates):
        G_ode[:, m] = (phi * wt[None, :]) @ (nt @ (wa * np.exp(c * a)))
    for m, prof in enumerate(birth_profiles):
        G_ode[:, rates.size + m] = (phi * wt[None, :]) @ (nt @ (wa * prof))
    G = np.vstack([G_pde, G_ode])
    b = np.concatenate([b_pde, b_ode])
    return np.column_stack([G, b])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
h = 0.05
a = np.arange(501) * h
n0 = np.where(a <= 15.0, 1.0 - np.cos(2.0 * np.pi * a / 15.0), 0.0)
clean = np.zeros((101, a.size))
clean[0] = n0
surv = np.exp(-0.1 * np.exp(0.08 * a[:-1]) * np.expm1(0.08 * h) / 0.08)
beta = np.exp(-(a - 10.0) ** 2 / 50.0)
wa = np.full(a.size, h); wa[0] = wa[-1] = h / 2
for i in range(100):
    clean[i + 1, 1:] = clean[i, :-1] * surv
    clean[i + 1, 0] = np.dot(wa[1:] * beta[1:], clean[i + 1, 1:]) / (1.0 - wa[0] * beta[0])
rates = np.array([0.08, 0.4, 0.72, 1.04, 1.36])
means = np.array([5.0, 10.0, 15.0])
""",
            "call": 'assemble_weak_system(clean.copy(), h, h, 0.0, 1.0, rates.copy(), means.copy(), 5.0, 26, 126, 0.5, 0.5, 14)',
            "gold_call": '_oracle_assemble_weak_system(clean.copy(), h, h, 0.0, 1.0, rates.copy(), means.copy(), 5.0, 26, 126, 0.5, 0.5, 14)',
        },
        {
            "setup": """import numpy as np
h = 0.1
a = np.arange(151) * h
n0 = np.where(a <= 8.0, 1.0 - np.cos(2.0 * np.pi * a / 8.0), 0.0)
t = np.arange(41) * h
clean = np.exp(-0.2 * t[:, None]) * np.interp(a[None, :] - t[:, None], a, n0, left=0.3, right=0.0)
rng = np.random.default_rng(4)
noisy = clean * np.exp(0.3 * rng.standard_normal(clean.shape))
rates = np.array([0.0, 0.1, 0.5])
means = np.array([3.0, 6.0])
""",
            "call": 'assemble_weak_system(noisy.copy(), h, h, 0.3, 1.0, rates.copy(), means.copy(), 3.0, 7, 9, 0.5, 0.4, 6)',
            "gold_call": '_oracle_assemble_weak_system(noisy.copy(), h, h, 0.3, 1.0, rates.copy(), means.copy(), 3.0, 7, 9, 0.5, 0.4, 6)',
        },
        {
            "setup": """import numpy as np
t = np.arange(21) * 0.2
a = np.arange(31) * 0.1
field = np.exp(-(a[None, :] - 1.0 - 0.5 * t[:, None]) ** 2) * (1.0 + 0.1 * t[:, None])
rates = np.array([0.3])
means = np.array([1.5, 2.5])
""",
            "call": 'assemble_weak_system(field.copy(), 0.2, 0.1, 0.1, 0.5, rates.copy(), means.copy(), 0.5, 2, 4, 1.0, 0.6, 3)',
            "gold_call": '_oracle_assemble_weak_system(field.copy(), 0.2, 0.1, 0.1, 0.5, rates.copy(), means.copy(), 0.5, 2, 4, 1.0, 0.6, 3)',
        },
        {
            "setup": """import numpy as np
zero = np.zeros((11, 13))
""",
            "call": 'assemble_weak_system(zero.copy(), 0.1, 0.1, 0.2, 1.0, np.array([0.1]), np.array([0.5]), 0.3, 3, 3, 0.5, 0.5, 4)',
            "gold_call": '_oracle_assemble_weak_system(zero.copy(), 0.1, 0.1, 0.2, 1.0, np.array([0.1]), np.array([0.5]), 0.3, 3, 3, 0.5, 0.5, 4)',
        },
        {
            "setup": """import numpy as np
field = np.ones((5, 6))
def run_model():
    try:
        assemble_weak_system(field.copy(), 0.1, 0.1, 0.0, 1.0, np.array([0.1]), np.array([0.5]), 0.3, 1, 3, 0.5, 0.5, 4)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
