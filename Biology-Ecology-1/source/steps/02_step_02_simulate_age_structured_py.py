"""
Compute the exact solution of a linear McKendrick–von Foerster age-structured model, whose mortality and fecundity are linear combinations of library functions, at the midpoints of the age classes.

In the age-structured (McKendrick–von Foerster) model the number density $n(t,a)$ of individuals of age $a$ is transported at unit aging speed and removed by an age-dependent per-capita mortality $d(a)$ (a source term $f=-d(a)\,n$), while the newborn density is set by a nonlocal renewal (birth) condition:

 

$$\partial_t n+\partial_a n=-d(a)\,n,\qquad n(t,0)=\int_0^{\infty} \beta(a)\,n(t,a)\,da,\qquad n(0,a)=n_0(a).$$

 

Here the source term is written in library form, $f(a,n)=\sum_m w_{f,m}\,e^{c_m a}\,n$, so the mortality is $d(a)=-\sum_m w_{f,m}\,e^{c_m a}$, and the fecundity is a sum of Gaussian bumps, $\beta(a)=\sum_m w_{\beta,m}\,e^{-(a-\mu_m)^2/(2s^2)}$. Along characteristics $a-t=\text{const}$ the density is multiplied by the survival ratio built from $S(a)=\exp\!\big(-\int_0^a d(\alpha)\,d\alpha\big)$, so the solution is known in closed form once the birth flux $B(t)=n(t,0)$ is known; $B$ itself satisfies a linear Volterra integral equation of the second kind (the renewal equation), which in general has no closed-form solution and must be solved numerically. The solution is continuous along characteristics but in general has a jump across the characteristic $a=t$ issuing from the origin, because $n_0(0)$ need not equal the initial birth flux.

 

$$
\partial_t n+\partial_a n=-d(a)\,n,\qquad n(t,0)=\int \beta(a)\,n(t,a)\,da,\qquad n(0,a)=n_0(a).
$$

 

Here the source term is written in library form, $f(a,n)=\sum_m w_{f,m}\,e^{c_m a}\,n$, so the mortality is $d(a)=-\sum_m w_{f,m}\,e^{c_m a}$, and the fecundity is a sum of Gaussian bumps, $\beta(a)=\sum_m w_{\beta,m}\,e^{-(a-\mu_m)^2/(2s^2)}$. With the time step equal to the age step, characteristics through grid nodes pass through grid nodes, so transport and survival along each characteristic are exact and only the renewal integral needs a quadrature rule.

Returns
-------
np.ndarray of shape (n_steps + 1, n_ages): exact density n(t_i, a_j) at times t_i = i h and age-class midpoints a_j = (j + 1/2) h
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_age_structured(initial_support: float, age_step: float, n_ages: int, n_steps: int, source_coefficients: "np.ndarray", source_rates: "np.ndarray", birth_coefficients: "np.ndarray", birth_means: "np.ndarray", birth_width: float) -> "np.ndarray":
    '''Return the exact age-structured density at age-class midpoints and grid times.
 
    With h = age_step, ages are the class midpoints a_j = (j + 1/2) h (j = 0..n_ages-1)
    of the age domain [0, A], A = n_ages h, and times are t_i = i h (i = 0..n_steps).
    The model is
 
        dn/dt + dn/da = -d(a) n,   n(t, 0) = integral_0^A beta(a) n(t, a) da,
        n(0, a) = n0(a) = 1 - cos(2 pi a / L0) for 0 <= a <= L0 and 0 for a > L0,
 
    with L0 = initial_support and
 
        d(a)    = -sum_m source_coefficients[m] * exp(source_rates[m] * a),
        beta(a) =  sum_m birth_coefficients[m] * exp(-(a - birth_means[m])^2 / (2 birth_width^2)).
 
    Because L0 + n_steps h <= A is required, no individual leaves [0, A] and the renewal
    integral equals the integral over [0, infinity). The returned values are those of the
    exact solution n(t_i, a_j) of this continuous model (no grid discretization), each
    accurate to within 1e-10 * (1 + |n(t_i, a_j)|). No grid point lies on the
    characteristic a = t, across which the exact solution may jump.
 
    Parameters
    ----------
    initial_support : float
        L0 > 0, the support length of the initial density.
    age_step : float
        Width h > 0 of the age classes; also the time step.
    n_ages : int
        Number of age classes (>= 1).
    n_steps : int
        Number of time steps (>= 0).
    source_coefficients : np.ndarray
        1-D array of library coefficients w_f[m] (negative values mean mortality).
    source_rates : np.ndarray
        1-D array of exponential rates c_m (same length; c_m = 0 is allowed).
    birth_coefficients : np.ndarray
        1-D array of Gaussian birth coefficients w_beta[m].
    birth_means : np.ndarray
        1-D array of Gaussian centres mu_m (same length as birth_coefficients).
    birth_width : float
        Common Gaussian standard deviation s > 0.
 
    Returns
    -------
    density : np.ndarray
        Array of shape (n_steps + 1, n_ages) with density[i, j] = n(t_i, a_j).
 
    Raises
    ------
    ValueError
        If array lengths are inconsistent, age_step, birth_width or initial_support is
        not positive, n_ages < 1, n_steps < 0, or L0 + n_steps h > n_ages h.
    '''
    return density

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _log_survival(a: "np.ndarray", wf: "np.ndarray", cf: "np.ndarray") -> "np.ndarray":
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    for w, c in zip(wf, cf):
        out += w * (a if c == 0.0 else np.expm1(c * a) / c)
    return out
 
def _birth_profile(a: "np.ndarray", wb: "np.ndarray", mb: "np.ndarray", s: float) -> "np.ndarray":
    a = np.asarray(a, dtype=float)
    out = np.zeros_like(a)
    for w, mu in zip(wb, mb):
        out += w * np.exp(-(a - mu) ** 2 / (2.0 * s * s))
    return out
 
def _initial_profile(u: "np.ndarray", support: float) -> "np.ndarray":
    u = np.asarray(u, dtype=float)
    return np.where((u >= 0.0) & (u <= support), 1.0 - np.cos(2.0 * np.pi * u / support), 0.0)
 
def _renewal_trapezoid(k: float, n_nodes: int, support: float, wf, cf, wb, mb, s) -> "np.ndarray":
    tau = np.arange(n_nodes + 1) * k
    x, gw = np.polynomial.legendre.leggauss(120)
    u = 0.5 * support * (x + 1.0)
    uw = 0.5 * support * gw
    arg = u[None, :] + tau[:, None]
    forcing = (_birth_profile(arg, wb, mb, s) * np.exp(_log_survival(arg, wf, cf) - _log_survival(u, wf, cf)[None, :]) * _initial_profile(u, support)[None, :]) @ uw
    kernel = _birth_profile(tau, wb, mb, s) * np.exp(_log_survival(tau, wf, cf))
    flux = np.empty(n_nodes + 1)
    flux[0] = forcing[0]
    denom = 1.0 - 0.5 * k * kernel[0]
    for n in range(1, n_nodes + 1):
        memory = np.dot(kernel[1:n], flux[n - 1:0:-1]) + 0.5 * kernel[n] * flux[0]
        flux[n] = (forcing[n] + k * memory) / denom
    return flux
 
def _oracle_simulate_age_structured(initial_support: float, age_step: float, n_ages: int, n_steps: int, source_coefficients: "np.ndarray", source_rates: "np.ndarray", birth_coefficients: "np.ndarray", birth_means: "np.ndarray", birth_width: float) -> "np.ndarray":
    wf = np.asarray(source_coefficients, dtype=float).ravel()
    cf = np.asarray(source_rates, dtype=float).ravel()
    wb = np.asarray(birth_coefficients, dtype=float).ravel()
    mb = np.asarray(birth_means, dtype=float).ravel()
    if wf.size != cf.size or wb.size != mb.size:
        raise ValueError("coefficient and parameter arrays must have matching lengths")
    if not (age_step > 0 and birth_width > 0 and initial_support > 0):
        raise ValueError("age_step, birth_width and initial_support must be positive")
    if int(n_ages) != n_ages or n_ages < 1 or int(n_steps) != n_steps or n_steps < 0:
        raise ValueError("n_ages must be >= 1 and n_steps >= 0")
    h, L0 = float(age_step), float(initial_support)
    n_ages, n_steps = int(n_ages), int(n_steps)
    if L0 + n_steps * h > n_ages * h * (1.0 + 1e-12):
        raise ValueError("individuals would leave the age domain")
    a = (np.arange(n_ages) + 0.5) * h
    t = np.arange(n_steps + 1) * h
    T, A = np.meshgrid(t, a, indexing="ij")
    out = np.empty_like(T)
    old = A > T
    out[old] = _initial_profile(A[old] - T[old], L0) * np.exp(_log_survival(A[old], wf, cf) - _log_survival(A[old] - T[old], wf, cf))
    if (~old).any():
        # birth flux on a grid of step h/4 containing all t - a = (i - j - 1/2) h, with
        # Richardson extrapolation of the trapezoidal Volterra scheme (error ~ k^6)
        base = h / 4.0
        levels = []
        for level in range(3):
            k = base / 2 ** level
            n_nodes = int(round(n_steps * h / k))
            levels.append(_renewal_trapezoid(k, n_nodes, L0, wf, cf, wb, mb, birth_width)[::2 ** level])
        power = 2
        while len(levels) > 1:
            levels = [(2 ** power * levels[i + 1] - levels[i]) / (2 ** power - 1) for i in range(len(levels) - 1)]
            power += 2
        flux = levels[0]
        idx = np.rint((T[~old] - A[~old]) / base).astype(int)
        out[~old] = flux[idx] * np.exp(_log_survival(A[~old], wf, cf))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
""",
            "call": 'simulate_age_structured(15.0, 0.05, 500, 200, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([10.0]), 5.0)',
            "gold_call": '_oracle_simulate_age_structured(15.0, 0.05, 500, 200, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([10.0]), 5.0)',
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
""",
            "call": 'simulate_age_structured(3.0, 0.1, 80, 40, np.array([-0.05, 0.02, -0.001]), np.array([0.08, 0.4, 1.36]), np.array([0.3, 1.2, -0.1]), np.array([0.0, 3.0, 6.0]), 1.5)',
            "gold_call": '_oracle_simulate_age_structured(3.0, 0.1, 80, 40, np.array([-0.05, 0.02, -0.001]), np.array([0.08, 0.4, 1.36]), np.array([0.3, 1.2, -0.1]), np.array([0.0, 3.0, 6.0]), 1.5)',
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
""",
            "call": 'simulate_age_structured(15.0, 0.05, 500, 200, np.array([-0.0999]), np.array([0.08]), np.array([0.64, -1.90, 4.14]), np.array([5.0, 10.0, 15.0]), 5.0)',
            "gold_call": '_oracle_simulate_age_structured(15.0, 0.05, 500, 200, np.array([-0.0999]), np.array([0.08]), np.array([0.64, -1.90, 4.14]), np.array([5.0, 10.0, 15.0]), 5.0)',
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
""",
            "call": 'simulate_age_structured(1.0, 0.25, 8, 4, np.array([-0.3]), np.array([0.0]), np.array([0.4]), np.array([1.0]), 1.0)',
            "gold_call": '_oracle_simulate_age_structured(1.0, 0.25, 8, 4, np.array([-0.3]), np.array([0.0]), np.array([0.4]), np.array([1.0]), 1.0)',
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
""",
            "call": 'simulate_age_structured(2.0, 0.5, 6, 0, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([1.0]), 1.0)',
            "gold_call": '_oracle_simulate_age_structured(2.0, 0.5, 6, 0, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([1.0]), 1.0)',
            "tol": 1e-10,
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        simulate_age_structured(15.0, 0.05, 500, 201, np.array([-0.1]), np.array([0.08]), np.array([1.0]), np.array([10.0]), 5.0)
        return 0
    except ValueError:
        return 1
""",
            "call": 'run_model()',
            "gold_call": '1',
        },
    ]
