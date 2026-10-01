# Biology-Ecology-1

## Background

Physiologically structured population models describe how individual traits such as age or size shape the growth of a population. In the age-structured setting, individuals age deterministically while dying at an age-dependent rate, and newborns enter at age zero at a rate that depends on how many individuals of reproductive age are present. The resulting first-order hyperbolic equation, with a nonlocal renewal condition at the age boundary, is a standard model in demography, fisheries and wildlife ecology. Its vital rates are hard to measure directly, because individual-level studies are expensive and census data are noisy, coarse and nonnegative.
 
Equation-learning methods treat vital-rate identification as a model-selection problem: candidate functional forms suggested by ecological knowledge are collected in a library, and a sparse regression chooses the few terms that explain the observed dynamics. Weak (integral) formulations make such regressions robust to noise, because derivatives are transferred from the data onto smooth, compactly supported test functions. The same compact support removes boundary information, so birth processes that act only at the boundary of the structure domain need a separate route into the regression. Recovering the right terms, rather than an analytically different but similarly fitting combination, is a central concern when candidate functions are similar or when one part of the stacked regression problem vastly outnumbers the other.

## Problem

Age-dependent mortality and fecundity determine how a structured population changes, but inferring them from population census data usually requires repeated forward simulation of a hyperbolic PDE for every candidate model, which is slow and ill-posed under noise. A recent weak-form sparse-regression approach instead selects the model ingredients of a McKendrick–von Foerster age-structured model directly from noisy time series of the age histogram, including the nonlocal birth (boundary) process that compactly supported test functions cannot see and that must therefore be learned by coupling the PDE to the total-population equation. The method takes a noisy density time series and libraries of candidate mortality and birth functions and returns a sparse coefficient vector, whose quality is judged by coefficient errors, a true positivity ratio and the prediction error of the learned model outside the training window.
 
The benchmark population obeys $\partial_t n+\partial_a n=-d^\star(a)\,n$ with $n(t,0)=\int_0^{25}\beta^\star(a)\,n(t,a)\,da$, $d^\star(a)=0.1\,e^{0.08a}$, $\beta^\star(a)=e^{-(a-10)^2/50}$ and $n(0,a)=1-\cos(2\pi a/15)$ for $0\le a\le 15$ and $0$ for $15<a\le 25$, on ages $a\in[0,25]$ and times $t\in[0,10]$, so no individual reaches age 25 before $t=10$. The source-term (mortality) library is $\{e^{c a}\,n\}$ with $c=(0.08,0.40,0.72,1.04,1.36)$ and the birth library is $\{e^{-(a-\mu)^2/50}\}$ with $\mu=(5,10,15)$; order the coefficients as $w=(w_f,w_\beta)$, so the truth is $w^\star=(-0.1,0,0,0,0,0,1,0)$ and the mortality is $d(a)=-\sum_m w_{f,m}e^{c_m a}$. The aging speed is known to equal 1 and is moved to the data side of the weak form, the birth coefficients are identified by stacking the PDE weak form with the weak form of the total-population equation, and the sparse solution is obtained with modified sequential-thresholding least squares (MSTLS) followed by the source paper's cross-validation of the boundary (birth) terms.
 
Use these conventions. (i) Data: the noise-free data are values of the exact solution of this continuous model (no grid discretization, accurate to at least $10^{-10}$ relative) at the 500 age-class midpoints $a_j=(j+\tfrac12)h$ and the times $t_i=ih$, with $h=0.05$. (ii) Noise: the training rows $t\le 5$ (array of shape $101\times 500$) are multiplied by $e^{\sigma Z}$ with $Z$ equal to `numpy.random.default_rng(5).standard_normal((101, 500))`, where $\sigma$ corresponds to the source paper's expected noise-to-signal ratio $\sigma_{NR}=0.1$. (iii) Weak form on the training window $[0,5]\times[0,25]$: the source paper's piecewise-polynomial test functions with its default exponents, normalization and support ratios in time and age; 26 temporal and 126 age test functions whose left endpoints are evenly spaced from the domain start to the last position that keeps the support inside the domain; tensor products for the PDE rows and the temporal functions alone for the total-population rows; the composite trapezoidal rule in time and the midpoint rule over the age classes for every integral. (iv) The total population and every total-population row are built from the de-biased density $n/\mathbb{E}[e^{\sigma z}]$, using the known $\sigma$, while the PDE rows use the noisy density as observed. (v) MSTLS follows the weak-form PDE identification reference that the source paper cites for it (scale-aware thresholding bounds, least-squares refits, and the loss and selection rule of that reference), with the threshold grid $\lambda_j=10^{-4+4j/49}$, $j=0,\dots,49$, and minimum-norm least squares. (vi) Prediction: solve the learned model exactly from $$n(0,a)$$ on $$[0,10]$$ at the same points and compute the relative $L^2$ prediction error over the testing window $(5,10)\times(0,25)$ with the trapezoidal rule in time and the midpoint rule in age.
 
State the renewal (integral) equation for the newborn density $B(t)=n(t,0)$ that determines the data, its value $B(0)$, the PDE weak-form equation and the total-population weak-form equation that you regress on, and the size of the stacked system. Report $\sigma$ and the de-biasing factor; the coefficient vector and selected threshold of the first MSTLS fit of the full stacked system, the prediction error that this uncorrected fit would give, and the reason its birth coefficients are or are not sparse; the re-selected birth coefficients from the total-population block and the birth terms retained for the refit; the final learned coefficient vector with its $E_\infty$, $E_2$, true positivity ratio (as defined in the source paper) and relative residual $R$ of the full stacked system; and the prediction error $E_p$ of the final learned model. Your final answer must be a single number: $E_p$.
 
Output Format Requirements:
 
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
 
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
 
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_polynomial_test_functions.py

Goal
----
Evaluate a family of sup-norm-normalized piecewise-polynomial test functions and their exact first derivatives on a uniform grid.

```python
def polynomial_test_functions(grid: "np.ndarray", left_endpoints: "np.ndarray", support_length: float, power: int) -> "np.ndarray":
    '''Evaluate normalized piecewise-polynomial test functions and their derivatives.
 
    For each left endpoint x1_k the test function is
 
        phi_k(x) = C (x - x1_k)^p (x1_k + ell - x)^p   on the open interval (x1_k, x1_k + ell),
 
    and zero elsewhere (including the two support endpoints), where ell = support_length,
    p = power and C is chosen so that max_x phi_k(x) = 1 (the maximum is attained at the
    support midpoint). The derivative d phi_k / dx must be the exact analytic derivative
    of this piecewise polynomial, evaluated at the grid points (also zero outside the open
    support). Grid points and endpoints are arbitrary real numbers; supports need not lie
    inside the grid range.
 
    Parameters
    ----------
    grid : np.ndarray
        1-D array of n evaluation points.
    left_endpoints : np.ndarray
        1-D array of K left support endpoints x1_k.
    support_length : float
        Common support length ell > 0.
    power : int
        Exponent p >= 2 applied to both factors.
 
    Returns
    -------
    values : np.ndarray
        Array of shape (2, K, n): values[0, k, i] = phi_k(grid[i]) and
        values[1, k, i] = phi_k'(grid[i]).
 
    Raises
    ------
    ValueError
        If grid or left_endpoints is not a nonempty 1-D array, support_length <= 0,
        or power is not an integer >= 2.
    '''
    return values
```

### Step 2

02_simulate_age_structured.py

Goal
----
Compute the exact solution of a linear McKendrick–von Foerster age-structured model, whose mortality and fecundity are linear combinations of library functions, at the midpoints of the age classes.

```python
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
```

### Step 3

03_add_lognormal_noise.py

Goal
----
Corrupt a population density with seeded multiplicative log-normal noise whose strength is specified by the expected noise-to-signal ratio.

```python
def add_lognormal_noise(clean_density: "np.ndarray", noise_to_signal_ratio: float, seed: int) -> "np.ndarray":
    '''Return the density corrupted by seeded multiplicative log-normal noise.
 
    The noise level sigma >= 0 is the unique nonnegative solution of
 
        E[(exp(z) - 1)^2] = noise_to_signal_ratio,   z ~ N(0, sigma^2),
 
    accurate to within 1e-12 relative. The noisy density is
 
        noisy = clean_density * exp(sigma * Z),
 
    where Z = np.random.default_rng(seed).standard_normal(clean_density.shape) is drawn in
    a single call immediately after constructing the generator (so Z[i, j] is the
    (i, j) entry of that one draw in C order).
 
    Parameters
    ----------
    clean_density : np.ndarray
        Array of noise-free densities (any shape).
    noise_to_signal_ratio : float
        Expected mean-square noise-to-signal ratio sigma_NR >= 0.
    seed : int
        Seed for np.random.default_rng.
 
    Returns
    -------
    noisy : np.ndarray
        Noisy density with the shape of clean_density.
 
    Raises
    ------
    ValueError
        If noise_to_signal_ratio is negative or not finite.
    '''
    return noisy
```

### Step 4

04_assemble_weak_system.py

Goal
----
Assemble the stacked weak-form linear system that couples the transport PDE with the total-population ODE for learning mortality and fecundity terms of an age-structured model with known aging speed.

```python
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
```

### Step 5

05_mstls_sparse_regression.py

Goal
----
Solve a sparse regression problem with modified sequential-thresholding least squares (MSTLS) and select the sparsity threshold by minimizing a fit-plus-sparsity loss over a grid.

```python
def mstls_sparse_regression(G: "np.ndarray", b: "np.ndarray", lambdas: "np.ndarray") -> "np.ndarray":
    '''Return the MSTLS coefficient vector followed by the selected threshold.
 
    Let J be the number of columns, w_LS the minimum-norm least-squares solution of
    G w ~ b, and G_j the j-th column (all norms are Euclidean). For a threshold lambda:
      * bounds  L_j = lambda * max(1, ||b|| / ||G_j||),
                U_j = (1 / lambda) * min(1, ||b|| / ||G_j||);
      * iterate from w^0 = w_LS and I^{-1} = {0..J-1}:
            I^l     = {j : L_j <= |w^l_j| <= U_j},
            w^{l+1} = minimum-norm least-squares solution of G w ~ b with w_j = 0 for j
                      not in I^l (w^{l+1} = 0 if I^l is empty),
        stopping at the first l with I^l = I^{l-1}; the result w^lambda is the current
        iterate w^l (restricted least squares on the stable index set).
    The loss is
        loss(lambda) = ||G (w^lambda - w_LS)|| / ||G w_LS|| + |I^lambda| / J,
    where |I^lambda| is the size of the final index set. The selected threshold is the
    smallest lambda in `lambdas` whose loss equals the minimum loss over `lambdas`.
 
    Parameters
    ----------
    G : np.ndarray
        Matrix of shape (R, J) with R >= 1, J >= 1 and no zero column.
    b : np.ndarray
        Right-hand side of shape (R,).
    lambdas : np.ndarray
        1-D nonempty array of positive thresholds (any order).
 
    Returns
    -------
    result : np.ndarray
        Float array of shape (J + 1,): result[:J] is w^lambda_hat, with exact zeros
        outside the final index set, and result[J] is lambda_hat.
 
    Raises
    ------
    ValueError
        If shapes are inconsistent, G has a zero column, lambdas is empty or contains a
        nonpositive value, or G w_LS = 0 (the loss is undefined).
    '''
    return result
```

### Step 6

06_boundary_bagging_regression.py

Goal
----
Identify source and boundary (birth) coefficients from a stacked PDE/ODE weak-form system, using a cross-validated boundary-term selection that corrects the PDE-dominated joint sparse regression.

```python
def boundary_bagging_regression(G: "np.ndarray", b: "np.ndarray", n_pde_rows: int, n_source_terms: int, lambdas: "np.ndarray") -> "np.ndarray":
    '''Return the learned coefficient vector w = (w_f, w_beta) after boundary cross-validation.
 
    Notation: G has J = M_f + M_beta columns (the first M_f = n_source_terms are source
    terms, the rest birth terms); rows 0..n_pde_rows-1 form the PDE block and the
    remaining rows form the ODE block, whose source and birth sub-matrices are Xi_f and
    Xi_beta and whose right-hand side is b_ode. MSTLS(A, y) denotes the modified
    sequential-thresholding least-squares estimate with threshold chosen from `lambdas`:
    minimum-norm least squares w_LS; bounds L_j = lambda max(1, ||y||/||A_j||) and
    U_j = min(1, ||y||/||A_j||)/lambda; iterate I = {j : L_j <= |w_j| <= U_j} and
    restricted least-squares refits (zeros off I) from w_LS until I stops changing;
    loss(lambda) = ||A (w^lambda - w_LS)|| / ||A w_LS|| + |I^lambda| / (number of columns);
    choose the smallest lambda attaining the minimum loss. supp(v) is the set of
    indices of nonzero entries and R(v) = ||b - G v|| / ||b|| is the relative residual of
    the full system.
 
    Procedure:
      1. (w_f, w_beta) = MSTLS(G, b).
      2. w_beta_hat = MSTLS(Xi_beta, b_ode - Xi_f w_f).
      3. If supp(w_beta_hat) != supp(w_beta): let S = supp(w_beta_hat) & supp(w_beta) if
         this intersection is nonempty, otherwise the union. Recompute
         (w_f, w_beta_S) = MSTLS(G restricted to all source columns plus the birth
         columns in S, b), and return w_f with w_beta_S placed at the birth positions in
         S and zeros at the other birth positions.
      4. Otherwise return (w_f, w_beta_hat) if R((w_f, w_beta_hat)) < R((w_f, w_beta)),
         else (w_f, w_beta).
 
    Parameters
    ----------
    G : np.ndarray
        Stacked matrix of shape (R, J) with no zero column.
    b : np.ndarray
        Stacked right-hand side of shape (R,), nonzero.
    n_pde_rows : int
        Number of PDE rows (0 < n_pde_rows < R).
    n_source_terms : int
        Number of source columns M_f (0 < M_f < J).
    lambdas : np.ndarray
        Threshold grid (positive values).
 
    Returns
    -------
    w : np.ndarray
        Learned coefficients of shape (J,), exact zeros for inactive terms.
 
    Raises
    ------
    ValueError
        If the block sizes are inconsistent with G, or any MSTLS subproblem is invalid
        (zero column, empty or nonpositive thresholds, vanishing least-squares fit).
    '''
    return w
```

### Step 7

07_prediction_error.py

Goal
----
Measure the out-of-sample prediction error of a learned age-structured model by solving it exactly and comparing with the true density over the testing interval.

```python
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
```

### Step 8

08_wsindy_prediction_pipeline.py

Goal
----
Run the complete weak-form sparse identification pipeline on a noisy synthetic age-structured population and return the prediction error of the learned model.

```python
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
```
