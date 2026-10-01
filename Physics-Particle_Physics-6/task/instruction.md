# Physics-Particle_Physics-6

## Background

Measurements at hadron colliders compare observed event counts with predictions assembled from simulated samples. The prediction for every bin of an observable is a sum of templates, one per physics process, whose shapes come from Monte Carlo generation followed by a detector simulation. Because the simulated samples are finite, every template carries a statistical uncertainty of its own, and because the simulation depends on calibrations and modelling choices, every template also carries shape uncertainties that are estimated by regenerating it at shifted values of the corresponding parameters. In the likelihood these uncertainties are represented by nuisance parameters, profiled together with the parameter of interest such as a signal strength, and the width of the resulting confidence interval depends on how many nuisance parameters the model carries and how well they are constrained.

The underlying distributions that templates approximate are smooth functions of the observable. Statistical models that exploit this smoothness pool the information of neighbouring bins, which reduces the effective number of free parameters, tightens the template where the simulated sample is small, and removes the bin-by-bin noise that would otherwise be mistaken for a physical shape variation. Gaussian processes are a standard tool for such smoothing: a covariance kernel encodes the correlation length of the function, hyperparameters can be chosen by the evidence of the data, and for count data the Poisson likelihood is combined with the Gaussian prior on the logarithm of the rate through a Gaussian approximation around the most probable configuration. The eigendecomposition of the resulting covariance orders the directions of template uncertainty by their variance, so that a few leading directions describe most of the uncertainty of a smooth template.

## Problem

Statistical inference at the LHC rests on template histograms: the expected count in every bin of an observable is assembled from Monte Carlo predictions, and deviations from the nominal prediction enter the likelihood through nuisance parameters. The standard treatment introduces one multiplicative factor per bin, with an independent Gaussian constraint, for the statistical uncertainty of the Monte Carlo template, and one interpolation parameter per systematic source for its shape variation, so that the number of nuisance parameters grows with the number of bins and the constraints ignore the smoothness of the underlying distribution. A recent proposal replaces the histogram template by a smooth functional representation derived from the posterior of a log-Gaussian Cox process fitted to the binned Monte Carlo counts, augments the posterior covariance by one rank-one contribution per systematic shape variation, and truncates the eigendecomposition of the combined covariance so that a small number of Gaussian-constrained mode amplitudes replaces the full set of per-bin factors and interpolation parameters. The inputs below are the nominal and systematically varied Monte Carlo counts of one background, the signal template and the observed counts of a single channel, and the output is the residual between the signal strength fitted with the proposal's template and the signal strength fitted with the histogram template on the same data, the quantity whose ensemble mean measures the plug-in penalty of smoothing the template once before the fit.

The Monte Carlo counts $a_j$ in bins of width $w_j$ centred at $x_j$ are modelled as Poisson variables with means $e^{f(x_j)} w_j$, where the log-rate $f$ is a Gaussian process with a Matérn kernel of order $5/2$ of amplitude $\sigma = 1$ and a parametric mean function that absorbs the large-scale shape. The mean function is a cubic B-spline basis of four functions on the open uniform knot vector of the range $[0, 1]$, with the boundary knots of multiplicity four and no interior knot, whose coefficients are integrated out under independent Gaussian priors of standard deviation $b = 10$, so that the mean function becomes an additional term of the prior covariance. The length scale of the kernel is selected among the candidates $\{0.1, 0.2, 0.3, 0.5, 0.8\}$ by maximising the Laplace approximation to the log marginal likelihood of the nominal counts, and the same prior covariance is used for every Monte Carlo template. The posterior of the log-rate is taken in the Laplace approximation: its mode maximises the Poisson log-likelihood plus the Gaussian log-prior and its covariance is the inverse curvature of that objective at the mode. The smooth background template of a bin is the fitted rate of the nominal sample scaled to the data luminosity, the Monte Carlo sample being $\tau = 10$ times larger than the data.

Three systematic sources are described by Monte Carlo templates at their $+1\sigma$ and $-1\sigma$ points: a calibration shift that moves events between bins, a resolution change that broadens or narrows the peak, and a normalisation change of the peaked component. For each source the proposal obtains the systematic shape direction in log-rate space from the Gaussian process fits to its two variation templates, so that the smoothing is applied before the differencing, and adds the outer product of that direction to the statistical posterior covariance. The eigenmodes of the combined covariance are kept in decreasing order of eigenvalue up to the smallest number of modes whose eigenvalues sum to at least 95 percent of the trace. The expected count of bin $j$ is then the signal strength $\mu$ times the signal template plus the smooth background template deformed multiplicatively by the exponential of the retained modes, each mode scaled by the square root of its eigenvalue and multiplied by an amplitude that carries a unit Gaussian constraint, and the likelihood is the product of the Poisson terms over bins and the constraints. The signal strength is fitted jointly with the amplitudes, and its uncertainty is the parabolic one from the inverse Hessian of the negative log-likelihood at the minimum. The same data are also fitted with the histogram template $a_j/\tau$, the standard per-bin multiplicative factors with their Gaussian constraints, and one unit-Gaussian constrained parameter per source that interpolates between the nominal and the two variation histograms with the standard framework's interpolation code 4, exponential beyond one standard deviation and polynomial inside, for comparison.

The twelve bins are of equal width on $x \in [0, 1]$. The nominal Monte Carlo counts $a_j$, the varied counts at the $+1\sigma$ and $-1\sigma$ points of the calibration (cal), resolution (res) and normalisation (norm) sources, the expected signal counts $s_j$ at unit signal strength, and the observed counts $n_j$ are

| Bin | $a_j$ | cal $+$ | cal $-$ | res $+$ | res $-$ | norm $+$ | norm $-$ | $s_j$ | $n_j$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2947 | 3375 | 2557 | 3016 | 2985 | 3098 | 2938 | 0.00 | 295 |
| 2 | 2193 | 2420 | 1887 | 2107 | 2086 | 2180 | 2160 | 0.29 | 238 |
| 3 | 1526 | 1672 | 1339 | 1507 | 1556 | 1550 | 1463 | 10.09 | 137 |
| 4 | 1094 | 1202 | 1056 | 1120 | 1141 | 1186 | 1093 | 40.82 | 169 |
| 5 | 1026 | 1037 | 1115 | 1136 | 1032 | 1162 | 1053 | 28.79 | 110 |
| 6 | 1288 | 1242 | 1471 | 1348 | 1299 | 1414 | 1262 | 12.63 | 132 |
| 7 | 1572 | 1498 | 1441 | 1623 | 1514 | 1727 | 1389 | 3.67 | 151 |
| 8 | 1133 | 1384 | 939 | 1175 | 1126 | 1246 | 1029 | 0.41 | 115 |
| 9 | 561 | 750 | 397 | 578 | 526 | 594 | 477 | 0.02 | 50 |
| 10 | 244 | 301 | 151 | 257 | 230 | 209 | 210 | 0.00 | 10 |
| 11 | 110 | 140 | 92 | 138 | 108 | 125 | 127 | 0.00 | 13 |
| 12 | 74 | 79 | 67 | 80 | 83 | 69 | 86 | 0.00 | 11 |

Report as the single final scalar the residual between the signal strength fitted with the proposal's eigenmode template and the signal strength fitted with the histogram template, the former minus the latter. Alongside it, give the two fitted signal strengths with their parabolic uncertainties, the mean over bins of the ratio between the posterior standard deviation of the log-rate and the relative statistical uncertainty of the histogram template, and explain the exact posterior-variance bound in terms of the fitted Monte Carlo means, distinguishing it from the comparison with the raw-count histogram uncertainties.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_effective_kernel_matrix

Goal
----
Build the effective prior covariance matrix of the log-rate at the bin centres of a template: a Matérn 5/2 kernel plus the contribution of a cubic B-spline mean function whose coefficients are integrated out under a broad Gaussian prior.

```python
def effective_kernel_matrix(x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, ell: float, prior_scale: float) -> "np.ndarray":
    r"""Return the effective prior covariance matrix of the log-rate at the given positions.

    Parameters
    ----------
    x : np.ndarray
        Positions of shape (N,), N at least one, finite, each within the
        closed range [lo, hi].
    lo : float
        Lower end of the template range, finite.
    hi : float
        Upper end of the template range, finite and above lo.
    n_basis : int
        Number of cubic B-spline basis functions of the mean function, at
        least four.
    sigma : float
        Amplitude of the Matérn 5/2 kernel, finite and above zero.
    ell : float
        Length scale of the Matérn 5/2 kernel, finite and above zero.
    prior_scale : float
        Standard deviation b of the independent Gaussian prior on every
        coefficient of the mean function, finite and above zero.

    Returns
    -------
    K_eff : np.ndarray
        Array of shape (N, N) with entries k(x_i, x_j) + b ** 2 * sum_m
        H_im H_jm, where k(x, x') = sigma ** 2 * (1 + sqrt(5) r + 5 r ** 2 / 3)
        * exp(-sqrt(5) r) with r = |x - x'| / ell is the Matérn 5/2 kernel
        and H_im is the value at x_i of the m-th cubic B-spline of the open
        uniform knot vector on [lo, hi] (boundary knots of multiplicity four,
        n_basis - 4 interior knots equally spaced between them, so that the
        basis functions sum to one at every position of the closed range,
        including at hi). The result is symmetric and positive definite.

    Raises
    ------
    ValueError
        If x is not a finite one-dimensional array with at least one entry
        inside [lo, hi], if lo or hi is not finite or hi is not above lo,
        if n_basis is below four, or if sigma, ell or prior_scale is not
        finite or not above zero.
    """
    return K_eff
```

### Step 2

02_lgcp_posterior_mode

Goal
----
Compute the posterior mode of the log-rate of a log-Gaussian Cox process fitted to binned Monte Carlo counts, the penalised maximum-likelihood estimate of the Laplace approximation.

```python
def lgcp_posterior_mode(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    r"""Return the log-rate vector maximising the penalised Poisson log-likelihood.

    Parameters
    ----------
    counts : np.ndarray
        Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers (an integer or float dtype).
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite.

    Returns
    -------
    f_hat : np.ndarray
        Array of shape (N,) maximising sum_j (a_j f_j - exp(f_j) w_j)
        - f^T K^{-1} f / 2 over f, with a_j the counts and w_j the widths.
        The objective is strictly concave, so the maximiser is unique;
        it is returned with every component converged to within 1e-10 of
        the exact maximiser (the gradient a_j - exp(f_j) w_j
        - (K^{-1} f)_j vanishes to that accuracy). A bin with zero counts
        is valid and is kept finite by the prior rather than driven to
        minus infinity.

    Raises
    ------
    ValueError
        If counts is not a finite one-dimensional array of non-negative
        integer values, if widths is not a finite one-dimensional array of
        the same length with every entry above zero, or if K is not a
        finite (N, N) matrix that is symmetric to within 1e-9 and positive
        definite.
    """
    return f_hat
```

### Step 3

03_laplace_posterior_covariance

Goal
----
Compute the Laplace approximation to the posterior covariance of the log-rate of a log-Gaussian Cox process at its mode.

```python
def laplace_posterior_covariance(f_hat: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    r"""Return the Laplace posterior covariance of the log-rate at the given mode.

    Parameters
    ----------
    f_hat : np.ndarray
        Posterior mode of the log-rate, shape (N,), N at least one, finite.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite.

    Returns
    -------
    Sigma : np.ndarray
        Array of shape (N, N) equal to the inverse of K^{-1} + W, with W
        the diagonal matrix of the fitted rates exp(f_hat_j) * w_j. The
        result is symmetric and positive definite, and its diagonal
        entries never exceed 1 / (exp(f_hat_j) * w_j).

    Raises
    ------
    ValueError
        If f_hat is not a finite one-dimensional array, if widths is not
        a finite one-dimensional array of the same length with every entry
        above zero, or if K is not a finite (N, N) matrix that is
        symmetric to within 1e-9 and positive definite.
    """
    return Sigma
```

### Step 4

04_laplace_log_marginal_likelihood

Goal
----
Evaluate the Laplace approximation to the log marginal likelihood of binned Monte Carlo counts under a log-Gaussian Cox process with a given prior covariance.

```python
def laplace_log_marginal_likelihood(counts: "np.ndarray", widths: "np.ndarray", K: "np.ndarray") -> float:
    r"""Return the Laplace approximation to the log marginal likelihood of the counts.

    Parameters
    ----------
    counts : np.ndarray
        Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite.

    Returns
    -------
    log_marginal : float
        The value, as a native Python float, of sum_j (a_j log mu_j - mu_j
        - log Gamma(a_j + 1)) - f^T K^{-1} f / 2 - log det(I + K W) / 2,
        evaluated at the posterior mode f of the log-rate for this K, with
        mu_j = exp(f_j) * w_j the fitted rates and W their diagonal matrix.
        The first term is the Poisson log-likelihood of the counts at the
        mode including the log-factorial of every count, so that the value
        is a proper log-probability.

    Raises
    ------
    ValueError
        If counts is not a finite one-dimensional array of non-negative
        integer values, if widths is not a finite one-dimensional array of
        the same length with every entry above zero, or if K is not a
        finite (N, N) matrix that is symmetric to within 1e-9 and positive
        definite.
    """
    return log_marginal
```

### Step 5

05_select_length_scale

Goal
----
Select the length scale of the kernel from a grid of candidates by maximising the Laplace approximation to the log marginal likelihood of the Monte Carlo counts.

```python
def select_length_scale(counts: "np.ndarray", widths: "np.ndarray", x: "np.ndarray", lo: float, hi: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray") -> float:
    r"""Return the length scale of the grid with the largest Laplace log marginal likelihood.

    Parameters
    ----------
    counts : np.ndarray
        Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    x : np.ndarray
        Bin centres of shape (N,), finite, within [lo, hi].
    lo : float
        Lower end of the template range, finite.
    hi : float
        Upper end of the template range, finite and above lo.
    n_basis : int
        Number of cubic B-spline basis functions of the mean function, at
        least four.
    sigma : float
        Amplitude of the Matérn 5/2 kernel, finite and above zero.
    prior_scale : float
        Standard deviation of the prior on the mean-function coefficients,
        finite and above zero.
    ell_grid : np.ndarray
        Candidate length scales of shape (M,), M at least one, finite and
        above zero.

    Returns
    -------
    ell : float
        The candidate, as a native Python float, whose effective prior
        covariance (Matérn 5/2 kernel with the given amplitude and that
        length scale plus the integrated mean function) gives the largest
        Laplace approximation to the log marginal likelihood of the
        counts. On an exact tie the candidate listed first in ell_grid is
        returned.

    Raises
    ------
    ValueError
        If counts, widths or x violate their stated requirements, if lo
        or hi is not finite or hi is not above lo, if n_basis is below
        four, if sigma or prior_scale is not finite or not above zero, or
        if ell_grid is empty or contains a value that is not finite or
        not above zero.
    """
    return ell
```

### Step 6

06_combined_template_covariance

Goal
----
Build the combined covariance of the log-rate of a template, the Laplace posterior covariance of the nominal Monte Carlo sample plus one rank-1 contribution for every systematic shape variation.

```python
def combined_template_covariance(counts: "np.ndarray", variations: list, widths: "np.ndarray", K: "np.ndarray") -> "np.ndarray":
    r"""Return the combined statistical and systematic covariance of the log-rate.

    Parameters
    ----------
    counts : np.ndarray
        Nominal Monte Carlo counts of shape (N,), N at least one, finite,
        non-negative integers.
    variations : list
        Sequence of pairs (counts_plus, counts_minus), possibly empty, one
        pair per systematic source, each entry a count array with the
        same requirements and length as counts.
    widths : np.ndarray
        Bin widths of shape (N,), finite and above zero.
    K : np.ndarray
        Prior covariance of the log-rate, shape (N, N), finite, symmetric
        to within an absolute tolerance of 1e-9 and positive definite,
        used for the nominal and for every variation template.

    Returns
    -------
    Sigma_comb : np.ndarray
        Array of shape (N, N): the Laplace posterior covariance of the
        log-rate at the posterior mode of the nominal counts, plus for
        every systematic source the outer product delta delta^T with
        delta = (f_plus - f_minus) / 2, where f_plus and f_minus are the
        posterior modes of the log-rate for the two variation templates
        under the same prior covariance K. The result is symmetric and
        positive definite.

    Raises
    ------
    ValueError
        If counts, widths or K violate their stated requirements, if
        variations is not a sequence of pairs, or if any variation array
        is not a finite one-dimensional array of non-negative integers of
        the same length as counts.
    """
    return Sigma_comb
```

### Step 7

07_truncated_eigenvalues

Goal
----
Decompose a template covariance into eigenmodes and keep the leading modes that capture a target fraction of the total variance.

```python
def truncated_eigenvalues(Sigma: "np.ndarray", fraction: float) -> "np.ndarray":
    r"""Return the leading eigenvalues that together reach the target variance fraction.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance of shape (N, N), N at least one, finite, symmetric to
        within an absolute tolerance of 1e-9, with a trace above zero and
        no eigenvalue below -1e-9 times the trace.
    fraction : float
        Target fraction of the total variance, above zero and at most
        one.

    Returns
    -------
    eigenvalues : np.ndarray
        Array of shape (k,) holding the eigenvalues of Sigma in decreasing
        order, truncated to the smallest k for which their sum is at least
        fraction times the trace of Sigma (the sum of all eigenvalues).
        Eigenvalues are those of the symmetrised matrix (Sigma + Sigma^T)
        / 2; the comparison uses the cumulative sum divided by the trace,
        so that a fraction of exactly one returns every eigenvalue.

    Raises
    ------
    ValueError
        If Sigma is not a finite square matrix that is symmetric to within
        1e-9, if its trace is not above zero or an eigenvalue lies below
        -1e-9 times the trace, or if fraction is not finite, not above
        zero or above one.
    """
    return eigenvalues
```

### Step 8

08_eigenmode_template_fit

Goal
----
Fit the signal strength of a single-channel template likelihood in which the background template is deformed along the leading eigenmodes of its covariance, with one unit-Gaussian constrained amplitude per retained mode.

```python
def eigenmode_template_fit(observed: "np.ndarray", signal: "np.ndarray", template: "np.ndarray", Sigma: "np.ndarray", fraction: float) -> tuple:
    r"""Return the fitted signal strength and its parabolic uncertainty.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts of shape (N,), N at least one, finite,
        non-negative integers.
    signal : np.ndarray
        Expected signal counts at unit signal strength, shape (N,), finite,
        non-negative, with at least one entry above zero.
    template : np.ndarray
        Smooth background template, shape (N,), finite and above zero.
    Sigma : np.ndarray
        Covariance of the background log-rate, shape (N, N), with the
        requirements of the eigenmode truncation (finite, symmetric to
        within 1e-9, trace above zero, no eigenvalue below -1e-9 times the
        trace).
    fraction : float
        Fraction of the total variance retained in the eigenmodes, above
        zero and at most one.

    Returns
    -------
    result : tuple
        Two native Python floats (mu_hat, sigma_mu). The k leading
        eigenpairs (lambda_i, v_i) of Sigma are retained, k being the
        smallest number of modes whose eigenvalues sum to at least
        fraction times the trace, and the expected count of bin j is
        nu_j = mu * s_j + b_j * exp(sum_i sqrt(lambda_i) z_i v_ij). The
        negative log-likelihood sum_j (nu_j - n_j log nu_j) + sum_i z_i^2
        / 2 is minimised jointly over the signal strength mu, which may
        take any real value that keeps every nu_j above zero, and the k
        amplitudes z_i. mu_hat is the minimising signal strength,
        converged to within 1e-9, and sigma_mu is the square root of the
        (mu, mu) entry of the inverse Hessian of the negative
        log-likelihood at the minimum, with the Hessian taken over
        (mu, z_1, ..., z_k). Eigenvector signs do not affect the result.

    Raises
    ------
    ValueError
        If observed is not a finite one-dimensional array of non-negative
        integers, if signal or template is not a finite one-dimensional
        array of the same length satisfying the stated positivity
        requirements, if Sigma is not a valid covariance of matching size,
        or if fraction is not finite, not above zero or above one.
    """
    return result
```

### Step 9

09_barlow_beeston_fit

Goal
----
Fit the signal strength of a single-channel template likelihood with the histogram background template, the Barlow–Beeston treatment of its Monte Carlo statistical uncertainty and the interpolation of its systematic shape variations by the standard framework's code 4.

```python
def barlow_beeston_fit(observed: "np.ndarray", signal: "np.ndarray", counts: "np.ndarray", variations: list, tau: float) -> tuple:
    r"""Return the fitted signal strength and its parabolic uncertainty with per-bin gamma factors and interpolated shape systematics.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts of shape (N,), N at least one, finite,
        non-negative integers.
    signal : np.ndarray
        Expected signal counts at unit signal strength, shape (N,), finite,
        non-negative, with at least one entry above zero.
    counts : np.ndarray
        Raw Monte Carlo background counts of shape (N,), finite, integers
        above zero.
    variations : list
        Sequence of pairs (counts_plus, counts_minus), possibly empty, one
        pair per systematic source, each a finite array of integers above
        zero of length N.
    tau : float
        Ratio of the Monte Carlo to the data luminosity, finite and above
        zero.

    Returns
    -------
    result : tuple
        Two native Python floats (mu_hat, sigma_mu). With h_j = a_j / tau
        the histogram template and h_j^{k,+}, h_j^{k,-} the variation
        templates of source k scaled in the same way, the expected count
        of bin j is nu_j = mu * s_j + gamma_j * h_j * prod_k F_jk(alpha_k).
        With r_+ = h_j^{k,+} / h_j and r_- = h_j^{k,-} / h_j, the
        interpolation factor is F_jk = r_+ ** alpha_k for alpha_k >= 1,
        r_- ** (-alpha_k) for alpha_k <= -1, and for |alpha_k| < 1 the
        polynomial 1 + sum_{i=1}^{6} c_i alpha_k ** i whose six
        coefficients are fixed by matching the value, the first and the
        second derivative of the two exponential branches at alpha_k = 1
        and alpha_k = -1. The negative log-likelihood sum_j (nu_j
        - n_j log nu_j) + sum_j a_j (gamma_j - 1)^2 / 2 + sum_k alpha_k^2
        / 2 is minimised jointly over the signal strength mu, which may
        take any real value that keeps every nu_j above zero, the N
        factors gamma_j and the source parameters alpha_k. mu_hat is the
        minimising signal strength, converged to within 1e-9, and sigma_mu
        is the square root of the (mu, mu) entry of the inverse Hessian of
        the negative log-likelihood at the minimum, with the Hessian taken
        over (mu, gamma_1, ..., gamma_N, alpha_1, ..., alpha_K).

    Raises
    ------
    ValueError
        If observed is not a finite one-dimensional array of non-negative
        integers, if signal is not a finite non-negative array of the same
        length with an entry above zero, if counts or any variation array
        is not a finite array of the same length of integers above zero,
        if variations is not a sequence of pairs, or if tau is not finite
        or not above zero.
    """
    return result
```

### Step 10

10_orchestrate_template_comparison

Goal
----
Run the complete pipeline: select the kernel length scale from the nominal Monte Carlo template, build the smooth template and its combined statistical and systematic covariance, fit the signal strength with the truncated eigenmodes, fit it again with the histogram template that uses per-bin Barlow–Beeston factors and interpolated shape systematics, and report the residual between the two fitted signal strengths.

```python
def orchestrate_template_comparison(edges: "np.ndarray", counts: "np.ndarray", variations: list, signal: "np.ndarray", observed: "np.ndarray", tau: float, n_basis: int, sigma: float, prior_scale: float, ell_grid: "np.ndarray", fraction: float) -> tuple:
    r"""Return the Gaussian process eigenmode fit, the histogram fit, their residual and the template diagnostics.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges of shape (N + 1,), N at least one, finite and strictly
        increasing.
    counts : np.ndarray
        Nominal Monte Carlo background counts of shape (N,), finite,
        integers above zero.
    variations : list
        Sequence of pairs (counts_plus, counts_minus), possibly empty, one
        pair per systematic source, each a finite array of integers above
        zero of length N.
    signal : np.ndarray
        Expected signal counts at unit signal strength, shape (N,), finite,
        non-negative, with at least one entry above zero.
    observed : np.ndarray
        Observed counts of shape (N,), finite, non-negative integers.
    tau : float
        Ratio of the Monte Carlo to the data luminosity, finite and above
        zero.
    n_basis : int
        Number of cubic B-spline basis functions of the mean function, at
        least four.
    sigma : float
        Amplitude of the Matérn 5/2 kernel, finite and above zero.
    prior_scale : float
        Standard deviation of the prior on the mean-function coefficients,
        finite and above zero.
    ell_grid : np.ndarray
        Candidate length scales, finite and above zero, at least one.
    fraction : float
        Fraction of the total variance retained in the eigenmodes, above
        zero and at most one.

    Returns
    -------
    result : tuple
        Eight native Python floats (mu_gp, sigma_gp, n_modes, mu_bb,
        sigma_bb, ell, ratio, residual). The bin centres are the midpoints of the
        edges and the widths their differences. ell is the length scale
        of ell_grid selected on the nominal counts by the Laplace log
        marginal likelihood with the given amplitude, mean-function basis
        on the range [edges[0], edges[-1]] and prior scale. With the
        effective prior covariance at that length scale, the smooth
        template of bin j is exp(f_j) w_j / tau with f the posterior mode
        of the nominal counts, and the combined covariance adds one
        rank-1 term per systematic source to the Laplace posterior
        covariance. mu_gp and sigma_gp are the fitted signal strength and
        its parabolic uncertainty from the eigenmode template fit at the
        given fraction, n_modes the number of retained eigenmodes as a
        float, mu_bb and sigma_bb the corresponding results of the
        Barlow–Beeston histogram fit with the same signal, observed counts,
        nominal counts, variation pairs and tau, ratio the mean over bins of the
        posterior standard deviation of the log-rate (square root of the
        diagonal of the Laplace posterior covariance of the nominal
        template) divided by the histogram relative uncertainty
        1 / sqrt(a_j), and residual the difference mu_gp - mu_bb between
        the two fitted signal strengths.

    Raises
    ------
    ValueError
        If edges is not a finite strictly increasing one-dimensional array
        with at least two entries, or if any other argument violates the
        requirements stated for the steps that consume it.
    """
    return result
```
