# Short-maturity Bachelier matrix Gamma under lognormal volatility

## Background

In a normal-return model with stochastic volatility, the option value can be expressed by conditioning on the volatility-driving Brownian motion. The remaining independent Brownian component then produces a conditional normal call. Realized variance and the correlated stochastic integral determine its random variance and shift.

A finite matrix expansion organizes the correction in correlation and moneyness powers. The same coefficient matrix supplies option values and the first two spot derivatives across a strike grid. Its model-specific inputs are mixed moments derived from the joint law of terminal lognormal volatility and its integrated square. At short maturity, the real integral representation suffers severe cancellation. Stable deterministic moment evaluation is necessary before the finite matrix is assembled. The task asks for this specified finite expansion.

## Problem

An undiscounted asset follows $dX_t=\sigma_t(\rho\,dW_t+\sqrt{1-\rho^2}\,dB_t)$, where $W$ and $B$ are independent standard Brownian motions and $\sigma_t=\sigma_0\exp(\nu W_t-\nu^2t/2)$; use $X_0=100$, $\sigma_0=35$, $\nu=0.7$, $\rho=-0.6$, $T=0.125$, and strikes $95,97.5,100,102.5,105$.
Compute the call values and their first two derivatives with respect to $X_0$ using the recent strike-independent coefficient-matrix method for this stochastic-volatility Bachelier model, truncated at correlation order $M=4$ and inner order $N=12$ (five rows and 26 columns).
The required scalar is the Gamma at strike $95$ from that finite expansion, rather than from a converged model pricer.
Evaluate the model expectations from the Brownian exponential-functional joint law deterministically to relative accuracy $10^{-10}$; this short maturity requires a stable treatment of the strongly cancelling oscillatory representation, and neither simulation nor a small-time asymptotic truncation defines the requested result.
Use signed correlated moments, with $\xi_T=(\sigma_T-\sigma_0)/\nu$, $v_T^2=T^{-1}\int_0^T\sigma_s^2ds$, and $\tau=\nu^2T$.
For numerical checkpoints define $R_{\beta,\alpha}(\tau)=\tau^\alpha\mathbb E[(e^{W_\tau-\tau/2}-1)^\beta(A_\tau)^{-\alpha}]$, where $A_\tau=\int_0^\tau e^{2W_s-s}ds$ for a standard Brownian motion; report $R_{0,-1/2}$ and $R_{1,1/2}$.
In short reasoning identify the matrix construction and stable expectation evaluation, and report $\tau$, RMS realized volatility, the inner-series moneyness radius, mean realized volatility, $A_{0,0}$, $A_{1,1}$, the constant-volatility reference Gamma, and the corrected price at strike $95$.
The full strike grid is computed internally; only these checkpoints and the requested final scalar need be reported, with at least eight significant figures for the final scalar.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_scale_constants

Goal
----
Scale constants of the lognormal-volatility normal-return model.

```python
import numpy as np

def scale_constants(sigma0: float, nu: float, rho: float, T: float) -> tuple:
    """Return the three scale constants of the model.

    The instantaneous volatility is ``sigma_t = sigma0 * exp(nu * W_t - nu**2 * t / 2)``
    for a standard Brownian motion ``W``, and the return process is driven by
    ``rho * dW_t + sqrt(1 - rho**2) * dB_t`` with ``B`` independent of ``W``.

    Compute, in this order:

    * ``tau = nu**2 * T``;
    * ``v = sigma0 * sqrt((exp(tau) - 1) / tau)``, the root-mean-square level
      ``sqrt(E[(1 / T) * int_0^T sigma_s**2 ds])``;
    * ``radius = sigma0 * sqrt(1 - rho**2) / nu``.

    Parameters
    ----------
    sigma0 : float
        Initial volatility level, strictly positive and finite.
    nu : float
        Volatility-of-volatility, strictly positive and finite.
    rho : float
        Correlation coefficient with ``-1 < rho < 1``.
    T : float
        Maturity in years, strictly positive and finite.

    Returns
    -------
    tuple
        ``(tau, v, radius)``, three finite floats.

    Raises
    ------
    ValueError
        If ``sigma0``, ``nu`` or ``T`` is not strictly positive and finite, or
        if ``rho`` is not a finite number with ``-1 < rho < 1``.
    """
    return (0.0, 0.0, 0.0)
```

### Step 2

02_normal_call_terms

Goal
----
Closed-form normal-model call value and its two underlying-price derivatives.

```python
import numpy as np

def normal_call_terms(T: float, X0: float, strikes: np.ndarray, sigma: float) -> tuple:
    """Return the constant-volatility normal call value and its first two
    derivatives with respect to the underlying level.

    With ``s = sigma * sqrt(T)`` and ``d_i = (X0 - k_i) / s``, let ``N`` be the
    standard normal cumulative distribution function and
    ``n(z) = exp(-z**2 / 2) / sqrt(2 * pi)`` its density. Return

    * ``price[i] = (X0 - k_i) * N(d_i) + s * n(d_i)``,
    * ``delta[i] = N(d_i)``,
    * ``gamma[i] = n(d_i) / s``.

    Parameters
    ----------
    T : float
        Maturity, strictly positive and finite.
    X0 : float
        Current underlying level, finite.
    strikes : numpy.ndarray
        One-dimensional, non-empty vector of finite strikes, shape ``(r,)``.
    sigma : float
        Constant absolute volatility, strictly positive and finite.

    Returns
    -------
    tuple
        ``(price, delta, gamma)``, three float64 arrays of shape ``(r,)``.

    Raises
    ------
    ValueError
        If ``T`` or ``sigma`` is not strictly positive and finite, if ``X0`` is
        not finite, or if ``strikes`` is not a one-dimensional non-empty array
        of finite values.
    """
    return (np.zeros(1), np.zeros(1), np.zeros(1))
```

### Step 3

03_normalized_moment

Goal
----
Signed normalized Brownian exponential-functional moments at short maturity.

```python
import numpy as np

def normalized_moment(beta: int, alpha: float, tau: float) -> float:
    """Compute R = tau**alpha E[(exp(W_tau-tau/2)-1)**beta A_tau**(-alpha)],
    where W is standard Brownian motion and
    A_tau = integral_0^tau exp(2*W_s-s) ds.

    A deterministic integral representation is R = tau**alpha Gamma(alpha+1)
    exp(-tau/8+pi**2/(2*tau))/sqrt(2*pi**3*tau) times

    integral over x in R, eta in (0,infinity) of
    (exp(x)-1)**beta exp(-(alpha+1/2)*x-eta**2/(2*tau))
    sinh(eta) sin(pi*eta/tau) / (cosh(x)+cosh(eta))**(alpha+1).

    For beta=0, the power is one. For odd beta, preserve its sign. This is
    the integral over the infinite domain, not a specified finite grid sum.
    At these small tau values the oscillatory real integral has severe
    cancellation. Use a deterministic, numerically stable evaluation;
    mathematically equivalent integral transformations are allowed. If a
    complex contour is used, continue the denominator power analytically
    from the real contour without crossing a zero or branch cut. Simulation
    and a small-tau truncation are not the target. Aim for relative error
    at most 1e-10, with absolute error 1e-12 allowed for near-zero values:
    abs(computed-R) <= 1e-12 + 1e-10*abs(R).
    Cost contract: one call must return in well under a second on a single
    core, and later steps of this task evaluate R at more than a hundred
    parameter combinations inside one shared process wall-clock limit.
    Reaching the stated accuracy by raising the working precision instead
    of by choosing a better representation does not meet that contract.

    Parameters
    ----------
    beta : int
        Integer from 0 through 4, excluding booleans.
    alpha : float
        Finite, -0.5 <= alpha <= 13.5 and 2*alpha+1.5 > beta.
        The last inequality makes the displayed double integral absolutely
        integrable before the oscillatory cancellations are taken.
    tau : float
        Finite dimensionless maturity in [0.04, 0.20].

    Returns
    -------
    float
        The signed normalized moment R.

    Raises
    ------
    ValueError
        If beta, alpha or tau is outside its stated domain.
    """
    return 0.0
```

### Step 4

04_expectation_vector

Goal
----
The physical moment family consumed by the truncated coefficient matrix.

```python
import numpy as np

def expectation_vector(M_max: int, N_max: int, sigma0: float, nu: float, T: float) -> dict:
    """Collect physical moments E(beta,alpha)=E[xi_T**beta*v_T**(-2*alpha)],
    with xi_T=(sigma_T-sigma0)/nu, v_T**2=(1/T)integral_0^T sigma_s**2 ds,
    sigma_t=sigma0*exp(nu*W_t-nu**2*t/2), tau=nu**2*T.
    In terms of normalized_moment, E(beta,alpha) equals
    sigma0**(beta-2*alpha)*nu**(-beta)*normalized_moment(beta,alpha,tau).
    For n=0,...,N_max return the following named families:
    base_even[n]=E(0,n-.5); base_odd[n]=E(1,n+.5);
    high_even[p][n]=E(2*p+2,n+p+.5), p=0,...,M_max//2-1;
    high_odd[p][n]=E(2*p+3,n+p+1.5), p=0,...,(M_max-1)//2-1.
    This step reuses normalized_moment and never depends on the strike grid.

    Parameters
    ----------
    M_max : int
        Integer from 1 through 4, excluding booleans.
    N_max : int
        Integer from 0 through 12, excluding booleans.
    sigma0, nu, T : float
        Strictly positive finite model constants; .04 <= nu**2*T <= .20.

    Returns
    -------
    dict
        base_even and base_odd are float arrays of shape (N_max+1,).
        high_even is a list of M_max//2 arrays of that shape; high_odd is
        a list of (M_max-1)//2 arrays of that shape. Keys are exactly
        'base_even', 'base_odd', 'high_even', 'high_odd'.

    Raises
    ------
    ValueError
        If an order or model parameter is outside the stated domain.
    """
    return {}
```

### Step 5

05_coefficient_matrix

Goal
----
Assemble the strike-independent coefficient matrix of the expansion.

```python
import numpy as np

def coefficient_matrix(exps: dict, M_max: int, N_max: int, rho: float,
                       T: float, v: float) -> np.ndarray:
    """Return the coefficient matrix whose row ``m`` multiplies ``rho**m`` and
    whose column ``j`` multiplies the ``j``-th power of the moneyness.

    Let ``q = 1 - rho**2``, ``g[j] = log(j!)``, ``n = 0, 1, ..., N_max``, and let
    ``base_even``, ``base_odd``, ``high_even`` and ``high_odd`` be the entries of
    ``exps``. Build an all-zero array of shape ``(M_max + 1, 2 * N_max + 2)`` and
    fill only these entries. Algebraically equivalent stable evaluations of
    the stated factorial ratios and powers are allowed:

    ``A[0, 2 * n] = -sqrt(T) / sqrt(2 * pi) * (-1)**n
      * (base_even[n] - v**(1 - 2 * n))
      / ((2 * n - 1) * exp(g[n] + n * log(2 * T) + (n - 0.5) * log(q)))``

    ``A[1, 2 * n + 1] = 1 / (2 * sqrt(2 * pi)) * (-1)**n * base_odd[n]
      * (4 / (4 * n + 2))
      / exp(g[n] + n * log(2) + (n + 0.5) * log(T) + (n + 0.5) * log(q))``

    ``A[2 * p + 2, 2 * n] = 1 / sqrt(2 * pi) * (-1)**(n + p) * high_even[p][n]
      * exp(g[2 * n + 2 * p] - g[2 * n] - g[n + p] - (n + p) * log(2)
            - (n + p + 0.5) * log(q * T)) / (2 * p + 2)!``

    ``A[2 * p + 3, 2 * n + 1] = -1 / sqrt(2 * pi) * (-1)**(n + p)
      * high_odd[p][n]
      * exp(g[2 * n + 2 * p + 2] - g[2 * n + 1] - g[n + p + 1]
            - (n + p + 1) * log(2) - (n + p + 1.5) * log(q * T)) / (2 * p + 3)!``

    Every other entry stays zero.

    Parameters
    ----------
    exps : dict
        Mixed-moment family with keys ``"base_even"``, ``"base_odd"``,
        ``"high_even"`` and ``"high_odd"``. The two arrays have length
        ``N_max + 1``; ``"high_even"`` holds ``M_max // 2`` such arrays and
        ``"high_odd"`` holds ``(M_max - 1) // 2``.
    M_max : int
        Outer truncation order, an integer ``>= 1``.
    N_max : int
        Inner truncation order, an integer ``>= 0``.
    rho : float
        Correlation with ``-1 < rho < 1``.
    T : float
        Maturity, strictly positive and finite.
    v : float
        Root-mean-square volatility level, strictly positive and finite.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(M_max + 1, 2 * N_max + 2)``.

    Raises
    ------
    ValueError
        If ``M_max`` is not an integer ``>= 1``, if ``N_max`` is not an integer
        ``>= 0``, if ``exps`` lacks a required key or carries arrays of the
        wrong length or lists of the wrong count, if ``rho`` is not finite with
        ``-1 < rho < 1``, or if ``T`` or ``v`` is not strictly positive and
        finite.
    """
    return np.zeros((1, 1))
```

### Step 6

06_matrix_values

Goal
----
One coefficient polynomial gives prices and both spot derivatives.

```python
import numpy as np

def matrix_values(A: np.ndarray, X0: float, strikes: np.ndarray, rho: float, base: tuple) -> tuple:
    """For d=X0-k define P(d)=sum_m sum_j rho**m*A[m,j]*d**j.
    Return base_price+P(d), base_delta+P'(d), base_gamma+P''(d) on all
    strikes. The coefficient array is common to all strikes and derivatives;
    differentiate its monomial polynomial analytically, not by finite
    differences of prices. Equivalent Horner or shift-operator evaluation
    is allowed. Input coefficient conventions are those of coefficient_matrix.

    Parameters
    ----------
    A : np.ndarray
        Finite real 2D array, at least one row and one column.
    X0 : float
        Finite initial asset level.
    strikes : np.ndarray
        Nonempty finite real 1D strike array; preserve order and duplicates.
    rho : float
        Finite correlation with abs(rho)<1.
    base : tuple
        Three finite real 1D arrays (price, delta, gamma), each the same
        shape as strikes, such as returned by normal_call_terms.

    Returns
    -------
    tuple
        Three float arrays (prices, deltas, gammas), each shaped like strikes.

    Raises
    ------
    ValueError
        If any input is nonfinite, A or strikes has the wrong dimensionality
        or is empty, abs(rho)>=1, or base does not contain three arrays with
        the strike shape.
    """
    return ()
```

### Step 7

07_matrix_expansion_gamma

Goal
----
Short-maturity matrix valuation: final orchestrator.

```python
import numpy as np

def matrix_expansion_gamma(X0: float=100., sigma0: float=35., nu: float=.7, rho: float=-.6,
                           T: float=.125, M_max: int=4, N_max: int=12,
                           strikes: tuple=(95.,97.5,100.,102.5,105.), report_strike: float=95.) -> float:
    """Compute the selected Gamma from the finite coefficient-matrix method.
    Reuse scale_constants, normal_call_terms, normalized_moment,
    expectation_vector, coefficient_matrix and matrix_values. The base
    normal-call volatility is sqrt(1-rho**2)*v, with v the RMS level from
    scale_constants. Evaluate the signed physical moment family, assemble
    the common coefficient matrix and compute all prices and both spot
    derivatives; return the Gamma at report_strike. This is the finite
    M_max,N_max expansion, not the converged stochastic-volatility value.

    Parameters
    ----------
    X0, report_strike : float
        Finite asset level and a strike occurring exactly once in strikes.
    sigma0, nu, T : float
        Positive finite parameters with .04<=nu**2*T<=.20.
    rho : float
        Finite and abs(rho)<1/sqrt(2).
    M_max, N_max : int
        Integers excluding booleans, 1<=M_max<=4 and 0<=N_max<=12.
    strikes : tuple
        Nonempty finite distinct strikes, in arbitrary order. All satisfy
        abs(X0-k)<sigma0*sqrt(1-rho**2)/nu.

    Returns
    -------
    float
        Gamma at report_strike, in the input asset units.

    Raises
    ------
    ValueError
        If any parameter violates its stated domain, report_strike is not
        present exactly once.
    """
    return 0.0
```
