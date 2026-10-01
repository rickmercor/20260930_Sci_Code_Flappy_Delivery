# Mathematics-Computational_Finance-29

## Background

A second-order short-tenor expansion of the characteristic function of a stochastic-volatility model whose spot volatility carries a piecewise-constant deterministic displacement, calibrated to a daily at-the-money term structure and priced by Fourier inversion into a five-day put-minus-call implied-volatility difference.

## Problem

A five-day implied-volatility smile is to be computed from a second-order short-tenor expansion of a stochastic-volatility model whose spot volatility carries a piecewise-constant deterministic displacement across daily tenors, and the put-minus-call implied-volatility difference at the last tenor is to be reported.

Rates and dividends are zero, the spot is 100, the tenor grid is tau_k = k/365 for k = 1, ..., 5 with tau_0 = 0, and the at-the-money implied volatilities at those tenors are 0.19, 0.215, 0.205, 0.23 and 0.22. The displacement is constant on each interval [tau_{k-1}, tau_k) at a level a_{k-1} with a_0 = 0, and the spot volatility together with the remaining four levels are the ones the displaced Black-Scholes recursion of Remark 1 of the source implies from that term structure. The time-zero model coefficients are a spot volatility-of-volatility of 0.6, a spot leverage of -0.65, eta_0 = 0.4 (the loading of the volatility-of-volatility on the price Brownian motion), alpha_0 = 0.10 (the drift coefficient of the spot volatility) and delta_0 = 0.0741 (the loading of the log-price drift on the price Brownian motion), with the price-side and orthogonal volatility-of-volatility coefficients following from the first two as in equation (2) of the source.

The source is a recent treatment of jointly pricing implied-volatility smiles over tenors of a few days, in which a deterministic displacement of the spot volatility carries the at-the-money term structure across tenors and the continuous component is priced through an expansion of its characteristic function. Use its Theorem 1, the expansion to second order in the square root of the tenor of the characteristic function of the standardised and demeaned continuous log-return, at the fifth tenor, with the nested time integrals of the displacement profile that appear in the theorem evaluated exactly for the piecewise-constant displacement of equation (4); the single-interval closed forms of Corollary 2 are not equal to those integrals when the displacement changes level and must not be used, and no jump component is included.

Price European calls with the Fourier representation of equation (9) of the source, computing each of its two integrals with the midpoint rule on (0, 20] with 4000 equal cells, and convert prices to Black-Scholes implied volatilities by bisection on [1e-4, 5] with 200 halvings. Strikes are set by log-moneyness in the convention of equation (12) of the source with the fifth tenor's at-the-money implied volatility: m = -2 on the put side and m = +2 on the call side.

Report the implied volatility at m = -2 minus the implied volatility at m = +2, as an annualised decimal.

State the conventions you adopted and justify each from the source.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

displacement_levels

Goal
----
Given the tenor grid tau_0 = 0 < tau_1 < ... < tau_n and the at-the-money implied volatilities sigma_BS(tau_1), ..., sigma_BS(tau_n), return the displacement levels a_0, a_1, ..., a_{n-1} of the piecewise-constant displacement of equation (4) of the source, with a_0 = 0 and the remaining levels fixed by the recursion of Remark 1 (the displaced Black-Scholes calibration to the at-the-money term structure). The spot volatility of that recursion is the first at-the-money volatility.

```python
import numpy as np


def displacement_levels(tenors, atm_vols):
    """Return the piecewise-constant displacement levels implied by the ATM term structure."""
    return np.zeros(len(atm_vols))
```

### Step 2

displacement_functionals

Goal
----
Let phi_tilde(s) = 1 + a_k / sigma_0 on each tenor interval [tau_k, tau_{k+1}) be the normalised displacement profile of Theorem 1 of the source, supplied here as the array levels (one level per interval). For a tenor tau that is one of the grid points, return the eight distinct time-integral functionals of phi_tilde over [0, tau] that appear in the statement of Theorem 1, in the order in which they first appear there, each evaluated exactly (the profile is piecewise constant, so every integrand is a piecewise polynomial). Do not use the single-interval closed forms of Corollary 2 of the source: for a profile that changes level they are not equal to the integrals of Theorem 1.

```python
import numpy as np


def displacement_functionals(tenors, levels, tau):
    """Return the eight distinct time-integral functionals of the displacement profile."""
    return np.zeros(8)
```

### Step 3

expansion_blocks

Goal
----
At Fourier frequency u and tenor tau, return the six correction blocks that appear inside the large parentheses of the expansion of Theorem 1 of the source, in their order of appearance, each as a complex number: the block driven by the price-side volatility-of-volatility, the block driven by delta_0, the block driven by alpha_0, the block driven by eta_0, the block quadratic in the price-side volatility-of-volatility, and the block quadratic in the orthogonal volatility-of-volatility. The price-side and orthogonal volatility-of-volatility coefficients follow from the spot volatility-of-volatility vov and the spot leverage rho as in equation (2) of the source. The displacement enters only through functionals, the array returned by displacement_functionals. u is a complex scalar or a one-dimensional array of frequencies; in the array case the blocks are evaluated elementwise.

```python
import numpy as np


def expansion_blocks(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, functionals):
    """Return the six correction blocks of the source's Theorem 1 at the frequency or frequencies u."""
    return np.zeros(6, dtype=complex)
```

### Step 4

continuous_cf_expansion

Goal
----
Assemble the second-order expansion of Theorem 1 of the source for the characteristic function of the standardised, demeaned continuous log-return at tenor tau (a tenor grid point) and complex frequency u: normalise the displacement levels shifts into the profile of Theorem 1, evaluate the displacement functionals and the six correction blocks, and combine them with the Gaussian leading factor exactly as the theorem states. u is a complex scalar or a one-dimensional array of frequencies, evaluated elementwise.

```python
import numpy as np


def continuous_cf_expansion(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, tenors, shifts):
    """Return the Theorem 1 expansion of the standardised continuous log-return's characteristic function at u."""
    return 0j
```

### Step 5

fourier_call_price

Goal
----
Price a European call with spot s0, strike strike and tenor tau (a tenor grid point) by the Fourier representation of equation (9) of the source, with zero interest rate and no jump component, using continuous_cf_expansion as the characteristic function. model is a dict with keys 'sigma0', 'vov', 'rho', 'eta0', 'alpha0', 'delta0', 'tenors' and 'shifts'. Evaluate each of the two integrals over the positive frequency axis with the midpoint rule on (0, u_max] with n_nodes equal cells.

```python
import numpy as np


def fourier_call_price(model, s0, strike, tau, n_nodes, u_max):
    """Return the call price of equation (9) of the source under the expanded characteristic function."""
    return 0.0
```

### Step 6

implied_volatility

Goal
----
Invert the zero-rate Black-Scholes call formula for the volatility that reproduces price at spot s0, strike strike and tenor tau, by bisection on the interval [1e-4, 5] with exactly 200 halvings, returning the midpoint of the final bracket. Reject prices outside the no-arbitrage bounds.

```python
import numpy as np


def implied_volatility(price, s0, strike, tau):
    """Return the Black-Scholes implied volatility of a call price at zero rate."""
    return 0.0
```

### Step 7

risk_reversal

Goal
----
Integrate the pipeline and report one number. config holds 'tenors' (tau_0 = 0 first), 'atm_vols', the model coefficients 'vov', 'rho', 'eta0', 'alpha0', 'delta0', the spot 's0', the two log-moneyness levels 'm_put' and 'm_call', and the quadrature settings 'n_nodes' and 'u_max'. Calibrate the displacement levels from the at-the-money term structure with displacement_levels, take the spot volatility as the first at-the-money volatility, check the assembled expansion at zero frequency, set the two strikes at the last tenor by the log-moneyness convention of equation (12) of the source with that tenor's at-the-money volatility, price both calls with fourier_call_price, invert them with implied_volatility, and return the implied volatility at m_put minus the implied volatility at m_call.

```python
import numpy as np


def risk_reversal(config):
    """Return the put-minus-call implied volatility difference at the last tenor."""
    return 0.0
```
