# Physics-Particle_Physics-11

## Background

Searches for new phenomena in particle physics often reduce to a Poisson counting experiment in which the expected background is itself estimated from a control measurement. The evidence for a signal is quantified by the $p$-value of the background-only hypothesis, quoted as an equivalent number of Gaussian standard deviations, and the sensitivity of a planned experiment by the median of that significance for a nominal signal. Approximations based on the large-sample distribution of likelihood-ratio statistics make such sensitivities available in closed form but degrade when only a handful of events are expected, which is the regime in which searches are designed and selection cuts optimised. Higher-order asymptotic corrections and exact discrete reference calculations are the tools for assessing the sensitivity reliably at low counts.

## Problem

A search for a new process counts $n$ events in a signal region, Poisson distributed with mean $s + b$, where $s$ and $b$ are the expected numbers of signal and background events. The background is not known a priori but is constrained by a control measurement that counts $m$ events, Poisson distributed with mean $\tau b$ for a known scale factor $\tau$, in a control region believed to be free of signal. Discovery is claimed from the $p$-value of the background-only hypothesis $s = 0$, expressed as the Gaussian significance $Z = \Phi^{-1}(1 - p)$, and the sensitivity of a planned experiment is characterised by the median of $Z$ under the assumption that the signal is present with its nominal strength. Closed-form approximations to this median rest on the large-sample distribution of the profile likelihood ratio and lose accuracy at the low event yields typical of the most interesting searches. A higher-order treatment of the likelihood root extends their validity to lower counts, and the question below is how much error remains in the corrected approximation for one concrete low-count configuration. The inputs are the nominal signal strength, the expected background and the relative uncertainty of the background estimate from the control measurement, and the output is a single number in units of standard deviations.

The discovery test uses the profile likelihood ratio $\lambda(0) = L(0, \hat{\hat{b}}_0) / L(\hat{s}, \hat{b})$ of the product of the two Poisson likelihoods, with the background profiled at $s = 0$, and the statistic $q_0 = -2 \ln \lambda(0)$ for $\hat{s} \ge 0$ and $q_0 = 0$ otherwise, so that the first-order significance is $Z = \sqrt{q_0} = \max\{0, r(0)\}$ in terms of the signed likelihood-ratio root $r(0)$. Its median for a given $s$ is approximated by the Asimov data set, in which both counts are replaced by their expectation values. The higher-order treatment replaces $r(0)$ by the modified signed likelihood-ratio root of Barndorff-Nielsen, which adds to $r(0)$ a logarithmic adjustment involving a model-dependent auxiliary statistic $u(0)$ obtained from derivatives of the log-likelihood, and builds the corrected discovery statistic $q_0^*$ from the modified root. The auxiliary statistic of the two-count on/off likelihood, the handling of the sample-space boundaries and the treatment of discreteness in the corrected statistic are the parts of the treatment to be established. The exact reference against which both approximations are judged is the profile construction, in which the background-only $p$-value of an observed pair $(n, m)$ is taken from the distribution of the first-order discovery statistic at $s = 0$ with the background fixed to its value profiled from that pair. The median of the resulting significance is taken over the joint Poisson distribution of the two counts at the nominal signal strength.

Take $s = 5$, $b = 0.8$ and a control measurement whose background estimate has a relative standard deviation $\sigma_b / b = 1$, which fixes the scale factor $\tau$. Evaluate the first-order Asimov significance $Z_A(q_0)$ and the higher-order Asimov significance $Z_A(q_0^*)$ at the real-valued Asimov counts, following the higher-order treatment's own prescriptions for a model with two discrete counts. Then compute the exact median discovery significance $\mathrm{med}[Z \mid s]$ of the profile construction by summing Poisson probabilities rather than by sampling toy experiments. In that construction the $p$-value of a pair $(n, m)$ is the probability, under $N \sim \mathrm{Poisson}(\hat{\hat{b}}_0)$ and $M \sim \mathrm{Poisson}(\tau \hat{\hat{b}}_0)$ with $\hat{\hat{b}}_0$ the background profiled from $(n, m)$ at $s = 0$, of all pairs whose first-order signed root is at least that of $(n, m)$, the observed pair included. Each pair carries $Z = \max\{0, \Phi^{-1}(1 - p)\}$, and a pair with no excess, $n \le m/\tau$, has $Z = 0$. The median is the smallest attainable value $z$ with $P(Z \le z) \ge 1/2$ under $N \sim \mathrm{Poisson}(s + b)$ and $M \sim \mathrm{Poisson}(\tau b)$. Every Poisson sum may be truncated where the omitted upper tail has probability below $10^{-12}$.

Report as the single final scalar the residual $Z_A(q_0^*) - \mathrm{med}[Z \mid s]$ of the higher-order Asimov approximation, positive when it overstates the sensitivity, give the residual of the first-order approximation alongside it for comparison, and explain what the remaining residual reflects in terms of the approximations inherent in the Asimov estimate.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

profile_background_estimate

Goal
----
Compute the profiled background estimate of the on/off counting experiment for a fixed signal strength.

```python
def profile_background_estimate(n: float, m: float, tau: float, s: float) -> float:
    r"""Return the conditional maximum-likelihood background for a fixed signal strength.

    Parameters
    ----------
    n : float
        Count in the signal region, finite and at least zero. Real values are
        allowed for Asimov data.
    m : float
        Count in the control region, finite and at least zero. Real values are
        allowed for Asimov data.
    tau : float
        Scale factor between the control and signal regions, finite and above
        zero, so that the control count has mean tau times the background.
    s : float
        Signal strength at which the background is profiled, finite and at
        least zero.

    Returns
    -------
    b_profiled : float
        The conditional estimate of the background for the given s, the
        non-negative root of the profiling condition, as a native Python
        float, accurate to a relative error of 1e-9 for every valid input,
        including signal strengths far above n + m. For s equal to zero it
        is the pooled estimate (n + m) / (1 + tau).

    Raises
    ------
    ValueError
        If n or m is negative or not finite, if tau is not finite or not above
        zero, or if s is negative or not finite.
    """
    return b_profiled
```

### Step 2

signed_likelihood_root

Goal
----
Evaluate the signed likelihood-ratio root of the background-only test for arrays of on/off counts.

```python
def signed_likelihood_root(n: "np.ndarray", m: "np.ndarray", tau: float) -> "np.ndarray":
    r"""Return the signed root r(0) of the background-only profile likelihood ratio.

    Parameters
    ----------
    n : np.ndarray
        Signal-region counts, an array of finite values of at least zero.
        Real values are allowed for Asimov data.
    m : np.ndarray
        Control-region counts, an array of finite values of at least zero,
        broadcastable against n.
    tau : float
        Scale factor between the control and signal regions, finite and
        above zero.

    Returns
    -------
    r : np.ndarray
        Float array with the broadcast shape of n and m. Entry-wise it is the
        signed likelihood-ratio root of the hypothesis s = 0, positive for an
        excess n > m / tau, negative for a deficit and zero when n equals
        m / tau or both counts are zero. Zero counts contribute nothing to
        the log-likelihood difference, and a round-off negative value of the
        likelihood-ratio statistic is treated as zero. Every entry is accurate
        to a relative error of 1e-9 for counts up to 1e9.

    Raises
    ------
    ValueError
        If tau is not finite or not above zero, or if any count is negative
        or not finite.
    """
    return r
```

### Step 3

auxiliary_statistic

Goal
----
Evaluate the auxiliary statistic of the higher-order likelihood-root correction for the on/off counting experiment.

```python
def auxiliary_statistic(n: float, m: float, tau: float, s: float) -> float:
    r"""Return the auxiliary statistic u(s) of the higher-order correction for the on/off model.

    Parameters
    ----------
    n : float
        Count in the signal region, finite and at least zero. Real values are
        allowed for Asimov data.
    m : float
        Count in the control region, finite and at least zero. Real values are
        allowed for Asimov data.
    tau : float
        Scale factor between the control and signal regions, finite and above
        zero.
    s : float
        Tested signal strength, finite and at least zero. The background is
        profiled at this value.

    Returns
    -------
    u : float
        The auxiliary statistic u(s) as a native Python float. It carries the
        sign of the signed likelihood-ratio root r(s). It is zero when the
        signal region is empty, when the control region is empty at s equal to
        zero or with n above (1 + tau) s, and when the signal region sits
        exactly at its expectation under the tested hypothesis. For an empty
        control region with positive s it is sqrt(n) ln(n / s) when n is
        below (1 + tau) s and sqrt(n / 2) ln(n / s) when n equals (1 + tau) s
        exactly.

    Raises
    ------
    ValueError
        If n or m is negative or not finite, if tau is not finite or not above
        zero, or if s is negative or not finite.
    """
    return u
```

### Step 4

corrected_discovery_significance

Goal
----
Apply the higher-order likelihood-root correction and the discovery convention to obtain the corrected significance.

```python
def corrected_discovery_significance(r: float, u: float) -> float:
    r"""Return the discovery significance max(0, r*) from a signed root and its auxiliary statistic.

    Parameters
    ----------
    r : float
        Signed likelihood-ratio root r(0) of the background-only test, finite.
    u : float
        Auxiliary statistic u(0) of the higher-order correction, finite. It is
        zero at a sample-space boundary and otherwise has the sign of r.

    Returns
    -------
    z : float
        The corrected discovery significance as a native Python float, the
        larger of zero and r*, with r* = r + ln(u / r) / r. When u is zero
        or r is zero the adjustment is undefined and r* equals r.

    Raises
    ------
    ValueError
        If r or u is not finite, or if r and u are both non-zero with
        opposite signs, for which the logarithm is undefined.
    """
    return z
```

### Step 5

first_order_asimov_significance

Goal
----
Compute the first-order Asimov median discovery significance of the on/off experiment from the background uncertainty.

```python
def first_order_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    r"""Return the first-order Asimov median significance in terms of the background uncertainty.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    sigma_b : float
        Standard deviation of the background estimate inferred from the
        control measurement, finite and above zero.

    Returns
    -------
    z_asimov : float
        The first-order Asimov significance Z_A as a native Python float,
        zero for s equal to zero, accurate to a relative error of 1e-9 for
        every valid sigma_b, including sigma_b far below b, where it tends
        to the known-background value. A round-off negative value of the
        bracket under the square root is treated as zero.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or sigma_b is not finite or
        not above zero.
    """
    return z_asimov
```

### Step 6

corrected_asimov_significance

Goal
----
Compute the higher-order corrected Asimov median discovery significance of the on/off experiment.

```python
def corrected_asimov_significance(s: float, b: float, sigma_b: float) -> float:
    r"""Return the Asimov median significance from the higher-order corrected discovery statistic.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    sigma_b : float
        Standard deviation of the background estimate inferred from the
        control measurement, finite and above zero.

    Returns
    -------
    z_asimov_corrected : float
        The corrected Asimov significance, the larger of zero and the
        refined root r*(0) evaluated at the Asimov counts n = s + b and
        m = b^2 / sigma_b^2 with scale factor b / sigma_b^2, as a native
        Python float, accurate to 1e-8. For s equal to zero the corrected
        root is its continuous limit as s tends to zero from above along the
        Asimov path, a function of b and the scale factor alone that is
        positive for a scale factor above one, negative below one and zero
        at one, and the returned value is the larger of zero and that limit.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or sigma_b is not finite or
        not above zero.
    """
    return z_asimov_corrected
```

### Step 7

profile_construction_significance

Goal
----
Evaluate the exact profile-construction discovery significance of an observed on/off outcome by Poisson tail summation.

```python
def profile_construction_significance(n: int, m: int, tau: float) -> float:
    r"""Return the exact profile-construction discovery significance of an observed pair of counts.

    Parameters
    ----------
    n : int
        Observed count in the signal region, an integer of at least zero.
        An integral value carried by a float or a NumPy integer is accepted.
    m : int
        Observed count in the control region, an integer of at least zero.
        An integral value carried by a float or a NumPy integer is accepted.
    tau : float
        Scale factor between the control and signal regions, finite and
        above zero.

    Returns
    -------
    z_reference : float
        The discovery significance as a native Python float, the larger of
        zero and the standard-normal quantile of one minus the background-only
        tail probability of the first-order signed root under the profiled
        null with b fixed to (n + m) / (1 + tau). It is zero when n is at most
        m / tau. Each Poisson sum omits an upper tail of probability below
        1e-12 and extends at least to the observed count.

    Raises
    ------
    ValueError
        If n or m is negative or not an integer value, or if tau is not
        finite or not above zero.
    """
    return z_reference
```

### Step 8

median_discovery_significance

Goal
----
Compute the exact median discovery significance of the on/off experiment for a nominal signal strength.

```python
def median_discovery_significance(s: float, b: float, tau: float) -> float:
    r"""Return the exact median of the profile-construction discovery significance for a nominal signal.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    tau : float
        Scale factor between the control and signal regions, finite and
        above zero.

    Returns
    -------
    z_median : float
        The median discovery significance as a native Python float, the
        smallest attainable significance whose cumulative probability under
        the joint Poisson distribution of the two counts reaches one half.
        Each Poisson sum over the outcomes omits an upper tail of probability
        below 1e-12.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or tau is not finite or not
        above zero.
    """
    return z_median
```

### Step 9

orchestrate_asimov_residual

Goal
----
Orchestrate the sensitivity calculation and report the residual of the corrected Asimov estimate against the exact median significance.

```python
def orchestrate_asimov_residual(s: float, b: float, sigma_b: float) -> tuple:
    r"""Return the residual of the corrected Asimov significance together with its ingredients.

    Parameters
    ----------
    s : float
        Nominal expected number of signal events, finite and at least zero.
    b : float
        Expected number of background events in the signal region, finite
        and above zero.
    sigma_b : float
        Standard deviation of the background estimate inferred from the
        control measurement, finite and above zero. The scale factor of the
        control region is b / sigma_b^2.

    Returns
    -------
    result : tuple
        A tuple of four native Python floats. The first entry is the residual,
        the corrected Asimov significance minus the exact median discovery
        significance. The second is the corrected Asimov significance, the
        third the exact median and the fourth the first-order Asimov
        significance.

    Raises
    ------
    ValueError
        If s is negative or not finite, or if b or sigma_b is not finite or
        not above zero.
    """
    return result
```
