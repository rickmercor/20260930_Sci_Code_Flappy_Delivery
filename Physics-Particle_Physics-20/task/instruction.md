# Physics-Particle_Physics-20

## Background

Counting experiments are among the oldest and most transparent instruments of particle physics. A search observes $n$ events in a signal region, modelled as a Poisson variable with mean $s + b$; a discovery claim is a rejection of the background-only hypothesis $s = 0$, quantified by a $p$-value and its Gaussian equivalent $Z = \Phi^{-1}(1 - p)$, with $Z = 5$ the conventional discovery threshold. When designing an experiment or optimising its selection cuts, one wants the expected sensitivity: the median of $Z$ under the assumption that a signal of nominal strength is present. Because the $p$-value and $Z$ are related monotonically but nonlinearly, the median is the appropriate summary of the sensitivity.

The modern treatment builds the test on the profile likelihood ratio $\lambda(s) = L(s, \hat{\hat{b}}(s)) / L(\hat{s}, \hat{b})$, where the background rate $b$ is a nuisance parameter, and on the statistic $q_0 = -2 \ln \lambda(0)$ for $\hat{s} \geq 0$ (zero otherwise). By the theorems of Wilks (1938) and Wald (1943), $q_0$ has a simple asymptotic distribution and the discovery significance is $Z = \sqrt{q_0}$ in the large-sample limit. The median significance can be approximated by evaluating the statistic on the data set in which every observed count is replaced by its expectation value under the assumed signal - the Asimov data set (Cowan, Cranmer, Gross and Vitells, Eur. Phys. J. C 71, 1554 (2011)).

In practice the expected background is rarely known exactly. In the on/off configuration, familiar from gamma-ray astronomy (Li and Ma, Astrophys. J. 272, 317 (1983)) and collider searches alike, a control region with exposure ratio $\tau$ yields a second Poisson count $m$ with mean $\tau b$, and $b$ is estimated by profiling over the joint likelihood of $(n, m)$. The widely used approximations $s/\sqrt{b}$ and $s/\sqrt{b + \sigma_b^2}$ arise as leading-order expansions of the profile-likelihood significance and are reliable only when $s \ll b$ and the background uncertainty is small; outside that regime they overestimate the sensitivity, sometimes badly. First-order asymptotic results themselves degrade when the expected event yields are small - precisely the regime of rare-process searches - and refinements of the signed likelihood-ratio root developed in the higher-order asymptotics literature restore accuracy there; their form is model-specific and must be taken from the literature for the two-measurement Poisson problem considered here.

The same on/off data admit a Bayesian description: the unknown Poisson means are given non-informative priors and one computes the posterior probability that the on-region mean exceeds the background. Different non-informative priors yield materially different frequentist error rates for the resulting detection criterion, and the comparison singling out the preferred prior, together with the closed form of the posterior probability, is likewise established in the literature. Used as a channel-selection rule, this criterion need not agree with a fixed threshold on the frequentist significance.

Reproducibility note. The instance is fully deterministic: the only random draw in the whole task is the single call `u = numpy.random.default_rng(21).random((120, 3))`, whose columns feed, in order, $s_i = 0.8 + 4.2 u_{i0}$, $b_i = 0.3 + 5.7 u_{i1}$, $\tau_i = 0.5 + 2.5 u_{i2}$ for channels $i = 0, \ldots, 119$. No other random numbers are drawn anywhere; all subsequent quantities are deterministic functions of these 360 values, evaluated in IEEE double precision at the Asimov data $n_i = s_i + b_i$, $m_i = \tau_i b_i$.

## Problem

In a particle-physics search, one observes a number of events $n$ in a signal ("on") region, modelled as Poisson-distributed with mean $s + b$, where $s$ and $b$ are the expected signal and background yields. The background rate is not known a priori: it is constrained by a control ("off") measurement $m$, Poisson-distributed with mean $\tau b$, where $\tau$ is the known ratio of the control-to-signal exposures. The sensitivity of such a counting experiment is characterised by the median significance with which the background-only hypothesis $s = 0$ would be rejected if the signal were truly present, and at small event yields the popular approximations $s/\sqrt{b}$ and $s/\sqrt{b + \sigma_b^2}$ overestimate this sensitivity.

Your task is to solve one concrete deterministic instance of a multi-channel sensitivity computation. Use the following configuration:

- 120 independent on/off counting channels, indexed $i = 0, \ldots, 119$.
- Channel parameters from a single draw `u = numpy.random.default_rng(21).random((120, 3))`, columns used in this order: $s_i = 0.8 + 4.2 u_{i0}$, $b_i = 0.3 + 5.7 u_{i1}$, $\tau_i = 0.5 + 2.5 u_{i2}$.
- Every channel-level quantity is evaluated at that channel's Asimov data, $n_i = s_i + b_i$ and $m_i = \tau_i b_i$, treated as real-valued counts; the median significance is approximated by evaluating at these expectation values.
- Per channel, the discovery significance $Z_i$ is obtained from the profile-likelihood-ratio test of $s = 0$ in the two-measurement model, with the signed likelihood-ratio root improved by the higher-order asymptotic correction prescribed for this model in the literature; $Z_i = \max\{0, r_{\mathrm{corrected}}\}$. No continuity or discreteness adjustment is applied in this two-measurement setting.
- Per channel, the posterior probability that the on-region mean exceeds the background, $P_i = P(\mu_{\mathrm{on}} > b \mid n_i, m_i)$ with $\mu_{\mathrm{on}} = s + b$, is computed with the non-informative prior that the reference comparison of Bayesian on/off criteria recommends for on/off analysis at relatively small background a channel is selected if and only if $P_i > 0.95$ (strict inequality).
- The combined sensitivity is $Z_{\mathrm{comb}} = \sqrt{\sum_{i \in \mathrm{selected}} Z_i^2}$. All arithmetic is IEEE double precision.

Compute the per-channel significances and posterior signal probabilities, apply the selection, and combine the selected channels.

In your reasoning, report the following scalars: the number of selected channels; the smallest and the largest per-channel significance among the selected channels, each to at least 6 significant figures; the posterior signal probability of channel 0 and whether channel 0 is selected; and the smallest and largest posterior signal probabilities across all 120 channels. Report each of the three posterior-probability checkpoints to at least 6 significant figures. Also state, in one or two lines each: the form of the higher-order correction applied to the signed root; the form of the auxiliary statistic entering that correction in the two-measurement model; the closed-form expression used for the posterior signal probability; which non-informative prior was used, and the quantitative error-rate evidence by which the reference comparison prefers it, including how its statistical power compares with the frequentist criterion and over what range of background; and the form of the signed likelihood-ratio root and of the profiled background estimate at s = 0 in the two-measurement model.

Your final answer must be a single number: $Z_{\mathrm{comb}}$.

The scalars and the five statements listed above are required inside <reasoning> and are the only intermediate content it must contain; the brevity guidance in the block below applies to everything beyond that list. Do not paste the full channel parameter table, per-channel significance lists, or per-channel probability lists.

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

01_generate_channel_parameters

Goal
----
Generate the deterministic ensemble of on/off channel parameters (s_i, b_i, tau_i) for the multi-channel sensitivity study from a single seeded pseudo-random draw.

```python
def generate_channel_parameters(n_channels: int, seed: int):
    '''Generate the (n_channels, 3) array of channel parameters.

    Parameters
    ----------
    n_channels : int
        Number of on/off channels; must be a positive integer.
    seed : int
        Seed of the single numpy.random.default_rng draw; must be an
        integer (booleans are not accepted).

    Raises
    ------
    ValueError
        If n_channels is not a positive integer, or seed is not an
        integer, or either argument is a boolean.

    Returns
    -------
    params : numpy.ndarray
        Array of shape (n_channels, 3) whose columns are, in order,
        s = 0.8 + 4.2*u[:, 0], b = 0.3 + 5.7*u[:, 1],
        tau = 0.5 + 2.5*u[:, 2], with
        u = numpy.random.default_rng(seed).random((n_channels, 3)).
    '''
    return result  # placeholder
```

### Step 2

02_compute_asimov_counts

Goal
----
Replace each channel's observable counts by their expectation values under the nominal signal hypothesis (the Asimov data of the channel).

```python
def compute_asimov_counts(params):
    '''Compute the Asimov counts (n, m) for every channel.

    Parameters
    ----------
    params : numpy.ndarray
        Array of shape (N, 3) with columns [s, b, tau]; all entries must
        be finite, strictly positive and at most 1e12.

    Raises
    ------
    ValueError
        If params is not a two-dimensional array with exactly 3 columns,
        or contains a non-finite entry, or any s, b or tau is not
        strictly positive, or any entry exceeds 1e12, which is the
        supported domain of the counting model.

    Returns
    -------
    counts : numpy.ndarray
        Array of shape (N, 2), float64, columns [n, m] with n = s + b
        and m = tau * b.
    '''
    return result  # placeholder
```

### Step 3

03_compute_profiled_background

Goal
----
Compute the constrained (profiled) maximum-likelihood estimate of the background rate b for a specified signal value s in the two-measurement on/off model.

```python
def compute_profiled_background(n: float, m: float, tau: float, s: float) -> float:
    '''Profiled background estimate bhh(s) of the on/off model.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 <= n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 <= m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.
    s : float
        Hypothesised signal value; must be finite and satisfy
        0 <= s <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n < 0 or m < 0 or s < 0, if
        tau <= 0, or if any of n, m, tau, s exceeds 1e12, which is the
        supported domain of the counting model.

    Returns
    -------
    bhh : float
        The profiled background estimate bhh(s), the nonnegative
        constrained maximum-likelihood estimate, as a
        native Python float.
    '''
    return result  # placeholder
```

### Step 4

04_compute_signed_root

Goal
----
Compute the signed likelihood-ratio root r(0) for a test of s = 0 in the two-measurement on/off model.

```python
def compute_signed_root(n: float, m: float, tau: float) -> float:
    '''Signed likelihood-ratio root r(0) of the on/off model.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 < n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 < m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n <= 0 or m <= 0 or tau <= 0,
        or if any of n, m, tau exceeds 1e12, which is the supported
        domain of the counting model.

    Returns
    -------
    r0 : float
        The signed root r(0), positive when n > m/tau and negative when
        n < m/tau, as a native Python float.
    '''
    return result  # placeholder
```

### Step 5

05_compute_auxiliary_statistic

Goal
----
Compute the auxiliary statistic u(0) that enters the higher-order correction of the signed likelihood-ratio root, in the form specific to the two-measurement on/off model.

```python
def compute_auxiliary_statistic(n: float, m: float, tau: float) -> float:
    '''Auxiliary statistic u(0) of the two-measurement on/off model.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 < n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 < m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n <= 0 or m <= 0 or tau <= 0,
        or if any of n, m, tau exceeds 1e12, which is the supported
        domain of the counting model.

    Returns
    -------
    u0 : float
        The auxiliary statistic u(0) for a test of s = 0, in the form
        prescribed for the two-measurement model, as a native Python
        float.
    '''
    return result  # placeholder
```

### Step 6

06_compute_corrected_significance

Goal
----
Combine the signed root r(0) and the auxiliary statistic u(0) into the higher-order corrected discovery significance of one channel.

```python
def compute_corrected_significance(r0: float, u0: float) -> float:
    '''Higher-order corrected discovery significance from r(0) and u(0).

    Parameters
    ----------
    r0 : float
        Signed likelihood-ratio root r(0); must be finite.
    u0 : float
        Auxiliary statistic u(0); must be finite.

    Raises
    ------
    ValueError
        If r0 or u0 is non-finite.

    Returns
    -------
    z : float
        The channel significance Z = max{0, corrected root}, where the
        corrected root equals r0 whenever r0 = 0 or u0 = 0, as a native
        Python float.
    '''
    return result  # placeholder
```

### Step 7

07_compute_signal_probability

Goal
----
Compute the posterior probability that the on-region mean exceeds the background, under the prescribed non-informative prior of the reference Bayesian on/off analysis.

```python
def compute_signal_probability(n: float, m: float, tau: float) -> float:
    '''Posterior probability P(mu_on > b | n, m) of the on/off channel.

    Parameters
    ----------
    n : float
        On-region count (real-valued Asimov counts allowed); must be
        finite and satisfy 0 <= n <= 1e12.
    m : float
        Control-region count; must be finite and satisfy 0 <= m <= 1e12.
    tau : float
        Exposure ratio; must be finite and satisfy 0 < tau <= 1e12.

    Raises
    ------
    ValueError
        If any argument is non-finite, if n < 0 or m < 0, if tau <= 0,
        or if any of n, m, tau exceeds 1e12, which is the supported
        domain of the counting model.

    Returns
    -------
    p : float
        The posterior probability that the on-region mean exceeds the
        background, under the prescribed non-informative prior, as a
        native Python float in [0, 1]. The mathematical probability is
        strictly between 0 and 1; IEEE double precision may round it
        to an endpoint.
    '''
    return result
```

### Step 8

08_compute_combined_sensitivity

Goal
----
Run the full multi-channel pipeline: generate the channel ensemble, evaluate each channel at its Asimov data, compute the higher-order corrected significance and the posterior signal probability per channel, select channels by the Bayesian criterion, and combine the selected significances in quadrature.

```python
def compute_combined_sensitivity(n_channels: int, seed: int, threshold: float) -> float:
    '''Combined multi-channel sensitivity Z_comb of the ensemble.

    Parameters
    ----------
    n_channels : int
        Number of on/off channels; must be a positive integer.
    seed : int
        Seed of the single numpy.random.default_rng draw; must be an
        integer (booleans are not accepted).
    threshold : float
        Selection threshold on the posterior signal probability; must be
        finite and satisfy 0 < threshold < 1. Selection uses the strict
        inequality P > threshold.

    Raises
    ------
    ValueError
        If n_channels is not a positive integer, or seed is not an
        integer, or either of those arguments is a boolean, or threshold
        is non-finite or not strictly between 0 and 1.

    Returns
    -------
    z_comb : float
        The combined sensitivity Z_comb = sqrt(sum of squared channel
        significances over the selected channels), 0.0 if no channel is
        selected, as a native Python float.
    '''
    return result  # placeholder
```
