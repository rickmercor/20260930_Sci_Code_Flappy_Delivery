# Physics-Optics-46

## Background

A reconstructive spectrometer encodes a spectrum into the intensities measured at several output ports. Interference inside a disordered optical cavity gives different frequencies different spatial patterns. Recovering the spectrum requires those patterns to carry distinguishable information in the presence of detector noise.

Opening the cavity to external channels changes both its resonances and the fraction of incident power reaching the detectors. Stronger coupling can broaden spectral features while changing throughput, so performance cannot be inferred from linewidth alone. A finite cavity may also have several locally favorable design regimes, and a device may expose more than one tunable parameter. Detectors respond linearly only up to a maximum incident power, so the throughput a design may deliver is itself bounded.

For a linear reconstruction, the nonzero singular values of the intensity-transmission matrix determine how detector noise is amplified. An underdetermined measurement has an additional unobserved subspace, whose reconstruction bias is conceptually separate from the variance caused by detector noise. Maintaining that distinction is essential when interpreting a performance bound. A device is often judged by the reconstruction variance of the frequency channels inside a specified passband rather than across the whole comb, so individual channels can carry unequal weight in the figure of merit.

Statistical descriptions connect average noise amplification to spectral correlations and the number of measurement channels. Such descriptions rest on assumptions about the scattering statistics of a typical device. Comparing an individual device with an ensemble description is a way of testing how far those assumptions carry.

## Problem

Ensemble estimates for reconstructive spectrometers relate a device's average noise amplification to its spectral correlations and channel counts. Test such an estimate against a deterministic open-cavity benchmark by comparing its exact in-band noise amplification, at the best design its detector allows, with the literature's Lorentzian, finite-port-window prediction. Use a lossless cavity with $$P=32$$ internal modes, $$M=6$$ measured output ports and one input port, and $$N=48$$ equally spaced, endpoint-inclusive angular frequencies $$\omega_n$$ from $$-0.8$$ to $$0.8$$ in dimensionless units; construct its fixed realization with a fresh NumPy $$\texttt{default\_rng}(17)$$ by drawing standard-normal arrays $$X,Y$$ of shape $$(P,P)$$ and then $$V$$ of shape $$(P,M+1)$$, and setting $$H=[X+X^{\mathsf T}+i(Y-Y^{\mathsf T})]/(2\sqrt P)$$ and $$W_0=V/\sqrt P$$. The device has two design parameters: the logarithmic coupling $$q\in[0.3,1]$$ and a comb detuning $$s\in[-0.06,0.06]$$ that rigidly shifts the sampled frequencies. With $$W=e^qW_0$$ and $$H_{\rm eff}=H-iWW^\dagger/2$$, the flux-normalized scattering matrix is $$S(\omega,q,s)=I-iW^\dagger[(\omega+s)I-H_{\rm eff}]^{-1}W$$, where $$\dagger$$ denotes conjugate transpose and channel zero is the input.

At fixed detector-noise variance, the noise-induced variance of the minimum-norm spectral estimate inside the device passband is $$F(q,s)=\sum_{n\in\mathcal K}[(A^{\mathsf T}A)^+]_{nn}$$ for $$A_{mn}=|S_{m0}(\omega_n,q,s)|^2$$ with measured ports $$m=1,\ldots,M$$, where $$+$$ denotes the Moore–Penrose pseudoinverse and $$\mathcal K$$ collects the $$K=24$$ central frequency channels, zero-based indices $$12$$ through $$35$$; null-space bias is excluded and the intensity matrix remains full row rank throughout the rectangle. Detector linearity admits only those designs whose frequency-averaged measured throughput $$T_0(q,s)=N^{-1}\sum_{m,n}A_{mn}$$ satisfies $$T_0\le0.76$$, and a design at which $$T_0=0.76$$ is itself admissible. Minimize $$F$$ over the admissible designs; the admissible minimum over the rectangle is unique and lies at least one percent below every other admissible candidate in objective value.

Report the selected design $$q_\star,s_\star$$ and $$F_\star$$, and characterize it by the multiplier $$\mu\ge0$$ of the throughput constraint in the first-order optimality conditions of this constrained problem, and by the second derivative of $$F$$ along the constraint curve, parameterized by arclength, at that design. At the optimized design, take $$T_0$$ as defined above and let $$\rho_k$$ be the mean dot product over the $$N-k$$ pairs of columns separated by $$k$$, after separately subtracting each column's port mean and normalizing it to unit Euclidean norm; fit $$a\in[0.05,20]$$ by minimizing the unweighted residual $$\sum_{k=1}^{8}[a^2/(k^2+a^2)-\rho_k]^2$$. Interpret $$a=\Gamma_{\rm corr}/\Delta\omega$$ using the Lorentzian **half-width at half maximum**, and apply the primary literature's underdetermined inverse-Gram-trace approximation, including its fitted finite-window coefficient, to obtain the whole-comb prediction from $$a,T_0,M,N$$; take the in-band prediction to be $$F_{\rm pred}=(K/N)$$ times that whole-comb value. Compute the signed discrepancy $$D=100(F_{\rm pred}/F_\star-1)$$, and give the scalar diagnostics $$q_\star,s_\star,F_\star,T_0,a$$, fit residual, $$F_{\rm pred}$$, $$\mu$$ and the tangential curvature, together with the source-based origin of the window coefficient and an explanation of the size of this discrepancy.

## Output format

```
Output Format Requirements:
A single final numeric answer wrapped in <final_answer>...</final_answer> tags, followed by scientific reasoning wrapped in <reasoning>...</reasoning> tags. The final answer is the signed discrepancy $$D$$ in percent, written as a finite decimal with no units or words. The reasoning gives the requested scalar diagnostics, the source-based origin of the window coefficient and the explanation of the discrepancy, with the relations and intermediate values that establish the optimization and the comparison, and does not reproduce input matrices, full spectra, coefficient vectors or iteration histories.
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

cavity_realization

Goal
----
Construct the fixed Hamiltonian and channel-coupling realization of an open optical cavity.

```python
def cavity_realization(n_modes: int, n_ports: int, seed: int) -> "np.ndarray":
    r"""Create one reproducible cavity in dimensionless frequency units.
    
    Parameters
    ----------
    n_modes : int
        Mode count P >= 2.
    n_ports : int
        Measured output count M >= 1 with M + 1 <= P.
    seed : int
        Nonnegative seed for numpy.random.default_rng. Draw X, then Y, then V
        using standard_normal with the shapes in the background; draw order
        defines this fixed realization, rather than a distribution-only test.
    
    Returns
    -------
    result : np.ndarray
        Complex array of shape (P, P + M + 1), containing H followed by W0
        horizontally. The final M + 1 columns are real-valued couplings.
    
    Notes
    -----
    Inputs satisfy the stated domain. No global random state is read or changed.
    """
    return result
```

### Step 2

scattering_jet

Goal
----
Evaluate the scattering matrix and its first and second derivatives in the two design parameters.

```python
def scattering_jet(cavity: "np.ndarray", frequencies: "np.ndarray", log_coupling: float, detuning: float) -> "np.ndarray":
    r"""Compute a two-parameter scattering jet at the supplied design point.
    
    Parameters
    ----------
    cavity : np.ndarray
        Complex array (P, P + C), with Hermitian H in the first P columns
        and arbitrary complex W0 in the remaining C >= 2 columns.
    frequencies : np.ndarray
        Finite real vector of length N >= 1; its ordering is preserved.
    log_coupling : float
        Finite real q. All shifted-frequency resolvents are nonsingular.
    detuning : float
        Finite real s, added to every sampled frequency.
    
    Returns
    -------
    result : np.ndarray
        Complex array (6, N, C, C) containing S, dS/dq, dS/ds, d2S/dq2,
        d2S/dqds, d2S/ds2, in that order. Matrix indices are outgoing,
        incoming channels.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Results must
    resolve the ordinary partial derivatives, including near zero; analytic
    differentiation or an equivalently accurate method is accepted.
    """
    return result
```

### Step 3

transmission_jet

Goal
----
Convert the two-parameter scattering jet into the intensity-transmission jet seen by the measured ports.

```python
def transmission_jet(scattering: "np.ndarray") -> "np.ndarray":
    r"""Extract the measured intensity matrix and its design derivatives.
    
    Parameters
    ----------
    scattering : np.ndarray
        Finite complex array (6, N, C, C), with orders S, dS/dq, dS/ds,
        d2S/dq2, d2S/dqds, d2S/ds2 and channel indices outgoing then
        incoming; C >= 2 and N >= 1.
    
    Returns
    -------
    result : np.ndarray
        Real array (6, C - 1, N) containing A, dA/dq, dA/ds, d2A/dq2,
        d2A/dqds, d2A/ds2. Only outgoing ports 1 through C - 1 for incoming
        port zero are used.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Jets are ordinary
    real-parameter partial derivatives, with no factorial normalization.
    """
    return result
```

### Step 4

inverse_gram_jet

Goal
----
Evaluate a channel-weighted inverse-Gram variance and its gradient and Hessian in the two design parameters.

```python
def inverse_gram_jet(transmission: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    r"""Compute F, its gradient and its Hessian in the two design parameters.
    
    Parameters
    ----------
    transmission : np.ndarray
        Finite real array (6, M, N), orders A, dA/dq, dA/ds, d2A/dq2,
        d2A/dqds, d2A/ds2, M,N >= 1. A is full rank min(M,N), with that rank
        constant nearby, and its condition number, the ratio of its largest
        to smallest nonzero singular value, is at most 50. Negative entries
        in derivative slices are allowed.
    weights : np.ndarray
        Finite nonnegative real vector of length N, one weight per frequency
        channel, indexed like the columns of A. Weights are independent of the
        design parameters.
    
    Returns
    -------
    result : np.ndarray
        Real vector [F, dF/dq, dF/ds, d2F/dq2, d2F/dqds, d2F/ds2], shape (6,).
        The weighted sum runs over frequency channels; it is not divided by M,
        N or the weight sum.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Square,
    overdetermined and underdetermined matrices use the same scientific definition.
    """
    return result
```

### Step 5

spectral_correlation

Goal
----
Measure throughput and a lag-dependent spectral correlation from a finite transmission matrix.

```python
def spectral_correlation(transmission: "np.ndarray", max_lag: int) -> "np.ndarray":
    r"""Return throughput and the normalized finite-record correlation profile.
    
    Parameters
    ----------
    transmission : np.ndarray
        Finite nonnegative real matrix A, shape (M,N), M >= 2, N >= 2.
        Every port-centered frequency column has positive Euclidean norm.
    max_lag : int
        Largest lag L, with 1 <= L < N.
    
    Returns
    -------
    result : np.ndarray
        Real vector of length L + 2, [T0, rho_0, rho_1, ..., rho_L].
        rho_0 equals one; lag k averages exactly N-k normalized dot products.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged.
    """
    return result
```

### Step 6

fit_lorentzian

Goal
----
Fit a normalized Lorentzian half-width to a measured finite-record correlation profile.

```python
def fit_lorentzian(profile: "np.ndarray", width_interval: "np.ndarray") -> "np.ndarray":
    r"""Fit a half-width in units of frequency-channel spacing.
    
    Parameters
    ----------
    profile : np.ndarray
        Real vector [T0, rho_0, ..., rho_L], L >= 1, finite T0 > 0,
        rho_0 = 1 and all rho values in [-1,1]. The loss described in the
        background has at most one interior stationary point on the interval.
    width_interval : np.ndarray
        Real vector [a_lower,a_upper], with 0 < a_lower < a_upper <= 30.
    
    Returns
    -------
    result : np.ndarray
        Real vector [a_fit, residual_sum_of_squares], shape (2,).
        The fit includes both endpoints and uses unweighted positive lags.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. Return an
    endpoint exactly when it minimizes the loss. The unique minimum is supported.
    """
    return result
```

### Step 7

asymptotic_trace

Goal
----
Evaluate the literature's underdetermined inverse-trace prediction for a fitted Lorentzian correlation.

```python
def asymptotic_trace(normalized_width: float, throughput: float, n_ports: int, n_channels: int) -> float:
    r"""Compute the source model's trace prediction.
    
    Parameters
    ----------
    normalized_width : float
        Positive fitted Lorentzian half-width divided by channel spacing;
        0 < normalized_width <= 30.
    throughput : float
        Finite positive frequency-averaged total measured transmittance T0.
    n_ports : int
        Measured port count M >= 1.
    n_channels : int
        Frequency count N > M.
    
    Returns
    -------
    result : float
        Finite positive inverse-Gram trace prediction in the stated convention.
    
    Notes
    -----
    Inputs satisfy the stated domain. The source's closed form is evaluated
    throughout that domain, even where a particular cavity violates its
    statistical assumptions.
    """
    return result
```

### Step 8

stationary_design

Goal
----
Select the least-noise admissible design over the two-parameter coupling and detuning rectangle.

```python
def stationary_design(cavity: "np.ndarray", frequencies: "np.ndarray", weights: "np.ndarray", bounds: "np.ndarray", throughput_limit: float, scan_points: "np.ndarray") -> "np.ndarray":
    r"""Find the minimum admissible noise amplification over the design rectangle.
    
    Parameters
    ----------
    cavity : np.ndarray
        Cavity array (P,P+M+1), as in scattering_jet.
    frequencies : np.ndarray
        Finite real vector of N > M channels, before detuning. At every design
        in the rectangle the intensity matrix is full row rank, meets
        inverse_gram_jet's conditioning bound, and its shifted resolvents are
        nonsingular.
    weights : np.ndarray
        Finite nonnegative channel weights of length N, as in inverse_gram_jet.
    bounds : np.ndarray
        Real array [[q_lower,q_upper],[s_lower,s_upper]], each lower < upper.
    throughput_limit : float
        Positive detector throughput ceiling T_max. At least one admissible
        design exists in the rectangle.
    scan_points : np.ndarray
        Two integers >= 3, the numbers of uniformly spaced endpoint-inclusive
        nodes along the coupling and the detuning axis. Each supported
        instance is supplied with a scan for which the outcome conditions in
        the background hold; a coarser scan carries no such guarantee.
    
    Returns
    -------
    result : np.ndarray
        Real vector [q_star, s_star, F_star, active, q_pinned, s_pinned],
        shape (6,). The fourth entry is 1.0 when the throughput at the selected
        design equals the limit and 0.0 otherwise; the last two are 1.0 when the
        corresponding design coordinate sits on a bound of the rectangle and 0.0
        otherwise.
    
    Notes
    -----
    Compose scattering_jet, transmission_jet and inverse_gram_jet using their
    returned values. The throughput and its design derivatives come from the
    same intensity jet, each order being the average over frequency channels of
    the summed measured intensities of the corresponding intensity derivative,
    which matches the throughput convention of spectral_correlation. Inputs
    satisfy the stated domain and must remain unchanged.
    """
    return result
```

### Step 9

constrained_sensitivity

Goal
----
Evaluate the throughput multiplier, the limit response and the constrained curvature of the selected two-parameter design.

```python
def constrained_sensitivity(transmission: "np.ndarray", weights: "np.ndarray", throughput_active: bool, pinned_axes: "np.ndarray") -> "np.ndarray":
    r"""Return the constraint multiplier, the limit response and the tangential curvature.
    
    Parameters
    ----------
    transmission : np.ndarray
        Finite real array (6, M, N), the intensity jet at the selected design,
        as returned by transmission_jet. A is full rank min(M,N), M,N >= 1,
        and meets inverse_gram_jet's conditioning bound.
    weights : np.ndarray
        Finite nonnegative channel weights of length N, as in inverse_gram_jet.
    throughput_active : bool
        True when the throughput limit binds at the selected design; False
        otherwise. When True, not every coordinate sits on a bound and the
        throughput multiplier is uniquely determined.
    pinned_axes : np.ndarray
        Two booleans, True where the corresponding design coordinate sits on a
        bound of the design rectangle.
    
    Returns
    -------
    result : np.ndarray
        Real vector [multiplier, dF_star/dT_max, tangential_curvature],
        shape (3,): the throughput constraint's multiplier in the first-order
        optimality conditions, nonnegative at a binding minimum; the derivative
        of the optimized in-band variance with respect to the detector limit;
        and the curvature defined in the background.
    
    Notes
    -----
    Inputs satisfy the stated domain and must remain unchanged. The returned
    quantities are exact, not finite-difference estimates of a re-solved
    optimization.
    """
    return result
```

### Step 10

optimized_bound_comparison

Goal
----
Compose the two-parameter throughput-limited optimization and the empirical in-band bound comparison into the complete benchmark.

```python
def optimized_bound_comparison(n_modes: int, n_ports: int, n_channels: int, seed: int, bandwidth: float, bounds: "np.ndarray", throughput_limit: float, band_channels: int, max_lag: int, width_interval: "np.ndarray", scan_points: "np.ndarray") -> "np.ndarray":
    r"""Orchestrator for the two-parameter finite-cavity model comparison.
    
    Parameters
    ----------
    n_modes : int
        Mode count P >= M + 1.
    n_ports : int
        Measured output count M >= 2.
    n_channels : int
        Frequency count N > M.
    seed : int
        Nonnegative realization seed, as in cavity_realization.
    bandwidth : float
        Positive total frequency span B; sample N equally spaced frequencies
        from -B/2 through B/2, including both endpoints, before detuning.
    bounds : np.ndarray
        Real array [[q_lower,q_upper],[s_lower,s_upper]], the design rectangle,
        satisfying stationary_design's resolution and full-rank preconditions.
    throughput_limit : float
        Positive detector throughput ceiling, as in stationary_design.
    band_channels : int
        Count K, with 1 <= K <= N, of unit-weight central frequency channels
        placed as described in the background.
    max_lag : int
        Largest correlation lag L, with 1 <= L < N.
    width_interval : np.ndarray
        Positive [a_lower,a_upper] with a_upper <= 30. The optimized profile
        satisfies fit_lorentzian's uniqueness precondition.
    scan_points : np.ndarray
        Two integers >= 3, the scan node counts of stationary_design.
    
    Returns
    -------
    result : np.ndarray
        Real vector [D, q_star, s_star, F_star, T0, a_fit, fit_loss, F_pred,
        multiplier, tangential_curvature], shape (10,). D is in percent and
        F_star and F_pred are in-band sums.
    
    Notes
    -----
    Compose all earlier public functions and consume their returned values.
    Inputs satisfy the stated domain and the supplied arrays remain unchanged.
    No null-space bias, stochastic noise sampling, or realization averaging is added.
    """
    return result
```
