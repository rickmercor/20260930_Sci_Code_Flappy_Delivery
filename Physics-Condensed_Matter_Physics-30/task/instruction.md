# Physics-Condensed_Matter_Physics-30

## Background

Thermal transport in insulating solids is primarily mediated by lattice vibrations. At low temperature, quantum effects become increasingly important, and perturbative descriptions based only on independently propagating phonons and equilibrium lifetimes may not capture every contribution to the energy-current dynamics.

Path-integral Monte Carlo provides access to equilibrium correlation functions in imaginary time without requiring a perturbative treatment of anharmonicity. Within linear-response theory, thermal conductivity can be related to the real-frequency energy-current spectrum through the Green--Kubo formalism. Recovering that spectrum from imaginary-time correlation data is an analytic-continuation problem and is generally ill-conditioned, particularly when the correlation data contain statistical uncertainty.

A harmonic description provides a structured baseline for the energy-current spectrum, while the full microscopic current may contain contributions not represented by the harmonic approximation. Comparing harmonic and total imaginary-time correlations therefore provides a way to isolate an additional spectral contribution while preserving information already established by the harmonic reconstruction.

The dc thermal conductivity depends on the zero-frequency limit of the current spectral density. Reconstructing an additional spectral contribution from imaginary-time data therefore makes it possible to quantify how beyond-harmonic current processes alter the transport response.

## Problem

At low temperature, the total energy-current correlation of an insulating solid can contain contributions from higher-order phonon transitions that are absent from the harmonic-current approximation. Consider a reduced isotropic system in dimensionless units with

$$
k_B=\hbar=V=1,\qquad \beta=1.5.
$$

A harmonic-current spectral reconstruction has already been obtained, with zero-frequency value

$$
\Lambda_h(0)=6.485568295814.
$$

At six imaginary times,

$$
\boldsymbol{\tau}=
\begin{pmatrix}
0.075 & 0.165 & 0.285 & 0.420 & 0.570 & 0.735
\end{pmatrix},
$$

the reconstructed harmonic correlation is

$$
\mathbf C_h=
\begin{pmatrix}
1.520000000000 &
1.470000000000 &
1.420000000000 &
1.380000000000 &
1.355000000000 &
1.345000000000
\end{pmatrix},
$$

while the measured total-current correlation is

$$
\mathbf C_{\mathrm{tot}}=
\begin{pmatrix}
1.589244048566 &
1.518820260964 &
1.454425796087 &
1.403371336138 &
1.373713928683 &
1.360610293767
\end{pmatrix},
$$

with standard errors

$$
\boldsymbol{\sigma}=
\begin{pmatrix}
0.0014 & 0.0012 & 0.0010 & 0.0009 & 0.0008 & 0.0008
\end{pmatrix}.
$$

Using the total-current extension of the source methodology, determine how the previously reconstructed harmonic spectrum should be treated, what spectral contribution accounts for the additional higher-order transitions, and which imaginary-time region is expected to be most sensitive to that additional contribution.

Represent the additional smooth spectral density by seven positive Gaussian basis functions,

$$
\widetilde{\Lambda}(\omega)=\sum_{j=1}^{7}a_j\exp\left[-\frac{(\omega-\mu_j)^2}{2s_j^2}\right],\qquad a_j>0,
$$

with fixed centers

$$
\boldsymbol{\mu}=
\begin{pmatrix}
0.35 & 0.80 & 1.40 & 2.20 & 3.20 & 4.50 & 6.00
\end{pmatrix},
$$

and fixed widths

$$
\mathbf s=
\begin{pmatrix}
0.18 & 0.25 & 0.32 & 0.42 & 0.55 & 0.72 & 0.95
\end{pmatrix}.
$$

Recover from the source the positive-frequency spectral representation relating an energy-current spectral density to its imaginary-time correlation, including its normalization and both thermal branches, and use it to construct the Gaussian amplitude-to-correlation map. Define

$$
d_i=C_{\mathrm{tot}}(\tau_i)-C_h(\tau_i),
$$

and

$$
\chi^2(\mathbf a)=\sum_{i=1}^{6}\left[\frac{\widetilde C(\tau_i;\mathbf a)-d_i}{\sigma_i}\right]^2.
$$

Use the positive default spectrum weights

$$
\mathbf m=
\begin{pmatrix}
0.003 & 0.006 & 0.012 & 0.020 & 0.030 & 0.025 & 0.015
\end{pmatrix},
$$

and the relative-entropy penalty

$$
D(\mathbf a\Vert\mathbf m)=\sum_{j=1}^{7}\left[a_j\ln\left(\frac{a_j}{m_j}\right)-a_j+m_j\right].
$$

For $\alpha>0$, define

$$
Q_\alpha(\mathbf a)=\frac{1}{2}\chi^2(\mathbf a)+\alpha D(\mathbf a\Vert\mathbf m).
$$

Determine the unique positive stationary spectrum for each $\alpha$ and locate the discrepancy-matched solution satisfying

$$
\chi^2(\mathbf a_\alpha)=6
$$

within

$$
1100\leq\alpha\leq1200.
$$

Treat $u=\log\alpha$ as the continuation coordinate. Starting from the fixed-$\alpha$ stationarity equations, derive the local continuation equations needed to follow the positive stationary branch and report the discrepancy-matched regularization strength, fitted amplitudes, and amplitude tangent $d\mathbf a_\alpha/du$.

Then assess whether the zero-frequency correction is uniquely identified by the six imaginary-time observations independently of the entropy regularization. Determine the unit-Euclidean-norm amplitude direction that leaves all six predicted imaginary-time corrections unchanged, choosing its sign so that its first component is positive. Quantify whether this data-null direction changes $\widetilde{\Lambda}(0)$, determine the complete interval over which the perturbed amplitudes remain strictly positive, and determine the corresponding limiting range of zero-frequency spectral corrections compatible with exactly the same six fitted correlation values.

Explain how the entropy penalty selects a unique spectrum from this data-equivalent family, and relate this finite-dimensional ambiguity to the source discussion of the difficulty of constraining the precise zero-frequency value of a spectrum from imaginary-time correlations.

Finally, quantify the first-order leverage of each measured imaginary-time correction on the selected dc spectral value. Perturb one $d_i$ at a time while allowing both the stationary amplitudes and $u=\log\alpha$ to respond so that the stationarity equations and $\chi^2=6$ remain satisfied to first order. Derive the coupled sensitivity equations and evaluate

$$
\ell_i=\sigma_i\frac{\partial\widetilde{\Lambda}(0)}{\partial d_i},\qquad i=1,\ldots,6.
$$

Identify which imaginary-time measurement has the largest $|\ell_i|$ and explain why its leverage on the dc extrapolation does not contradict the source observation that the additional higher-frequency total-current contribution is most visible at short imaginary times.

Recover from the source the zero-frequency Green--Kubo relation for the isotropic thermal conductivity and determine

$$
\Delta_\kappa=100\frac{\kappa_{\mathrm{tot}}-\kappa_h}{\kappa_h}.
$$

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

01_gaussian_laplace_transform.py

Goal
----
Evaluate the half-line Laplace transform of Gaussian spectral basis functions for supplied nonnegative Laplace parameters.

```python
def compute_gaussian_laplace(
    c: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Evaluate half-line Laplace transforms of Gaussian basis functions.

    Parameters
    ----------
    c : np.ndarray
        One-dimensional finite array of nonnegative Laplace parameters with
        shape (n_c,).
    centers : np.ndarray
        One-dimensional finite array of nonnegative Gaussian centers with
        shape (n_basis,).
    widths : np.ndarray
        One-dimensional finite array of strictly positive Gaussian widths
        with shape (n_basis,).

    Returns
    -------
    result : np.ndarray
        Transform matrix with shape (n_c, n_basis), where entry (i, j)
        is the half-line Laplace transform for c[i], centers[j], and
        widths[j].

    Raises
    ------
    ValueError
        If an input is not one-dimensional or nonempty, if centers and
        widths have different lengths, if any input is nonfinite, if any
        c or center is negative, or if any width is not strictly positive.
    """
    return result
```

### Step 2

02_build_imaginary_time_kernel.py

Goal
----
Construct the imaginary-time spectral kernel for Gaussian basis functions using the forward and thermally reflected Laplace contributions.

```python
def build_imaginary_time_kernel(
    tau: "np.ndarray",
    beta: float,
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> "np.ndarray":
    """Construct the Gaussian imaginary-time kernel.

    Parameters
    ----------
    tau : np.ndarray
        One-dimensional finite array of imaginary times with shape
        (n_tau,), each satisfying 0 <= tau[i] <= beta.
    beta : float
        Positive inverse temperature.
    centers : np.ndarray
        One-dimensional finite array of nonnegative Gaussian centers with
        shape (n_basis,).
    widths : np.ndarray
        One-dimensional finite array of strictly positive Gaussian widths
        with shape (n_basis,).

    Returns
    -------
    kernel : np.ndarray
        Kernel matrix with shape (n_tau, n_basis).

    Raises
    ------
    ValueError
        If tau is not a nonempty one-dimensional finite array, if beta is
        not finite and strictly positive, or if any tau lies outside
        [0, beta]. Validation of centers and widths follows the transform
        requirements from the preceding step.
    """
    return kernel
```

### Step 3

03_solve_kkt_newton.py

Goal
----
Solve the positive entropy-regularized spectral stationarity equations at a fixed regularization strength using a safeguarded Newton iteration.

```python
def solve_kkt_newton(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    alpha: float,
    initial: "np.ndarray",
) -> "np.ndarray":
    """Solve the positive fixed-alpha KKT equations with safeguarded Newton steps.

    Parameters
    ----------
    kernel : np.ndarray
        Finite two-dimensional array with shape (n_obs, n_basis).
    correction : np.ndarray
        Finite one-dimensional correction data with shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors with shape (n_obs,).
    default : np.ndarray
        Finite strictly positive default spectrum with shape (n_basis,).
    alpha : float
        Finite strictly positive regularization strength.
    initial : np.ndarray
        Finite strictly positive initial amplitudes with shape (n_basis,).

    Returns
    -------
    amplitudes : np.ndarray
        Unique strictly positive stationary amplitudes with shape
        (n_basis,).

    Raises
    ------
    ValueError
        If the inputs have inconsistent shapes, contain nonfinite values,
        violate positivity requirements, or alpha is not strictly positive.
    RuntimeError
        If the safeguarded Newton iteration fails to converge.
    """
    return amplitudes
```

### Step 4

04_compute_regularization_sensitivity.py

Goal
----
Compute the implicit derivative of the regularized spectral amplitudes and chi-square discrepancy with respect to the logarithm of the regularization strength.

```python
def compute_regularization_sensitivity(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    alpha: float,
    amplitudes: "np.ndarray",
) -> "np.ndarray":
    """Compute the logarithmic-alpha continuation tangent.

    Parameters
    ----------
    kernel : np.ndarray
        Finite array with shape (n_obs, n_basis).
    correction : np.ndarray
        Finite correction vector with shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors with shape (n_obs,).
    default : np.ndarray
        Finite strictly positive default spectrum with shape (n_basis,).
    alpha : float
        Finite strictly positive regularization strength.
    amplitudes : np.ndarray
        Finite strictly positive stationary amplitudes with shape
        (n_basis,).

    Returns
    -------
    result : np.ndarray
        One-dimensional array of length n_basis + 1. The first n_basis
        entries contain d amplitudes / d log(alpha), and the last entry
        contains d chi-square / d log(alpha).

    Raises
    ------
    ValueError
        If dimensions or shapes are inconsistent, if required values are
        nonfinite, or if positivity conditions are violated.
    """
    return result
```

### Step 5

05_trace_discrepancy_continuation.py

Goal
----
Trace the entropy-regularized solution branch in log regularization strength and locate the spectrum satisfying a prescribed chi-square discrepancy using predictor-corrector continuation and safeguarded Newton refinement.

```python
def trace_discrepancy_continuation(
    kernel: "np.ndarray",
    correction: "np.ndarray",
    sigma: "np.ndarray",
    default: "np.ndarray",
    target_chi2: float,
    alpha_lower: float,
    alpha_upper: float,
    n_steps: int,
) -> "np.ndarray":
    """Locate the discrepancy-matched spectrum by log-alpha continuation.

    Parameters
    ----------
    kernel : np.ndarray
        Finite array with shape (n_obs, n_basis).
    correction : np.ndarray
        Finite correction data with shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors with shape (n_obs,).
    default : np.ndarray
        Finite strictly positive default spectrum with shape (n_basis,).
    target_chi2 : float
        Finite nonnegative discrepancy target.
    alpha_lower : float
        Finite strictly positive lower continuation bound.
    alpha_upper : float
        Finite strictly positive upper continuation bound greater than
        alpha_lower.
    n_steps : int
        Number of positive log-alpha continuation intervals. Must be at
        least 2.

    Returns
    -------
    result : np.ndarray
        One-dimensional array of length n_basis + 2 containing the matched
        alpha, achieved chi-square, and fitted amplitudes.

    Raises
    ------
    ValueError
        If scalar conditions are violated or the continuation interval
        does not bracket the requested discrepancy.
    RuntimeError
        If continuation or safeguarded root refinement fails.
    """
    return result
```

### Step 6

06_compute_zero_frequency_spectrum.py

Goal
----
Evaluate the reconstructed additional spectral density at zero frequency from the fitted Gaussian amplitudes.

```python
def compute_zero_frequency_spectrum(
    amplitudes: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
) -> float:
    """Evaluate the additional spectral density at zero frequency.

    Parameters
    ----------
    amplitudes : np.ndarray
        One-dimensional finite array of nonnegative Gaussian amplitudes with
        shape (n_basis,).
    centers : np.ndarray
        One-dimensional finite array of nonnegative Gaussian centers with
        shape (n_basis,).
    widths : np.ndarray
        One-dimensional finite array of strictly positive Gaussian widths
        with shape (n_basis,).

    Returns
    -------
    lambda_zero : float
        Additional reconstructed spectral density at zero frequency.

    Raises
    ------
    ValueError
        If an input is not a nonempty one-dimensional array, if the three
        arrays do not have the same shape, if any input is nonfinite, if any
        amplitude or center is negative, or if any width is not strictly
        positive.
    """
    return lambda_zero
```

### Step 7

07_compute_conductivity_change.py

Goal
----
Convert harmonic and reconstructed zero-frequency spectral densities into harmonic conductivity, total conductivity, and percentage conductivity change.

```python
def compute_conductivity_change(
    beta: float,
    harmonic_lambda_zero: float,
    additional_lambda_zero: float,
) -> "np.ndarray":
    """Compute conductivity values from zero-frequency spectral densities.

    Parameters
    ----------
    beta : float
        Finite strictly positive inverse temperature.
    harmonic_lambda_zero : float
        Finite strictly positive harmonic spectral density at zero frequency.
    additional_lambda_zero : float
        Finite nonnegative additional spectral density at zero frequency.

    Returns
    -------
    result : np.ndarray
        One-dimensional array of length 3 containing harmonic conductivity,
        total conductivity, and percentage conductivity change, in that order.

    Raises
    ------
    ValueError
        If beta is not finite and strictly positive, if the harmonic
        zero-frequency spectral density is not finite and strictly positive,
        or if the additional zero-frequency spectral density is not finite
        and nonnegative.
    """
    return result
```

### Step 8

08_run_total_current_reconstruction.py

Goal
----
Run the complete total-current spectral reconstruction using log-alpha continuation and return the percentage change in thermal conductivity relative to the fixed harmonic reconstruction.

```python
def run_total_current_reconstruction(
    tau: "np.ndarray",
    beta: float,
    harmonic_correlation: "np.ndarray",
    total_correlation: "np.ndarray",
    sigma: "np.ndarray",
    centers: "np.ndarray",
    widths: "np.ndarray",
    default: "np.ndarray",
    harmonic_lambda_zero: float,
    target_chi2: float,
    alpha_lower: float,
    alpha_upper: float,
) -> float:
    """Run the complete total-current reconstruction in reduced units.

    Parameters
    ----------
    tau : np.ndarray
        Nonempty finite array of shape (n_obs,), with 0 <= tau[i] <= beta.
    beta : float
        Finite strictly positive inverse temperature.
    harmonic_correlation : np.ndarray
        Finite fixed harmonic correlation, shape (n_obs,).
    total_correlation : np.ndarray
        Finite measured total-current correlation, shape (n_obs,).
    sigma : np.ndarray
        Finite strictly positive standard errors, shape (n_obs,).
    centers : np.ndarray
        Nonempty finite nonnegative Gaussian centers, shape (n_basis,).
    widths : np.ndarray
        Finite strictly positive Gaussian widths, shape (n_basis,).
    default : np.ndarray
        Finite strictly positive default amplitudes, shape (n_basis,).
    harmonic_lambda_zero : float
        Finite strictly positive harmonic spectral density at zero frequency.
    target_chi2 : float
        Finite nonnegative discrepancy target.
    alpha_lower : float
        Finite strictly positive lower continuation bound.
    alpha_upper : float
        Finite upper continuation bound strictly greater than alpha_lower.
        The bounds must contain a spectrum meeting the discrepancy target.

    Returns
    -------
    delta_percent : float
        Percentage conductivity change relative to the fixed harmonic
        reconstruction. Compose the preceding public functions to obtain it.

    Raises
    ------
    ValueError
        If array shapes, finiteness or the stated domains are invalid, or
        if the continuation interval does not bracket the target discrepancy.
    RuntimeError
        If an upstream stationary solve or continuation refinement fails.
    """
    return delta_percent
```
