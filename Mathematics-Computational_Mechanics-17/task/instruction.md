# Mathematics-Computational_Mechanics-17

## Background

Dynamic homogenisation of composites for waves, applicable at all wavelengths and frequencies, showed that the effective response of a heterogeneous medium is generally nonlocal in space and time and carries cross-couplings, the Willis couplings of elastodynamics and the bianisotropic terms of electromagnetics, that none of its constituents possesses, and that a cell without a centre of symmetry keeps them even in the long-wavelength limit. Heat conduction differs from wave propagation in being diffusive, so its thermal waves decay as they travel and its effective description has two constitutive relations to couple, Fourier's law and the entropy-temperature relation. Earlier homogenisations of layered conductors obtained a single coupling, and only in the presence of a time modulation of the material; because effective parameters of this kind are not unique, a numerical value of a coupling has meaning only together with the construction that defines it.

## Problem

A periodic laminate whose layers each obey Fourier's law, with no cross-coupling of any kind, homogenises exactly to an effective conductor whose mean heat flux and mean entropy depend nonlocally in space and time on both the mean temperature and its gradient, so that its effective description carries bianisotropic, Willis-type couplings. Such effective parameters are not unique, because the free-wave dispersion of the laminate does not fix how its response divides between direct and coupling terms, and a different definition gives a different number: the quantity required here is the coupling that the source-driven homogenisation construction defines, with effective fields taken as ensemble means over a uniformly distributed translation of the infinite laminate. In each layer $-q = \kappa \partial_x \theta$ and $\theta_R \eta = c \theta$, with energy balance $-\partial_x q + r = \partial_t (\theta_R \eta)$, where $\theta$ is the temperature increment above the reference temperature $\theta_R$, $q$ the heat flux along $+x$, $\eta$ the entropy increment per unit volume, $c$ the volumetric heat capacity and $r$ a heat input; temperature and heat flux are continuous at every interface. The effective relations are written $-\langle q \rangle = \tilde\kappa \partial_x \langle \theta \rangle + \tilde\chi \langle \theta \rangle$ and $\langle \theta_R \eta \rangle = \tilde\xi \partial_x \langle \theta \rangle + \tilde c \langle \theta \rangle$, whose coefficients, acting on fields that vary as $\exp(i k x + i \omega t)$, are functions of $k$ and $\omega$.

The period, of length $l = 2.00$ micrometres, consists of four layers listed in order along $+x$:

| layer | material | thickness (micrometres) | $\kappa$ (W m$^{-1}$ K$^{-1}$) | $c$ (J m$^{-3}$ K$^{-1}$) |
| --- | --- | --- | --- | --- |
| 1 | silica | 0.70 | 1.38 | $1.65 \times 10^{6}$ |
| 2 | copper | 0.30 | 400 | $3.45 \times 10^{6}$ |
| 3 | alumina | 0.60 | 35 | $3.06 \times 10^{6}$ |
| 4 | silicon | 0.40 | 148 | $1.66 \times 10^{6}$ |

Take $\omega = 3.75 \times 10^{6}$ rad s$^{-1}$ and $k = 1.25 \times 10^{6}$ rad m$^{-1}$, and report the real part of the dimensionless coupling $\tilde\chi(k, \omega) l / \kappa_0$, with $\kappa_0 = 1$ W m$^{-1}$ K$^{-1}$, to four decimal places; the answer is graded within $0.005$. Inside the reasoning, report also the imaginary part of $\tilde\chi(k, \omega) l / \kappa_0$ to four decimal places, the real and imaginary parts of $\tilde\kappa(k, \omega) / \kappa_0$ to three decimal places, and those of $K_B l$ to four decimal places, where $K_B$ is the complex Bloch wavenumber of source-free thermal waves of the form $p(x) \exp(\pm K_B x + i \omega t)$ in the laminate at the same frequency, $p$ periodic with period $l$, taken with positive real part and with the imaginary part of $K_B l$ in $(-\pi, \pi]$.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
Output Format Requirements:
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

cell_transfer_matrix

Goal
----
Work in units in which the period, a reference conductivity and a reference volumetric heat capacity are one. The laminate repeats a cell of layers listed in order along $+x$, the first occupying $0 < x < h_1$. Return the transfer matrix that carries the state vector $(-q, \theta)$ from the start of the period beginning at $x = \mathrm{start}$, taken modulo the period, to the end of that period, together with its determinant and its trace.

```python
def cell_transfer_matrix(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    r"""Transfer matrix of the flux-temperature state $(-q, \theta)$ across one period of a periodic laminate.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$ of the time dependence $e^{s t}$.
    start : float
        Position at which the period begins.

    Returns
    -------
    dict
        Under the keys matrix, determinant and trace.

    Raises
    ------
    ValueError
        When the layer arrays are not one-dimensional, non-empty, finite, above zero and of equal length, when the Laplace variable is not finite, is zero or has a negative real part, or when start is not finite.
    """
    return
```

### Step 2

bloch_trace_retrieval

Goal
----
From the transfer matrix of the period beginning at $x = \mathrm{start}$ (step 01), return the Bloch wavenumber $K_B$ of the free thermal waves $p(x) e^{\pm K_B x + s t}$ of the laminate, $p$ periodic, taking the root with non-negative real part and with the imaginary part of $K_B l$ in $(-\pi, \pi]$, and the parameters $\kappa_R$, $c_R$ and $\chi_R$ of the uniform bianisotropic slab of length $l$ whose entropy coupling is $\chi_R / s$, whose transfer matrix equals that of the period and whose wavenumber equals that $K_B$. Units are those of step 01, and $l$ is the sum of the thicknesses.

```python
def bloch_trace_retrieval(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    start: float,
) -> dict:
    r"""Bloch wavenumber and one-period transfer-matrix retrieval of local bianisotropic parameters.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$; the period $l$ is their sum.
    laplace_s : complex
        Laplace variable $s$.
    start : float
        Position at which the period begins.

    Returns
    -------
    dict
        Under the keys bloch_wavenumber ($K_B$), kappa ($\kappa_R$), capacity ($c_R$), chi ($\chi_R$) and xi ($\xi_R = \chi_R / s$).

    Raises
    ------
    ValueError
        When an input is invalid for the transfer matrix, or when the period transfer matrix $T$ has a vanishing lower-left entry or $\operatorname{tr} T = \pm 2$, at which the retrieval is undefined.
    """
    return
```

### Step 3

forced_bloch_cell_response

Goal
----
For sources $r = r_0 e^{K x}$, $\zeta = \zeta_0 e^{K x}$ and $\varphi = \varphi_0 e^{K x}$ acting on the periodic laminate whose cell is listed in order along $+x$ from $x = 0$, return the amplitudes of the ensemble means over a uniformly distributed translation of the laminate, the sources being held fixed: $\langle \theta \rangle = \Theta e^{K x}$, $\langle -q \rangle = Q e^{K x}$ and $\langle \theta_R \eta \rangle = H e^{K x}$. Units are those of step 01.

```python
def forced_bloch_cell_response(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
    heat_source: complex,
    residual_gradient: complex,
    residual_temperature: complex,
) -> dict:
    r"""Ensemble-mean response of a periodic laminate to a heat source and to residual fields of one wavenumber.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.
    wavenumber : complex
        Wavenumber $K$ of the source dependence $e^{K x}$.
    heat_source : complex
        Amplitude $r_0$ of the heat input $r$.
    residual_gradient : complex
        Amplitude $\zeta_0$ of the residual temperature gradient $\zeta$.
    residual_temperature : complex
        Amplitude $\varphi_0$ of the residual temperature $\varphi$.

    Returns
    -------
    dict
        Under the keys mean_temperature, mean_flux and mean_entropy.

    Raises
    ------
    ValueError
        When an input is invalid for the transfer matrix, when the wavenumber or a source amplitude is not a number or is not finite, when $|\kappa_j K^2 - s c_j| \le 10^{-12} \max(|\kappa_j K^2|, |s c_j|)$ in some layer $j$, or when the linear solve of the forced cell problem fails or returns non-finite amplitudes, which happens only when the wavenumber lies on the Bloch dispersion to working precision.
    """
    return
```

### Step 4

source_driven_constitutive_matrix

Goal
----
At complex wavenumber $K$ and Laplace variable $s$, return the source-driven effective matrix $L_{\mathrm{eff}}$ of the laminate, the 2 by 2 matrix with rows $(\kappa_{\mathrm{eff}}, \chi_{\mathrm{eff}})$ and $(L_{21}, c_{\mathrm{eff}})$, $L_{21}$ being the coefficient of the mean temperature gradient in the mean entropy, that maps the mean kinematic amplitudes $(K \Theta - \zeta_0, \Theta - \varphi_0)$ to the mean kinetic amplitudes $(Q, H)$ of step 03 for every combination of the three sources, and a consistency figure: the largest modulus of the deviation of $L_{\mathrm{eff}} (K \Theta - \zeta_0, \Theta - \varphi_0)$ from $(Q, H)$ for a unit residual temperature, divided by the largest modulus of $Q$ and $H$ over the responses to a unit heat source, a unit residual gradient and a unit residual temperature. Units are those of step 01.

```python
def source_driven_constitutive_matrix(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
) -> dict:
    r"""Nonlocal effective constitutive matrix $L_{\mathrm{eff}}(K, s)$ of a periodic laminate from source-driven ensemble means.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.
    wavenumber : complex
        Wavenumber $K$ of the dependence $e^{K x}$.

    Returns
    -------
    dict
        Under the keys kappa ($\kappa_{\mathrm{eff}}$), chi ($\chi_{\mathrm{eff}}$), xi ($\xi_{\mathrm{eff}} = L_{21}$), capacity ($c_{\mathrm{eff}}$) and consistency.

    Raises
    ------
    ValueError
        When an input is invalid for the forced cell problem, when the wavenumber is not a number, or when the mean kinematic vectors of the unit heat source and the unit residual gradient are linearly dependent to working precision.
    """
    return
```

### Step 5

willis_symmetry_certificate

Goal
----
Certify the effective matrix of the laminate at $(K, s)$ by computing it also at $-K$, for the mirrored cell (layers in reverse order) at $-K$, and for the period cut at $x = \mathrm{start}$, and report three relative residuals, all measured on the scaled matrix with rows $(\kappa_{\mathrm{eff}}, \chi_{\mathrm{eff}})$ and $(s L_{21}, s \, c_{\mathrm{eff}})$, $L_{21}$ being the lower-left entry of the matrix of step 04. The adjoint residual is the largest of $|\chi_{\mathrm{eff}}(K) - s L_{21}(-K)|$, $|\kappa_{\mathrm{eff}}(K) - \kappa_{\mathrm{eff}}(-K)|$ and $|s \, c_{\mathrm{eff}}(K) - s \, c_{\mathrm{eff}}(-K)|$; the mirror residual is the largest of $|\chi^{m}(-K) + \chi_{\mathrm{eff}}(K)|$, $|s L_{21}^{m}(-K) + s L_{21}(K)|$, $|\kappa^{m}(-K) - \kappa_{\mathrm{eff}}(K)|$ and $|s \, c^{m}(-K) - s \, c_{\mathrm{eff}}(K)|$, the superscript $m$ marking the mirrored cell; the translation residual is the largest entry-wise modulus of the difference between the scaled matrices of the two cuts. Each is divided by the largest entry modulus of the scaled matrix at $(K, s)$. Units are those of step 01.

```python
def willis_symmetry_certificate(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    wavenumber: complex,
    start: float,
) -> dict:
    r"""Adjoint-pair, mirror and translation residuals of the source-driven effective matrix.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.
    wavenumber : complex
        Wavenumber $K$.
    start : float
        Position at which the alternative period begins.

    Returns
    -------
    dict
        Under the keys adjoint_residual, mirror_residual, translation_residual and chi.

    Raises
    ------
    ValueError
        When an input is invalid for the effective matrix at $K$ or at $-K$, or when start is not finite.
    """
    return
```

### Step 6

local_directional_impedance

Goal
----
Evaluate the effective matrix of step 04 at $K = 0$ and return its entries, the local wavenumber $k_{\mathrm{eff}}$, with non-negative real part, of source-free disturbances of the local effective medium, and the forward and backward thermal impedances $Z_{+}$ and $Z_{-}$, defined as the temperature divided by the heat flux $q$ (not $-q$) of a disturbance $\theta \propto e^{-k_{\mathrm{eff}} x}$ decaying along $+x$ and of a disturbance $\theta \propto e^{k_{\mathrm{eff}} x}$ decaying along $-x$. Report also the pair residual $|\chi_{\mathrm{eff}} - s L_{21}| / |Z^{-1}|$, with $L_{21}$ the lower-left entry and $Z^{-1} = \kappa_{\mathrm{eff}} k_{\mathrm{eff}}$. Units are those of step 01.

```python
def local_directional_impedance(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
) -> dict:
    r"""Local ($K = 0$) source-driven effective parameters and the direction-dependent thermal impedances $Z_{\pm}$.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.

    Returns
    -------
    dict
        Under the keys kappa, chi, xi ($\xi_{\mathrm{eff}} = L_{21}$ at $K = 0$), capacity, local_wavenumber ($k_{\mathrm{eff}}$), impedance_forward ($Z_{+}$), impedance_backward ($Z_{-}$) and pair_residual.

    Raises
    ------
    ValueError
        When an input is invalid for the effective matrix at $K = 0$, or when an impedance is undefined because its denominator vanishes.
    """
    return
```

### Step 7

effective_bloch_dispersion_root

Goal
----
Find a zero of the effective dispersion function of the laminate, $D(K, s) = -1 / \Theta_r(K, s)$, where $\Theta_r$ is the mean-temperature amplitude of the response to a unit heat source (step 03), by the secant method started from the two given complex wavenumbers. Stop when the secant update is at most tolerance times the modulus of the new iterate, returning that iterate without evaluating $D$ there; raise an error if this does not happen within max_iterations updates. Units are those of step 01.

```python
def effective_bloch_dispersion_root(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    first_guess: complex,
    second_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    r"""Zero of the dispersion function $D = -1/\Theta_r$ of the source-driven effective medium, by the secant method.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.
    first_guess : complex
        First starting wavenumber $K_0$.
    second_guess : complex
        Second starting wavenumber $K_1$.
    tolerance : float
        Relative stopping tolerance on the secant update.
    max_iterations : int
        Largest number of secant updates.

    Returns
    -------
    dict
        Under the keys root, dispersion_at_guess and iterations.

    Raises
    ------
    ValueError
        When an input is invalid for the forced cell problem, when the guesses coincide or are not finite numbers, when the tolerance is not a finite real number above zero, when the iteration limit is not an integer of at least 1, when two successive values of $D$ coincide, or when the iteration does not converge.
    """
    return
```

### Step 8

willis_heat_coupling_report

Goal
----
Given the physical layer data of the cell, listed in order along $+x$, an angular frequency and a wavenumber for fields varying as $e^{i k x + i \omega t}$, and the reference scales, return the normalised source-driven effective matrix at that frequency and wavenumber, whose coupling $\chi_{\mathrm{eff}} l / \kappa_0$ is the reported result, together with the certificates and contrasts of the other stages.

```python
def willis_heat_coupling_report(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    angular_frequency: float,
    wavenumber: float,
    reference_conductivity: float,
    reference_capacity: float,
    root_tolerance: float,
    max_iterations: int,
    certificate_threshold: float,
) -> dict:
    r"""Report the source-driven bianisotropic coupling $\tilde\chi l / \kappa_0$ of a periodic laminate at a given frequency and wavenumber.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities in $\mathrm{W\,m^{-1}\,K^{-1}}$, in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities in $\mathrm{J\,m^{-3}\,K^{-1}}$.
    thicknesses : np.ndarray
        Layer thicknesses in $\mathrm{m}$.
    angular_frequency : float
        Angular frequency $\omega$ in $\mathrm{rad\,s^{-1}}$.
    wavenumber : float
        Wavenumber $k$ in $\mathrm{rad\,m^{-1}}$ of the dependence $e^{i k x + i \omega t}$.
    reference_conductivity : float
        Reference conductivity $\kappa_0$.
    reference_capacity : float
        Reference volumetric heat capacity $c_0$.
    root_tolerance : float
        Relative tolerance of the dispersion root.
    max_iterations : int
        Largest number of secant updates.
    certificate_threshold : float
        Largest accepted certificate residual.

    Returns
    -------
    dict
        Under the keys chi_real, chi_imag, dimensionless_frequency, dimensionless_wavenumber, adjoint_residual, mirror_residual, translation_residual, root_mismatch, chi, kappa, xi, capacity, bloch_wavenumber, dispersion_root, retrieval_chi, local_chi, impedance_forward and impedance_backward.

    Raises
    ------
    ValueError
        When any input is invalid for the stage that uses it, when a scalar input is not a real number or a layer array holds non-real values, when the frequency, the wavenumber, a reference scale or the certificate threshold is out of range, when a certificate residual exceeds the threshold, or when the dispersion root disagrees with the trace-formula Bloch wavenumber.
    """
    return
```
