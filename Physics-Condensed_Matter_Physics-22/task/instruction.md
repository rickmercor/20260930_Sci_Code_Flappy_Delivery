# Physics-Condensed_Matter_Physics-22

## Background

Thermal transport in insulating solids is commonly described using vibrational modes and their lifetimes. At temperatures well below the Debye temperature, quantum effects become important, and classical or semiclassical approaches may no longer provide an adequate description of heat transport.

Path-integral Monte Carlo provides equilibrium information without requiring a perturbative treatment of anharmonicity. Heat transport can then be studied through imaginary-time correlations of the energy current together with Green--Kubo linear response. Recovering a real-frequency spectrum from imaginary-time correlation data is an analytic-continuation problem that becomes poorly conditioned when the data contain statistical uncertainty.

A physically constrained spectral reconstruction can reduce this ambiguity and provide access to the low-frequency response that controls thermal transport. This approach also allows transport behavior inferred from current correlations to be compared with predictions based on equilibrium phonon properties.

## Problem

At low temperature, the thermal conductivity of an insulating solid can be obtained from imaginary-time heat-current correlations without assuming that the equilibrium phonon lifetimes also determine the decay of the heat current. Consider a reduced isotropic system in dimensionless units with

$$
k_B=\hbar=V=1,\qquad \beta=1.5.
$$

The system contains four effective vibrational modes with frequencies

$$
\boldsymbol{\omega}=
\begin{pmatrix}
0.75 & 1.20 & 1.85 & 2.55
\end{pmatrix}.
$$

For one Cartesian direction, the harmonic heat-current coupling matrix is

$$
\boldsymbol{\nu}=
\begin{pmatrix}
1.20 & 0.55 & 0.20 & 0.10\\
0.55 & 1.00 & 0.50 & 0.18\\
0.20 & 0.50 & 0.78 & 0.40\\
0.10 & 0.18 & 0.40 & 0.58
\end{pmatrix}.
$$

The diagonal elements of this matrix are used as the mode velocities in the Peierls--Boltzmann relaxation-time calculation,

$$
\mathbf{v}=
\begin{pmatrix}
1.20 & 1.00 & 0.78 & 0.58
\end{pmatrix}.
$$

The equilibrium phonon linewidths are

$$
\boldsymbol{\Gamma}^{ph}=
\begin{pmatrix}
0.18 & 0.22 & 0.27 & 0.34
\end{pmatrix},
$$

with phonon lifetimes

$$
\tau_n^{ph}=\frac{1}{2\Gamma_n^{ph}}.
$$

The harmonic heat-current correlation is available at six imaginary times. The supplied correlation values and their standard errors are

$$
\begin{array}{c|c|c|c}
i & \tau_i/\beta & C_{xx}^{h}(\tau_i) & \sigma_i\\
\hline
1 & 0.04 & 1.514473717829 & 0.0018\\
2 & 0.10 & 1.462768236466 & 0.0016\\
3 & 0.18 & 1.416173007618 & 0.0014\\
4 & 0.28 & 1.376243100066 & 0.0012\\
5 & 0.39 & 1.353349579459 & 0.0011\\
6 & 0.50 & 1.345265301976 & 0.0011
\end{array}.
$$

Reconstruct the harmonic heat-current spectrum using its singular and regular spectral contributions and the default two-parameter spectral prior of the PIMC Green--Kubo reconstruction. Use one mode-independent transport damping rate $\Gamma^{tr}$, with

$$
\Gamma_n^{tr}=\Gamma^{tr},\qquad \Gamma_{nm}=2\Gamma^{tr},
$$

and determine the global minimum of

$$
\chi^2(\Gamma^{tr},\xi)=\sum_{i=1}^{6}\left[\frac{C_{\mathrm{model}}^{h}(\tau_i;\Gamma^{tr},\xi)-C_{xx}^{h}(\tau_i)}{\sigma_i}\right]^2
$$

within

$$
0.03\leq\Gamma^{tr}\leq0.30,\qquad 0.5\leq\xi\leq2.5.
$$

Using the fitted spectrum, calculate the isotropic thermal conductivity from its zero-frequency limit. Independently calculate the Peierls--Boltzmann relaxation-time conductivity using the equilibrium phonon linewidths, and return the ratio

$$
R_{\kappa}=\frac{\kappa_{\mathrm{rec}}}{\kappa_{\mathrm{PB-RTA}}}.
$$

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_compute_modal_properties.py

Goal
----
Compute the quantum harmonic heat capacities and equilibrium phonon lifetimes for the supplied vibrational modes. These modal properties provide the equilibrium-phonon quantities needed later for the Peierls--Boltzmann relaxation-time conductivity.

```python
def compute_modal_properties(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
) -> "np.ndarray":
    """Compute modal heat capacities and equilibrium phonon lifetimes.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive vibrational frequencies.
    beta : float
        Positive inverse temperature in units with k_B = hbar = 1.
    gamma_ph : np.ndarray
        One-dimensional array of positive equilibrium phonon linewidths
        with the same length as omega.

    Returns
    -------
    properties : np.ndarray
        Array of shape (2, N). Row 0 contains the modal heat capacities
        and row 1 contains the equilibrium phonon lifetimes.
    """
    return properties
```

### Step 2

02_construct_spectral_components.py

Goal
----
Construct the pairwise coefficients, resonance centers, and transport broadenings that define the singular and regular harmonic heat-current spectral contributions.

```python
def construct_spectral_components(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
) -> "np.ndarray":
    """Construct pairwise terms defining the harmonic heat-current spectrum.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix of shape (N, N).
    beta : float
        Positive inverse temperature.
    gamma_tr : float
        Positive mode-independent transport damping rate.

    Returns
    -------
    components : np.ndarray
        Array of shape (5, N, N). The slices contain the singular
        coefficients, singular centers, regular coefficients, regular
        centers, and pair broadenings, respectively.
    """
    return components
```

### Step 3

03_transform_to_imaginary_time.py

Goal
----
Transform the singular and regular harmonic spectral contributions into their corresponding imaginary-time heat-current correlation components.

```python
def transform_to_imaginary_time(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    tau: "np.ndarray",
) -> "np.ndarray":
    """Transform the spectral components to imaginary-time correlations.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    gamma_tr : float
        Positive mode-independent transport damping rate.
    tau : np.ndarray
        One-dimensional array of imaginary times satisfying
        0 < tau_i < beta.

    Returns
    -------
    correlation : np.ndarray
        Array of shape (2, M). Row 0 contains the singular imaginary-time
        contribution and row 1 contains the unscaled regular contribution.
    """
    return correlation
```

### Step 4

04_fit_transport_reconstruction.py

Goal
----
Infer the mode-independent transport damping rate and regular-spectrum scale by minimizing the weighted mismatch between the reconstructed and supplied imaginary-time heat-current correlations over the specified parameter bounds.

```python
def fit_transport_reconstruction(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    tau: "np.ndarray",
    c_obs: "np.ndarray",
    sigma: "np.ndarray",
    gamma_bounds: tuple[float, float],
    xi_bounds: tuple[float, float],
) -> "np.ndarray":
    """Fit the two-parameter transport spectral reconstruction.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    tau : np.ndarray
        Imaginary times satisfying 0 < tau_i < beta.
    c_obs : np.ndarray
        Supplied imaginary-time heat-current correlation values.
    sigma : np.ndarray
        Positive standard errors corresponding to c_obs.
    gamma_bounds : tuple[float, float]
        Lower and upper bounds for Gamma_tr.
    xi_bounds : tuple[float, float]
        Lower and upper bounds for xi.

    Returns
    -------
    fit : np.ndarray
        Length-3 array containing fitted Gamma_tr, fitted xi,
        and the minimum weighted chi-square.

    Notes
    -----
    The supported input domain contains a nonzero regular spectral
    contribution over the fitted imaginary-time points.
    """
    return fit
```

### Step 5

05_evaluate_zero_frequency_spectrum.py

Goal
----
Evaluate the singular, regular, and total fitted heat-current spectral density at zero frequency and obtain the transport lifetime associated with the fitted damping rate.

```python
def evaluate_zero_frequency_spectrum(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    gamma_tr: float,
    xi: float,
) -> "np.ndarray":
    """Evaluate the fitted heat-current spectrum at zero frequency.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    gamma_tr : float
        Positive fitted transport damping rate.
    xi : float
        Scale applied to the regular spectral contribution.

    Returns
    -------
    result : np.ndarray
        Length-4 array containing the transport lifetime,
        singular zero-frequency spectrum, unscaled regular
        zero-frequency spectrum, and fitted total zero-frequency spectrum.
    """
    return result
```

### Step 6

06_compute_reconstructed_conductivity.py

Goal
----
Convert the fitted zero-frequency heat-current spectral density into the reconstructed isotropic thermal conductivity.

```python
def compute_reconstructed_conductivity(
    beta: float,
    lambda_zero: float,
) -> float:
    """Compute the reconstructed isotropic thermal conductivity.

    Parameters
    ----------
    beta : float
        Positive inverse temperature.
    lambda_zero : float
        Fitted total zero-frequency heat-current spectral density.

    Returns
    -------
    kappa_rec : float
        Reconstructed thermal conductivity in the dimensionless
        units of the task.
    """
    return kappa_rec
```

### Step 7

07_compute_pb_rta_conductivity.py

Goal
----
Compute the independent Peierls--Boltzmann relaxation-time thermal conductivity from the modal quantum heat capacities, supplied mode velocities, and equilibrium phonon lifetimes.

```python
def compute_pb_rta_conductivity(
    omega: "np.ndarray",
    beta: float,
    gamma_ph: "np.ndarray",
    velocity: "np.ndarray",
) -> float:
    """Compute Peierls--Boltzmann relaxation-time conductivity.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    beta : float
        Positive inverse temperature.
    gamma_ph : np.ndarray
        Positive equilibrium phonon linewidths.
    velocity : np.ndarray
        Mode velocities with the same length as omega.

    Returns
    -------
    kappa_pb : float
        Peierls--Boltzmann relaxation-time conductivity.
    """
    return kappa_pb
```

### Step 8

08_run_transport_reconstruction_pipeline.py

Goal
----
Run the complete transport reconstruction, evaluate the fitted zero-frequency Green--Kubo conductivity, independently evaluate the Peierls--Boltzmann relaxation-time conductivity, and return their ratio.

```python
def run_transport_reconstruction_pipeline(
    omega: "np.ndarray",
    nu: "np.ndarray",
    beta: float,
    tau: "np.ndarray",
    c_obs: "np.ndarray",
    sigma: "np.ndarray",
    gamma_ph: "np.ndarray",
    velocity: "np.ndarray",
    gamma_bounds: tuple[float, float],
    xi_bounds: tuple[float, float],
) -> float:
    """Run the complete reconstruction and return the conductivity ratio.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of positive mode frequencies.
    nu : np.ndarray
        Square heat-current coupling matrix.
    beta : float
        Positive inverse temperature.
    tau : np.ndarray
        Imaginary-time sampling points satisfying 0 < tau_i < beta.
    c_obs : np.ndarray
        Supplied imaginary-time heat-current correlation values.
    sigma : np.ndarray
        Positive standard errors corresponding to c_obs.
    gamma_ph : np.ndarray
        Positive equilibrium phonon linewidths.
    velocity : np.ndarray
        Mode velocities.
    gamma_bounds : tuple[float, float]
        Bounds for Gamma_tr.
    xi_bounds : tuple[float, float]
        Bounds for xi.

    Returns
    -------
    ratio : float
        Ratio of reconstructed conductivity to PB-RTA conductivity.
    """
    return ratio
```
