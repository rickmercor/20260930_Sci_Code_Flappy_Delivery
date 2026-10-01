# Physics-Optics-36

## Background

Optical coherence tomography probes the depth-dependent return of light from scattering material. A focused illumination and collection geometry makes detection depend on the position and propagation history of the returning light. Numerical photon transport can represent multiple scattering, but many launched packets contribute little to the collected signal, making statistical efficiency an important practical issue.

## Problem

Consider detector-side replay of photon transport in homogeneous tissue using a Monte Carlo OCT detection model that replaces the photon's path after its last scattering event with a spherical-wave source coupled to a Gaussian single-mode fiber. Use the model's paraxial power-transfer expression and its flux, attenuation and optical-path corrections, with beam waist radius $w_0=0.03$ mm, Rayleigh range $z_R=0.5$ mm, focus depth $z_f=0.6$ mm, absorption $\mu_a=0.15$ mm$^{-1}$, scattering $\mu_s=3.0$ mm$^{-1}$ and refractive index $n=1.33$, and accept only outward packets whose internal incidence angle is strictly below $50$ degrees (unit launch weight, no additional Fresnel factor). The sample surface is $z=0$, positive $z$ points into the sample, and each row below specifies exit coordinates $(x_e,y_e)$, unit direction $(v_x,v_y,v_z)$ toward that surface, distance $s$ from the last scattering site to the exit, total physical path $L$ inside the tissue, and the unconditional probability $p$ of that outcome per launch; all lengths are in mm, these histories include all transport scattering probabilities, and the remaining probability is a zero-signal outcome.

| $x_e$ | $y_e$ | $v_x$ | $v_y$ | $v_z$ | $s$ | $L$ | $p$ |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.0 | 0.0 | 0.0 | -1.0 | 0.2 | 0.42 | 2/64 |
| 0.15 | 0.02 | 0.6 | 0.0 | -0.8 | 0.25 | 0.49 | 3/64 |
| -0.13 | 0.03 | -0.6 | 0.0 | -0.8 | 0.35 | 0.84 | 2/64 |
| 0.02 | 0.17 | 0.0 | 0.6 | -0.8 | 0.4 | 0.97 | 4/64 |
| 0.22 | 0.0 | 0.6 | 0.0 | -0.8 | 0.5 | 1.18 | 3/64 |
| -0.24 | 0.01 | -0.6 | 0.0 | -0.8 | 0.5 | 1.27 | 2/64 |
| 0.03 | 0.34 | 0.0 | 0.6 | -0.8 | 0.65 | 1.68 | 3/64 |
| 0.43 | 0.04 | 0.6 | 0.0 | -0.8 | 0.7 | 1.84 | 2/64 |
| 0.015 | -0.02 | 0.0 | 0.0 | -1.0 | 0.8 | 1.85 | 4/64 |
| 0.55 | 0.0 | 0.8 | 0.0 | -0.6 | 0.7 | 1.7 | 3/64 |
| 0.02 | 0.01 | 0.0 | 0.0 | -1.0 | 0.1 | 2.4 | 2/64 |
| 0.02 | -0.01 | 0.0 | 0.0 | -1.0 | 0.15 | 0.3 | 2/64 |
| 0.24 | 0.26 | 0.48 | 0.64 | -0.6 | 0.4 | 1.0 | 3/64 |
| 0.01 | 0.38 | 0.0 | 0.6 | -0.8 | 0.6 | 1.5 | 2/64 |
| -0.35 | 0.0 | -0.6 | 0.0 | -0.8 | 0.6 | 1.61 | 2/64 |
| 0.0 | 0.0 | 0.0 | 0.0 | -1.0 | 0.6 | 1.2 | 4/64 |

Treat the tabulated population as exact and launches as independent; absorption is applied to each supplied full path, while the probabilities already account for scattering survival and must not receive another full-path scattering attenuation. Form the six incoherent OCT intensity bins with one-way optical resolution $l_c=0.25$ mm and centres $j l_c$ for $j=1,\ldots,6$, using the model's corrected optical path and the model's strict coherence gate (an outcome exactly on a bin boundary is excluded). Let $I_j(N)$ denote the summed detected intensity in bin $j$ for $N$ launches and let $m_j$ be its exact population mean per launch, so the noise-free reference at that launch count is $N m_j$. Use the normalized root mean squared deviation of the simulated profile from this reference, normalized by the mean reference intensity across all six bins, and determine the dimensionless RMS noise coefficient $\alpha_{\rm RMS}=\sqrt{N\,\mathbb{E}[\operatorname{NRMSD}(N)^2]}$, which is independent of $N$ for this population, to six significant figures.

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

recover_scatter_sites

Goal
----
Recover the final scattering positions from exiting photon histories.

```python
def recover_scatter_sites(records: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7), K >= 1. Columns are exit x, exit y, outward unit
        direction vx, vy, vz < 0, final segment length s >= 0, and total
        in-medium physical path L >= s. Lengths are in mm.

    Returns
    -------
    sites : np.ndarray
        Shape (K, 3), final scattering positions x, y, z in row order.
    """
    return
```

### Step 2

compute_mode_transfer

Goal
----
Compute spherical-wave power transfer to the Gaussian collection mode.

```python
def compute_mode_transfer(sites: "np.ndarray", waist: float, rayleigh: float, focus: float) -> "np.ndarray":
    """Parameters
    ----------
    sites : np.ndarray
        Shape (K, 3), last-scatter x, y, z in mm, with z >= 0.
    waist, rayleigh : float
        Positive Gaussian waist radius and Rayleigh range in mm.
    focus : float
        Focus depth in mm.

    Returns
    -------
    transfer : np.ndarray
        Shape (K,), dimensionless power-coupling coefficients in row order,
        using the selected model's paraxial expression, including its
        normalization. The expression is continuously extended to z = 0.
    """
    return
```

### Step 3

compute_detected_weights

Goal
----
Compute accepted hybrid detected packet intensities from transport histories and mode transfer.

```python
def compute_detected_weights(records: "np.ndarray", sites: "np.ndarray", transfer: "np.ndarray", absorption: float, scattering: float, angle_limit: float) -> "np.ndarray":
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7), columns exit x, exit y, vx, vy, vz < 0, final
        segment length s, and full in-medium path L; directions are unit
        vectors and lengths are in mm.
    sites : np.ndarray
        Shape (K, 3), corresponding last-scatter positions in mm.
    transfer : np.ndarray
        Shape (K,), nonnegative spherical-wave power coupling.
    absorption, scattering : float
        Nonnegative coefficients in inverse mm.
    angle_limit : float
        Internal incidence cutoff in degrees, strictly between 0 and 90.

    Returns
    -------
    detected : np.ndarray
        Shape (K,), hybrid detected intensities per unit launch energy.
        Packets on or above the angular cutoff have zero intensity.
        No extra Fresnel transmission is included.
    """
    return
```

### Step 4

compute_hybrid_opl

Goal
----
Compute round-trip optical pathlengths assigned to hybrid detected packets.

```python
def compute_hybrid_opl(records: "np.ndarray", sites: "np.ndarray", refractive_index: float) -> "np.ndarray":
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7): exit x, exit y, vx, vy, vz, final segment s,
        total physical path L. Lengths are in mm.
    sites : np.ndarray
        Shape (K, 3), last-scatter x, y, z in mm.
    refractive_index : float
        Positive constant refractive index of the sample.

    Returns
    -------
    opl : np.ndarray
        Shape (K,), corrected round-trip optical pathlengths in mm,
        in the same packet order.
    """
    return
```

### Step 5

compute_gated_responses

Goal
----
Compute each packet outcome’s incoherent contribution to each OCT optical-depth bin.

```python
def compute_gated_responses(opl: "np.ndarray", detected: "np.ndarray", resolution: float, bins: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    opl : np.ndarray
        Shape (K,), nonnegative corrected round-trip optical paths in mm.
    detected : np.ndarray
        Shape (K,), nonnegative detected intensities.
    resolution : float
        Positive one-way optical bin resolution in mm.
    bins : np.ndarray
        Shape (M,), M >= 1, strictly increasing nonnegative integer bin
        indices. The one-way optical centre of index j is j * resolution.

    Returns
    -------
    response : np.ndarray
        Shape (K, M), intensity contributed by each outcome to each bin,
        preserving both supplied orders. Outside-bin and exact-boundary
        contributions are zero. Intensities are not squared or coherently
        combined at this stage.
    """
    return
```

### Step 6

compute_noise_coefficient

Goal
----
Compute the exact RMS normalized-profile-noise coefficient of a finite packet population.

```python
def compute_noise_coefficient(responses: "np.ndarray", probabilities: "np.ndarray") -> float:
    """Parameters
    ----------
    responses : np.ndarray
        Shape (K, M), K,M >= 1, nonnegative per-launch outcome intensities.
    probabilities : np.ndarray
        Shape (K,), nonnegative unconditional outcome probabilities with
        sum at most one. Remaining probability contributes a zero vector.
        The population mean profile must have positive total intensity.

    Returns
    -------
    coefficient : float
        Dimensionless sqrt(N times expected squared NRMSD), where NRMSD
        is the root mean squared bin residual relative to the exact
        N-launch population profile divided by that profile's mean
        intensity across the M bins. No finite-reference noise is included.
    """
    return
```

### Step 7

compute_oct_noise

Goal
----
Compute the RMS noise coefficient of a hybrid particle-wave OCT detector replay.

```python
def compute_oct_noise(records: "np.ndarray", probabilities: "np.ndarray", waist: float, rayleigh: float, focus: float, absorption: float, scattering: float, refractive_index: float, angle_limit: float, resolution: float, bins: "np.ndarray") -> float:
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7), K >= 1, columns exit x, exit y, outward unit vx, vy,
        vz < 0, final segment s >= 0 and full in-medium physical path L >= s.
        All lengths are in mm, surface z = 0 and positive z is inward.
    probabilities : np.ndarray
        Shape (K,), unconditional nonnegative probabilities, sum <= 1;
        remaining probability has zero signal. These include scattering
        survival but exclude absorption, and launches have unit energy.
    waist, rayleigh : float
        Positive in-medium Gaussian waist radius and Rayleigh range in mm.
    focus : float
        In-medium focus depth in mm.
    absorption, scattering : float
        Nonnegative coefficients in inverse mm.
    refractive_index : float
        Positive homogeneous sample index.
    angle_limit : float
        Strict incidence cutoff in degrees, between 0 and 90; no additional
        Fresnel transmission is included.
    resolution : float
        Positive one-way optical bin resolution in mm.
    bins : np.ndarray
        Shape (M,), M >= 1, increasing nonnegative integer bin indices.
        Exact coherence-bin boundaries are excluded. The resulting
        population profile must have positive total intensity.

    Returns
    -------
    coefficient : float
        Dimensionless RMS noise coefficient against the exact population
        reference, as defined by compute_noise_coefficient.
    """
    return
```
