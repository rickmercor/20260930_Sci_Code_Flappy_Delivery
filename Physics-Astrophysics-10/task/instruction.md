# Physics-Astrophysics-10

## Background

Small-body searches often use repeated images to gain depth below the single-exposure detection limit, but a distant object is not fixed on a detector when the observatory itself is moving. In a finite exposure, the observer's displacement changes the apparent position across the integration, while rotational variability and the object's spectral response change the expected source flux. A physically specified orbit turns the topocentric midpoint and endpoint directions into a barycentric path and exposure displacement, allowing information from heterogeneous observations to be accumulated on one reference canvas.

For point sources with independent Gaussian pixel errors, correlation with the exposure-averaged source profile produces a flux statistic and its precision. Those quantities can be registered and combined with the source's expected response in each calibrated band, without discarding the information carried by different seeing, noise levels, exposure displacements, brightness states, and filters. Injection and recovery experiments then convert the end-to-end detection behavior into a selection function, commonly summarized by the magnitude at which the fitted recovery probability is one half. Because objects of the same magnitude differ in color, rotation state, and how closely their motion follows the trajectory actually searched, that selection function is a smooth curve whose width carries as much information as its midpoint.

## Problem

## Setup

Residual low-Earth-orbit parallax can move a distant minor planet by several detector pixels even after ordinary linear pointing drift has been removed, so a controlled sequence of faint-source difference stamps is used to estimate the depth reached along a single searched trajectory.

## Inputs

The solver-visible records specify six finite-duration exposures with heterogeneous seeing, variance, and calibrated band codes, residual barycentric observer positions and velocities across 2.03 days, and twenty-four seeded injected sources spanning 26.80 to 29.41 magnitude whose canvas offsets, residual pixel drifts, rotational brightness parameters, and red-to-blue flux ratios all differ from one another.

| Quantity | Value |
|:---|:---:|
| Reference distance | 42.60 AU |
| Angular pixel scale | 0.040 arcsec/pixel |
| Solar gravitational parameter | $2.959122082855911\times10^{-4}$ AU$^3$/day$^2$ |
| Speed of light | 173.144632674240 AU/day |
| Flux at magnitude 27 | 455 common flux units |
| Rotation period | 0.73 day |
| Linear WCS rate | $(502.65520879,-2.27302087)$ pixel/day |

The graded orbit is specified by longitude, latitude, distance, radial speed, tangential speed, inclination, and branch

$$
(\phi,\theta,r,v_r,v_\Omega,i,\kappa)=(0.640,0.061,42.60,-0.00014,0.00418,0.0614,-1),
$$

with angles in radians, distance in AU, speeds in AU/day, and signed spherical branch angle $\psi=\kappa\arccos(\cos i/\cos\theta)$.

The exposure rows are ordered as $(t,x_{obs},y_{obs},z_{obs},v_x,v_y,v_z,s,V,\Delta t,b)$, with positions in AU, velocities in AU/day, PSF width $s$ in pixels, variance $V$, duration in days, and blue/red band code $b=0/1$:

| $t$ | $x_{obs}$ | $y_{obs}$ | $z_{obs}$ | $v_x$ | $v_y$ | $v_z$ | $s$ | $V$ | $\Delta t$ | $b$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 4.54e-5 | 0.00e-5 | 1.10e-5 | 0.00015 | 0.00425 | 0.00020 | 0.76 | 92.0 | 0.0045 | 1 |
| 0.31 | 0.72e-5 | 4.48e-5 | -1.25e-5 | -0.00420 | 0.00060 | -0.00020 | 0.93 | 118.0 | 0.0052 | 1 |
| 0.69 | -4.22e-5 | 1.45e-5 | 0.95e-5 | -0.00120 | -0.00410 | 0.00015 | 1.12 | 86.0 | 0.0048 | 0 |
| 1.08 | -1.85e-5 | -4.17e-5 | -0.82e-5 | 0.00390 | -0.00170 | -0.00020 | 0.84 | 141.0 | 0.0054 | 0 |
| 1.56 | 4.10e-5 | -1.72e-5 | 1.31e-5 | 0.00170 | 0.00400 | 0.00025 | 1.04 | 104.0 | 0.0046 | 0 |
| 2.03 | 1.13e-5 | 4.39e-5 | -1.04e-5 | -0.00410 | 0.00100 | -0.00010 | 0.88 | 127.0 | 0.0050 | 1 |

The injected-source rows are ordered as $(m,seed,x_0,y_0,d_x,d_y,A,\varphi,c)$, where $(d_x,d_y)$ is residual drift in pixels/day, $A$ and $\varphi$ define the stated 0.73-day rotational modulation, and $c$ is the red-to-blue flux ratio:

| $m$ | seed | $x_0$ | $y_0$ | $d_x$ | $d_y$ | $A$ | $\varphi$ | $c$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 26.80 | 101 | -1.60 | -1.10 | -0.045 | 0.030 | 0.30 | 0.1500000000 | 1.20 |
| 26.99 | 211 | 1.35 | -1.45 | 0.310 | -0.240 | 0.35 | 1.9300000000 | 0.85 |
| 27.12 | 307 | -0.55 | 1.72 | 0.060 | 0.020 | 0.45 | 3.4200000000 | 1.05 |
| 27.20 | 419 | 1.78 | 0.54 | 2.900 | -2.200 | 0.30 | 4.7500000000 | 1.60 |
| 27.31 | 523 | -1.88 | 0.36 | -0.030 | 0.055 | 0.90 | 5.6200000000 | 0.62 |
| 27.44 | 631 | 0.70 | -1.84 | -0.520 | -0.180 | 0.15 | 2.5300000000 | 1.80 |
| 27.55 | 743 | -0.18 | 0.84 | -2.600 | 2.500 | 0.40 | 0.8700000000 | 0.95 |
| 27.66 | 463 | 1.07 | 1.18 | 0.420 | -0.310 | 0.55 | 3.8800000000 | 0.75 |
| 27.79 | 967 | -1.22 | -0.31 | -0.061 | 0.011 | 0.85 | 5.1000000000 | 2.30 |
| 27.90 | 189 | 0.31 | -0.68 | 0.015 | -0.063 | 0.25 | 1.3200000000 | 0.55 |
| 28.02 | 1193 | 1.63 | -0.14 | 0.570 | -0.270 | 0.35 | 1.9634954085 | 1.45 |
| 28.08 | 1301 | -0.91 | 1.39 | 3.100 | 1.900 | 0.35 | 4.4505895926 | 1.75 |
| 28.13 | 1427 | 0.92 | 0.25 | -1.450 | 0.920 | 0.70 | 0.4800000000 | 0.65 |
| 28.18 | 1543 | -1.47 | -1.52 | 0.048 | -0.035 | 0.20 | 2.1800000000 | 2.30 |
| 28.33 | 1657 | 0.08 | 1.51 | -0.370 | 0.520 | 0.45 | 3.5700000000 | 0.90 |
| 28.50 | 1777 | 1.47 | -0.83 | -1.180 | 0.740 | 0.30 | 5.9100000000 | 1.10 |
| 28.60 | 1889 | -0.73 | 0.05 | 1.620 | 1.050 | 0.60 | 1.0500000000 | 2.20 |
| 28.71 | 1999 | 0.53 | -1.18 | -0.026 | 0.044 | 0.20 | 4.9400000000 | 0.80 |
| 28.83 | 2111 | -1.31 | 0.97 | 0.510 | -0.310 | 0.95 | 2.7100000000 | 1.70 |
| 28.96 | 2237 | 1.16 | -0.42 | 0.019 | 0.058 | 0.40 | 0.6200000000 | 0.60 |
| 29.10 | 2351 | -0.34 | -1.66 | -1.100 | -0.750 | 0.25 | 3.1500000000 | 1.30 |
| 28.26 | 2477 | 1.72 | 0.68 | 0.028 | -0.022 | 0.15 | 5.3800000000 | 2.40 |
| 29.25 | 2593 | -1.05 | 1.24 | 0.310 | 0.500 | 0.75 | 1.4700000000 | 2.00 |
| 29.41 | 2707 | 0.44 | -0.95 | -0.017 | -0.045 | 0.30 | 4.1300000000 | 0.70 |

## Physical model

Build the signed two-body state and retarded moving-observer midpoint and endpoint directions, register the resulting finite-exposure path relative to the stated WCS drift, form each exposure's likelihood statistic from its own exposure-averaged sampled PSF, calibrated source response, variance, and deterministic seeded independent Gaussian pixel noise, transport numerator and precision separately to a common canvas, use the retained coadd peak only as the location for robust per-exposure forced photometry, compare that remeasured significance with the paper's empirical purity floor, and fit the graded recovery produced by the sources' differing color, rotation, and residual drift.

## Task

Use the complete coupled calculation to fit the binary recovery curve and return **the 50-percent-completeness magnitude $m_{50}$**. In the brief reasoning, report the source-calibrated retention significance, the number of recovered injections, and the fitted coefficient pair $(\beta_0,\beta_1)$ that determine the final value.

## Numerical conventions

Use the following deterministic conventions.

| Stage | Contract |
|:---|:---|
| Propagation | Classical fourth-order Runge-Kutta; $N=\max(1,\lceil|\tau|/0.01\rceil)$ equal substeps; reference-state geometric light-time initialization; five fixed-point updates |
| Observer and path | Linear observer advance to midpoint and endpoints; stated WCS line subtracted; trail computed directly as apparent stop minus apparent start minus $\Delta t$ times the WCS rate; path values rounded to $10^{-6}$ pixel |
| Exposure nodes | Seven equally spaced fractions from $-0.5$ to $0.5$; total reference flux $F=455\,10^{-0.4(m-27)}$; $S_f=10^{-0.4A\sin[2\pi(t+f\Delta t)/0.73+\varphi]}$ and multiplied by $c$ for red exposures |
| PSF and profile | Normalized circular Gaussian $P_f=\exp[-(x^2+y^2)/(2s^2)]/(2\pi s^2)$; finite-exposure source profile is the arithmetic mean, not the sum, of the seven $F S_f P_f$ profiles |
| Stamp and noise | $25\times25$ stamp centered at $(12,12)$; response-weighted unit-sum $11\times11$ correlation PSF; no correlated residual; independent `np.random.default_rng(int(seed + 1009*j))` Gaussian array with standard deviation $\sqrt V$ and shape `(25, 25)` for exposure index $j$ |
| Registration | $w_{kj}$ is the arithmetic mean of the seven $S_f$ values; zero-padded bilinear sampling; numerator weight $w_{kj}$; precision weight $w_{kj}^2$; first row-major maximum in the $23\times23$ interior |
| Forced photometry | Canonical per-exposure flux $f_j$ and its own error $\sigma_j$ at the retained location; begin with every positive-precision measurement; at most ten passes; arithmetic mean $\bar f$ over retained values; retain exactly $|f_j-\bar f|\leq3\sigma_j$; stop on an unchanged set; an empty proposal leaves the preceding nonempty set |
| Recovery and fit | Arithmetic-mean significance compared with the paper's remeasured purity floor; $p(m)=\operatorname{expit}[\beta_0+\beta_1(m-28.7)]$; start $(0,-2.2)$; at most 80 damped Newton steps; predictors clipped to $[-40,40]$; Hessian ridge $10^{-8}I$; loss-decreasing step halving; scaled-step tolerance $10^{-11}$; require $\beta_1<-10^{-6}$; return $m_{50}=28.7-\beta_0/\beta_1$ in magnitudes |

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.
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

observation plan

Goal
----
Deterministic exposure and injection tables for the compact search.

```python
"""Deterministic exposure and injection tables for the compact search."""

import numpy as np


def observation_plan() -> tuple[np.ndarray, np.ndarray]:
    """Return the exposure table and injected-source table for the search.

    Returns
    -------
    tuple of numpy.ndarray
        ``(exposures, injections)`` with numerical columns documented in the
        solver-visible step specification.
    """
    return ()
```

### Step 2

spherical orbit state

Goal
----
Map spherical orbit coordinates to a Cartesian Kepler state.

```python
"""Map spherical orbit coordinates to a Cartesian Kepler state."""

import numpy as np


def spherical_orbit_state(phi: float, theta: float, distance_au: float, radial_speed_au_day: float, tangential_speed_au_day: float, inclination_rad: float, kappa: int) -> np.ndarray:
    """Return the Cartesian state implied by a signed spherical orbit.

    Parameters are longitude, latitude, distance, radial and tangential speed,
    inclination, and the ascending/descending branch sign ``kappa``.
    """
    return np.array([], dtype=float)
```

### Step 3

apparent registration path

Goal
----
Light-time-corrected apparent offsets along a two-body orbit.

```python
"""Light-time-corrected apparent offsets along a two-body orbit."""

import numpy as np

_MU = 2.959122082855911e-4


def apparent_registration_path(state: np.ndarray, exposures: np.ndarray, pixel_scale_arcsec: float, wcs_rate_px_day: np.ndarray) -> np.ndarray:
    """Return paths rounded to 1e-6 pixel; reject invalid or non-finite inputs."""
    return np.empty((0, 4), dtype=float)


def _acceleration(position: np.ndarray) -> np.ndarray:
    radius = float(np.linalg.norm(position))
    return -_MU * position / radius**3


def _propagate_rk4(position: np.ndarray, velocity: np.ndarray, elapsed_day: float) -> np.ndarray:
    """Propagate a Kepler state with fixed, short RK4 substeps."""
    count = max(1, int(np.ceil(abs(elapsed_day) / 0.01)))
    h = elapsed_day / count
    r = position.astype(float).copy()
    v = velocity.astype(float).copy()
    for _ in range(count):
        k1r, k1v = v, _acceleration(r)
        k2r, k2v = v + 0.5 * h * k1v, _acceleration(r + 0.5 * h * k1r)
        k3r, k3v = v + 0.5 * h * k2v, _acceleration(r + 0.5 * h * k2r)
        k4r, k4v = v + h * k3v, _acceleration(r + h * k3r)
        r += h * (k1r + 2 * k2r + 2 * k3r + k4r) / 6
        v += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    return r
```

### Step 4

matched filter statistics

Goal
----
Construct per-exposure PSF-matched numerator and precision maps.

```python
"""Construct per-exposure PSF-matched numerator and precision maps."""

import numpy as np

_ROTATION_PERIOD_DAY = 0.73
_FRACTIONS = np.linspace(-0.5, 0.5, 7)


def matched_filter_statistics(path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return matched maps; reject non-finite inputs, nonpositive seeing, or variance."""
    return ()


def _gaussian_psf(sigma: float) -> np.ndarray:
    axis = np.arange(-5, 6, dtype=float)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def _relative_flux(amplitude_mag: float, phase_rad: float, sample_times_day: np.ndarray) -> np.ndarray:
    """Return flux factors for a sinusoidal magnitude light curve."""
    return 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * sample_times_day / _ROTATION_PERIOD_DAY + phase_rad))


def _relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, band_code: float, sample_times_day: np.ndarray) -> np.ndarray:
    """Return the rotating source response in the specified calibrated band."""
    if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
        raise ValueError("invalid source color or band code")
    return _relative_flux(amplitude_mag, phase_rad, sample_times_day) * (red_blue_flux_ratio if band_code == 1.0 else 1.0)


def _trailed_psf(sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
    """Return a flux-weighted, unit-sum finite-exposure template."""
    axis = np.arange(-5, 6, dtype=float)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    template = np.zeros_like(xx)
    for fraction, weight in zip(_FRACTIONS, relative_flux):
        template += weight * np.exp(-((xx - fraction * trail[0]) ** 2 + (yy - fraction * trail[1]) ** 2) / (2 * sigma**2))
    return template / template.sum()


def _source_profile(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float, trail: np.ndarray, relative_flux: np.ndarray) -> np.ndarray:
    """Return the light-curve-weighted continuous source profile on a stamp."""
    profile = np.zeros_like(xx)
    for fraction, weight in zip(_FRACTIONS, relative_flux):
        profile += weight * np.exp(-((xx - x0 - fraction * trail[0]) ** 2 + (yy - y0 - fraction * trail[1]) ** 2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return profile / 7.0


def _correlate_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode="constant")
    output = np.empty_like(image, dtype=float)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            output[y, x] = float(np.sum(padded[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel))
    return output
```

### Step 5

registered coadd

Goal
----
Register matched statistics on a barycentric canvas and coadd them.

```python
"""Register matched statistics on a barycentric canvas and coadd them."""

import numpy as np


def registered_coadd(xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray) -> np.ndarray:
    """Return source-response-weighted significance maps; reject non-finite inputs."""
    return np.empty((0, 0, 0), dtype=float)


def _bilinear(image: np.ndarray, y: float, x: float) -> float:
    if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
        return 0.0
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
    fy, fx = y - y0, x - x0
    return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))


def _mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
    """Return the seven-node source response in the exposure's calibrated band."""
    if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
        raise ValueError("invalid source color or band code")
    fractions = np.linspace(-0.5, 0.5, 7)
    samples = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
    return float(np.mean(samples) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))
```

### Step 6

deduplicated candidates

Goal
----
Select one local likelihood maximum per injected trajectory.

```python
"""Select one local likelihood maximum per injected trajectory."""

import numpy as np


def deduplicated_candidates(significance_maps: np.ndarray) -> np.ndarray:
    """Return one ``(x, y, significance)`` candidate per coadded map."""
    return np.empty((0, 3), dtype=float)
```

### Step 7

injection recoveries

Goal
----
Measure robust forced fluxes at the retained candidate positions.

```python
"""Measure robust forced fluxes at the retained candidate positions."""

import numpy as np


def injection_recoveries(candidates: np.ndarray, xi: np.ndarray, zeta: np.ndarray, path: np.ndarray, exposures: np.ndarray, injections: np.ndarray, threshold: float) -> np.ndarray:
    """Return robust recovery flags; reject non-finite inputs or nonpositive threshold."""
    return np.empty((0, 2), dtype=float)


def _bilinear(image: np.ndarray, y: float, x: float) -> float:
    """Return a zero-padded bilinear sample from one statistic map."""
    if y < 0 or x < 0 or y > image.shape[0] - 1 or x > image.shape[1] - 1:
        return 0.0
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, image.shape[0] - 1), min(x0 + 1, image.shape[1] - 1)
    fy, fx = y - y0, x - x0
    return float((1 - fy) * ((1 - fx) * image[y0, x0] + fx * image[y0, x1]) + fy * ((1 - fx) * image[y1, x0] + fx * image[y1, x1]))


def _mean_relative_response(amplitude_mag: float, phase_rad: float, red_blue_flux_ratio: float, midpoint_day: float, duration_day: float, band_code: float) -> float:
    """Return the canonical-flux response for one calibrated exposure."""
    if red_blue_flux_ratio <= 0 or band_code not in (0.0, 1.0):
        raise ValueError("invalid source color or band code")
    fractions = np.linspace(-0.5, 0.5, 7)
    rotation = 10.0 ** (-0.4 * amplitude_mag * np.sin(2.0 * np.pi * (midpoint_day + fractions * duration_day) / 0.73 + phase_rad))
    return float(np.mean(rotation) * (red_blue_flux_ratio if band_code == 1.0 else 1.0))
```

### Step 8

logistic completeness fit

Goal
----
Fit the unbinned logistic recovery efficiency curve.

```python
"""Fit the unbinned logistic recovery efficiency curve."""

import numpy as np


def logistic_completeness_fit(recoveries: np.ndarray) -> float:
    """Return the fitted 50-percent-completeness magnitude from binary recoveries.

    Stop when the accepted scaled Newton step has norm below 1e-11.
    """
    return float("nan")
```

### Step 9

completeness magnitude

Goal
----
Run the complete deterministic pipeline and return the 50-percent-completeness magnitude.

```python
"""Final end-to-end completeness calculation."""

import numpy as np


def completeness_magnitude(inclination_rad: float = 0.0614, branch_sign: int = -1) -> float:
    """Return the fitted 50-percent recovery magnitude for the stated search.

    The forced-photometry stage is evaluated at the retention significance
    fixed by the registered source. That significance is a fixed calibration
    of the search rather than a configurable argument, so it is supplied
    internally and the same value is used for every call.

    Parameters
    ----------
    inclination_rad : float, optional
        Orbit inclination in radians used to build the signed spherical state.
    branch_sign : int, optional
        Ascending or descending tangential branch sign, either ``-1`` or ``+1``.

    Returns
    -------
    float
        The fitted 50-percent-completeness magnitude.
    """
    return float("nan")
```
