# Physics-Astrophysics-30

## Background

The extended solar corona is heated well above the temperature that thermal
conduction from the base can supply, and the leading candidate mechanism is
the dissipation of Alfvenic fluctuations launched from the photosphere. Those
fluctuations propagate outward along open magnetic field lines into the solar
wind. Because a purely outward-propagating Alfvenic population is an exact
nonlinear solution, some process must generate a counter-propagating component
before a turbulent cascade, and hence heating, can occur; in the open corona
that process is the partial reflection of the outward fluctuations off the
inhomogeneity of the background medium.

Open-field regions in the corona are not smooth tubes. Coronal holes contain
plumes, interplume lanes, and the funnel geometry that carries the field from
supergranular boundaries into the open corona, so a field line sits in a
background that varies across the field as well as along its own length.

The three model regions in this task are analytically constructed
divergence-free fields: two are structured across the field and the third has
no structure beyond its axisymmetric expansion. They share a single plasma
background - a radially falling density and an accelerating outflow - so the
field geometry is the only thing that changes from one region to the next.

Astrophysical units are used throughout: lengths in solar radii, magnetic
field in gauss, density in g cm^-3, speeds in km/s where stated, and the
resulting volumetric heating rate in erg cm^-3 s^-1.

## Problem

Alfvenic fluctuations launched at the coronal base of open magnetic field lines are a leading candidate for heating the extended solar corona: travelling outward through an inhomogeneous background they are partly reflected, and the resulting counter-propagating pair cascades and dissipates where it is produced. Treat the fluctuations with the transport theory of reflection-driven Alfvenic turbulence generalised to arbitrary (non-axisymmetric) field geometry, in which the transverse deformation of the field direction contributes to the reflection rate alongside the along-line Alfven-speed gradient, and take the heating rate to be the outward-fluctuation energy dissipated at the total reflection rate. The target uses the theory's finite-flow closure consistently for both fluctuation transport and heating. Your task is to compute that heating along three model open coronal regions and report a single number: the base-10 logarithm of the largest end-of-line volumetric heating rate (in erg cm^-3 s^-1) among the three field lines.

Each region is a divergence-free magnetic field in Sun-centred Cartesian coordinates (units of R_sun = 6.957e10 cm, field in gauss): with u3 = z - 1 the height above the footpoint (0, 0, 1) and u_perp = (x, y), the field is B_perp = M(u3) u_perp and B_z = B3(u3), where B3(u3) = B0 (1 + u3/hB)^(-aB) and M(u3) = (t(u3)/2) I + [[p(u3), q(u3)], [q(u3), -p(u3)]], with the trace t(u3) fixed by requiring div B = 0 exactly and p, q the Gaussian profiles p(u3) = p0 exp(-((u3-up)/wp)^2), q(u3) = q0 exp(-((u3-uq)/wq)^2). The plasma background is shared by all regions: mass density rho(r) = 2e-16 (r/1)^(-4) g cm^-3 and outflow speed U(r) = 10 + 640 (1 - exp(-(r-1)/3)) km/s with r the heliocentric distance in R_sun, and the rms outward Elsasser amplitude at each footpoint is 30 km/s.

Follow each region's field line from its footpoint for an arclength of 8 R_sun, sampled at 2001 equally spaced stations, measuring all field gradients by central differences with half-step 2e-5 R_sun; each region's heating rate is evaluated at the final station. The shared axial parameters are B0 = 3.0 gauss, hB = 0.8 R_sun and aB = 1.9 (dimensionless); the transverse amplitudes p0 and q0 are in gauss per R_sun and the offsets and widths up, wp, uq, wq in R_sun:

* Region 1: p0 = 0.06, up = 8.0, wp = 1.2, q0 = 0.03, uq = 7.0, wq = 1.5
* Region 2: p0 = 0.12, up = 3.0, wp = 1.0, q0 = 0.07, uq = 3.5, wq = 1.0
* Region 3: p0 = 0 and q0 = 0, so the transverse block is purely the trace part (up, wp, uq, wq are then immaterial; take wp = wq = 1.0)

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

evaluate_field

Goal
----
Implement evaluate_field, the model coronal magnetic field of one open region.

```python
import numpy as np

def evaluate_field(points: np.ndarray, region_params: dict) -> np.ndarray:
    '''Evaluate the divergence-free model region field in gauss.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) Cartesian positions in units of R_sun.
    region_params : dict
        Keys "B0", "hB", "aB", "z_foot", "p0", "up", "wp", "q0", "uq", "wq",
        all floats. B0 in gauss, p0 and q0 in gauss per R_sun, the length
        scales in R_sun. B0 and hB must be positive.

    Returns
    -------
    B : np.ndarray
        (N, 3) magnetic field in gauss at each input position.

    Raises
    ------
    ValueError
        If points does not have shape (N, 3), if region_params is missing any
        of the ten required keys, if B0 or hB is not positive, or if any point
        fails 1 + (z - z_foot) / hB > 0.
    '''
    return B  # placeholder
```

### Step 2

trace_field_line

Goal
----
Implement trace_field_line, which follows a field line of the region field

from a seed point for a fixed arclength.

```python
import numpy as np

def trace_field_line(region_params: dict, seed: np.ndarray, ell_max: float,
                     n_points: int) -> np.ndarray:
    '''Trace one field line with fixed-step RK4 (8 substeps per interval).

    Parameters
    ----------
    region_params : dict
        Field parameters accepted by evaluate_field (step 01).
    seed : np.ndarray
        (3,) starting position in R_sun.
    ell_max : float
        Positive finite arclength to follow, in R_sun.
    n_points : int
        Number of equally spaced stations, >= 2, including both endpoints.

    Returns
    -------
    positions : np.ndarray
        (n_points, 3) station positions in R_sun; positions[0] equals seed.

    Raises
    ------
    ValueError
        If seed does not have shape (3,), if ell_max is not a positive finite
        number, or if n_points is less than 2. A region_params dict rejected by
        step 01 propagates that ValueError.
    '''
    return positions  # placeholder
```

### Step 3

plasma_background

Goal
----
Implement plasma_background, the spherically symmetric plasma state of the

open corona evaluated along a field line.

```python
import numpy as np

def plasma_background(positions: np.ndarray, bg_params: dict) -> np.ndarray:
    '''Evaluate the plasma density and outflow speed at each position.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) Cartesian positions in units of R_sun.
    bg_params : dict
        Keys "rho0" (g cm^-3, > 0), "r0" (R_sun, > 0), "alpha_rho",
        "u0" (km/s), "uinf" (km/s), "lu" (R_sun, > 0).

    Returns
    -------
    out : np.ndarray
        (N, 2) array; column 0 is rho in g cm^-3, column 1 is U in cm/s.

    Raises
    ------
    ValueError
        If positions does not have shape (N, 3), if bg_params is missing any of
        the six required keys, if rho0, r0 or lu is not positive, or if any
        position has non-positive heliocentric distance.
    '''
    return out  # placeholder
```

### Step 4

alfven_speed_gradient

Goal
----
Implement alfven_speed_gradient: the Alfven speed profile along a line and

its logarithmic arclength gradient.

```python
import numpy as np

def alfven_speed_gradient(bmag: np.ndarray, rho: np.ndarray, ds_cm: float) -> np.ndarray:
    '''Alfven speed and its logarithmic arclength derivative.

    Parameters
    ----------
    bmag : np.ndarray
        (N,) magnetic field magnitude in gauss, N >= 3, all positive.
    rho : np.ndarray
        (N,) mass density in g cm^-3, same length, all positive.
    ds_cm : float
        Positive spacing between consecutive stations, in cm.

    Returns
    -------
    out : np.ndarray
        (N, 2) array; column 0 is v_A in cm/s, column 1 is K_vA in 1/cm.

    Raises
    ------
    ValueError
        If bmag and rho are not 1-D arrays of the same length, if they hold
        fewer than 3 stations, if any entry of bmag or rho is not positive, or
        if ds_cm is not a positive finite number.
    '''
    return out  # placeholder
```

### Step 5

deformation_rate

Goal
----
Implement deformation_rate: the transverse deformation rate of the field

direction along a line.

```python
import numpy as np

def deformation_rate(region_params: dict, positions: np.ndarray,
                     h_fd: float) -> np.ndarray:
    '''Transverse deformation rate of the field direction at each station.

    Parameters
    ----------
    region_params : dict
        Field parameters accepted by evaluate_field (step 01).
    positions : np.ndarray
        (N, 3) station positions in units of R_sun.
    h_fd : float
        Positive central-differencing half-step in R_sun.

    Returns
    -------
    smag : np.ndarray
        (N,) non-negative deformation rate in 1/R_sun; independent of the
        choice of perpendicular basis.

    Raises
    ------
    ValueError
        If positions does not have shape (N, 3), or if h_fd is not a positive
        finite number. A region_params dict rejected by step 01 propagates that
        ValueError.
    '''
    return smag  # placeholder
```

### Step 6

wave_amplitude

Goal
----
Implement wave_amplitude: the rms amplitude of the outward Alfvenic

fluctuation along a line, in the presence of an inhomogeneous background

flow and reflection-driven damping.

```python
import numpy as np

def wave_amplitude(zp0_cms: float, va: np.ndarray, u: np.ndarray,
                   damping: np.ndarray, s_cm: np.ndarray) -> np.ndarray:
    '''Outward Elsasser amplitude at each station along the line.

    Parameters
    ----------
    zp0_cms : float
        Positive rms amplitude of the outward fluctuation at the first
        station, in cm/s.
    va : np.ndarray
        (N,) Alfven speed in cm/s, all positive.
    u : np.ndarray
        (N,) outflow speed in cm/s, all positive.
    damping : np.ndarray
        (N,) damping rate in 1/cm.
    s_cm : np.ndarray
        (N,) strictly increasing arclength in cm with s_cm[0] = 0.

    Returns
    -------
    zp : np.ndarray
        (N,) rms outward amplitude in cm/s.

    Raises
    ------
    ValueError
        If zp0_cms is not a positive finite number, if va, u, damping and s_cm
        are not all 1-D arrays, if they do not all have the same length, if any
        entry of va or u is not positive, or if s_cm does not start at 0 and
        increase strictly.
    '''
    return zp  # placeholder
```

### Step 7

heating_rate

Goal
----
Implement heating_rate: the local volumetric heating rate of reflection-driven

Alfvenic turbulence.

```python
import numpy as np

def heating_rate(rho: np.ndarray, zp: np.ndarray, u: np.ndarray,
                 va: np.ndarray, eta: np.ndarray) -> np.ndarray:
    '''Volumetric heating rate in erg cm^-3 s^-1 at each station.

    Parameters
    ----------
    rho : np.ndarray
        (N,) mass density in g cm^-3, all positive.
    zp : np.ndarray
        (N,) rms outward amplitude in cm/s, all non-negative.
    u : np.ndarray
        (N,) field-aligned outflow speed in cm/s, all non-negative.
    va : np.ndarray
        (N,) Alfven speed in cm/s, all positive.
    eta : np.ndarray
        (N,) reflection rate per unit length in 1/cm, all non-negative.

    Returns
    -------
    q : np.ndarray
        (N,) heating rate in erg cm^-3 s^-1.

    Raises
    ------
    ValueError
        If rho, zp, u, va and eta are not all 1-D arrays, if they do not all
        have the same length, if any entry is non-finite, if any entry of rho
        or va is not positive, or if any entry of zp, u or eta is negative.
    '''
    return q  # placeholder
```

### Step 8

coronal_heating_summary

Goal
----
Implement coronal_heating_summary, the end-to-end pipeline (final step,

orchestrator).

```python
import numpy as np

def coronal_heating_summary(config: dict) -> float:
    '''End-to-end heating summary over all configured regions.

    Parameters
    ----------
    config : dict
        Keys "regions" (non-empty list of region parameter dicts),
        "bg_params" (background parameter dict), "zp0_kms" (float > 0),
        "ell_max" (float > 0, R_sun), "n_points" (int >= 3), "h_fd"
        (float > 0, R_sun), "r_sun_cm" (float > 0).

    Returns
    -------
    summary : float
        log10 of the largest end-of-line heating rate in erg cm^-3 s^-1.

    Raises
    ------
    ValueError
        If config is missing any of the seven required keys, if
        config["regions"] is not a non-empty list, if zp0_kms or r_sun_cm is
        not a positive finite number, or if n_points is less than 3. Values
        rejected by an earlier step propagate that step's ValueError.
    '''
    return summary  # placeholder
```
