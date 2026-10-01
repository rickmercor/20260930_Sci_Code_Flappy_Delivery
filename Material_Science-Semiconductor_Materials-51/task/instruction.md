# Material_Science-Semiconductor_Materials-51

## Background

Predicting carrier mobility in semiconductor nanowires is essential for designing next-generation nanoscale electronic devices, since spatial confinement introduces surface scattering effects absent in bulk materials. A fully ab initio framework combining bulk electron-phonon scattering with geometric surface scattering reveals that nanowire mobility follows a systematic empirical dependence on diameter, governed by two system-specific parameters: a characteristic length scale (comparable in magnitude to the carrier mean free path) and an exponent capturing the competition between the two scattering mechanisms. Because carrier transport in real crystals is anisotropic, the mean free path varies by crystallographic direction, and a physically meaningful single representative value must properly weight each direction's contribution by how much carrier density actually flows along it -- an unweighted average would misrepresent the transport-relevant length scale. Separately, the temperature dependence of bulk mobility follows an approximate power-law behavior over ranges where the dominant scattering mechanism remains unchanged. Combining these effects requires correctly sequencing several distinct calculations: establishing the transport-weighted characteristic length scale, correcting the bulk mobility for temperature, applying the diameter-dependent empirical relation using the temperature-corrected bulk value, and finally extracting the surface-scattering-only contribution via Matthiessen's rule, which combines independent scattering mechanisms as the reciprocal sum of their individual mobility contributions.

A complete transport model must also account for ionized-impurity scattering, a distinct mechanism from both bulk phonon scattering and surface scattering, arising when dopant atoms introduce charged scattering centers into the crystal. The Brooks-Herring model describes this contribution using a screened-Coulomb-scattering framework, characterizing the strength of Coulomb screening via a dimensionless parameter that depends on temperature and impurity concentration, and yielding an impurity-limited mobility that must be combined with the size-limited mobility (itself already reflecting both bulk and surface scattering) via a second, independent application of Matthiessen's rule.

## Problem

Predicting how multiple independent physical mechanisms limit charge transport in a confined semiconductor structure is essential for nanoscale electronic device design. A baseline transport-limiting mechanism sets a reference mobility that decreases with temperature; a second, geometry-dependent mechanism further suppresses this reference value once a critical structural dimension approaches the characteristic transport length scale, through an empirical relation combining the two via Matthiessen's rule; and a third, independent mechanism -- arising from charged scattering centers in the material -- contributes an additional limiting effect whose strength depends on scattering-center concentration and a temperature- and concentration-dependent screening parameter. Combining all three mechanisms requires two separate applications of Matthiessen's rule.

Your task is to compute this fully combined mobility, relative to the bulk mobility, for one concrete, deterministic instance. The representative mean free path must be weighted by carrier density across crystallographic directions (6 directions, seeds 421 and 433 respectively for the per-direction MFP values and their weights). 
Given: per-direction MFP values (nm) produced by numpy's default_rng(421).uniform(20, 150, 6), per-direction carrier-density weights produced by numpy's default_rng(433).uniform(0.5, 2.0, 6), proportionality constant kappa=0.72, reference bulk mobility mu_bulk_ref=1090.0 cm^2/V/s at T_ref=245 K, temperature-scaling exponent alpha=1.58, target temperature T=370 K, nanowire diameter d=190 nm, diameter-dependence exponent beta=1.44, ionized impurity concentration N_I=3.1e18 cm^-3, and two further fixed proportionality constants C_b=4.2e14 and C_i=6.5e17 governing the impurity-scattering screening parameter and mobility magnitude respectively. Find the ratio of the fully combined mobility (incorporating bulk, surface, and impurity scattering together) to the temperature-scaled bulk mobility alone. Your final answer must be a single number: this ratio.

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

weighted_mfp_average

Goal
----
Carrier transport in real crystals is anisotropic, so the mean free path varies by crystallographic direction. A single representative value must weight each direction's contribution by how much carrier density actually flows along it, since directions carrying more current are more relevant to the effective transport behavior -- an unweighted average would misrepresent this transport-relevant length scale.

```python
def weighted_mfp_average(MFP: list[float], weights: list[float]) -> float:
    '''Compute the carrier-density-weighted average mean free path.

    Parameters
    ----------
    MFP : list[float]
        1D array of per-direction mean free path values (nm).
    weights : list[float]
        1D array of per-direction carrier-density weights, same length as MFP.

    Returns
    -------
    MFP_avg : float
        The weighted average MFP, as a native Python float.

    Raises
    ------
    ValueError
        If MFP and weights are not equal-length non-empty 1D arrays, if
        any weight is negative, or if the weights sum to zero.
    '''
    return MFP_avg  # placeholder
```

### Step 2

d0_from_mfp

Goal
----
Implement d0_from_mfp, which computes the characteristic length scale governing the onset of surface scattering as a fixed proportion of the weighted-average mean free path.

```python
def d0_from_mfp(MFP_avg: float, kappa: float) -> float:
    '''Compute the characteristic length scale from the weighted MFP average.

    Parameters
    ----------
    MFP_avg : float
        The weighted average MFP.
    kappa : float
        The proportionality constant.

    Returns
    -------
    d0 : float
        The characteristic length scale, as a native Python float.

    Raises
    ------
    ValueError
        If MFP_avg or kappa is not finite, or if either is not positive.
    '''
    return d0  # placeholder
```

### Step 3

bulk_mobility_temperature_scaling

Goal
----
Implement bulk_mobility_temperature_scaling, which computes the temperature-corrected bulk carrier mobility from a reference value using the observed power-law temperature dependence.

```python
def bulk_mobility_temperature_scaling(mu_bulk_ref: float, T_ref: float, alpha: float, T: float) -> float:
    '''Compute the temperature-scaled bulk mobility.

    Parameters
    ----------
    mu_bulk_ref : float
        Reference bulk mobility at T_ref.
    T_ref : float
        Reference temperature.
    alpha : float
        Power-law temperature-scaling exponent.
    T : float
        Target temperature.

    Returns
    -------
    mu_bulk_T : float
        The temperature-scaled bulk mobility, as a native Python float.

    Raises
    ------
    ValueError
        If any input is not finite, or if mu_bulk_ref, T_ref, or T is not positive.
    '''
    return mu_bulk_T  # placeholder
```

### Step 4

diameter_dependent_mobility

Goal
----
Implement diameter_dependent_mobility, which computes the effective 1D nanowire mobility from the temperature-scaled bulk mobility, using the empirical diameter-dependence relation.

```python
def diameter_dependent_mobility(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    '''Compute the diameter-dependent nanowire mobility.

    Parameters
    ----------
    mu_bulk_T : float
        Temperature-scaled bulk mobility.
    d : float
        Nanowire diameter.
    d0 : float
        Characteristic length scale.
    beta : float
        Diameter-dependence exponent.

    Returns
    -------
    mu_1D : float
        The diameter-dependent nanowire mobility, as a native Python float.

    Raises
    ------
    ValueError
        If any input is not finite, or if d does not exceed d0.
    '''
    return mu_1D  # placeholder
```

### Step 5

surface_limited_mobility

Goal
----
Implement surface_limited_mobility, which extracts the surface-scattering-only contribution to mobility from the combined 1D mobility and the bulk mobility, using Matthiessen's rule.

```python
def surface_limited_mobility(mu_1D: float, mu_bulk_T: float) -> float:
    '''Compute the surface-limited mobility via Matthiessen's rule.

    Parameters
    ----------
    mu_1D : float
        Diameter-dependent 1D nanowire mobility.
    mu_bulk_T : float
        Temperature-scaled bulk mobility.

    Returns
    -------
    mu_s : float
        The surface-limited mobility, as a native Python float.

    Raises
    ------
    ValueError
        If either mobility is non-finite or non-positive, or if mu_1D is
        greater than or equal to mu_bulk_T.
    '''
    return mu_s  # placeholder
```

### Step 6

surface_mobility_direct_check

Goal
----
Implement surface_mobility_direct_check, which computes the surface-scattering-only mobility using the paper's alternate direct form (Eq. 6), rather than deriving it via Matthiessen's rule, as an independent consistency check on the pipeline.

```python
def surface_mobility_direct_check(mu_bulk_T: float, d: float, d0: float, beta: float) -> float:
    '''Compute surface-scattering-only mobility via the direct Eq. 6 form.

    Parameters
    ----------
    mu_bulk_T : float
        Temperature-scaled bulk mobility.
    d : float
        Nanowire diameter.
    d0 : float
        Characteristic length scale.
    beta : float
        Diameter-dependence exponent.

    Returns
    -------
    mu_s_direct : float
        The directly computed surface-limited mobility, as a native
        Python float.

    Raises
    ------
    ValueError
        If any input is not finite, or if d does not exceed d0.
    '''
    return mu_s_direct  # placeholder
```

### Step 7

ionized_impurity_screening_parameter

Goal
----
Implement ionized_impurity_screening_parameter, which computes the dimensionless Brooks-Herring screening parameter characterizing the strength of Coulomb screening by free carriers around ionized impurities.

```python
def ionized_impurity_screening_parameter(C_b: float, T: float, N_I: float) -> float:
    '''Compute the Brooks-Herring screening parameter b.

    Returns
    -------
    b : float
        The dimensionless screening parameter.

    Raises
    ------
    ValueError
        If any input is not finite, or if C_b, T, or N_I is not positive.
    '''
    return b  # placeholder
```

### Step 8

ionized_impurity_mobility

Goal
----
Implement ionized_impurity_mobility, which computes the ionized-impurity-limited carrier mobility using the Brooks-Herring screening function.

```python
def ionized_impurity_mobility(C_i: float, T: float, N_I: float, b: float) -> float:
    '''Compute the ionized-impurity-limited mobility.

    Returns
    -------
    mu_i : float
        The impurity-limited mobility.

    Raises
    ------
    ValueError
        If any input is not finite, if C_i, T, N_I, or b is not positive,
        or if the resulting screening function G(b) is not positive.
    '''
    return mu_i  # placeholder
```

### Step 9

final_combined_mobility_ratio

Goal
----
Implement final_combined_mobility_ratio, the final orchestrator that chains all eight prior sub-problems: computing the weighted MFP average, characteristic length scale, temperature-scaled bulk mobility, diameter-dependent mobility, surface-limited mobility (with direct-form consistency check), the ionized-impurity screening parameter, and the impurity-limited mobility, then combines the diameter-dependent nanowire mobility (which already includes bulk and surface scattering) with the impurity-limited mobility via a second application of Matthiessen's rule, and returns the ratio of this fully combined mobility to the temperature-scaled bulk mobility.

```python
def final_combined_mobility_ratio(MFP: np.ndarray, weights: np.ndarray, kappa: float, mu_bulk_ref: float, T_ref: float, alpha: float, T: float, d: float, beta: float, C_b: float, C_i: float, N_I: float) -> float:
    '''Compute the fully combined mobility ratio (orchestrator).

    Returns
    -------
    ratio : float
        mu_total / mu_bulk(T), as a native Python float.

    Raises
    ------
    ValueError
        If any earlier step's input validity conditions are violated.
    '''
    return ratio  # placeholder
```
