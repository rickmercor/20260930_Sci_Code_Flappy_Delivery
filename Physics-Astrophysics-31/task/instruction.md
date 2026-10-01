# Physics-Astrophysics-31

## Problem

A magnetically structured coronal flux bundle extends 120 Mm above its base, and is made of overdense strands of circular cross-section embedded in ambient plasma, the strands filling the same fraction f = 0.16 of the bundle's cross-sectional area at every height. Two populations of magnetohydrodynamic waves are injected into this bundle and are damped as they travel along it, depositing their energy as heat, while the plasma cools by optically thin radiation; you are asked for the single height in Mm at which the two rates come into balance.

The background is static and entirely prescribed. Writing z for the height above the base in Mm and taking R_sun = 695.7 Mm, the cross-sectionally averaged hydrogen number density is n_H(z) = 1.5e15 exp(-z/42) m^-3, the mass density follows from it as rho = m_p (1 + 4 A_He) n_H with A_He = 0.1 and m_p = 1.6726219e-27 kg, the temperature is T(z) = 0.62 + 0.83 [1 - exp(-z/45)] MK and is uniform across the cross-section, the magnetic field strength is B(z) = 12.5 [695.7/(z + 695.7)]^2 G, the radius of a single strand is R(z) = 0.8 sqrt(B(0)/B(z)) Mm, and the ratio of strand-interior to ambient mass density is zeta(z) = 2.6 exp(-z/400) + 1. The first wave population consists of transverse displacement waves of the strands, of which only an outward-travelling population is present, injected at the base with energy density 0.9 mJ m^-3. The second population consists of Alfven waves, injected outward at the base with energy density 0.5 mJ m^-3 and inward at the top of the domain with energy density 0.025 mJ m^-3. No wave energy is reflected or converted from one population or propagation direction to another anywhere inside the domain, so these three injections are the only sources of wave energy.

Radiation is local and optically thin: at every point in the plasma the volumetric loss rate is the square of the hydrogen number density at that point, multiplied by Lambda(T) = 1.15e-35 (T / 1 MK)^(-1/2) W m^3. Work in the steady state on this static background, with no bulk flow anywhere in the domain, and take the bundle as a whole to have a fixed cross-sectional area, even though the individual strands expand with the field. Report, in Mm, the lowest height above the base at which the total volumetric heating rate deposited by the two wave populations together first rises to equal the volumetric radiative loss rate, both rates taken per unit volume of the bundle as a whole.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

stratified_background

Goal
----
Step 01: evaluate the prescribed static background on the height grid.

Contract
--------
The atmosphere is completely prescribed; nothing here is solved for. On a
height grid $z$ measured from the base of the domain, evaluate

- the hydrogen number density, falling off exponentially with scale height
$H$ from its base value $n_base$;
- the mass density, obtained from the hydrogen number density by adding the
helium mass, with a helium-to-hydrogen abundance ratio $A_He$, so that
$rho = m_p (1 + 4 A_He) n_H$;
- the temperature, rising from $T_base$ towards $T_base + dT$ as
1 - exp(-z / LT);
- the magnetic field strength, falling as the inverse square of the distance
from the solar centre, $B = B0 (R_sun / (z + R_sun))^2$ with
`R_sun = 695.7` Mm;
- the strand radius, which follows the field through conservation of the
magnetic flux threading one strand, R = R0 sqrt(B0 / B);
- the density contrast, relaxing from `zeta0` towards 1 as
zeta = (zeta0 - 1) exp(-z / Lzeta) + 1.

```python
def stratified_background(z: "ArrayLike", n_base: float, H: float, B0: float,
                          R0: float, zeta0: float, Lzeta: float, T_base: float,
                          dT: float, LT: float, A_He: float) -> "np.ndarray":
    '''Evaluate the prescribed background profiles on the height grid.

    Parameters
    ----------
    z : array_like
        One-dimensional height grid in Mm, strictly increasing, z[0] >= 0.
    n_base : float
        Hydrogen number density at the base, in 1e15 m^-3. Finite and > 0.
    H : float
        Density scale height in Mm. Finite and > 0.
    B0 : float
        Field strength at the base in G. Finite and > 0.
    R0 : float
        Strand radius at the base in Mm. Finite and > 0.
    zeta0 : float
        Density contrast at the base. Finite and not less than 1.
    Lzeta : float
        Relaxation length of the density contrast in Mm. Finite and > 0.
    T_base : float
        Temperature at the base in MK. Finite and > 0.
    dT : float
        Temperature rise across the domain in MK. Finite and >= 0.
    LT : float
        Height scale of the temperature rise in Mm. Finite and > 0.
    A_He : float
        Helium-to-hydrogen abundance ratio. Finite and >= 0.

    Returns
    -------
    np.ndarray
        Shape (6, N) float array. Rows in order: hydrogen number density in
        1e15 m^-3, mass density in 1e-12 kg m^-3, temperature in MK, field
        strength in G, strand radius in Mm, density contrast (dimensionless).

    Raises
    ------
    ValueError
        On a malformed grid or a non-finite or out-of-range parameter.
    '''
    return out  # placeholder
```

### Step 2

cross_section_structure

Goal
----
Step 02: resolve the cross-sectional density structure of the strand bundle.

Contract
--------
Seen end-on, the bundle is a two-component medium: a fraction $f$ of the
cross-sectional area is strand interior at mass density $rho_i$, the
remaining $1 - f$ is ambient material at $rho_e$, and the two are related
by the density contrast $zeta = rho_i / rho_e$. The single density carried by
the one-dimensional model is the area-weighted mean over the cross-section,
$rho_avg$, which is what this step is given.

Given $rho_avg$, `zeta` and $f$, return the two component densities that
reproduce that area-weighted mean.

```python
def cross_section_structure(rho_avg: "ArrayLike", zeta: "ArrayLike",
                            f: float) -> "np.ndarray":
    '''Split the area-weighted density into its two cross-sectional components.

    Parameters
    ----------
    rho_avg : array_like
        Area-weighted mean mass density over the bundle cross-section, in
        units of 1e-12 kg m^-3. Finite and strictly positive.
    zeta : array_like
        Density contrast rho_i / rho_e, broadcastable against ``rho_avg``.
        Finite and not less than 1.
    f : float
        Fraction of the cross-sectional area occupied by strand interior.
        Finite, with 0 < f < 1.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array in units of 1e-12 kg m^-3. Row 0 is the
        ambient density rho_e, row 1 the strand-interior density rho_i.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or out-of-range input.
    '''
    return out  # placeholder
```

### Step 3

channel_speeds

Goal
----
Step 03: the speed at which each wave channel is carried along the bundle.

Contract
--------
Two families of waves travel along the bundle and they do not share a
propagation speed.

- Row 0: the speed at which the Alfven channel is carried. It is built on the
field strength and on the single density the one-dimensional model carries,
the area-weighted mean.
- Row 1: the speed at which the transverse displacement channel is carried.
That channel is a collective oscillation in which a strand and the material
surrounding it move together, so its speed is set by the field strength and
by BOTH component densities, and it is not the Alfven speed of either
component alone. The same field threads the strand interior and the ambient
medium.

Neither speed is written out here; supply both.

```python
def channel_speeds(B: "ArrayLike", rho_avg: "ArrayLike", rho_i: "ArrayLike",
                   rho_e: "ArrayLike") -> "np.ndarray":
    '''Propagation speed of each wave channel, in Mm s^-1.

    Parameters
    ----------
    B : array_like
        Magnetic field strength in G. Finite and strictly positive.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_i : array_like
        Strand-interior mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3. Finite and > 0.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array in Mm s^-1. Row 0 is the speed of the Alfven
        channel; row 1 is the speed of the transverse displacement channel.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or non-positive input.
    '''
    return out  # placeholder
```

### Step 4

perpendicular_correlation_lengths

Goal
----
Step 04: the perpendicular correlation length of each wave channel.



Contract

--------

Evaluate the two correlation lengths specified in the Scientific background.

Return the transverse displacement (kink) channel length in row 0 and the

Alfven channel length in row 1. The first depends on strand radius, density

contrast and filling factor; the second depends on local temperature and

field strength.

```python
def perpendicular_correlation_lengths(R: "ArrayLike", zeta: "ArrayLike",
                                      f: float, T: "ArrayLike",
                                      B: "ArrayLike") -> "np.ndarray":
    '''Perpendicular correlation length of each channel, in Mm.

    Parameters
    ----------
    R : array_like
        Strand radius in Mm. Finite and strictly positive.
    zeta : array_like
        Density contrast, broadcastable against ``R``. Finite and > 1.
    f : float
        Fraction of the cross-sectional area occupied by strand interior.
        Finite, with 0 < f < 1.
    T : array_like
        Temperature in MK, broadcastable against ``R``. Finite and > 0.
    B : array_like
        Field strength in G, broadcastable against ``R``. Finite and > 0.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array in Mm. Row 0 is the perpendicular correlation
        length of the transverse displacement channel; row 1 is that of the
        Alfven channel.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or out-of-range input.
    '''
    return out  # placeholder
```

### Step 5

deposition_rates

Goal
----
Step 05: the volumetric rate at which each wave population deposits its energy.



Contract

--------

Evaluate the three local dissipation closures specified in the Scientific

background. Given the wave energy densities, local mass densities and

correlation lengths, return the deposition rates of the transverse

displacement channel, the outward Alfven population and the inward Alfven

population, in that row order. Each rate depends only on the supplied

local arguments.

```python
def deposition_rates(W_kink: "ArrayLike", W_out: "ArrayLike",
                     W_in: "ArrayLike", rho_avg: "ArrayLike",
                     rho_e: "ArrayLike", L_kink: "ArrayLike",
                     L_alfven: "ArrayLike") -> "np.ndarray":
    '''Volumetric deposition rate of each wave population, in uW m^-3.

    Parameters
    ----------
    W_kink : array_like
        Energy density of the transverse displacement channel, in mJ m^-3.
        Finite.
    W_out : array_like
        Energy density of the outward-travelling Alfven population, in
        mJ m^-3. Finite.
    W_in : array_like
        Energy density of the inward-travelling Alfven population, in mJ m^-3.
        Finite.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3. Finite and > 0.
    L_kink : array_like
        Perpendicular correlation length of the transverse channel, in Mm.
        Finite and > 0.
    L_alfven : array_like
        Perpendicular correlation length of the Alfven channel, in Mm. Finite
        and > 0.

    Returns
    -------
    np.ndarray
        Shape (3, N) float array in uW m^-3. Row 0 is the deposition rate of
        the transverse displacement channel, row 1 that of the outward Alfven
        population, row 2 that of the inward Alfven population.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or non-positive input.
    '''
    return out  # placeholder
```

### Step 6

transverse_channel_profile

Goal
----
Step 06: steady-state profile of the transverse displacement channel.

Contract
--------
Only an outward-travelling population of this channel is present, injected at
the base with energy density $W_base$, so its profile follows from a single
integration from the base to the top of the grid.

The background is static and there is no bulk flow, so in steady state the
height derivative of this channel's energy flux balances minus the volumetric
rate at which the channel deposits energy. The flux is the channel's energy
density times the speed at which it is carried. The deposition rate is the one
returned in row 0 by $deposition_rates$; call that function, do not
re-derive it here.

```python
def transverse_channel_profile(z: "ArrayLike", v_kink: "ArrayLike",
                               rho_avg: "ArrayLike", rho_e: "ArrayLike",
                               L_kink: "ArrayLike",
                               W_base: float) -> "np.ndarray":
    '''Steady-state energy density and deposition rate of the transverse channel.

    Parameters
    ----------
    z : array_like
        One-dimensional uniform height grid in Mm, strictly increasing, at
        least four points.
    v_kink : array_like
        Speed at which the transverse channel is carried, in Mm s^-1, one
        value per grid point. Finite and strictly positive.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3, one value per grid
        point. Finite and strictly positive.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3, one value per grid point.
        Finite and strictly positive.
    L_kink : array_like
        Perpendicular correlation length of this channel in Mm, one value per
        grid point. Finite and strictly positive.
    W_base : float
        Energy density injected at the base of the domain, in mJ m^-3. Finite
        and non-negative.

    Returns
    -------
    np.ndarray
        Shape (2, N) float array. Row 0 is the channel energy density in
        mJ m^-3, row 1 its volumetric deposition rate in uW m^-3.

    Raises
    ------
    ValueError
        On a malformed grid, a malformed background array or a negative
        injection.
    '''
    return out  # placeholder
```

### Step 7

alfven_channel_profile

Goal
----
Step 07: steady-state profile of the two-population Alfven channel.

Contract
--------
This channel carries TWO populations. An outward-travelling one is injected at
the base of the domain with energy density $W_out_base$; an inward-travelling
one enters at the top with energy density $W_in_top$ and propagates towards
the base. Both are carried at the Alfven speed.

As in step 06 the background is static, so in steady state the height
derivative of a population's energy flux balances minus the volumetric rate at
which that population deposits energy, and the flux is that population's energy
density times the Alfven speed. Those two rates are the ones returned in rows 1
and 2 by $deposition_rates$; call that function, do not re-derive them here.
Because the two boundary conditions sit at opposite ends of the domain, this is
a two-point boundary-value problem and must be relaxed rather than integrated
once.

```python
def alfven_channel_profile(z: "ArrayLike", v_alfven: "ArrayLike",
                           rho_avg: "ArrayLike", L_alfven: "ArrayLike",
                           W_out_base: float, W_in_top: float,
                           tol: float = 1e-15,
                           itmax: int = 500) -> "np.ndarray":
    '''Steady-state energy densities and deposition rate of the Alfven channel.

    Parameters
    ----------
    z : array_like
        One-dimensional uniform height grid in Mm, strictly increasing, at
        least four points.
    v_alfven : array_like
        Alfven speed in Mm s^-1, one value per grid point. Finite and > 0.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3, one value per grid
        point. Finite and strictly positive.
    L_alfven : array_like
        Perpendicular correlation length of this channel in Mm, one value per
        grid point. Finite and strictly positive.
    W_out_base : float
        Outward energy density injected at the base, in mJ m^-3. Finite and
        non-negative.
    W_in_top : float
        Inward energy density entering at the top, in mJ m^-3. Finite and
        non-negative.
    tol : float, optional
        Relative convergence tolerance of the Picard relaxation. Finite, > 0.
    itmax : int, optional
        Maximum number of Picard sweeps. Integer, at least 1.

    Returns
    -------
    np.ndarray
        Shape (3, N) float array. Row 0 outward energy density, row 1 inward
        energy density, both in mJ m^-3; row 2 the total volumetric deposition
        rate of the channel in uW m^-3.

    Raises
    ------
    ValueError
        On a malformed grid, a malformed background array, a negative
        injection or a malformed relaxation control.
    '''
    return out  # placeholder
```

### Step 8

radiative_loss_profile

Goal
----
Step 08: optically thin radiative loss rate of the structured bundle.

Contract
--------
Radiation is a local, optically thin, two-body process: at every point inside
the bundle the plasma loses energy at a volumetric rate equal to the radiative
loss function evaluated at the local temperature, multiplied by the square of
the LOCAL hydrogen number density at that point.

The bundle is not homogeneous across its cross-section. A fraction $f$ of the
area is strand interior at mass density $rho_i$, the rest is ambient material
at $rho_e$, and the one-dimensional model carries only the area-weighted mean
$rho_avg$ and the corresponding area-weighted mean hydrogen number density
$n_H$. The composition is the same everywhere, so at any point the hydrogen
number density is in the same ratio to $n_H$ as the local mass density is to
$rho_avg$. The temperature is uniform across the cross-section.

Return the loss rate per unit volume of the bundle as a whole. How that rate
follows from the arguments is NOT written out here; supply it.

The radiative loss function is prescribed, not modelled. It is the single
power law

$Lambda(T) = lambda0 (T / 1 MK)^lambda_exponent$

with `lambda0` given in units of 1e-35 W m^3 and $lambda_exponent$ the
power-law index.

```python
def radiative_loss_profile(n_H: "ArrayLike", T: "ArrayLike",
                           rho_avg: "ArrayLike", rho_e: "ArrayLike",
                           rho_i: "ArrayLike", f: float, lambda0: float,
                           lambda_exponent: float) -> "np.ndarray":
    '''Volumetric optically thin radiative loss rate of the bundle, in uW m^-3.

    Parameters
    ----------
    n_H : array_like
        Area-weighted mean hydrogen number density over the cross-section, in
        1e15 m^-3. Finite and non-negative.
    T : array_like
        Temperature in MK, broadcastable against ``n_H``. Finite and > 0.
    rho_avg : array_like
        Area-weighted mean mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_e : array_like
        Ambient mass density in 1e-12 kg m^-3. Finite and > 0.
    rho_i : array_like
        Strand-interior mass density in 1e-12 kg m^-3. Finite and > 0.
    f : float
        Fraction of the cross-sectional area occupied by strand interior.
        Finite, with 0 < f < 1.
    lambda0 : float
        Radiative loss function at 1 MK, in units of 1e-35 W m^3. Finite and
        strictly positive.
    lambda_exponent : float
        Power-law index of the radiative loss function. Finite.

    Returns
    -------
    np.ndarray
        Volumetric radiative loss rate in uW m^-3, of the broadcast shape.

    Raises
    ------
    ValueError
        On non-broadcastable, non-finite or out-of-range input.
    '''
    return out  # placeholder
```

### Step 9

first_balance_height

Goal
----
Step 09: locate the lowest height at which heating balances losses.

Contract
--------
Given a heating rate and a loss rate sampled on a common uniform height grid,
find the LOWEST height at which their difference changes sign, that is the
first crossing of heating and losses as one moves up from the base.

```python
def first_balance_height(z: "ArrayLike", heating: "ArrayLike",
                         losses: "ArrayLike") -> float:
    '''Lowest height at which the heating rate equals the loss rate.

    Parameters
    ----------
    z : array_like
        One-dimensional uniform height grid in Mm, strictly increasing, at
        least four points.
    heating : array_like
        Volumetric heating rate on that grid, finite, one value per point.
    losses : array_like
        Volumetric loss rate on that grid in the same units as ``heating``,
        finite, one value per point.

    Returns
    -------
    float
        The lowest height in Mm at which heating and losses are equal.

    Raises
    ------
    ValueError
        On a malformed grid, a malformed rate array, or a residual that never
        changes sign inside the grid.
    '''
    return z_star  # placeholder
```

### Step 10

orchestrator

Goal
----
Step 10 (FINAL ORCHESTRATOR): the two-channel radiative balance height.

Contract
--------
Run the whole chain end to end and return one scalar: the lowest height at
which the total wave heating deposited by BOTH channels first rises to equal
the local optically thin radiative loss rate of the bundle.

The chain is: the prescribed background on a uniform height grid (step 01);
the cross-sectional density split (step 02); the two propagation speeds
(step 03); the two perpendicular correlation lengths (step 04); the
dissipation closures (step 05); the steady-state profile and deposition rate of
the transverse displacement channel (step 06); the same for the two-population
Alfven channel (step 07); the radiative loss rate (step 08); and the first
crossing of their sum with the losses (step 09).

The driver composes the functions of steps 01-09; it does not re-implement
them. Called with no arguments it runs the benchmark configuration of the
problem statement.

```python
def two_channel_balance_height(
        n_points: int = 4001, z_top: float = 120.0,
        n_base: float = 1.5, H: float = 42.0, B0: float = 12.5,
        R0: float = 0.8, zeta0: float = 3.6, Lzeta: float = 400.0,
        T_base: float = 0.62, dT: float = 0.83, LT: float = 45.0,
        A_He: float = 0.1, f: float = 0.16, lambda0: float = 1.15,
        lambda_exponent: float = -0.5, W_kink_base: float = 0.9,
        W_alfven_out_base: float = 0.5, W_alfven_in_top: float = 0.025,
        tol: float = 1e-15, itmax: int = 500) -> float:
    '''Height at which two-channel wave heating first balances radiative losses.

    Runs the whole chain: the prescribed background, the cross-sectional
    density structure, the two propagation speeds, the two perpendicular
    correlation lengths, the dissipation closures, the steady-state profile of
    each wave channel, the radiative losses, and the first crossing of total
    heating with losses.

    Called with no arguments it runs the benchmark configuration of the
    problem statement.

    Parameters
    ----------
    n_points : int, optional
        Number of nodes on the uniform height grid. Integer, at least 4.
    z_top : float, optional
        Top of the domain in Mm. Finite and strictly positive.
    n_base : float, optional
        Base hydrogen number density in 1e15 m^-3. Finite and > 0.
    H : float, optional
        Density scale height in Mm. Finite and > 0.
    B0 : float, optional
        Base field strength in G. Finite and > 0.
    R0 : float, optional
        Base strand radius in Mm. Finite and > 0.
    zeta0 : float, optional
        Base density contrast. Finite and > 1.
    Lzeta : float, optional
        Relaxation length of the contrast in Mm. Finite and > 0.
    T_base : float, optional
        Base temperature in MK. Finite and > 0.
    dT : float, optional
        Temperature rise across the domain in MK. Finite and >= 0.
    LT : float, optional
        Height scale of the temperature rise in Mm. Finite and > 0.
    A_He : float, optional
        Helium-to-hydrogen abundance ratio. Finite and >= 0.
    f : float, optional
        Strand-interior area filling factor, 0 < f < 1.
    lambda0 : float, optional
        Radiative loss function at 1 MK in 1e-35 W m^3. Finite and > 0.
    lambda_exponent : float, optional
        Power-law index of the radiative loss function. Finite.
    W_kink_base : float, optional
        Transverse-channel energy density injected at the base, in mJ m^-3.
        Finite and non-negative.
    W_alfven_out_base : float, optional
        Outward Alfven energy density injected at the base, in mJ m^-3.
        Finite and non-negative.
    W_alfven_in_top : float, optional
        Inward Alfven energy density entering at the top, in mJ m^-3. Finite
        and non-negative.
    tol : float, optional
        Relative convergence tolerance of the Alfven relaxation. Finite, > 0.
    itmax : int, optional
        Maximum number of relaxation sweeps. Integer, at least 1.

    Returns
    -------
    float
        The balance height in Mm.

    Raises
    ------
    ValueError
        If any certification fails, or if any argument fails the validation of
        the step it is passed to.
    '''
    return z_star  # placeholder
```
