# Physics-Astrophysics-17

## Background

The solar wind and the solar corona are nearly collisionless plasmas in which the protons and heavier ions are observed to be hotter than adiabatic expansion allows, and to be heated mainly in the direction perpendicular to the magnetic field. The free energy that could supply this heating is carried by turbulent electromagnetic fluctuations, which cascade from large injection scales toward the ion gyroradius and below, where the character of the fluctuations changes from magnetohydrodynamic Alfven waves to kinetic Alfven waves.

A charged particle gyrating in a slowly varying field conserves its magnetic moment, an adiabatic invariant, so fluctuations that are slow and gentle compared with its gyromotion cannot change its perpendicular energy in a lasting way. Perpendicular heating therefore requires fluctuations that violate this adiabatic behaviour, either through resonance with the gyromotion or through non-resonant, sufficiently rapid or sufficiently large variations of the fields along the particle's orbit. Several theoretical pictures of this non-resonant, magnetic-moment-breaking heating have been developed and compared with spacecraft measurements and kinetic simulations.

Such estimates matter well beyond the solar wind. The partition of turbulent dissipation between ions and electrons controls the thermodynamics of the expanding corona and wind, and it sets the ion-to-electron temperature ratio assumed when interpreting emission from remote astrophysical plasmas such as accretion flows. Quantitative predictions of the ion heating rate from measurable fluctuation spectra are therefore a central goal of space and astrophysical plasma physics.

## Problem

A fast solar-wind stream observed close to the Sun is a plasma of protons and electrons only, with a mean magnetic field of 180 nT, a proton number density of 90 cm^-3, a proton thermal speed v_thp = sqrt(2 k_B T_p / m_p) of 92 km/s, and an electron-to-proton temperature ratio T_e/T_p = 2.0. Its turbulence is balanced and critically balanced kinetic-Alfven turbulence in the low-beta regime with isothermal electrons, every scale of interest is much larger than the electron inertial length, and at each perpendicular wavenumber the fluctuations are characterised by a single amplitude with no intermittency. Use e = 1.602176634e-19 C, m_p = 1.67262192369e-27 kg and mu_0 = 4 pi x 10^-7 H/m. Taking the two perpendicular components together, the measured root-mean-square amplitude of the perpendicular magnetic fluctuations at perpendicular wavenumber k_perp, in rad/km, is

    delta_B(k_perp) = 12.0 nT * (k_perp / 1.0e-3)^(-1/3) * [1 + (k_perp / 0.150)^2]^(-11/60).

Report the rate, in watts per kilogram, at which the perpendicular kinetic energy per unit mass of the protons grows because of the fluctuations at k_perp = 0.280 rad/km.

The protons are thermal, and the fluctuations at each scale act on them as structures that are coherent in both space and time, so that repeated uncorrelated encounters break the proton magnetic moment and spread the protons in perpendicular energy. Adopt c_1 = 0.75 and c_2 = 0.34 for the two dimensionless constants of order unity that this low-beta perpendicular ion-heating estimate carries.

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

plasma_reference_scales

Goal
----
Reference scales of the proton-electron plasma.

```python
def plasma_reference_scales(B0_nT: float, n_p_cm3: float, v_thp_kms: float) -> "np.ndarray":
    """Return the proton gyrofrequency, thermal gyroradius and Alfven speed.

    Parameters
    ----------
    B0_nT : float
        Mean magnetic field strength in nT. Finite and strictly positive.
    n_p_cm3 : float
        Proton number density in cm^-3. Finite and strictly positive.
    v_thp_kms : float
        Proton thermal speed sqrt(2 k_B T_p / m_p) in km/s. Finite and strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (3,) holding [Omega_p, rho_p, v_A]: the proton gyrofrequency
        Omega_p = e B0 / m_p in rad/s, the proton thermal gyroradius
        rho_p = v_thp / Omega_p in km, and the Alfven speed
        v_A = B0 / sqrt(mu_0 n_p m_p) in km/s.

    Raises
    ------
    ValueError
        If any argument is not a finite, strictly positive real scalar.
    """
    return scales  # placeholder
```

### Step 2

perpendicular_velocity_amplitude

Goal
----
The measured perpendicular fluctuation amplitude in velocity units.

```python
def perpendicular_velocity_amplitude(
    k_perp: "ArrayLike",
    B0_nT: float,
    v_A_kms: float,
    dB_ref_nT: float,
    k_ref: float,
    k_break: float,
    slope_large: float,
    slope_small: float,
) -> "np.ndarray":
    """Return the perpendicular magnetic fluctuation amplitude in velocity units.

    Parameters
    ----------
    k_perp : float or array_like
        Perpendicular wavenumber(s) in rad/km. Non-empty, every entry finite and strictly
        positive.
    B0_nT : float
        Mean magnetic field strength in nT. Finite and strictly positive.
    v_A_kms : float
        Alfven speed of the proton mass density in km/s (step 01). Finite and strictly
        positive.
    dB_ref_nT : float
        Amplitude scale of the magnetic spectrum in nT. Finite and strictly positive.
    k_ref : float
        Reference wavenumber of the spectrum in rad/km. Finite and strictly positive.
    k_break : float
        Break wavenumber of the spectrum in rad/km. Finite and strictly positive.
    slope_large : float
        Power-law index of delta_B well below the break (delta_B ~ k^-slope_large). Finite.
    slope_small : float
        Power-law index of delta_B well above the break (delta_B ~ k^-slope_small). Finite.

    Returns
    -------
    numpy.ndarray
        Array with the shape of ``k_perp`` holding delta_b = delta_B(k_perp) v_A / B0 in
        km/s, where
        delta_B(k) = dB_ref_nT (k / k_ref)^(-slope_large)
        [1 + (k / k_break)^2]^(-(slope_small - slope_large) / 2) in nT is the rms amplitude
        of both perpendicular components together.

    Raises
    ------
    ValueError
        If ``k_perp`` is empty or has a non-finite or non-positive entry, if ``B0_nT``,
        ``v_A_kms``, ``dB_ref_nT``, ``k_ref`` or ``k_break`` is not a finite, strictly
        positive real scalar, or if ``slope_large`` or ``slope_small`` is not a finite real
        scalar.
    """
    return delta_b  # placeholder
```

### Step 3

electron_flow_factor

Goal
----
Effective electron-flow amplitude of Alfvenic fluctuations at ion scales.

```python
def electron_flow_factor(k_rho_p: "ArrayLike", te_over_tp: float) -> "np.ndarray":
    """Return the effective electron-flow factor alpha_k at each k_perp rho_p.

    Parameters
    ----------
    k_rho_p : float or array_like
        Perpendicular wavenumber times the proton thermal gyroradius of step 01
        (dimensionless). Non-empty, every entry finite and strictly positive.
    te_over_tp : float
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Array with the shape of ``k_rho_p`` holding alpha_k = delta u_e,k / delta b_k, the
        ratio of the effective electron bulk-flow amplitude to the perpendicular magnetic
        fluctuation amplitude in velocity units for the linear Alfvenic eigenmodes of the
        model at that k_perp rho_p (singly charged ions). alpha_k tends to 1 as
        k_rho_p -> 0.

    Raises
    ------
    ValueError
        If ``k_rho_p`` is empty or has a non-finite or non-positive entry, or if
        ``te_over_tp`` is not a finite, non-negative real scalar.
    """
    return alpha  # placeholder
```

### Step 4

electric_amplitude_parameter

Goal
----
The electric-field amplitude seen by a thermal proton.

```python
def electric_amplitude_parameter(
    k_rho_p: "ArrayLike",
    delta_b_kms: "ArrayLike",
    v_thp_kms: float,
    te_over_tp: float,
) -> "np.ndarray":
    """Return the normalised electric amplitude epsilon_k of the fluctuations at each scale.

    Parameters
    ----------
    k_rho_p : float or array_like
        Perpendicular wavenumber times the proton thermal gyroradius of step 01
        (dimensionless). Non-empty, every entry finite and strictly positive.
    delta_b_kms : float or array_like
        Perpendicular magnetic fluctuation amplitude in velocity units at the same scale(s),
        in km/s (step 02). Non-empty, every entry finite and strictly positive; must
        broadcast with ``k_rho_p``.
    v_thp_kms : float
        Proton thermal speed sqrt(2 k_B T_p / m_p) in km/s, used as the proton's
        perpendicular speed. Finite and strictly positive.
    te_over_tp : float
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Array with the broadcast shape of ``k_rho_p`` and ``delta_b_kms`` holding
        epsilon_k = E_k / (B0 v_thp), where E_k = k_perp phi_k is the amplitude of the
        electrostatic perpendicular electric field of the Alfvenic fluctuations of step 03
        at that scale.

    Raises
    ------
    ValueError
        If ``k_rho_p`` or ``delta_b_kms`` is empty or has a non-finite or non-positive
        entry, if the two do not broadcast together, if ``v_thp_kms`` is not a finite,
        strictly positive real scalar, or if ``te_over_tp`` is not a finite, non-negative
        real scalar.
    """
    return epsilon  # placeholder
```

### Step 5

fluctuation_frequency_ratio

Goal
----
How fast the fluctuations evolve, measured in proton gyrofrequencies.

```python
def fluctuation_frequency_ratio(
    k_perp: "ArrayLike",
    delta_b_kms: "ArrayLike",
    omega_p: float,
    rho_p_km: float,
    te_over_tp: float,
) -> "np.ndarray":
    """Return the ratio of the fluctuation frequency to the proton gyrofrequency.

    Parameters
    ----------
    k_perp : float or array_like
        Perpendicular wavenumber(s) in rad/km. Non-empty, every entry finite and strictly
        positive.
    delta_b_kms : float or array_like
        Perpendicular magnetic fluctuation amplitude in velocity units at the same scale(s),
        in km/s (step 02). Non-empty, every entry finite and strictly positive; must
        broadcast with ``k_perp``.
    omega_p : float
        Proton gyrofrequency in rad/s (step 01). Finite and strictly positive.
    rho_p_km : float
        Proton thermal gyroradius in km (step 01). Finite and strictly positive.
    te_over_tp : float
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Array with the broadcast shape of ``k_perp`` and ``delta_b_kms`` holding
        omega_k / Omega_p, the critically balanced nonlinear frequency of the Alfvenic
        fluctuations of step 03 at that scale divided by the proton gyrofrequency, with no
        order-unity constant applied.

    Raises
    ------
    ValueError
        If ``k_perp`` or ``delta_b_kms`` is empty or has a non-finite or non-positive entry,
        if the two do not broadcast together, if ``omega_p`` or ``rho_p_km`` is not a
        finite, strictly positive real scalar, or if ``te_over_tp`` is not a finite,
        non-negative real scalar.
    """
    return ratio  # placeholder
```

### Step 6

gyroaveraging_weight

Goal
----
Gyro-averaging of the energy change a fluctuation can impart.

```python
def gyroaveraging_weight(k_rho: "ArrayLike") -> "np.ndarray":
    """Return the gyro-averaging weight of the mean-square perpendicular-energy change.

    Parameters
    ----------
    k_rho : float or array_like
        Perpendicular wavenumber times the proton gyroradius (dimensionless). Non-empty,
        every entry finite and strictly positive.

    Returns
    -------
    numpy.ndarray
        Array with the shape of ``k_rho`` holding the weight by which gyrophase averaging
        multiplies the mean-square perpendicular-energy change produced by the electrostatic
        perpendicular electric field of step 04 at that k_perp rho; it tends to 1/4 as
        k_rho -> 0.

    Raises
    ------
    ValueError
        If ``k_rho`` is empty or has a non-finite or non-positive entry.
    """
    return weight  # placeholder
```

### Step 7

proton_perpendicular_heating_rate

Goal
----
Perpendicular proton heating rate at one scale (final orchestrator step).

```python
def proton_perpendicular_heating_rate(
    k_perp: float = 0.280,
    B0_nT: float = 180.0,
    n_p_cm3: float = 90.0,
    v_thp_kms: float = 92.0,
    te_over_tp: float = 2.0,
    dB_ref_nT: float = 12.0,
    k_ref: float = 1.0e-3,
    k_break: float = 0.150,
    slope_large: float = 1.0 / 3.0,
    slope_small: float = 0.70,
    c1: float = 0.75,
    c2: float = 0.34,
) -> float:
    """Return the perpendicular proton heating rate from the fluctuations at k_perp.

    Parameters
    ----------
    k_perp : float, optional
        Perpendicular wavenumber of the fluctuations in rad/km. Finite and strictly positive.
    B0_nT : float, optional
        Mean magnetic field strength in nT (step 01). Finite and strictly positive.
    n_p_cm3 : float, optional
        Proton number density in cm^-3 (step 01). Finite and strictly positive.
    v_thp_kms : float, optional
        Proton thermal speed sqrt(2 k_B T_p / m_p) in km/s (step 01). Finite and strictly
        positive.
    te_over_tp : float, optional
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.
    dB_ref_nT, k_ref, k_break, slope_large, slope_small : float, optional
        Parameters of the measured amplitude spectrum of step 02 (nT, rad/km, rad/km and
        two finite indices).
    c1, c2 : float, optional
        The two dimensionless order-unity constants of the heating estimate. Finite and
        strictly positive.

    Returns
    -------
    float
        Rate of growth of the perpendicular kinetic energy per unit mass of thermal protons
        caused by the fluctuations at ``k_perp`` breaking their magnetic moment, in W/kg.

    Raises
    ------
    ValueError
        If ``k_perp``, ``c1`` or ``c2`` is not a finite, strictly positive real scalar, or
        for any argument that steps 01-06 reject.
    """
    return 0.0
```
