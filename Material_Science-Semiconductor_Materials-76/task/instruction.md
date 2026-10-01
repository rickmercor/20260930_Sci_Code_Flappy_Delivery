# Material_Science-Semiconductor_Materials-76

## Background

Quantum cascade lasers are unipolar semiconductor lasers in which electrons cascade through a stack of coupled quantum wells and emit a photon at each stage through an intersubband transition. The transition energy is set by layer thicknesses rather than by a material band gap, so the same material systems, most often GaAs/AlGaAs and InGaAs/InAlAs grown by molecular beam epitaxy, cover the mid-infrared and the terahertz range. A defining property of these devices is that the upper laser level empties by fast phonon scattering, so the gain recovers on a picosecond time scale, much faster than the photon round trip in a millimetre-long cavity and orders of magnitude faster than in interband quantum-well diode lasers.

Fast gain recovery changes how multimode emission arises. In lasers with slow gain, a population grating or a saturable absorber is usually needed to lock the modes, and short pulses are the typical outcome. Quantum cascade lasers instead often form frequency combs spontaneously, many of them with a nearly constant output intensity and a frequency-modulated field. Among the comb states observed experimentally are harmonic states, in which only every m-th cavity mode lases, so the lines are separated by an integer multiple of the free spectral range. Such states appear in both Fabry-Perot and ring resonators, their order often changes with bias current, and they are attractive as sources of microwave and terahertz beat notes and for on-chip spectroscopy.

The theoretical description of these lasers usually starts from Maxwell-Bloch equations for the slowly varying field, the macroscopic polarisation and the carrier population. For semiconductor media the two-level form is modified to include an asymmetric, carrier-density dependent gain and refractive index and the coupling between amplitude and phase expressed by the linewidth enhancement factor (the Henry factor), which is of order one in quantum cascade lasers. Ring geometries with a single propagation direction are a convenient test bed because they avoid spatial hole burning and have translationally invariant single-frequency solutions.

Whether single-frequency emission survives above threshold is a question of dynamical stability. Classical laser physics knows two routes. In two-level lasers, a strong intracavity field drives Rabi oscillations of polarisation and inversion, and when the Rabi frequency matches a cavity mode spacing the single-mode state becomes unstable at a second threshold, the Risken-Nummedal-Graham-Haken instability. Near threshold and for a large amplitude-phase coupling, laser dynamics can instead be reduced to a complex Ginzburg-Landau equation, whose single-frequency solutions lose stability through the long-wavelength Benjamin-Feir, or phase, instability. Linear stability analysis of the single-frequency state against sideband perturbations, followed by numerical integration of the full equations, is the standard way of deciding which regime a given device falls into.

## Problem

Quantum cascade lasers recover their gain within picoseconds, and ring lasers of this kind often start emitting harmonic frequency combs by themselves, with comb lines spaced by several free spectral ranges of the cavity. A recent theory explains this as a resonance between a cavity mode and an intrinsic oscillation of the semiconductor gain medium, at a characteristic frequency f_M, and backs the picture with a linear stability analysis of single-frequency emission in the effective semiconductor Maxwell-Bloch equations. For a unidirectional terahertz ring laser described by its scaled material and cavity parameters, the task is to find the pump at which the comb order predicted by linear stability first changes and, at that pump, how far the maximum of the parametric gain lies from f_M.

With time measured in units of the dephasing time tau_d and the propagation coordinate eta in units of v_g tau_d, the field F, the polarisation P and the carrier density D obey dF/deta + dF/dt = -sigma (F + P), dP/dt = -Gamma (1 + i alpha) [P + (1 + i alpha) D F] and dD/dt = b [mu - D + (F* P + F P*)/2], with the field periodic around the ring (unit reflectivity) and zero detuning at the maximum of the unsaturated gain. Use the following configuration:

- alpha = 0.95, Gamma = 0.06, sigma = 1.6e-3, b = 0.014 and tau_d = 0.1 ps;
- ring length L = 4.7 mm and group index 3.6, so that v_g = c/3.6 with c = 299792458 m/s;
- cavity sidebands n = 1, 2, ..., 16, sideband n lying n free spectral ranges (n v_g / L) from the wave;
- pump window 2 <= mu <= 12, and sideband offsets up to 400 GHz for the parametric gain.

The fundamental continuous wave is the solution F = F0 exp(-i k eta + i omega t), P = P0 exp(-i k eta + i omega t), D = D0 with k = 0 and F0 real, and a perturbation of it with wavenumber offset k_n is taken proportional to exp(-i k_n eta + lambda t) in the frame of the wave; the growth rate of a sideband is the largest real part of its allowed lambda, and the predicted comb order is the sideband that grows fastest. Raising the pump, the multimode threshold mu_c is the pump at which the first sideband starts to grow, and mu_star is the pump above mu_c at which the predicted order first changes. At mu_star, take f_M as the magnitude of the imaginary part of the complex-conjugate eigenvalue pair of the linearised P, P*, D dynamics with the field held fixed, real and at the reference frequency, with the intensity of the fundamental wave, and locate the maximum of that wave's parametric gain over sideband offsets treated as a continuous variable. The final answer is the frequency offset of the gain maximum minus f_M, both as ordinary frequencies in GHz, to at least three decimals. In the reasoning give, with pumps to four decimals, frequencies in GHz to at least three decimals and growth rates in 1/ns to four significant figures: the frequency omega/(2 pi tau_d) of the fundamental wave, with the sign of omega in the form above; mu_c and its sideband; mu_star and the sideband that takes over; the growth rate the two sidebands share at mu_star; f_M and its damping (the magnitude of the real part of the same pair) at mu_star; the frequency offset and growth rate of the gain maximum at mu_star and the lower edge of the band of growing offsets there; the cavity sideband closest to f_M at mu_star; the perturbation amplitudes that the linearised equations couple at one sideband offset; the characteristic polynomial whose complex roots give f_M; and, from the source's own simulation of this laser with alpha raised to 1.05 at mu = 5.9 (other scaled parameters unchanged), how long the first comb it forms lasts and how long the irregular stage after it lasts, and the order and central frequency offset of the comb the laser finally settles in.

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

01_cw_emission_state

Goal
----
Step 01 - Continuous-wave state of the ring laser equations.

```python
def cw_emission_state(k: float, Gamma: float, alpha: float, sigma: float, mu: float) -> "np.ndarray":
    '''Continuous-wave solution of the ring equations at scaled wavenumber k.

    Parameters
    ----------
    k : float
        Scaled wavenumber of the continuous wave (k = 0 is the reference
        frequency, the gain maximum).
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    sigma : float
        Scaled field loss rate, > 0.
    mu : float
        Pump parameter, > 0.

    Returns
    -------
    state : np.ndarray
        Shape (5,): [omega, D0, X, Re(P0), Im(P0)] with F0 = sqrt(X) real
        and positive; omega and the envelopes in the scaled units of the
        equations.

    Raises
    ------
    ValueError
        If any argument is not a finite real number, if Gamma, sigma or mu is
        not positive, if the wavenumber lies outside the gain band (no
        positive-gain root of the dispersion relation), or if mu does not
        exceed the lasing threshold of this continuous wave (X <= 0).
    '''
    return state
```

### Step 2

02_effective_rabi_frequency

Goal
----
Step 02 - Effective Rabi frequency of the gain medium.

```python
def effective_rabi_frequency(X: float, Gamma: float, alpha: float, b: float) -> "np.ndarray":
    '''Effective Rabi frequency and damping of the medium for a clamped field.

    Parameters
    ----------
    X : float
        Field intensity |F|^2 (scaled), >= 0.
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    b : float
        Scaled carrier recovery rate, > 0.

    Returns
    -------
    rates : np.ndarray
        Shape (2,): [ERF, damping], the angular frequency of the clamped-field
        medium's oscillation and the magnitude of the rate at which its amplitude
        changes, both positive, in units of 1/tau_d.

    Raises
    ------
    ValueError
        If any argument is not a finite real number, if X < 0, if Gamma <= 0
        or b <= 0, or if the clamped-field medium is overdamped, so that it has no
        oscillation.
    '''
    return rates
```

### Step 3

03_sideband_growth_rate

Goal
----
Step 03 - Growth rate of a sideband perturbation of the continuous wave.

```python
def sideband_growth_rate(kn: float, k: float, Gamma: float, alpha: float, sigma: float, b: float, mu: float) -> float:
    '''Growth rate of a sideband of offset kn on the continuous wave of wavenumber k.

    Parameters
    ----------
    kn : float
        Scaled wavenumber offset of the sideband from the continuous wave.
    k : float
        Scaled wavenumber of the continuous wave (step 01).
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    sigma : float
        Scaled field loss rate, > 0.
    b : float
        Scaled carrier recovery rate, > 0.
    mu : float
        Pump parameter, above the lasing threshold of the continuous wave.

    Returns
    -------
    rate : float
        Fastest exponential growth rate of a perturbation of offset kn, in
        units of 1/tau_d.

    Raises
    ------
    ValueError
        If kn or b is not a finite real number or b <= 0, or for any
        condition under which step 01 raises for (k, Gamma, alpha, sigma, mu).
    '''
    return rate
```

### Step 4

04_parametric_gain_maximum

Goal
----
Step 04 - Parametric-gain maximum of the fundamental continuous wave.

```python
def parametric_gain_maximum(Gamma: float, alpha: float, sigma: float, b: float, mu: float, tau_d_ps: float, f_max_ghz: float) -> "np.ndarray":
    '''Position and height of the parametric-gain maximum of the k = 0 wave.

    Parameters
    ----------
    Gamma : float
        Scaled gain bandwidth, > 0.
    alpha : float
        Linewidth enhancement factor.
    sigma : float
        Scaled field loss rate, > 0.
    b : float
        Scaled carrier recovery rate, > 0.
    mu : float
        Pump parameter, above the lasing threshold of the k = 0 wave.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    f_max_ghz : float
        Upper end of the searched sideband offsets in GHz, > 0.

    Returns
    -------
    peak : np.ndarray
        Shape (2,): [frequency offset of the maximum in GHz, growth rate at
        the maximum in 1/ns].

    Raises
    ------
    ValueError
        If tau_d_ps or f_max_ghz is not a finite positive number, for any
        condition under which step 03 raises for these parameters, or if the
        growth rate is nowhere positive on f_max_ghz/4000 <= f <= f_max_ghz.
    '''
    return peak
```

### Step 5

05_sideband_onset_pump

Goal
----
Step 05 - Onset pump of a cavity sideband.

```python
def sideband_onset_pump(n: int, L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, mu_min: float, mu_max: float) -> float:
    '''Lowest pump in [mu_min, mu_max] at which cavity sideband n of the k = 0 wave grows.

    Parameters
    ----------
    n : int
        Sideband index, n >= 1 (offset of n free spectral ranges).
    L_mm : float
        Ring length in millimetres, > 0.
    n_group : float
        Group index, > 0.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    Gamma, alpha, sigma, b : float
        Scaled gain bandwidth (> 0), linewidth enhancement factor, field loss
        rate (> 0) and carrier recovery rate (> 0).
    mu_min, mu_max : float
        Pump window, mu_min < mu_max, with mu_min above the lasing threshold of
        the k = 0 wave.

    Returns
    -------
    mu_on : float
        Onset pump of sideband n, to 1e-9.

    Raises
    ------
    ValueError
        If n is not an integer >= 1, if L_mm, n_group or tau_d_ps is not a
        finite positive number, if mu_min >= mu_max, for any condition under
        which step 03 raises at mu_min (mu_min at or below the lasing
        threshold included), if the sideband growth rate is already positive
        at mu_min, or if it does not become positive by mu_max.
    '''
    return mu_on
```

### Step 6

06_harmonic_order_switch

Goal
----
Step 06 - Multimode threshold and first change of the predicted comb order.

```python
def harmonic_order_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float) -> "np.ndarray":
    '''Multimode threshold, onset order, first order-change pump and new order.

    Parameters
    ----------
    L_mm : float
        Ring length in millimetres, > 0.
    n_group : float
        Group index, > 0.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    Gamma, alpha, sigma, b : float
        Scaled gain bandwidth (> 0), linewidth enhancement factor, field loss
        rate (> 0) and carrier recovery rate (> 0).
    n_max : int
        Highest sideband index considered, n_max >= 2.
    mu_min, mu_max : float
        Pump window, mu_min < mu_max, mu_min above the lasing threshold of the
        k = 0 wave, with every sideband 1 ... n_max damped at mu_min.

    Returns
    -------
    result : np.ndarray
        Shape (4,): [mu_c, n_c, mu_star, n_new] with n_c and n_new stored as
        floats holding integer values.

    Raises
    ------
    ValueError
        If n_max is not an integer >= 2, for any condition under which step 05
        raises for its other arguments, if some sideband 1 ... n_max is
        already growing at mu_min, if no sideband starts to grow inside the
        window, or if the selected sideband does not change before mu_max.
    '''
    return result
```

### Step 7

07_erf_gain_offset_at_switch

Goal
----
Step 07 - Gain maximum minus effective Rabi frequency at the order change (orchestrator).

```python
def erf_gain_offset_at_switch(L_mm: float, n_group: float, tau_d_ps: float, Gamma: float, alpha: float, sigma: float, b: float, n_max: int, mu_min: float, mu_max: float, f_max_ghz: float) -> float:
    '''Gain-maximum frequency minus effective Rabi frequency at the first order change.

    Parameters
    ----------
    L_mm : float
        Ring length in millimetres, > 0.
    n_group : float
        Group index, > 0.
    tau_d_ps : float
        Polarisation dephasing time in picoseconds, > 0.
    Gamma, alpha, sigma, b : float
        Scaled gain bandwidth (> 0), linewidth enhancement factor, field loss
        rate (> 0) and carrier recovery rate (> 0).
    n_max : int
        Highest cavity sideband index considered, >= 2.
    mu_min, mu_max : float
        Pump window of step 06.
    f_max_ghz : float
        Upper end of the sideband offsets searched for the gain maximum, GHz.

    Returns
    -------
    offset : float
        f(gain maximum) - ERF at mu_star, in GHz.

    Raises
    ------
    ValueError
        For any condition under which steps 01, 02, 04 or 06 raise for these
        arguments.
    '''
    return offset
```
