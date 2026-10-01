# Physics-Optics-33

## Background

Ultrashort, ultra-intense laser pulses are now routinely delivered at focal intensities where every solid material placed in the beam is ionized within the first few optical cycles. Knowing the duration of such a pulse matters because almost every quantity an experiment reports, from the peak intensity that drives an ionization channel to the ponderomotive energy of the electrons it liberates, scales with it. Established temporal diagnostics, including intensity autocorrelation, frequency-resolved optical gating, spectral shearing interferometry and dispersion scans, all measure a beam that has been picked off and recollimated ahead of the final focusing optic, and the result is transported to the interaction point through a model of the intervening optics. In systems that use large-aperture transmissive elements, the aberrations, material dispersion and spatio-temporal couplings that model must absorb are exactly the ones that are hardest to characterize.

An ionized gas is attractive as a diagnostic medium precisely because it has already been destroyed. Free electrons created by optical-field ionization lower the refractive index in proportion to their density, and that index change follows the ionizing intensity distribution with sub-cycle promptness. When two coherent beams cross in a gas, the interference fringes of their combined field ionize the medium unevenly, and the resulting periodic electron-density modulation behaves as a volume phase grating. Such structures tolerate intensities several orders of magnitude beyond the damage threshold of dielectric optics, persist for tens to hundreds of picoseconds before recombination and ambipolar diffusion erase them, and can be written and read at the repetition rate of the laser.

Because the ionization rate depends on intensity through a steeply nonlinear, tunneling-like law, the electron density written into the gas is a sharply weighted map of where and when the field was strong. That sensitivity is what makes an ionized structure useful as a recorder: modest changes in the local field translate into large changes in the free-electron population, and features of the driving field that a linear medium would average away are imprinted on the plasma. Reading the structure out with a weak probe that satisfies the Bragg condition of the grating, and imaging the diffracted order, converts the spatial distribution of the recorded electron density into a camera image, with the geometry of the diffraction and of the imaging system setting the scale factor between image coordinates and distances inside the plasma.

The gases used for this purpose are ordinary ones, air or a noble gas at or near atmospheric density, and their response to a strong low-frequency field is described by strong-field ionization theory. In the regime where the Keldysh parameter is of order unity, neither the multiphoton nor the pure tunneling limit is adequate on its own, and the non-adiabatic rate formulations developed for atoms in intense low-frequency fields are used instead. Depletion of the neutral population matters as soon as an appreciable fraction of the atoms is ionized, so the ionization yield saturates where the field is strongest, which in turn shapes how the recorded structure responds to changes in the driving pulse.

## Problem

Petawatt-class facilities still infer their pulse duration from a beam sampled ahead of the final focusing optic, because autocorrelators, frequency-resolved optical gating and spectral shearing interferometry all need solid nonlinear media that neither survive nor stay linear at focus, and the transfer function carrying a pre-focus number to the interaction point can be wrong by more than half in large-aperture systems. Writing an ionization grating directly in the focal volume removes that transfer function: the pulse under test interferes with a counter-propagating replica of itself, and the axial extent of the structure they ionize is fixed by how far apart in time the two replicas can drift and still ionize the gas, so a weak Bragg-matched probe turns a temporal quantity into the size of a single diffraction spot. Your task is to invert that encoding and report the temporal full width at half maximum of the pulse that wrote the recorded structure.

Both pumps are identical, carry 1.20 mJ at 800 nm with Gaussian temporal envelopes, counter-propagate along a common axis and are brought into exact temporal coincidence at the axial origin, focused to a common Gaussian waist of 40.0 um 1/e^2 intensity radius whose 6.3 mm Rayleigh range lets the on-axis intensity stand for the whole interaction. The gas is argon at 2.45e25 m^-3 with an ionization potential of 15.7596 eV, and its field ionization follows the cycle-averaged Perelomov-Popov-Terent'ev rate of the anchor method for the m = 0 channel, evaluated with kappa = sqrt(2 I_p), effective principal quantum number n* = Z/kappa at Z = 1, power-law exponent nu = 2 n* - 1 and unit prefactor amplitude in atomic units, with the field amplitude taken from the cycle-averaged intensity through F = sqrt(I / I_at) at I_at = 3.5094452e16 W/cm^2 and rates expressed per second using an atomic unit of time of 2.4188843265e-17 s. Ionization depletes the neutral population, and at the 1.2 ps readout delay, well inside the 30 to 40 ps plasma lifetime, recombination, diffusion and collisional ionization have not yet acted.

The readout is a collimated 5 mm diameter, 5 nm bandwidth probe at 400 nm in a gas of refractive index 1.000 at that wavelength, held below 1e12 W/cm^2 to avoid modifying the plasma, with first-order diffraction efficiency below 0.35; the probe meets the structure at its Bragg angle, close to 30 degrees, against a critical density of 6.97e27 m^-3 at the probe wavelength. The +1 order is imaged onto a camera with 3.45 um pixels at a magnification of 7.00, and after the point-spread function is removed the axial envelope of the recorded spot has a corrected full-width-at-half-maximum count of 24.0 pixels. Convert that count to a length inside the plasma, calibrate grating length against pulse duration at this pump energy and focusing geometry, and invert the calibration at the measured length.

Use the following configuration throughout:

- Axial sampling: 121 uniformly spaced positions from 0 to 30.0 um; the grating length is twice the position at which the axial envelope first falls to half its value at the origin, located by linear interpolation between the two bracketing samples.
- Fringe sampling: 128 uniformly spaced phases starting at zero and spanning one full period.
- Time sampling: 401 uniformly spaced instants on an interval symmetric about the midpoint of the two pump envelopes and extending 4 pulse durations beyond the outermost envelope centre, integrated by the composite trapezoidal rule.
- Calibration grid: pulse durations from 25 fs to 115 fs inclusive in 1 fs steps, inverted by linear interpolation of duration against grating length.

Report the retrieved pulse duration in femtoseconds.

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

Focal peak intensity of one pump

Goal
----
Compute the on-axis peak intensity of one focused pump pulse from its energy, focal spot size and temporal width.

```python
def compute_peak_intensity(pulse_energy_j: float, waist_radius_m: float, duration_fwhm_s: float) -> float:
    '''Return the on-axis peak intensity of a focused Gaussian pump pulse.

    Parameters
    ----------
    pulse_energy_j : float
        Energy carried by the pulse, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the focal spot, in metres. Must be positive.
    duration_fwhm_s : float
        Full width at half maximum of the Gaussian temporal intensity envelope,
        in seconds. Must be positive.

    Returns
    -------
    peak_intensity_wcm2 : float
        On-axis peak intensity, in watts per square centimetre.

    Raises
    ------
    ValueError
        If any of the energy, the radius or the duration is not positive.
    '''
    return peak_intensity_wcm2
```

### Step 2

Standing-wave period, Bragg geometry and object-space grating length

Goal
----
Compute the fringe period of the ionizing standing wave, the probe Bragg angle, the angle between the grating axis and the diffracted order, and the object-space grating length implied by the recorded pixel count.

```python
def compute_readout_geometry(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float) -> "np.ndarray":
    '''Return the readout geometry and the object-space grating length.

    Parameters
    ----------
    pump_wavelength_m : float
        Central wavelength of both counter-propagating pumps, in metres. Must be
        positive.
    probe_wavelength_m : float
        Central wavelength of the readout probe, in metres. Must be positive and
        small enough for the Bragg order to exist.
    medium_index : float
        Refractive index of the gas at the probe wavelength. Must be positive.
    pixel_pitch_m : float
        Physical pixel pitch of the camera, in metres. Must be positive.
    magnification : float
        Object-to-image magnification of the imaging system. Must be positive.
    corrected_pixel_count : float
        Corrected full width at half maximum of the recorded axial envelope, in
        pixels. Must be positive.

    Returns
    -------
    geometry : np.ndarray
        Array of shape (4,) holding the fringe period in metres, the Bragg angle in
        radians, the angle between the grating axis and the first diffracted order
        in radians, and the grating length in metres.

    Raises
    ------
    ValueError
        If any supplied quantity is not positive, or if the probe wavelength is too
        long for the first Bragg order of the structure to exist.
    '''
    return geometry
```

### Step 3

Fringe-resolved interference intensity at one axial position

Goal
----
Compute the instantaneous intensity seen by the gas at one axial position, as a function of time and of position within the fringe pattern.

```python
def compute_interference_intensity(times_s: "np.ndarray", fringe_phases_rad: "np.ndarray", axial_position_m: float, peak_intensity_wcm2: float, duration_fwhm_s: float) -> "np.ndarray":
    '''Return the instantaneous intensity on a fringe-phase by time grid.

    Parameters
    ----------
    times_s : np.ndarray
        One-dimensional, non-empty array of time samples in seconds.
    fringe_phases_rad : np.ndarray
        One-dimensional, non-empty array of fringe phases in radians.
    axial_position_m : float
        Axial coordinate at which the intensity is evaluated, in metres.
    peak_intensity_wcm2 : float
        On-axis peak intensity of each individual pump, in watts per square
        centimetre. Must be non-negative.
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.

    Returns
    -------
    intensity_wcm2 : np.ndarray
        Array of shape (fringe_phases_rad.size, times_s.size) holding the
        instantaneous intensity in watts per square centimetre.

    Raises
    ------
    ValueError
        If either input array is empty or not one-dimensional, if the peak intensity
        is negative, or if the duration is not positive.
    '''
    return intensity_wcm2
```

### Step 4

Cycle-averaged field-ionization rate of the gas

Goal
----
Compute the cycle-averaged field-ionization rate of the gas at each supplied instantaneous intensity.

```python
def compute_ionization_rate(intensity_wcm2: "np.ndarray", ionization_potential_ev: float, pump_wavelength_m: float) -> "np.ndarray":
    '''Return the field-ionization rate for each supplied intensity.

    Parameters
    ----------
    intensity_wcm2 : np.ndarray
        Instantaneous cycle-averaged intensity in watts per square centimetre. Any
        shape is accepted. All entries must be finite.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of the driving field, in metres. Must be positive.

    Returns
    -------
    rate_per_s : np.ndarray
        Ionization rate in inverse seconds, with the same shape as the supplied
        intensity.

    Raises
    ------
    ValueError
        If any supplied intensity is not finite, or if the ionization potential or
        the wavelength is not positive.
    '''
    return rate_per_s
```

### Step 5

Fringe-resolved free-electron density after both pumps

Goal
----
Compute the free-electron density left behind across one fringe of the pattern at a single axial position, after both pumps have passed.

```python
def compute_electron_density(axial_position_m: float, duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    '''Return the fringe-resolved free-electron density at one axial position.

    Parameters
    ----------
    axial_position_m : float
        Axial coordinate at which the density is evaluated, in metres.
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.
    peak_intensity_wcm2 : float
        On-axis peak intensity of each individual pump, in watts per square
        centimetre. Must be non-negative.
    neutral_density_m3 : float
        Neutral number density of the gas before the pulses arrive, in inverse cubic
        metres. Must be non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
    n_fringe_phase : int
        Number of uniformly spaced fringe phases covering one full period. Must be
        at least one.
    n_time : int
        Number of uniformly spaced time samples. Must be at least two.
    time_window_factor : float
        Multiple of the pulse duration by which the time interval extends beyond the
        outermost envelope centre. Must be positive.

    Returns
    -------
    electron_density_m3 : np.ndarray
        Array of shape (n_fringe_phase,) holding the free-electron density in
        inverse cubic metres.

    Raises
    ------
    ValueError
        If the neutral density is negative, if the number of fringe phases is below
        one, if the number of time samples is below two, if the window factor is not
        positive, or if any quantity forwarded to the intensity or rate evaluation is
        invalid.
    '''
    return electron_density_m3
```

### Step 6

First spatial harmonic of the written electron density

Goal
----
Compute the amplitude of the first spatial harmonic of the written electron density at each supplied axial position.

```python
def compute_first_harmonic(axial_positions_m: "np.ndarray", duration_fwhm_s: float, peak_intensity_wcm2: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    '''Return the first-harmonic density amplitude at each axial position.

    Parameters
    ----------
    axial_positions_m : np.ndarray
        One-dimensional, non-empty array of axial coordinates in metres.
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.
    peak_intensity_wcm2 : float
        On-axis peak intensity of each individual pump, in watts per square
        centimetre. Must be non-negative.
    neutral_density_m3 : float
        Neutral number density of the gas before the pulses arrive, in inverse cubic
        metres. Must be non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
    n_fringe_phase : int
        Number of uniformly spaced fringe phases covering one full period. Must be
        at least one.
    n_time : int
        Number of uniformly spaced time samples. Must be at least two.
    time_window_factor : float
        Multiple of the pulse duration by which the time interval extends beyond the
        outermost envelope centre. Must be positive.

    Returns
    -------
    harmonic_amplitude_m3 : np.ndarray
        Array of shape (axial_positions_m.size,) holding the first-harmonic density
        amplitude in inverse cubic metres.

    Raises
    ------
    ValueError
        If the array of axial positions is empty or not one-dimensional, or if any
        quantity forwarded to the fringe-resolved density evaluation is invalid.
    '''
    return harmonic_amplitude_m3
```

### Step 7

Effective grating length for one pump duration

Goal
----
Compute the effective grating length written by a pump pulse of the supplied duration at fixed pump energy and focusing geometry.

```python
def compute_grating_length(duration_fwhm_s: float, pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    '''Return the effective grating length for one pump duration.

    Parameters
    ----------
    duration_fwhm_s : float
        Full width at half maximum of each pump's Gaussian temporal intensity
        envelope, in seconds. Must be positive.
    pulse_energy_j : float
        Energy of each individual pump, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the common focal spot, in metres. Must be positive.
    neutral_density_m3 : float
        Neutral number density of the gas, in inverse cubic metres. Must be
        non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
    z_max_m : float
        Largest sampled axial position, in metres. Must be positive.
    n_axial : int
        Number of uniformly spaced axial positions from zero to z_max_m inclusive.
        Must be at least two.
    n_fringe_phase : int
        Number of uniformly spaced fringe phases covering one full period. Must be
        at least one.
    n_time : int
        Number of uniformly spaced time samples. Must be at least two.
    time_window_factor : float
        Multiple of the pulse duration by which the time interval extends beyond the
        outermost envelope centre. Must be positive.

    Returns
    -------
    grating_length_m : float
        Effective grating length, in metres.

    Raises
    ------
    ValueError
        If z_max_m is not positive, if n_axial is below two, if the axial envelope
        does not fall to half its value at the origin anywhere within the sampled
        range, or if any quantity forwarded to the earlier evaluations is invalid.
    '''
    return grating_length_m
```

### Step 8

Grating-length calibration against pump duration

Goal
----
Compute the calibration of effective grating length against pump duration for one pump energy and focusing geometry.

```python
def compute_calibration_curve(durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, pump_wavelength_m: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> "np.ndarray":
    '''Return the grating length for each trial pump duration.

    Parameters
    ----------
    durations_s : np.ndarray
        One-dimensional, non-empty array of trial durations in seconds. All entries
        must be positive.
    pulse_energy_j : float
        Energy of each individual pump, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the common focal spot, in metres. Must be positive.
    neutral_density_m3 : float
        Neutral number density of the gas, in inverse cubic metres. Must be
        non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
    z_max_m : float
        Largest sampled axial position, in metres. Must be positive.
    n_axial : int
        Number of uniformly spaced axial positions from zero to z_max_m inclusive.
        Must be at least two.
    n_fringe_phase : int
        Number of uniformly spaced fringe phases covering one full period. Must be
        at least one.
    n_time : int
        Number of uniformly spaced time samples. Must be at least two.
    time_window_factor : float
        Multiple of the pulse duration by which the time interval extends beyond the
        outermost envelope centre. Must be positive.

    Returns
    -------
    grating_lengths_m : np.ndarray
        Array of shape (durations_s.size,) holding the grating length in metres for
        each trial duration.

    Raises
    ------
    ValueError
        If the array of durations is empty, not one-dimensional or contains a
        non-positive entry, or if any quantity forwarded to the single-duration
        evaluation is invalid.
    '''
    return grating_lengths_m
```

### Step 9

Retrieved far-field pulse duration

Goal
----
Compute the pulse duration recorded by the diagnostic, from the imaged extent of the first diffraction order and the numerical calibration of the plasma structure.

```python
def retrieve_pulse_duration(pump_wavelength_m: float, probe_wavelength_m: float, medium_index: float, pixel_pitch_m: float, magnification: float, corrected_pixel_count: float, calibration_durations_s: "np.ndarray", pulse_energy_j: float, waist_radius_m: float, neutral_density_m3: float, ionization_potential_ev: float, z_max_m: float, n_axial: int, n_fringe_phase: int, n_time: int, time_window_factor: float) -> float:
    '''Return the retrieved pulse duration in femtoseconds.

    Parameters
    ----------
    pump_wavelength_m : float
        Central wavelength of both pumps, in metres. Must be positive.
    probe_wavelength_m : float
        Central wavelength of the readout probe, in metres. Must be positive and
        small enough for the Bragg order to exist.
    medium_index : float
        Refractive index of the gas at the probe wavelength. Must be positive.
    pixel_pitch_m : float
        Physical pixel pitch of the camera, in metres. Must be positive.
    magnification : float
        Object-to-image magnification of the imaging system. Must be positive.
    corrected_pixel_count : float
        Corrected full width at half maximum of the recorded axial envelope, in
        pixels. Must be positive.
    calibration_durations_s : np.ndarray
        One-dimensional array of at least two trial durations in seconds, strictly
        increasing and all positive.
    pulse_energy_j : float
        Energy of each individual pump, in joules. Must be positive.
    waist_radius_m : float
        1/e^2 intensity radius of the common focal spot, in metres. Must be positive.
    neutral_density_m3 : float
        Neutral number density of the gas, in inverse cubic metres. Must be
        non-negative.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    z_max_m : float
        Largest sampled axial position, in metres. Must be positive.
    n_axial : int
        Number of uniformly spaced axial positions from zero to z_max_m inclusive.
        Must be at least two.
    n_fringe_phase : int
        Number of uniformly spaced fringe phases covering one full period. Must be
        at least one.
    n_time : int
        Number of uniformly spaced time samples. Must be at least two.
    time_window_factor : float
        Multiple of the pulse duration by which the time interval extends beyond the
        outermost envelope centre. Must be positive.

    Returns
    -------
    retrieved_duration_fs : float
        Retrieved pulse duration, in femtoseconds.

    Raises
    ------
    ValueError
        If fewer than two trial durations are supplied, if the calibrated lengths are
        not strictly increasing with duration, if the measured length falls outside
        the calibrated range, or if any quantity forwarded to the earlier evaluations
        is invalid.
    '''
    return retrieved_duration_fs
```
