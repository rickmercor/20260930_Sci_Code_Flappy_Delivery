# Chemistry-Quantum_Chemistry-3

## Background

Magic-angle spinning is the standard route to high-resolution spectra of powdered solids. Rotating the sample about an axis inclined at the magic angle to the static field averages second-rank anisotropic interactions to zero over a rotor period, which sharpens the lines but also removes the dipole-dipole couplings that carry internuclear distance information and that drive polarization transfer between nuclei. Experiments therefore apply radio-frequency irradiation designed to interfere with the sample rotation in a controlled way, so that a chosen coupling is reintroduced while the unwanted averaging is left intact.

The difficulty is that the same sample rotation modulates every other anisotropic interaction as well. When a nucleus carries a chemical shielding anisotropy many times larger than the coupling being reintroduced, as fluorine and many quadrupolar or heavy nuclei do, the resonance offset experienced during irradiation swings over hundreds of kilohertz within a single rotor period. Matching conditions that are narrow in offset then hold only for a small fraction of the crystallites in a powder, and the transfer efficiency collapses. Established remedies such as ramped or adiabatic contact, or symmetry-based sequences, buy robustness in one variable at the cost of sensitivity in another.

A useful way to think about a periodic irradiation element is through the constant field it generates. Over one short, rotor-synchronized modulation period the element produces a net rotation, and when the element is repeated that rotation behaves like a constant field applied to the spin. Describing the experiment in terms of these fields separates the design problem into single-spin pieces: each channel can be shaped independently as long as the fields generated on the two channels satisfy a matching relation, and the quality of an element on one channel is measured by how little its field varies across crystallite orientations and resonance offsets.

Because a single-spin rotation has only three degrees of freedom, it can be propagated with quaternions rather than with propagator matrices. Quaternion composition is exact, stays on the rotation group without renormalization, and is cheap enough that a long piecewise-constant element evaluated over many crystallites and many offsets remains tractable inside an optimization loop. Since the element is an ordered product of interval rotations, derivatives with respect to the parameters of any single interval follow from the product rule, so the design problem becomes a gradient-driven search over the amplitudes and phases of the intervals. Orientational averages in this setting are quadratures over the Euler angles that carry an interaction tensor from its principal axis frame into the rotor frame, with the polar angle integrated in its cosine.

## Problem

Heteronuclear polarization transfer in magic-angle-spinning solid-state NMR relies on reintroducing a dipole-dipole coupling that sample rotation would otherwise average away, and it degrades badly when one of the two spins carries an anisotropic chemical shielding far larger than the coupling being recoupled. A productive way to attack this is to stop treating the two-spin transfer as a single optimization and instead characterize the irradiation applied to the shielding-dominated spin on its own: a short, rotor-synchronized element of piecewise-constant amplitude and phase is judged entirely by the constant field it generates when the element is repeated, and a good element produces the same field for every crystallite in a powder and across the whole band of isotropic offsets the sample presents. The quantities entering the calculation are the interval amplitudes and phases of the element, the isotropic and anisotropic shielding of the irradiated spin, the spinning frequency, the crystallite set and the offset band; the quantity leaving it is one number characterizing the generated field.

Over one interval the irradiated spin evolves as a rigid rotation whose axis is fixed by the irradiation amplitude, by the irradiation phase relative to the rotating-frame x axis and by the instantaneous resonance offset, and whose angle is fixed by the length of the interval. The resonance offset is not constant: the anisotropic shielding of a given crystallite is modulated by the sample rotation, so it contributes a rotor-periodic term that must be reconstructed from the second-rank spatial tensor of the shielding, carried from its principal axis frame into the rotor frame and then into the laboratory frame, before being added to the isotropic offset. The element as a whole is the ordered composition of its interval rotations, its agreement with a chosen design rotation is the overlap between the two rotations averaged over crystallites and offsets, and that agreement is differentiable with respect to every interval amplitude and every interval phase.

Compute the following for a fluorine channel irradiated at a spinning frequency of 25 kHz with the rotor at the magic angle. The element spans a modulation period of 80 us as 40 intervals of 2 us each, with starting amplitudes in Hz and phases in degrees measured from the x axis given below, one interval per row in time order. The irradiated spin has zero isotropic shielding of its own, an anisotropic shielding of 80 kHz and an asymmetry parameter of 0.5; its resonance offset for interval j, counted from zero, is the isotropic offset plus the rotor-modulated anisotropic shielding evaluated at the end of that interval, at time (j+1) times 2 us. Second-rank Wigner rotation matrix elements are taken as D^(2)_{m',m}(alpha, beta, gamma) = exp(-i m' alpha) d^(2)_{m',m}(beta) exp(-i m gamma). The design rotation is a rotation through 120 degrees about the x axis accumulated over the whole modulation period, and the agreement is averaged over 7 isotropic offsets equally spaced from -35 kHz to +35 kHz with equal weight and over a crystallite set built as a product quadrature in the three Euler angles carrying the shielding principal axis frame into the rotor frame: 6 Gauss-Legendre nodes in cos(beta) carrying their Gauss-Legendre weights, 6 equally spaced values alpha_i = 2 pi i / 6 for i = 0..5 and 3 equally spaced values gamma_l = 2 pi l / 3 for l = 0,1,2, the two azimuthal sets carrying equal weights, with the crystallite weights normalized to sum to one. Starting from the element below, apply exactly 10 steepest-ascent iterations: in each iteration divide the amplitude derivatives of the agreement by the largest amplitude-derivative magnitude and the phase derivatives by the largest phase-derivative magnitude, add 200 Hz times the rescaled amplitude derivatives to the amplitudes and 0.01 rad times the rescaled phase derivatives to the phases, then clip every amplitude into the range 0 to 100 kHz. For the refined element, read off the constant field that generates its rotation over the modulation period, taking the magnitude in the branch below one full turn per modulation period, and average the x component over the same crystallites and offsets.

| interval | amplitude / Hz | phase / deg | interval | amplitude / Hz | phase / deg |
|---|---|---|---|---|---|
| 1 | 99882.6002 | 308.58 | 21 | 99192.2675 | 18.55 |
| 2 | 97593.2747 | 91.90 | 22 | 75896.7021 | 262.57 |
| 3 | 99938.7784 | 322.74 | 23 | 99512.2651 | 169.77 |
| 4 | 99788.5816 | 235.95 | 24 | 99014.8033 | 79.16 |
| 5 | 98277.6433 | 158.68 | 25 | 99886.4254 | 9.71 |
| 6 | 99577.6807 | 107.58 | 26 | 99490.0204 | 311.73 |
| 7 | 98881.0528 | 55.28 | 27 | 99904.9940 | 264.74 |
| 8 | 94404.7994 | 30.57 | 28 | 99625.5643 | 231.86 |
| 9 | 99306.0922 | 8.96 | 29 | 99500.3110 | 212.56 |
| 10 | 99774.6973 | 6.07 | 30 | 99651.1342 | 208.43 |
| 11 | 98040.4008 | 358.64 | 31 | 99852.4588 | 204.51 |
| 12 | 99810.2557 | 19.22 | 32 | 99754.5915 | 222.47 |
| 13 | 91823.9307 | 35.18 | 33 | 99205.2418 | 242.32 |
| 14 | 99879.0133 | 64.16 | 34 | 99857.1682 | 268.75 |
| 15 | 99432.6134 | 115.58 | 35 | 99945.0147 | 319.18 |
| 16 | 99913.2100 | 171.43 | 36 | 99554.7653 | 13.80 |
| 17 | 99930.7510 | 249.82 | 37 | 99728.1894 | 94.72 |
| 18 | 99937.6518 | 334.75 | 38 | 99434.0476 | 193.81 |
| 19 | 99206.1576 | 84.39 | 39 | 71090.5744 | 327.26 |
| 20 | 99416.0881 | 205.60 | 40 | 98881.1991 | 38.66 |

The element is intended for transfer to a directly bonded carbon spin with a dipole-dipole coupling constant of -12.1 kHz, matched on the carbon channel by a square pulse of 16.5 kHz amplitude over a mixing time of 400 us, and the carbon spin carries no anisotropic shielding of its own. Report along the way the same crystallite- and offset-averaged field for the unrefined element, the full averaged field vector of the refined element, and the averaged x component the unrefined element produces when the anisotropic shielding is removed. Your final answer must be a single number: the crystallite- and offset-averaged x component of the effective field of the refined element, expressed in Hz.

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

01_build_powder_grid

Goal
----
Implement build_powder_grid which returns the weighted crystallite orientation set used for powder averaging.

```python
def build_powder_grid(n_beta: int, n_alpha: int, n_gamma: int) -> "np.ndarray":
    '''Build the weighted crystallite orientation set for powder averaging.

    The rows are ordered with the polar index varying slowest, then the alpha
    index, then the gamma index varying fastest.

    Parameters
    ----------
    n_beta : int
        Number of Gauss-Legendre nodes in cos(beta). Must be >= 1.
    n_alpha : int
        Number of equally spaced alpha nodes, alpha_i = 2*pi*i/n_alpha for
        i = 0, ..., n_alpha - 1. Must be >= 1.
    n_gamma : int
        Number of equally spaced gamma nodes, gamma_l = 2*pi*l/n_gamma for
        l = 0, ..., n_gamma - 1. Must be >= 1.

    Returns
    -------
    grid : np.ndarray
        Real array of shape (n_beta*n_alpha*n_gamma, 4) whose columns are
        alpha, beta and gamma in radians and the normalized crystallite
        weight. beta is taken as arccos of the Gauss-Legendre node, so it
        lies in (0, pi). The weights sum to 1.

    Raises
    ------
    ValueError
        If any of n_beta, n_alpha or n_gamma is not a positive integer.
    '''
    return grid  # placeholder
```

### Step 2

02_compute_shielding_fourier_components

Goal
----
Implement compute_shielding_fourier_components which returns the rotor-modulated Fourier coefficients of a single-spin shielding Hamiltonian.

```python
def compute_shielding_fourier_components(omega_iso: float, omega_aniso: float, eta: float,
                                         alpha_pr: float, beta_pr: float, gamma_pr: float,
                                         beta_rl: float) -> "np.ndarray":
    '''Compute the five rotor-frame Fourier coefficients of a shielding interaction.

    The coefficients are defined so that the instantaneous secular shielding
    frequency of the spin is sum over m from -2 to 2 of
    components[m + 2] * exp(i * m * omega_r * t).

    Parameters
    ----------
    omega_iso : float
        Isotropic shielding frequency in rad/s, measured relative to the
        radio-frequency carrier.
    omega_aniso : float
        Anisotropic shielding frequency in rad/s.
    eta : float
        Shielding asymmetry parameter, 0 <= eta <= 1.
    alpha_pr : float
        First Euler angle in radians carrying the shielding principal axis
        frame into the rotor frame.
    beta_pr : float
        Second Euler angle in radians carrying the shielding principal axis
        frame into the rotor frame.
    gamma_pr : float
        Third Euler angle in radians carrying the shielding principal axis
        frame into the rotor frame.
    beta_rl : float
        Angle in radians between the rotor axis and the static field.

    Returns
    -------
    components : np.ndarray
        Complex array of shape (5,) holding the coefficients for
        m = -2, -1, 0, 1, 2 in that order.

    Raises
    ------
    ValueError
        If eta is outside the interval [0, 1], or if any argument is not
        finite.
    '''
    return components  # placeholder
```

### Step 3

03_compute_offset_trajectory

Goal
----
Implement compute_offset_trajectory which reconstructs the instantaneous shielding frequency seen by a spin during sample rotation.

```python
def compute_offset_trajectory(components: "np.ndarray", omega_r: float,
                              times: "np.ndarray") -> "np.ndarray":
    '''Evaluate the rotor-modulated shielding frequency at a set of instants.

    Parameters
    ----------
    components : np.ndarray
        Complex array whose trailing axis has length 5 and holds the Fourier
        coefficients for m = -2, -1, 0, 1, 2 in that order. Leading axes are
        treated as independent crystallites.
    omega_r : float
        Sample spinning frequency in rad/s.
    times : np.ndarray
        Real array of shape (n_steps,) holding the sampling instants in
        seconds.

    Returns
    -------
    trajectory : np.ndarray
        Real array of shape components.shape[:-1] + (n_steps,) holding the
        shielding frequency in rad/s at each instant, obtained as the real
        part of the reconstructed series.

    Raises
    ------
    ValueError
        If the trailing axis of components does not have length 5, if times
        is not one dimensional, if omega_r is not finite, or if times
        contains values that are not finite.
    '''
    return trajectory  # placeholder
```

### Step 4

04_build_interval_quaternion

Goal
----
Implement build_interval_quaternion which represents one piecewise-constant irradiation interval as a rotation quaternion.

```python
def build_interval_quaternion(omega_rf: "np.ndarray", phi_rf: "np.ndarray",
                              delta_omega: "np.ndarray", dt: float) -> "np.ndarray":
    '''Build the rotation quaternion of one constant-amplitude irradiation interval.

    Parameters
    ----------
    omega_rf : np.ndarray
        Radio-frequency amplitude in rad/s. Scalars and arrays are accepted
        and are broadcast against the other two spin parameters.
    phi_rf : np.ndarray
        Radio-frequency phase in radians, measured from the x axis of the
        rotating frame.
    delta_omega : np.ndarray
        Instantaneous resonance offset in rad/s, including any rotor-modulated
        shielding contribution.
    dt : float
        Interval duration in seconds. Must be non-negative and finite.

    Returns
    -------
    quaternion : np.ndarray
        Real array of shape broadcast_shape + (4,) holding the scalar-first
        quaternion of the interval. An interval with vanishing amplitude and
        vanishing offset returns the identity quaternion (1, 0, 0, 0).

    Raises
    ------
    ValueError
        If dt is negative or not finite, if any spin parameter is not finite,
        or if the three spin parameters cannot be broadcast together.
    '''
    return quaternion  # placeholder
```

### Step 5

05_compose_quaternion_sequence

Goal
----
Implement compose_quaternion_sequence which accumulates the rotation of a whole pulse sequence element from its per-interval quaternions.

```python
def compose_quaternion_sequence(quaternions: "np.ndarray") -> "np.ndarray":
    '''Compose an ordered set of interval quaternions into the element quaternion.

    The interval stored at index 0 acts first and the interval stored at the
    last index acts last, so the returned quaternion is the product in which
    the last interval stands leftmost.

    Parameters
    ----------
    quaternions : np.ndarray
        Real array of shape leading_shape + (n_steps, 4) holding scalar-first
        interval quaternions in time order. Leading axes are treated as
        independent sequences. n_steps must be at least 1.

    Returns
    -------
    q_total : np.ndarray
        Real array of shape leading_shape + (4,) holding the scalar-first
        quaternion of the whole sequence.

    Raises
    ------
    ValueError
        If quaternions has fewer than two axes, if its trailing axis does not
        have length 4, if it holds no intervals, or if it contains values
        that are not finite.
    '''
    return q_total  # placeholder
```

### Step 6

06_extract_effective_field

Goal
----
Implement extract_effective_field which converts the rotation of a periodic pulse element into the constant field that generates it.

```python
def extract_effective_field(q_total: "np.ndarray", tau_m: float) -> "np.ndarray":
    '''Extract the constant effective field generated by a periodic pulse element.

    The quaternion uses the same scalar-first convention as the interval
    quaternions, so the rotation angle is recovered from the scalar part and
    the norm of the vector part, and the rotation axis is recovered from the
    vector part. The magnitude is taken in the branch [0, 2*pi/tau_m], and an
    identity quaternion returns a vanishing field.

    Parameters
    ----------
    q_total : np.ndarray
        Real array of shape leading_shape + (4,) holding scalar-first
        quaternions of complete pulse sequence elements.
    tau_m : float
        Modulation period of the element in seconds. Must be positive and
        finite.

    Returns
    -------
    effective_field : np.ndarray
        Real array of shape leading_shape + (3,) holding the Cartesian
        components of the effective field in rad/s.

    Raises
    ------
    ValueError
        If tau_m is not positive and finite, if the trailing axis of q_total
        does not have length 4, or if q_total contains values that are not
        finite.
    '''
    return effective_field  # placeholder
```

### Step 7

07_compute_fidelity_and_gradient

Goal
----
Implement compute_fidelity_and_gradient which scores a pulse element against a target rotation and returns the exact derivatives of that score.

```python
def compute_fidelity_and_gradient(amplitudes: "np.ndarray", phases: "np.ndarray", dt: float,
                                  offsets: "np.ndarray", crystallites: "np.ndarray",
                                  omega_aniso: float, eta: float, beta_rl: float,
                                  omega_r: float, q_target: "np.ndarray") -> "np.ndarray":
    '''Score a pulse element against a target rotation and differentiate the score.

    The score is the weighted mean, over crystallites and over resonance
    offsets, of the Euclidean inner product between q_target and the
    scalar-first quaternion of the whole element. Crystallites carry the
    weights supplied in the grid, and the offsets are weighted equally.

    Parameters
    ----------
    amplitudes : np.ndarray
        Real array of shape (n_steps,) holding the radio-frequency amplitude
        of each interval in rad/s. n_steps must be at least 1.
    phases : np.ndarray
        Real array of shape (n_steps,) holding the radio-frequency phase of
        each interval in radians.
    dt : float
        Duration of one interval in seconds. Must be positive and finite. The
        rotor-modulated shielding of interval j is sampled at the end of that
        interval, at time (j + 1) * dt with j counted from zero.
    offsets : np.ndarray
        Real array of shape (n_offsets,) holding the isotropic resonance
        offsets in rad/s over which the score is averaged. n_offsets must be
        at least 1.
    crystallites : np.ndarray
        Real array of shape (n_crystallites, 4) whose columns are the three
        Euler angles in radians carrying the shielding principal axis frame
        into the rotor frame, and the crystallite weight.
    omega_aniso : float
        Anisotropic shielding frequency in rad/s.
    eta : float
        Shielding asymmetry parameter, 0 <= eta <= 1.
    beta_rl : float
        Angle in radians between the rotor axis and the static field.
    omega_r : float
        Sample spinning frequency in rad/s.
    q_target : np.ndarray
        Real array of shape (4,) holding the scalar-first target quaternion.

    Returns
    -------
    result : np.ndarray
        Real array of shape (2*n_steps + 1,). Element 0 is the score. Elements
        1 to n_steps are the derivatives of the score with respect to the
        interval amplitudes, in rad/s units of amplitude. Elements n_steps + 1
        to 2*n_steps are the derivatives with respect to the interval phases.

    Raises
    ------
    ValueError
        If amplitudes and phases are not one-dimensional arrays of the same
        positive length, if dt is not positive and finite, if offsets is not
        a non-empty one-dimensional array, if crystallites is not an array of
        shape (n_crystallites, 4) with at least one row, if q_target does not
        have shape (4,), or if eta lies outside [0, 1].
    '''
    return result  # placeholder
```

### Step 8

08_run_pipeline

Goal
----
Implement run_pipeline which refines a piecewise-constant irradiation element against a target rotation and reports the effective field it generates.

```python
def run_pipeline(amplitudes_hz: "np.ndarray", phases_deg: "np.ndarray", dt: float,
                 omega_r: float, omega_aniso: float, eta: float, offsets: "np.ndarray",
                 grid_shape: tuple, flip_angle: float, n_iter: int, amp_step: float,
                 phase_step: float, amp_max: float) -> float:
    '''Refine an irradiation element and report its mean effective-field x component.

    The rotor axis is inclined at the magic angle. The target rotation is a
    rotation through flip_angle about the x axis of the rotating frame, taken
    over the whole modulation period, which is the interval duration times the
    number of intervals.

    Each iteration evaluates the score and its derivatives for the current
    element, divides the amplitude derivatives by the largest absolute
    amplitude derivative and the phase derivatives by the largest absolute
    phase derivative, adds amp_step times the rescaled amplitude derivatives
    to the amplitudes and phase_step times the rescaled phase derivatives to
    the phases, and then clips the amplitudes to the interval from zero to
    amp_max. A derivative block whose largest magnitude is zero leaves its
    parameters unchanged.

    Parameters
    ----------
    amplitudes_hz : np.ndarray
        Real array of shape (n_steps,) holding the starting radio-frequency
        amplitude of each interval in Hz.
    phases_deg : np.ndarray
        Real array of shape (n_steps,) holding the starting radio-frequency
        phase of each interval in degrees.
    dt : float
        Duration of one interval in seconds.
    omega_r : float
        Sample spinning frequency in rad/s.
    omega_aniso : float
        Anisotropic shielding frequency in rad/s.
    eta : float
        Shielding asymmetry parameter, 0 <= eta <= 1.
    offsets : np.ndarray
        Real array of shape (n_offsets,) holding the isotropic resonance
        offsets in rad/s over which the element is scored and averaged.
    grid_shape : tuple
        Three positive integers giving the number of polar, first azimuthal
        and second azimuthal crystallite nodes.
    flip_angle : float
        Target rotation angle in radians about the x axis.
    n_iter : int
        Number of steepest-ascent iterations. Must be non-negative. Zero
        returns the field of the unrefined element.
    amp_step : float
        Largest amplitude increment applied in one iteration, in rad/s.
    phase_step : float
        Largest phase increment applied in one iteration, in radians.
    amp_max : float
        Upper bound on the radio-frequency amplitude in rad/s.

    Returns
    -------
    mean_field_x : float
        The crystallite- and offset-averaged x component of the effective
        field of the refined element, converted to Hz, as a native Python
        float.

    Raises
    ------
    ValueError
        If n_iter is negative or not an integer, if amp_max is not positive
        and finite, if amp_step or phase_step is negative, if grid_shape is
        not a sequence of three integers, or if any argument forwarded to an
        earlier stage fails its own validation.
    '''
    return mean_field_x  # placeholder
```
