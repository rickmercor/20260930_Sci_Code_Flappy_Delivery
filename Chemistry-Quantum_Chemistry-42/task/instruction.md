# Chemistry-Quantum_Chemistry-42

## Background

In the time-dependent picture of molecular spectroscopy, an electronic absorption band at zero temperature follows from the dynamics of the vibrational wavepacket that a vertical transition places on the excited-state potential energy surface. Within the Condon approximation the spectrum is a Fourier transform of the autocorrelation function between the initial vibrational state and the propagated wavepacket, so the spectrum's vibronic peak positions, spacings and envelope are set by how the wavepacket moves and spreads on the excited surface. Numerically exact propagation on a grid is feasible only for a few vibrational coordinates, which is why approximate semiclassical propagators matter for real molecules.

A widely used family of approximations represents the wavepacket by a single Gaussian with a time-dependent centre, momentum, complex width matrix and phase. The Gaussian is propagated exactly in a quadratic model of the surface built around its current position, so the quality of the result depends on which second-derivative matrix enters that model. Re-evaluating the curvature along the path follows the anharmonic surface closely but requires a Hessian at every step, which is costly when the surface comes from electronic-structure calculations, and it can also let the width grow without bound. Freezing the curvature at a positive-definite matrix removes that cost and keeps the width bounded, at the price of choosing a geometry at which to evaluate it.

Approximate spectra are commonly compared with an exact reference through overlap measures that treat spectra as vectors, which are insensitive to overall intensity but sensitive to peak positions and envelopes when no frequency shift is applied. Complementary diagnostics come from conserved quantities: the exact dynamics conserves the energy expectation value, whereas an approximate Gaussian propagation conserves at most the energy of its own effective Hamiltonian, and how far the true energy wanders reveals the stability of the approximation. Long propagations with such Gaussians are usually done with splitting schemes composed into higher-order, time-reversible integrators.

## Problem

Vibrationally resolved electronic spectra of polyatomic molecules can be simulated at the cost of one classical trajectory by propagating a single thawed Gaussian wavepacket, whose centre follows the classical trajectory on the anharmonic excited-state surface while its width evolves in a quadratic expansion of that surface about the centre. Re-evaluating the Hessian at every step is the expensive part of this approach, and replacing it by one constant reference Hessian keeps the anharmonic trajectory at the cost of classical dynamics. That reference has to be taken at some geometry, usually the excited-state minimum or the Franck-Condon point, and which of the two gives the better spectrum depends on how far the excited-state minimum is displaced from the initial geometry. For a two-mode model of an absorption band, the task is to find the displacement at which the Franck-Condon choice becomes exactly as accurate as the excited-state-minimum choice.

Use mass-weighted coordinates q = (q1, q2) with unit masses and hbar = 1. The initial electronic state is harmonic with its minimum at the origin and Hessian R diag(omega_1^2, omega_2^2) R^T, where R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], omega_1 = 1, omega_2 = 0.5 and theta = 20 degrees, and the wavepacket at t = 0 is its vibrational ground state psi_0 (zero temperature and the Condon approximation). On the excited state it evolves under H = -(1/2)(d^2/dq1^2 + d^2/dq2^2) + V(q1, q2) with V(q1, q2) = D [1 - exp(-a (q1 - d))]^2 + (1/2) omega_b^2 exp(-gamma (q1 - d)) (q2 - delta)^2, D = omega_e / (4 chi), a = (2 omega_e chi)^(1/2), omega_e = 0.85, chi = 0.02, omega_b = 0.45, gamma = 0.3 and delta = 0.8, where the stretch displacement d > 0 is the quantity to be varied. The exact autocorrelation function is C(t) = <psi_0| exp(-i H t) |psi_0>, computed to numerical convergence.

Each approximate calculation propagates one Gaussian that equals psi_0 at t = 0 and evolves exactly under the time-dependent quadratic potential V(q_t) + grad V(q_t) . (q - q_t) + (1/2) (q - q_t)^T K (q - q_t) about its instantaneous centre q_t, and four choices of K are compared: the Hessian of V at q_t (local harmonic), the Hessian of V at the excited-state minimum (d, delta) (adiabatic), the Hessian of V at the origin (vertical) and the initial-state Hessian (initial). For any autocorrelation, the broadened lineshape is I(omega) = Re of the integral from 0 to infinity of C(t) exp(i omega t) exp(-t^2 / (2 tau^2)) dt, with tau chosen so that every line of the exact stick spectrum becomes a Gaussian with half-width at half-maximum 0.25, and the accuracy of an approximate lineshape I_a against the exact lineshape I_x is the cosine of the unshifted spectral contrast angle, cos(theta_c) = (integral of I_a I_x d omega) / [(integral of I_a^2 d omega)(integral of I_x^2 d omega)]^(1/2), with every frequency integral taken over the whole real line. At small d the vertical cosine lies below the adiabatic cosine; find the smallest displacement d_x > 0 at which the two cosines are equal, and give d_x to three decimal places as the final answer.

In your reasoning, report the following scalars, which determine and check the answer: the damping time tau; at d = 1, the three independent elements of the vertical reference Hessian, the energy expectation value <psi_0|H|psi_0> of the initial vibrational state on the excited state and the vertical cosine; at d_x, the common value of the vertical and adiabatic cosines, the local harmonic cosine and the initial cosine; the three independent elements of the vertical reference Hessian at d_x; at d_x, the largest absolute deviation of the energy expectation value <psi_t|H|psi_t> from its value at t = 0 over 0 <= t <= 6 tau, for the adiabatic Gaussian and for the local harmonic Gaussian; and the smallest displacement at which the initial cosine equals the adiabatic cosine. For comparison with the literature on this family of methods, also give the unshifted spectral contrast-angle cosines that the original study introducing a constant reference Hessian into thawed Gaussian spectra reported for its one-dimensional Morse oscillator with anharmonicity 0.02 when the reference Hessian is the vertical one and when it is the initial one. These are the few scalars the reasoning should show. Give each as a number: computed cosines to five decimal places, the damping time, Hessian elements and the energy at d = 1 to four decimal places, energy deviations to four significant figures and displacements to three decimal places.

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

01_excited_surface_derivatives

Goal
----
Step 01: Excited-state surface energy, gradient and Hessian.

```python
def excited_surface_derivatives(points: "np.ndarray", displacement: float, surface_params: "np.ndarray") -> "np.ndarray":
    '''Potential energy, gradient and Hessian of the excited-state surface at a set of points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2), N >= 1, of mass-weighted coordinates (q1, q2).
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] with omega_e > 0, chi > 0, omega_b > 0, any real gamma and the bend
        coordinate delta of the minimum.

    Returns
    -------
    result : np.ndarray
        Array of shape (N, 6) whose rows are [V, dV/dq1, dV/dq2, d2V/dq1^2, d2V/dq1dq2, d2V/dq2^2] at the points.

    Raises
    ------
    ValueError
        If points is not a finite array of shape (N, 2) with N >= 1, if surface_params does not hold five finite numbers,
        or if omega_e, chi or omega_b is not strictly positive.
    '''
    return result  # placeholder
```

### Step 2

02_exact_autocorrelation

Goal
----
Step 02: Exact autocorrelation by split-operator grid propagation.

```python
def exact_autocorrelation(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, n_steps: int) -> "np.ndarray":
    '''Autocorrelation C(t_n) = <psi_0|psi_(t_n)>, t_n = n time_step, from second-order split-operator propagation on a periodic grid.

    The discretization is fixed because the returned values depend on it: psi_0 is the analytic ground state sampled on the
    grid without renormalization; one step applies exp(-i V dt / 2), then the kinetic propagator exp(-i (k1^2 + k2^2) dt / 2)
    in the discrete Fourier representation with angular wavenumbers k = 2 pi m / (N h) of the unshifted FFT ordering, then
    exp(-i V dt / 2) again; and C(t_n) is the sum over grid points of conj(psi_0) psi_(t_n) times h1 h2, the product of
    the grid spacings.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] with ground-state frequencies omega_1, omega_2 > 0 and Duschinsky angle theta in
        radians.
    q1_axis : np.ndarray
        Evenly spaced, increasing grid points of q1 (at least 4), treated as one period of a periodic grid.
    q2_axis : np.ndarray
        Evenly spaced, increasing grid points of q2 (at least 4), treated as one period of a periodic grid.
    time_step : float
        Time step dt > 0.
    n_steps : int
        Number of steps n_steps >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (n_steps + 1, 2) whose row n is [Re C(t_n), Im C(t_n)].

    Raises
    ------
    ValueError
        If either axis is not one-dimensional, evenly spaced and increasing with at least 4 points, if time_step <= 0, if
        n_steps < 0, or if omega_1 or omega_2 is not strictly positive.
    '''
    return result  # placeholder
```

### Step 3

03_thawed_gaussian_autocorrelation

Goal
----
Step 03: Autocorrelation of a thawed Gaussian with local or constant reference Hessian.

```python
def thawed_gaussian_autocorrelation(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int) -> "np.ndarray":
    '''Autocorrelation C(t_n) = <psi_0|psi_n>, t_n = n time_step, of a thawed Gaussian propagated by a fourth-order composition.

    psi_0(q) = pi^(-1/2) (omega_1 omega_2)^(1/4) exp(-(1/2) q^T R diag(omega_1, omega_2) R^T q) is the vibrational ground state
    of the initial electronic state, with R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]], and it is also the Gaussian
    at t = 0. One step of length dt applies, for each weight w in (w_1, w_0, w_1) with w_1 = 1 / (2 - 2^(1/3)) and
    w_0 = -2^(1/3) / (2 - 2^(1/3)), the exact kinetic flow for a time w dt / 2, the exact potential flow for a time w dt and the
    exact kinetic flow for a time w dt / 2. The kinetic flow is free evolution under -(1/2)(d^2/dq1^2 + d^2/dq2^2). The
    potential flow is evolution under V_eff with the centre q_c of the Gaussian at the start of that flow, where K is the
    Hessian of the excited-state surface at q_c when hessian_mode is 0, and a fixed matrix otherwise. psi_n is the Gaussian
    after n steps, including its full phase.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] with omega_1, omega_2 > 0 and theta in radians.
    hessian_mode : int
        Choice of K: 0 for the Hessian of the excited-state surface at the current centre (local harmonic), 1 for the
        Hessian at the excited-state minimum (d, delta), 2 for the Hessian of the excited-state surface at the origin,
        which is the initial-state minimum, and 3 for the initial-state Hessian R diag(omega_1^2, omega_2^2) R^T.
    time_step : float
        Time step dt > 0.
    n_steps : int
        Number of steps n_steps >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (n_steps + 1, 2) whose row n is [Re C(t_n), Im C(t_n)].

    Raises
    ------
    ValueError
        If hessian_mode is not 0, 1, 2 or 3, if time_step <= 0, if n_steps < 0, or if omega_1 or omega_2 is not strictly
        positive.
    '''
    return result  # placeholder
```

### Step 4

04_thawed_gaussian_energy

Goal
----
Step 04: True energy along a thawed Gaussian propagation.

```python
def thawed_gaussian_energy(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", hessian_mode: int, time_step: float, n_steps: int) -> "np.ndarray":
    '''Expectation value of the excited-state Hamiltonian -(1/2) Laplacian + V in the thawed Gaussian at every step.

    The Gaussian at step n is the one defined by the previous step: it starts as the vibrational ground state of the initial
    electronic state and is advanced by the same fourth-order composition of exact kinetic and potential flows with the same
    choice of K. The returned values are exact expectation values of the true Hamiltonian in those Gaussians, to rounding.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface V.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] with omega_1, omega_2 > 0 and theta in radians.
    hessian_mode : int
        Choice of K as in the previous step: 0 local harmonic, 1 excited-state minimum, 2 origin, 3 initial-state Hessian.
    time_step : float
        Time step dt > 0.
    n_steps : int
        Number of steps n_steps >= 0.

    Returns
    -------
    result : np.ndarray
        Array of shape (n_steps + 1,) with the energy expectation value E_n = <psi_n|H|psi_n>.

    Raises
    ------
    ValueError
        If hessian_mode is not 0, 1, 2 or 3, if time_step <= 0, if n_steps < 0, if omega_1 or omega_2 is not strictly
        positive, or if omega_e, chi or omega_b is not strictly positive.
    '''
    return result  # placeholder
```

### Step 5

05_spectral_contrast_cosine

Goal
----
Step 05: Unshifted spectral contrast cosine of broadened lineshapes.

```python
def spectral_contrast_cosine(approx_autocorrelation: "np.ndarray", reference_autocorrelation: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    '''Unshifted spectral contrast cosine between the broadened lineshapes of an approximate and a reference autocorrelation.

    Both autocorrelations are sampled at t_n = n time_step, n = 0, ..., N - 1. The frequency integrals of the products and
    squares of the two lineshapes are converted exactly into time integrals over [0, infinity) of the corresponding products
    of autocorrelations and damping factors, and those time integrals are evaluated over [0, t_(N-1)] with the composite
    trapezoidal rule on the samples.

    Parameters
    ----------
    approx_autocorrelation : np.ndarray
        Array of shape (N, 2), N >= 2, with rows [Re C, Im C] of the approximate autocorrelation.
    reference_autocorrelation : np.ndarray
        Array of shape (N, 2) with rows [Re C, Im C] of the reference autocorrelation on the same times.
    time_step : float
        Sampling interval dt > 0.
    broadening_time : float
        Broadening time tau > 0 of the Gaussian damping exp(-t^2 / (2 tau^2)).

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) holding [cos(theta), squared norm of the approximate lineshape, squared norm of the reference
        lineshape], where each squared norm is the frequency integral of the square of that lineshape.

    Raises
    ------
    ValueError
        If the arrays are not finite with equal shapes (N, 2) and N >= 2, if time_step or broadening_time is not strictly
        positive, or if either lineshape has zero norm.
    '''
    return result  # placeholder
```

### Step 6

06_method_diagnostics

Goal
----
Step 06: Spectral accuracy and energy excursions of four Gaussian variants.

```python
def method_diagnostics(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    '''Spectral contrast cosines of four thawed Gaussian variants and the energy excursions of two of them.

    The number of steps is n = round(6 broadening_time / time_step). The exact autocorrelation comes from the split-operator
    propagation on the given grid, each thawed Gaussian autocorrelation and energy series from the fourth-order composition propagation
    with the same time step and number of steps, and each cosine from the unshifted spectral contrast cosine with the exact
    autocorrelation as the reference. The energy excursion of a variant is the largest |E_k - E_0| over the n + 1 sampled
    energy expectation values E_k.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] of the initial state.
    q1_axis : np.ndarray
        Evenly spaced grid points of q1 for the exact propagation.
    q2_axis : np.ndarray
        Evenly spaced grid points of q2 for the exact propagation.
    time_step : float
        Common time step dt > 0.
    broadening_time : float
        Broadening time tau > 0.

    Returns
    -------
    result : np.ndarray
        Array [cos_local, cos_adiabatic, cos_vertical, cos_initial, excursion_adiabatic, excursion_local] of shape (6,).

    Raises
    ------
    ValueError
        If time_step or broadening_time is not strictly positive, or if round(6 broadening_time / time_step) < 1.
    '''
    return result  # placeholder
```

### Step 7

07_vertical_adiabatic_crossover

Goal
----
Step 07: Displacement where vertical and adiabatic reference Hessians are equally accurate (orchestrator).

```python
def vertical_adiabatic_crossover(surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float, d_min: float, d_max: float, n_scan: int) -> float:
    '''Displacement, refined in the first scan bracket, where the vertical and adiabatic spectral contrast cosines are equal.

    Parameters
    ----------
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface; the stretch coordinate of its minimum is
        the variable being searched.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] of the initial state.
    q1_axis : np.ndarray
        Evenly spaced grid points of q1 for the exact propagation.
    q2_axis : np.ndarray
        Evenly spaced grid points of q2 for the exact propagation.
    time_step : float
        Common time step dt > 0.
    broadening_time : float
        Broadening time tau > 0.
    d_min : float
        First scan displacement.
    d_max : float
        Last scan displacement, d_max > d_min.
    n_scan : int
        Number of evenly spaced scan displacements, n_scan >= 2.

    Returns
    -------
    result : float
        Crossover displacement d_x.

    Raises
    ------
    ValueError
        If d_max <= d_min or n_scan < 2, if g(d_min) >= 0 so that the scan starts at or past the crossover, or if g stays
        negative over the whole scan.
    '''
    return result  # placeholder
```
