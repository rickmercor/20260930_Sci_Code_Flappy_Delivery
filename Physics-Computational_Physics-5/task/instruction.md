# Physics-Computational_Physics-5

## Background

Kinetic theory describes a collisionless plasma by a distribution function over phase space, and it is expensive. Almost all large-scale plasma modelling is done instead with fluid equations, obtained by taking velocity moments of the kinetic equation. That hierarchy never terminates: the equation for each moment involves the next one, so the equation for the pressure calls on the heat flux, and something has to be supplied from outside to close it. The closure is not a numerical detail. Landau damping, the collisionless decay of a wave through resonance with the particles that travel at its phase speed, is a purely kinetic effect with no counterpart in a locally closed fluid model, and whether a fluid model exhibits it at all, let alone at the right rate, is decided entirely by what is written for the heat flux.

The modern approach makes the closure non-local in space, which in Fourier space means letting the heat flux depend on the wave number. The Hammett–Perkins closure did this first, tying the heat flux to the temperature gradient through a coefficient carrying the inverse wave number, which gives a fluid model that damps. Later work put this on a systematic footing by recognising that a linearised three-moment fluid model reproduces exactly whatever rational approximation of the Maxwellian kinetic response function its closure corresponds to, so that choosing a closure is choosing a Padé approximant. The coefficients of that approximant have conventionally been fixed by matching the asymptotic expansions of the kinetic response in the two limits where it is known in closed form, the adiabatic limit of slow waves and the fluid limit of fast ones, with the different members of the family differing only in how the available matching conditions are split between the two ends.

That construction has a structural weakness. Asymptotic matching constrains the approximant where the expansions converge and says nothing about the intermediate wave numbers where much of the interesting physics sits, so the error of a static closure is largest exactly in the middle of the spectrum. Data-driven closures learned from kinetic simulations have been proposed as a way round this, but they are opaque and expensive to train. The alternative pursued here is analytic: rather than matching series, require the approximant to pass through the exact kinetic response at the plasma's own least-damped roots. Those roots are the solutions of the kinetic dispersion relation with the smallest damping, they dominate the long-time behaviour of any initial perturbation, and they depend on the wave number, so the closure coefficients become functions of the wave number rather than constants.

Whether that reanchoring is worth the trouble is a quantitative question about a particular band of wave numbers, not a matter of principle, and it is answered by evolving the same initial perturbation under competing closures and comparing what they predict. Because the linearised system for a single Fourier mode is a small autonomous linear system, that comparison can be made exactly and cheaply: the kinetic roots follow from a transcendental equation solved to machine precision, the closure coefficients from a small linear solve, and the trajectories from a fixed number of explicit time steps. Everything about the comparison is deterministic, and the discrepancy between the two closures is a computed number rather than a fitted one.

## Problem

A fluid description of a collisionless plasma has to truncate the moment hierarchy at the heat flux, and how that truncation is made decides whether the fluid model reproduces the Landau damping of the kinetic system underneath it. The established three-moment closures obtain the coefficients of a three-pole Padé approximant of the Maxwellian kinetic response function by matching its adiabatic and fluid asymptotic series, which anchors the closure only at the two ends of the spectrum and lets its fidelity degrade in between; an alternative fixes those coefficients afresh at every wave number by demanding that the approximant reproduce the plasma's own least-damped kinetic root pair, so that the fluid dispersion relation is exact for the mode that survives longest.

The system is a one-dimensional electrostatic electron plasma on an immobile neutralising ion background, Maxwellian in equilibrium, carried by the density, velocity and pressure moments and closed by a heat flux. Work throughout in the three-pole Padé family whose quadratic denominator coefficient is held at $-2$, and at each wave number anchor its two remaining coefficients on the least-damped kinetic root $\zeta_0$ together with its mirror image $-\zeta_0^{*}$, which the parity of a Maxwellian supplies. Take the Hammett–Perkins closure as the conventional benchmark against which the anchored closure is measured.

At each of $k/k_p = 0.2,\ 0.3,\ 0.4,\ 0.5,\ 0.6$, where $k_p$ is the Debye wave number, perturb the equilibrium by a cosine density ripple of relative amplitude $0.02$, which multiplies the whole Maxwellian and so carries a pressure ripple of the same relative amplitude with it, and evolve the linearised moment system for that single Fourier mode with a second-order Runge–Kutta scheme of step $0.005\,\omega_{pe}^{-1}$ out to $40\,\omega_{pe}^{-1}$. Record every time level $t_j = j\,\Delta t$ for $j = 0,\dots,8000$, the initial level included, and let $d(k)$ be the root mean square over those levels of $\lvert E_{\mathrm{HP}}(k,t_j) - E_{\mathrm{ref}}(k,t_j)\rvert$, the modulus of the difference between the two closures' complex positive-wave-number Fourier coefficients of the perturbed electrostatic field, divided by the root mean square of $\lvert E_{\mathrm{ref}}(k,t_j)\rvert$ over the same levels; do not compare magnitude envelopes. Your final answer must be a single number, the dimensionless band aggregate

$$\mathcal{D} = \sqrt{\tfrac{1}{5}\sum_{k} d(k)^2},$$

quoted to at least six significant figures.

State, in one line each, the equation whose least-damped solution the anchoring uses, the form of the three-pole approximant you anchor, the heat-flux closure it corresponds to, the rule that turns the Padé coefficients into the coefficients of that closure, which of those closure coefficients vanishes identically in this Padé family, the Hammett–Perkins closure coefficients, why the mirror-image root pair makes the closure coefficients real, the normalised moment state the initial ripple sets, and how the Poisson equation fixes the field from the density perturbation; then report the least-damped kinetic root at $k = 0.4\,k_p$, the two surviving anchored closure coefficients there, the least-damped root the Hammett–Perkins model itself predicts at that wave number, the phase in radians that the real-frequency difference between those two roots accumulates over the length of the run, the kinetic damping rate at $k = 0.2\,k_p$, the residual by which the anchored fluid model's own least-damped root differs from the kinetic one, the five $d(k)$, whether $\mathcal{D}$ depends on the ripple amplitude, the value $\mathcal{D}$ takes when the asymptotic member of the same family that matches one order fewer in the fluid limit than Hammett–Perkins is assessed in its place, the value it takes when the published fitted expressions for the closure coefficients are used instead of the exact anchoring, and what those numbers say about where in the band a conventional closure can be trusted. Those are the short report this task wants: they are the few scalars that determine the final number and the few that establish it was computed rather than estimated, and nothing beyond them is required.

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

01_evaluate_kinetic_response

Goal
----
Evaluate the kinetic response function of a Maxwellian electron population at a set of normalised complex phase speeds.

```python
import numpy as np


def evaluate_kinetic_response(zeta: np.ndarray) -> np.ndarray:
    """Evaluate the Maxwellian kinetic response function.

    Parameters
    ----------
    zeta : np.ndarray
        Array of normalised complex phase speeds zeta = omega/(sqrt(2)*abs(k)*v_t),
        where v_t = sqrt(T0/m), T0 is the equilibrium temperature in energy
        units, and m is the electron mass. Real input is accepted and treated
        as having zero imaginary part. Any shape is accepted.

    Returns
    -------
    response : np.ndarray
        Complex array of the same shape as ``zeta``, holding the kinetic
        response function of a Maxwellian equilibrium at each phase speed. The
        function equals one at zero phase speed and behaves as minus one half
        the inverse square of the phase speed when the phase speed is large.

    Raises
    ------
    ValueError
        If ``zeta`` is empty or holds a non-finite entry.
    """
    return response  # placeholder
```

### Step 2

02_solve_kinetic_root

Goal
----
Locate the least-damped root of the exact kinetic dispersion relation of an electrostatic Maxwellian plasma at a prescribed normalised wave number.

```python
import numpy as np


def solve_kinetic_root(wavenumber: float) -> complex:
    """Locate the least-damped kinetic root at one normalised wave number.

    The dispersion relation is the statement that the Maxwellian kinetic
    response function of sub-problem 01 evaluated at the normalised phase speed
    equals minus the square of the normalised wave number.

    Parameters
    ----------
    wavenumber : float
        Perturbation wave number divided by the Debye wave number of the
        equilibrium. It must be at least 0.1: below that the damping of the
        least-damped root falls under the resolution of double precision, the
        root can no longer be located reliably, and the function is not defined.

    Returns
    -------
    root : complex
        The normalised phase speed satisfying the dispersion relation that has
        a strictly positive real part and, among all such solutions, the
        largest (least negative) imaginary part.

    Raises
    ------
    ValueError
        If ``wavenumber`` is not a finite number, or is smaller than 0.1.
    """
    return root  # placeholder
```

### Step 3

03_solve_matched_pade_coefficients

Goal
----
Determine the coefficients of the three-pole Pade approximant of the kinetic response by forcing it to reproduce prescribed response values at two prescribed phase speeds.

```python
import numpy as np


def solve_matched_pade_coefficients(phase_speeds: np.ndarray,
                                    response_values: np.ndarray) -> np.ndarray:
    """Fix the Pade coefficients from two prescribed response values.

    The approximant is the three-pole rational form in the normalised phase
    speed whose numerator and denominator both equal one at zero phase speed,
    whose denominator has degree three with its cubic coefficient tied to the
    numerator coefficient, and whose quadratic denominator coefficient is held
    at minus two. The two remaining coefficients are fixed by requiring the
    approximant to take the given response value at the corresponding phase
    speed.

    Parameters
    ----------
    phase_speeds : np.ndarray
        Complex array of shape (2,) holding the two distinct normalised phase
        speeds at which the approximant is anchored.
    response_values : np.ndarray
        Complex array of shape (2,) holding the value the approximant must take
        at the corresponding entry of ``phase_speeds``.

    Returns
    -------
    pade_coefficients : np.ndarray
        Complex array of shape (3,) holding, in order, the numerator
        coefficient, the linear denominator coefficient and the quadratic
        denominator coefficient of the approximant.

    Raises
    ------
    ValueError
        If either input is not a finite array of shape (2,), if the two entries
        of ``phase_speeds`` are equal, or if the two anchoring conditions do not
        determine the coefficients.
    """
    return pade_coefficients  # placeholder
```

### Step 4

04_build_asymptotic_pade_coefficients

Goal
----
Return the coefficients of the conventional three-pole Pade approximants of the kinetic response, the ones obtained by matching its asymptotic power series rather than the plasma's own roots.

```python
import numpy as np


def build_asymptotic_pade_coefficients(label: str) -> np.ndarray:
    """Return the Pade coefficients of one conventional asymptotic closure.

    Three members are supported. ``"R30"`` matches the adiabatic expansion of
    the kinetic response through third order and the fluid expansion at leading
    order only. ``"R31"`` gives up one adiabatic order to gain one fluid order.
    ``"HP"`` gives up a second adiabatic order to gain a second fluid order and
    is the Hammett-Perkins closure.

    Parameters
    ----------
    label : str
        One of ``"R30"``, ``"R31"`` or ``"HP"``.

    Returns
    -------
    pade_coefficients : np.ndarray
        Complex array of shape (3,) holding, in order, the numerator
        coefficient, the linear denominator coefficient and the quadratic
        denominator coefficient, in the same convention as sub-problem 03.

    Raises
    ------
    ValueError
        If ``label`` is not one of the three supported strings, including when
        it is not a string at all. A lookup that propagates a ``KeyError`` or a
        ``TypeError`` instead does not satisfy this contract.
    """
    return pade_coefficients  # placeholder
```

### Step 5

05_evaluate_closure_parameters

Goal
----
Convert the coefficients of a three-pole Pade approximant of the kinetic response into the three real coefficients of the corresponding heat-flux closure.

```python
import numpy as np


def evaluate_closure_parameters(pade_coefficients: np.ndarray) -> np.ndarray:
    """Convert Pade coefficients into heat-flux closure parameters.

    Parameters
    ----------
    pade_coefficients : np.ndarray
        Complex array of shape (3,) holding the numerator coefficient, the
        linear denominator coefficient and the quadratic denominator
        coefficient of the approximant, in the convention of sub-problem 03.

    Returns
    -------
    closure_parameters : np.ndarray
        Array of shape (3,) of native floats holding, in order, the
        coefficient multiplying the normalised velocity, the coefficient
        multiplying the normalised pressure measured against the normalised
        electrostatic potential, and the coefficient multiplying the normalised
        temperature, in the heat-flux closure.

    Raises
    ------
    ValueError
        If ``pade_coefficients`` is not a finite array of shape (3,), if its
        numerator coefficient vanishes, or if any resulting closure parameter
        has an imaginary part exceeding 1e-6 times the larger of one and its
        own magnitude.
    """
    return closure_parameters  # placeholder
```

### Step 6

06_build_moment_evolution_matrix

Goal
----
Assemble the linear operator that advances the three normalised Fourier moments of an electrostatic electron plasma once a heat-flux closure has been supplied.

```python
import numpy as np


def build_moment_evolution_matrix(wavenumber: float,
                                  closure_parameters: np.ndarray) -> np.ndarray:
    """Assemble the three-moment evolution operator at one wave number.

    The state is ordered as the normalised density, velocity and pressure
    perturbations. Define v_t = sqrt(T0/m), where T0 is the equilibrium
    temperature in energy units and m is the electron mass. Density is
    normalised by the equilibrium density, velocity by sqrt(2)*v_t, pressure
    by the equilibrium pressure P0, heat flux by P0*sqrt(2)*v_t, and
    electrostatic potential by T0/e, where e is the elementary charge. Time
    is measured in inverse plasma frequencies and the wave vector points
    along the positive axis. The returned operator ``M`` is the one for
    which the state derivative equals ``M`` acting on the state.

    Parameters
    ----------
    wavenumber : float
        Perturbation wave number divided by the Debye wave number of the
        equilibrium, strictly positive.
    closure_parameters : np.ndarray
        Array of shape (3,) of real closure parameters as returned by
        sub-problem 05.

    Returns
    -------
    evolution_matrix : np.ndarray
        Complex array of shape (3, 3), the linear operator of the closed
        three-moment system in the state ordering above.

    Raises
    ------
    ValueError
        If ``wavenumber`` is not a finite number greater than zero, or if
        ``closure_parameters`` is not a finite real array of shape (3,).
    """
    return evolution_matrix  # placeholder
```

### Step 7

07_integrate_moment_history

Goal
----
Advance a linear Fourier-space moment system over a fixed number of equal steps with an explicit two-stage second-order Runge-Kutta scheme and record every time level.

```python
import numpy as np


def integrate_moment_history(evolution_matrix: np.ndarray,
                             initial_state: np.ndarray,
                             time_step: float,
                             step_count: int) -> np.ndarray:
    """Integrate a linear moment system and record every time level.

    The advance is the explicit two-stage second-order Runge-Kutta scheme, in
    which a trial derivative taken at the current level is used to form an
    intermediate state and the increment is then taken from the derivative
    there.

    Parameters
    ----------
    evolution_matrix : np.ndarray
        Complex array of shape (3, 3) as returned by sub-problem 06.
    initial_state : np.ndarray
        Array of shape (3,) holding the initial normalised density, velocity
        and pressure perturbations, in that order. Real input is accepted.
    time_step : float
        Time increment between successive levels in inverse plasma
        frequencies, strictly positive.
    step_count : int
        Number of increments to take, at least one.

    Returns
    -------
    moment_history : np.ndarray
        Complex array of shape (step_count + 1, 3) whose first row is
        ``initial_state`` and whose remaining rows are the successive levels.

    Raises
    ------
    ValueError
        If ``evolution_matrix`` is not a finite array of shape (3, 3), if
        ``initial_state`` is not a finite array of shape (3,), if ``time_step``
        is not a finite number greater than zero, or if ``step_count`` is not an
        integer greater than zero.
    """
    return moment_history  # placeholder
```

### Step 8

08_compute_field_amplitude_history

Goal
----
Reduce a recorded moment history to the history of the perturbed electrostatic field amplitude that the Poisson equation ties to it.

```python
import numpy as np


def compute_field_amplitude_history(moment_history: np.ndarray,
                                    wavenumber: float) -> np.ndarray:
    """Reduce a moment history to the perturbed electrostatic field amplitude.

    The field is normalised by E0 = T0*k_p/e, where T0 is the equilibrium
    temperature in energy units, k_p is the Debye wave number, and e is the
    elementary charge. This agrees with the potential normalisation T0/e in
    sub-problem 06. The wave vector points along the positive axis.

    Parameters
    ----------
    moment_history : np.ndarray
        Complex array of shape (n_levels, 3) as returned by sub-problem 07,
        holding the normalised density, velocity and pressure perturbations at
        each time level.
    wavenumber : float
        Perturbation wave number divided by the Debye wave number of the
        equilibrium, strictly positive.

    Returns
    -------
    field_history : np.ndarray
        Complex array of shape (n_levels,) holding the normalised perturbed
        electrostatic field amplitude at each time level.

    Raises
    ------
    ValueError
        If ``moment_history`` is not a finite two-dimensional array with three
        columns and at least one row, or if ``wavenumber`` is not a finite
        number greater than zero.
    """
    return field_history  # placeholder
```

### Step 9

09_compute_relative_field_deviation

Goal
----
Measure how far one recorded field history departs from a reference history over the whole run, as a single dimensionless number.

```python
import numpy as np


def compute_relative_field_deviation(reference_history: np.ndarray,
                                     test_history: np.ndarray) -> float:
    """Measure the relative departure of a field history from a reference.

    The measure is the root mean square, taken over all recorded time levels,
    of the magnitude of the difference between the two histories, divided by
    the root mean square of the magnitude of the reference history over the
    same levels.

    Parameters
    ----------
    reference_history : np.ndarray
        Complex array of shape (n_levels,) holding the reference field
        amplitude at each time level, as returned by sub-problem 08.
    test_history : np.ndarray
        Complex array of the same shape holding the field amplitude of the
        history being assessed.

    Returns
    -------
    deviation : float
        The dimensionless relative departure, as a native Python float.

    Raises
    ------
    ValueError
        If either input is not a finite one-dimensional array with at least one
        entry, if the two shapes differ, or if the reference history vanishes at
        every level.
    """
    return deviation  # placeholder
```

### Step 10

10_run_closure_fidelity_pipeline

Goal
----
Chain the sub-problem functions 01-09 end to end over a band of wave numbers and return the aggregate relative departure of a conventional asymptotic closure from the wave-number-dependent one.

```python
import numpy as np


def run_closure_fidelity_pipeline(wavenumbers: np.ndarray = (0.2, 0.3, 0.4, 0.5, 0.6),
                                  amplitude: float = 0.02,
                                  time_step: float = 0.005,
                                  step_count: int = 8000,
                                  benchmark: str = "HP") -> float:
    """Run the whole closure-fidelity comparison over a band of wave numbers.

    At each wave number the reference closure is the one anchored on the exact
    kinetic least-damped root pair, and the closure under assessment is the
    named conventional asymptotic member. Both are started from a Maxwellian
    equilibrium with a cosine density ripple of the given amplitude. This
    function uses cosine amplitudes internally: the initial normalised density
    and pressure perturbations both equal amplitude, and velocity is zero.
    These are twice the corresponding positive-k coefficients in a two-sided
    Fourier expansion. The common factor cancels in the relative departure.

    Parameters
    ----------
    wavenumbers : np.ndarray
        One-dimensional sequence of wave numbers divided by the Debye wave
        number, each at least 0.1. This is the root solver's admissible input
        bound; successful anchoring additionally requires a numerically
        nonsingular two-root fit. Exceptionally weak damping near the lower
        bound can make that fit numerically singular even for admissible inputs.
    amplitude : float
        Amplitude of the initial cosine density ripple, strictly positive.
    time_step : float
        Time increment in inverse plasma frequencies, strictly positive.
    step_count : int
        Number of increments to take, at least one.
    benchmark : str
        Which conventional asymptotic closure is assessed: ``"HP"``, ``"R31"``
        or ``"R30"``.

    Returns
    -------
    aggregate_deviation : float
        The root mean square, over the supplied wave numbers, of the relative
        departure of the benchmark closure's field history from the anchored
        closure's field history, dimensionless, as a native Python float.

    Raises
    ------
    ValueError
        If ``wavenumbers`` is not a non-empty one-dimensional array of finite
        entries of at least 0.1, if ``amplitude`` or ``time_step`` is not a
        finite number greater than zero, if ``step_count`` is not an integer
        greater than zero, if ``benchmark`` is not one of the three supported
        labels, if a located kinetic root fails to satisfy the dispersion
        relation to within 1e-8, or if the two-root anchoring system fails the
        numerical nonsingularity check of sub-problem 03.
    """
    return aggregate_deviation  # placeholder
```
