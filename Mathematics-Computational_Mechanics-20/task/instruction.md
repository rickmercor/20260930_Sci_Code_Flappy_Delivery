# Mathematics-Computational_Mechanics-20

## Background

Two fluids that do not mix are separated by a surface across which density and sound speed jump, sometimes by three orders of magnitude, and a compressible solver has to carry that surface through shocks, expansions and large deformations without letting it either smear away or fracture into numerical debris. Diffuse-interface methods take the pragmatic route: replace the surface with a thin layer of finite thickness, transport a volume fraction through it, and add a regularisation flux whose whole purpose is to hold the layer at the thickness it started with, balancing the diffusion that keeps it smooth against a sharpening term that pulls it back. Done correctly the two terms cancel exactly on the equilibrium profile, so the regularisation is invisible until numerics start to spread the interface, and it moves no mass across it.

The complication that this class of models has to face is algebraic rather than physical. Recovering a phasic density means dividing by that phase's volume fraction, and recovering a phasic pressure means inverting an equation of state that is only defined for positive density, so the pure phases cannot be allowed to reach exactly zero and one, which is precisely what the naive equilibrium profile does asymptotically and what finite-precision arithmetic then turns into a division by zero. The fix is to bound the volume fraction away from both ends by a small finite amount and rederive the regularisation so that the bounded profile, rather than the unbounded one, is the exact fixed point. That rederivation changes the interface-distance function, changes the prefactors on both terms of the flux, and is the difference between a scheme that runs and one that produces a negative density within a few dozen steps.

## Problem

Simulating a compressible flow of two immiscible fluids means tracking an interface that is itself a discontinuity in density and sound speed. Diffuse-interface methods replace that discontinuity with a thin smeared layer and add a regularisation flux whose job is to hold the layer at a fixed thickness against numerical diffusion without moving mass across it. The pressure and density algebra of a compressible two-phase model divides by the volume fraction, so the pure phases cannot be allowed to reach exactly zero and one; the source therefore reformulates the regularisation so that the volume fraction asymptotes to a small positive bound and its complement instead, and derives the interface profile that this bounded regularisation holds as an exact fixed point. Your job is to implement that bounded formulation exactly as the source specifies it and to audit the interface it produces when a prescribed two-dimensional vortex stretches and folds a circular blob, on the fixed configurations below. The construction rests on a handful of conventions that the source states and that a plausible alternative reading would get wrong, and at least one of them, the equilibrium profile itself together with its distance-like function, is derived in an appendix rather than in the main development; they are not reproduced here, and recovering them from the literature is the substance of the problem.

Work on the unit periodic square with a uniform mesh of N by N cells of spacing h = 1/N, cell centres at ((i + 1/2) h, (j + 1/2) h). The finite bound on the volume fraction is 1e-2 and the interface thickness is twice the grid spacing. Phase one has density 1.0e3 and stiffened-gas constants 4.4 and 6.0e8; phase two has density 1.2 and constants 1.4 and 0. Both phasic pressures are 1.0e5 everywhere. Each configuration is (N, R0, alpha, cx, cy, steps), the three being (96, 0.15, 0.4, 0.5, 0.5, 160), (128, 0.12, 0.7, 0.25, 0.5, 100) and (80, 0.18, 0.25, 0.3, 0.3, 107). Phase two moves with the prescribed velocity field vel_scale times (sin(2 pi x) cos(2 pi y), -cos(2 pi x) sin(2 pi y)) evaluated at the cell centres, and phase one moves with alpha times that field. The initial volume fraction of phase one is the source's bounded equilibrium profile for a circular blob of radius R0 centred at (cx, cy), written in the signed distance R0 minus the distance from the cell centre to (cx, cy), that distance measured on the periodic square with each coordinate difference reduced into [-1/2, 1/2). The phasic masses per unit volume start as the volume fractions times the phasic densities.

Fix the following numerical conventions so the results are reproducible. Take every spatial derivative with the second-order central difference along each axis on the periodic grid, including the gradients inside the regularisation flux and the interface normal; where the gradient magnitude used for the normal is exactly zero, the normal is the zero vector. The regularisation velocity is the largest velocity magnitude over both phases and all cells, and the time step is 0.25 times the grid spacing divided by it. Advance the configuration's number of steps with a two-stage strong-stability-preserving Runge-Kutta method, recovering each phasic density within a stage by dividing its mass by its volume fraction floored at the bound, and do not clip the volume fraction at any point. Where the source's distance-like function would divide by zero, add 1e-100 to numerator and denominator and clamp each of them at zero first, so the function stays finite in floating point at and beyond the bounds. Solve nothing iteratively; no tolerance enters the reported numbers.

Implement nine functions. stiffened_gas_pressure(rho, e, gamma, pi_) returns the phasic pressure. interface_closures(phi, p1, p2, rho1, rho2, u1, u2) returns the interface pressure and one component of the interface velocity. interface_distance(phi, eps, delta) returns the source's distance-like function for the bounded phase field. interface_normal(psi, h) returns the interface normal as a (2, N, N) array. interface_regularization_flux(phi, eps, delta, Gamma, h) returns the volumetric interface-regularisation flux as a (2, N, N) array. volume_fraction_rhs(phi, uI, eps, delta, Gamma, h) returns the right-hand side of the volume-fraction transport equation. phasic_mass_rhs(phi, rho, u, eps, delta, Gamma, h, phase_one) returns the right-hand side of one phase's mass equation. advance(phi, m1, m2, u1, u2, p1, p2, eps, delta, Gamma, h, dt) takes one Runge-Kutta step. vortex_audit(vel_scale), the final step, must be assembled by calling the earlier functions: it runs the three configurations and returns a float64 array (3, 6) whose rows are [the largest magnitude of the regularisation flux on the initial profile in units of 1e-6, the ratio of the space integral of the volume fraction at the end to its initial value, the largest magnitude of the volume-fraction gradient at the end, the smallest volume fraction at the end, the space integral of the absolute difference between the final and initial volume fraction in units of 1e-3, and the interface-length measure, the space integral of the volume fraction times its complement at the end, in units of 1e-3].

Evaluate vortex_audit with vel_scale = 1.0. All outputs are float64, finite and deterministic: two runs on identical inputs must agree exactly. In your reasoning report, give all six row entries of each configuration to at least six significant figures, then the running total of the sixth column across the three configurations. State what the first entry says about whether the initial profile is a fixed point of the regularisation and say in one sentence why the two terms of the flux cancel on that profile; say what the first entry would do under mesh refinement given how the interface thickness is set here. Say what the sixth entry measures and how it separates into a bulk part set by the bound and an interface part set by the interface length, and by what factor the interface length of the first configuration grew. Say what the fourth entry says about the finite bound and about the scheme, and what the second entry says about the space integral of the volume fraction and why it is not conserved here. State what the two phasic mass equations together imply for the mixture mass over the run. Also state, in ONE sentence each, the source conventions your numbers depend on, covering the bounded equilibrium profile the initial condition is built from, the two interface closures and how their weights differ, the ratio inside the distance-like function for the bounded phase field, the two bound-dependent factors in the regularisation flux and the fixed denominator in its sharpening term, the field the interface normal is computed from, the extra non-conservative term in the volume-fraction equation, and how the regularisation enters the two phasic mass equations, with its density weighting and its signs. For each, name the choice the source makes, say what an obvious alternative reading would have been, and say whether that alternative merely shifts the reported numbers or breaks the calculation outright. Report the values as a compact list of numbers and each convention as a single sentence, not as a derivation. As the final answer, report the sum over the three configurations of the sixth column to six significant figures. The eighteen row entries and the one-sentence conventions asked for above are the substance of the report: report all of them in full, including the entries in columns one to five, which do not enter the final number but are required output. They do not count against any length guidance below, and no instruction below to show only the scalars that determine the final number removes them. Keep the prose around them brief instead.

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

stiffened_gas_pressure

Goal
----
Return the pressure of one phase from its density and specific internal energy under the stiffened-gas equation of state the source uses, given that phase's two constants.

```python
def stiffened_gas_pressure(rho: "np.ndarray", e: "np.ndarray", gamma: float, pi_: float) -> "np.ndarray":
    """Return the pressure of one phase from its density and specific internal energy under
    the stiffened-gas equation of state the source uses, given that phase's two constants.

    Args:
        rho: Phasic density, any array shape.
        e: Phasic specific internal energy, shaped like rho.
        gamma: Ratio of specific heats of the phase, greater than one.
        pi_: Stiffening constant of the phase, non-negative.

    Returns:
        A float64 array shaped like rho holding the phasic pressure.

    Raises:
        ValueError: If gamma is not greater than one, if pi_ is negative, or if any density
            is not positive.
    """
    return None
```

### Step 2

interface_closures

Goal
----
Return the source's two interface quantities for a two-phase mixture, the interface pressure first and the interface velocity component second. Both are averages of the corresponding phasic quantities over the two phases, but they are NOT averaged with the same weights: one uses a weight built from the volume fractions alone and the other a weight built from the phasic masses. Which weight belongs to which quantity is the source's closure; recover it from the paper. The volume fraction given is that of phase one, so phase two carries its complement.

```python
def interface_closures(phi: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", rho1: "np.ndarray", rho2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray") -> "np.ndarray":
    """Return the source's two interface quantities for a two-phase mixture, the interface
    pressure first and one component of the interface velocity second, each an average of
    the phasic quantities with the source's own weights.

    Args:
        phi: Volume fraction of phase one, any array shape; phase two carries 1 - phi.
        p1: Pressure of phase one, shaped like phi.
        p2: Pressure of phase two, shaped like phi.
        rho1: Density of phase one, shaped like phi.
        rho2: Density of phase two, shaped like phi.
        u1: One velocity component of phase one, shaped like phi.
        u2: The same velocity component of phase two, shaped like phi.

    Returns:
        A float64 array of shape (2,) + phi.shape: the interface pressure first, the interface
        velocity component second.

    Raises:
        ValueError: If any volume fraction lies outside [0, 1] or any mixture density is not
            positive.
    """
    return None
```

### Step 3

interface_distance

Goal
----
Return the source's interface distance-like function for the phase field with FINITE NON-ZERO bounds, in which the volume fraction asymptotes to a small positive value and its complement rather than to zero and one. It is the interface thickness times the logarithm of a ratio formed from the volume fraction, and that ratio is NOT the one used for the unbounded phase field: the bound enters it. The exact ratio is derived in the source's appendix; recover it from there rather than from the main text. Add the very small constant 1e-100 to numerator and denominator to control the limit, and clamp each of them at zero first so the function stays finite in floating point at and beyond the bounds. Form each clamped quantity so that a volume fraction sitting exactly on either bound gives exactly zero before the constant is added, rather than a rounding residue of order 1e-16.

```python
def interface_distance(phi: "np.ndarray", eps: float, delta: float) -> "np.ndarray":
    """Return the source's interface distance-like function for the phase field with finite
    non-zero bounds, evaluated elementwise with the 1e-100 safeguard and the zero clamps
    described in the step text.

    Args:
        phi: Volume fraction of phase one, any array shape.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5); the volume fraction asymptotes to delta and
            1 - delta.

    Returns:
        A float64 array shaped like phi holding the distance-like function.

    Raises:
        ValueError: If eps is not positive or delta does not lie in [0, 0.5).
    """
    return None
```

### Step 4

interface_normal

Goal
----
Return the interface normal of a two-dimensional periodic field, computed from the interface distance-like function rather than from the volume fraction itself, as the source recommends. The gradient is taken with second-order central differences along each axis on the periodic grid, and the normal is that gradient divided by its own magnitude; where the magnitude is exactly zero the normal is the zero vector.

```python
def interface_normal(psi: "np.ndarray", h: float) -> "np.ndarray":
    """Return the interface normal of a two-dimensional periodic field from the distance-like
    function, using second-order central differences along each axis and normalising the
    gradient by its own magnitude.

    Args:
        psi: Interface distance-like function on the (N, N) periodic grid; index [i, j]
            refers to cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        h: Grid spacing, positive, the same along both axes.

    Returns:
        A (2, N, N) float64 array: entry 0 the x component and entry 1 the y component of the
        unit normal, the zero vector wherever the gradient magnitude is exactly zero.

    Raises:
        ValueError: If psi is not two-dimensional or h is not positive.
    """
    return None
```

### Step 5

interface_regularization_flux

Goal
----
Return the source's volumetric interface-regularisation flux for the bounded phase field on a two-dimensional periodic grid: a vector field with a diffusion term proportional to the gradient of the volume fraction and a sharpening term proportional to one minus the squared hyperbolic tangent of the distance-like function over twice the thickness, directed along the interface normal. Both terms carry a prefactor built from the bound, and the two prefactors are different powers of the same quantity; the sharpening term also carries a fixed denominator. Recover the prefactors from the source's bounded transport equation. Take gradients with second-order central differences along each axis on the periodic grid, and take the normal from the distance-like function.

```python
def interface_regularization_flux(phi: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    """Return the source's volumetric interface-regularisation flux of the bounded phase
    field on the (N, N) periodic grid, as a vector field with a diffusion term and a
    sharpening term directed along the normal taken from the distance-like function.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.

    Returns:
        A (2, N, N) float64 array: entry 0 the x component and entry 1 the y component of the
        flux.

    Raises:
        ValueError: If phi is not two-dimensional, if Gamma is negative, if eps is not
            positive, if delta does not lie in [0, 0.5), or if h is not positive.
    """
    return None
```

### Step 6

volume_fraction_rhs

Goal
----
Return the right-hand side of the source's bounded volume-fraction transport equation on a two-dimensional periodic grid, written for a time-stepper as the time derivative of the volume fraction: the negative divergence of the volume fraction times the interface velocity, plus the extra non-conservative term the source's equation carries, plus the divergence of the regularisation flux. Take every divergence with second-order central differences along each axis on the periodic grid.

```python
def volume_fraction_rhs(phi: "np.ndarray", uI: "np.ndarray", eps: float, delta: float, Gamma: float, h: float) -> "np.ndarray":
    """Return the time derivative of the volume fraction of phase one given by the source's
    bounded transport equation on the (N, N) periodic grid, including its non-conservative
    term and the divergence of the regularisation flux.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        uI: Interface velocity as a (2, N, N) array, entry 0 the x component and entry 1 the y
            component.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.

    Returns:
        An (N, N) float64 array holding the right-hand side.

    Raises:
        ValueError: If phi is not two-dimensional or uI is not a (2, N, N) array matching it,
            or if eps, delta, Gamma or h lies outside the domain accepted by the earlier steps.
    """
    return None
```

### Step 7

phasic_mass_rhs

Goal
----
Return the right-hand side of one phase's mass equation on a two-dimensional periodic grid, written for a time-stepper as the time derivative of the phasic mass per unit volume: the negative divergence of the phasic mass flux, plus the divergence of that phase's mass-regularisation flux. The source forms the mass-regularisation flux from the volumetric regularisation flux and the phase's own explicitly recovered density, and the two phases carry it with opposite signs. Take every divergence with second-order central differences along each axis on the periodic grid.

```python
def phasic_mass_rhs(phi: "np.ndarray", rho: "np.ndarray", u: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, phase_one: bool) -> "np.ndarray":
    """Return the time derivative of one phase's mass per unit volume on the (N, N) periodic
    grid: the convective term of that phase plus the divergence of its mass-regularisation
    flux, signed as the source assigns it to the phase.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        rho: Density of the phase being updated, shaped like phi, positive.
        u: Velocity of the phase being updated as a (2, N, N) array, entry 0 the x component
            and entry 1 the y component.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.
        phase_one: True when the phase being updated is phase one, whose volume fraction is
            phi; False for phase two, whose volume fraction is 1 - phi.

    Returns:
        An (N, N) float64 array holding the right-hand side.

    Raises:
        ValueError: If phi is not two-dimensional or u is not a (2, N, N) array matching it,
            if any density is not positive, or if eps, delta, Gamma or h lies outside the
            domain accepted by the earlier steps.
    """
    return None
```

### Step 8

advance

Goal
----
Take one time step of the coupled volume-fraction and phasic-mass system on a two-dimensional periodic grid with a two-stage strong-stability-preserving Runge-Kutta method, given the current volume fraction, the two phasic masses per unit volume, the two prescribed phasic velocity fields and the two uniform phasic pressures. Within each stage recover each phasic density by dividing its mass by its volume fraction floored at the bound, form the interface velocity from the source's closure, and evaluate the three right-hand sides; do not clip the volume fraction at any point.

```python
def advance(phi: "np.ndarray", m1: "np.ndarray", m2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", eps: float, delta: float, Gamma: float, h: float, dt: float) -> "np.ndarray":
    """Take one two-stage strong-stability-preserving Runge-Kutta step of the coupled
    volume-fraction and phasic-mass system on the (N, N) periodic grid, recovering each
    phasic density within a stage as its mass over its volume fraction floored at the bound,
    without clipping the volume fraction.

    Args:
        phi: Volume fraction of phase one on the (N, N) periodic grid; index [i, j] refers to
            cell centre (x_i, y_j) with x along axis 0 and y along axis 1.
        m1: Mass per unit volume of phase one, shaped like phi, positive.
        m2: Mass per unit volume of phase two, shaped like phi, positive.
        u1: Velocity of phase one as a (2, N, N) array, entry 0 the x component and entry 1
            the y component.
        u2: Velocity of phase two as a (2, N, N) array laid out like u1.
        p1: Pressure of phase one, shaped like phi.
        p2: Pressure of phase two, shaped like phi.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5).
        Gamma: Regularisation velocity scale, non-negative.
        h: Grid spacing, positive, the same along both axes.
        dt: Time step, positive.

    Returns:
        A (5, N, N) float64 array: the updated volume fraction, the updated mass per unit
        volume of phase one, the updated mass per unit volume of phase two, and the x and y
        components of the interface velocity evaluated at the start of the step.

    Raises:
        ValueError: If dt is not positive, if phi is not two-dimensional, if u1 or u2 is not
            a (2, N, N) array matching it, if m1 or m2 is not a positive field shaped like
            phi, or if eps, delta, Gamma or h lies outside the domain accepted by the earlier
            steps.
    """
    return None
```

### Step 9

vortex_audit

Goal
----
Run the source's bounded interface regularisation through a prescribed two-dimensional vortex on three fixed configurations and audit the interface it produces, assembling the run from the earlier steps: initialise a circular blob of phase one with the bounded equilibrium profile, advance the coupled volume-fraction and phasic-mass system with the two-stage Runge-Kutta step, and record one row of diagnostics per configuration. The interface-length measure in the last column, the space integral of the volume fraction times its complement, is the quantity the audit is built around.

```python
def vortex_audit(vel_scale: float) -> "np.ndarray":
    """Run the three fixed vortex configurations of the audit and return one row of
    diagnostics per configuration, assembling each run from the earlier steps.

    The domain is the unit periodic square with N x N cells of spacing h = 1 / N, cell
    centres at ((i + 1/2) h, (j + 1/2) h), index [i, j] with x along axis 0 and y along
    axis 1. The finite bound is delta = 1e-2 and the interface thickness is eps = 2 h. Phase
    one has density 1.0e3 and stiffened-gas constants 4.4 and 6.0e8, phase two has density
    1.2 and constants 1.4 and 0; both phasic pressures are 1.0e5 everywhere and the
    equation of state must invert consistently for both phases. The three configurations
    are (N, R0, alpha, cx, cy, steps) = (96, 0.15, 0.4, 0.5, 0.5, 160), (128, 0.12, 0.7,
    0.25, 0.5, 100) and (80, 0.18, 0.25, 0.3, 0.3, 107). The initial volume fraction of
    phase one is the bounded equilibrium profile delta + (1 - 2 delta) (1 + tanh(d / (2
    eps))) / 2 of the signed distance d = R0 - r, where r is the distance from the cell
    centre to (cx, cy) measured on the periodic square (each coordinate difference reduced
    into [-1/2, 1/2)). The velocity of phase two is vel_scale times (sin(2 pi x) cos(2 pi
    y), -cos(2 pi x) sin(2 pi y)) evaluated at the cell centres, and the velocity of phase
    one is alpha times that of phase two. The regularisation velocity Gamma is the largest
    velocity magnitude over both phases and all cells, and the time step is 0.25 h /
    Gamma. The phasic masses start as the volume fractions times the phasic densities, and
    the system is advanced for the configuration's number of steps with the earlier
    Runge-Kutta step, without clipping.

    Args:
        vel_scale: Positive finite factor multiplying both phasic velocity fields.

    Returns:
        A (3, 6) float64 array with one row per configuration in the order listed: the
        largest magnitude of the regularisation flux on the initial profile in units of 1e-6,
        the ratio of the space integral of the volume fraction at the end to its initial
        value, the largest magnitude of the volume-fraction gradient at the end (central
        differences), the smallest volume fraction at the end, the space integral of the
        absolute difference between the final and initial volume fraction in units of 1e-3,
        and the space integral of the volume fraction times its complement at the end in
        units of 1e-3.

    Raises:
        ValueError: If vel_scale is not a positive finite number, if the equation of state
            does not invert consistently for both phases, if the mixture mass drifts by more
            than 1e-10 relative over a run, or if the final volume fraction is not a finite
            field in [0, 1].
    """
    return None
```
