# Physics-Optics-44

## Background

In a medium with a nonlocal nonlinear response, the refractive-index change at a point is not set
by the local intensity but by an average of the intensity over a neighbourhood. The size of that
neighbourhood, relative to the beam, controls how the beam and the index it induces reinforce one
another. Nematic liquid crystals are the standard laboratory realization: the reorientation of the
molecules is elastic, so the index perturbation spreads well beyond the illuminated region, and
the range of that spreading can be tuned by an applied bias. Thermal media and media with a
long-range dipolar response behave similarly.

Nonlocality changes the stability of self-trapped beams. A purely local self-focusing medium in
two transverse dimensions collapses; averaging the response over a finite range arrests that
collapse and admits stable self-trapped beams over a wide parameter range. It also changes how a
beam responds when the medium is altered along the propagation direction, because the beam's width
and the range of the response set competing length scales.

A recurring idea in the control of such systems is that a beam will track a slowly moving
equilibrium. If a parameter of the medium is changed gradually enough with distance, the beam
remains close to the state it would settle into for the instantaneous parameter value. The price
is distance: the slower the change, the longer the sample. Techniques developed to beat that
trade-off elsewhere in physics prescribe the desired history of the system first and then ask what
time- or distance-dependent parameters would produce it, which turns a slow following problem into
an algebraic inversion. Whether such a prescription is available depends on having a faithful
reduced description of the dynamics in terms of a small number of collective coordinates, and on
the parameter one is free to modulate appearing in that description in an invertible way.

Comparing different physical knobs on the same footing is the practical question. A given medium
may allow the response range, the local nonlinearity or an external guiding profile to be varied
with distance, and these are not interchangeable: they enter the beam's effective dynamics
differently, they differ in how smoothly they can be realized, and they differ in how much the
true field departs from the reduced description when the protocol is pushed to short distances.

## Problem

An optical beam propagating in a medium whose nonlinear index response is spatially nonlocal
behaves, over a wide range of conditions, like a single breathing collective degree of freedom
rather than a continuum of field degrees of freedom, and if the medium's parameters change slowly
enough along the propagation direction it follows the instantaneous equilibrium of that reduced
description adiabatically, so it can be compressed by slowly shortening the range of the nonlocal
response. Doing the same compression over a much shorter distance is the interesting problem: the
beam no longer follows the equilibrium, and the compression has to be engineered rather than merely
ramped. The medium, the compression demanded of it and the numerical scheme are fixed below; what
comes out are the beam widths the engineered protocols actually reach and how hard each one has to
be driven to get there.

```
Propagation        the dimensionless beam envelope u(x, z) obeys

                     i u_z + (1/2) u_xx + u * (R conv |u|^2) + gamma |u|^2 u
                                                             - alpha^2 x^2 u = 0

                   where (R conv |u|^2)(x) = integral R(x - s) |u(s,z)|^2 ds and the
                   nonlocal response is the normalized Gaussian

                     R(x) = exp(-x^2 / sigma^2) / (sqrt(pi) sigma),   integral R dx = 1.

                   sigma is the response length, gamma the local Kerr coefficient and
                   alpha the strength of the parabolic confinement. Any one of sigma,
                   gamma or alpha^2 may be made a function of z; the other two are held
                   fixed at the values below.

Medium             power P = 12, gamma = 0.2, alpha = 0.15.

Compression        the response length is to be taken from sigma_i = 4.0 to
                   sigma_f = 0.8. The initial and final beam widths, a_i and a_f, are the
                   widths at which the beam would sit in equilibrium at sigma_i and at
                   sigma_f respectively, for this medium.

Width              for any field, the beam width is the intensity-weighted spread about
                   the field's own intensity centroid, normalized so that the profile
                   exp(-x^2 / (2 a^2)) has width exactly a.

Prescribed history the width is required to follow
                     a(z) = a_i + (a_f - a_i) (10 s^3 - 15 s^4 + 6 s^5),   s = z / z_f,
                   over the shortened distance z_f = 2.2.

Grid               1024 points on -30 <= x < 30, spaced uniformly, right endpoint
                   excluded, with the conjugate discrete-Fourier frequency grid.

Integrator         symmetric split-step in z: half of the diffraction operator applied in
                   the frequency domain, then the full nonlinear and confinement phase in
                   the coordinate domain, then the remaining half of diffraction. Use 2000
                   equal steps from 0 to z_f, and evaluate the z-dependent parameter once
                   per step, at the midpoint of that step.

Launch             the field at z = 0 is the unchirped Gaussian of width a_i carrying
                   power P, that is u(x, 0) = sqrt(P / (sqrt(pi) a_i)) exp(-x^2/(2 a_i^2)).

Root brackets      bisection searches use the bracket [1e-3, 50] for a width and
                   [1e-2, 20] for a distance unless another is given.
```

Build the three protocols that make the width follow the prescribed history: one that modulates
the response length sigma along z, one that modulates the Kerr coefficient gamma, and one that
modulates the squared confinement strength alpha^2. In each case the two parameters not being
modulated stay at their values above, and a protocol that modulates gamma or alpha^2 leaves sigma
at sigma_i throughout. Propagate the launched field under each protocol with the scheme above and
measure the beam width at z = z_f. In your reasoning report the two equilibrium widths a_i and
a_f, the final width reached by each of the three protocols, and the final width reached when the
response length is instead ramped linearly from sigma_i to sigma_f over the same distance, say
which single feature of the required relation is what separates an engineered protocol from that
linear ramp, give the width the beam would sit at halfway along that linear ramp if it were able
to follow the change arbitrarily closely, and account for why the final widths do not land on the
target width, and give the change in the effective potential of the reduced description from the
initial equilibrium width to the target equilibrium width, both evaluated at the initial response
length, that potential being the one whose
negative derivative with respect to the width is the predicted width acceleration. For that same linear ramp also report the largest value along it of the
dimensionless measure of adiabatic following for this reduced description, the one that
weighs how fast the instantaneous equilibrium width moves against the frequency of small
width oscillations about it, sampling that reference width at the 2001 equally spaced
distances from z = 0 to z = z_f inclusive, together with the threshold below which
following is taken to be adiabatic and the factor by which the ramp misses it, and, for the response-length protocol,
how accurately it lands, measured against the medium's own self-trapped profile at the
target response length rather than against a width, taking that profile from 2000
imaginary-distance steps of size 0.01 on the same grid, and say how it compares with the
value above which such a protocol is considered high fidelity. Also report the relative normalized total variation of each of the three
engineered control profiles, the measure that has been proposed for comparing how smoothly
such knobs must be driven, taking each profile at the 2001 equally spaced distances from
z = 0 to z = z_f inclusive. Your final answer is a single number: the beam width at z = z_f under the protocol
that modulates the response length.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 15 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

response_spectrum

Goal
----
Return the spectral transfer function of the medium's nonlocal response on the given angular-frequency grid. The response in real space is the normalized Gaussian with characteristic length sigma, so that convolving an intensity profile with it is a multiplication on this grid. Raise ValueError if sigma is not strictly positive.

```python
def response_spectrum(k: 'np.ndarray', sigma: float) -> 'np.ndarray':
    """Return the spectral transfer function of the medium's nonlocal response on the given angular-frequency grid. The response in real space is the normalized Gaussian with characteristic length sigma, so that convolving an intensity profile with it is a multiplication on this grid. Raise ValueError if sigma is not strictly positive.

    Returns
    -------
    ndarray of float64 with the same shape as k: the spectral transfer function.

    Raises
    ------
    ValueError
        If sigma is not strictly positive.
    """
    return None
```

### Step 2

width_acceleration

Goal
----
Return the second derivative of the beam width with respect to propagation distance that the reduced description of this medium predicts, for a beam of the given width carrying the given power in a medium with the given local Kerr coefficient, confinement strength and nonlocal length. The relation is the one the source derives by reducing the full propagation equation to the beam's collective coordinates. Raise ValueError if the width, the power or the nonlocal length is not strictly positive.

```python
def width_acceleration(a: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the second derivative of the beam width with respect to propagation distance that the reduced description of this medium predicts, for a beam of the given width carrying the given power in a medium with the given local Kerr coefficient, confinement strength and nonlocal length. The relation is the one the source derives by reducing the full propagation equation to the beam's collective coordinates. Raise ValueError if the width, the power or the nonlocal length is not strictly positive.

    Returns
    -------
    float: the second derivative of the width with respect to propagation distance.

    Raises
    ------
    ValueError
        If a, power or sigma is not strictly positive.
    """
    return None
```

### Step 3

width_curvature

Goal
----
Return the squared frequency of small oscillations of the width about the given width, defined as minus the derivative of the predicted width acceleration with respect to the width, evaluated at that width and at the given medium parameters. Raise ValueError if the width, the power or the response length is not strictly positive.

```python
def width_curvature(a: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the squared frequency of small oscillations of the width about the given width, defined as minus the derivative of the predicted width acceleration with respect to the width, evaluated at that width and at the given medium parameters. Raise ValueError if the width, the power or the response length is not strictly positive.

    Returns
    -------
    float: the squared oscillation frequency at the given width, which may be negative.

    Raises
    ------
    ValueError
        If a, power or sigma is not strictly positive.
    """
    return None
```

### Step 4

potential_change

Goal
----
Return the change in the effective potential of the reduced description between two widths, that is its value at the second width minus its value at the first. The potential is the function whose negative derivative with respect to the width is the predicted width acceleration, so only differences of it are defined. Raise ValueError if either width, the power or the response length is not strictly positive.

```python
def potential_change(a1: float, a2: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the change in the effective potential of the reduced description between two widths, that is its value at the second width minus its value at the first. The potential is the function whose negative derivative with respect to the width is the predicted width acceleration, so only differences of it are defined. Raise ValueError if either width, the power or the response length is not strictly positive.

    Returns
    -------
    float: the potential at the second width minus the potential at the first.

    Raises
    ------
    ValueError
        If a1, a2, power or sigma is not strictly positive.
    """
    return None
```

### Step 5

equilibrium_width

Goal
----
Return the beam width at which the reduced description predicts no width acceleration, for the given power, local Kerr coefficient, confinement strength and nonlocal length. Locate it by bisection on the bracket from lo to hi, halving the bracket two hundred times and returning its midpoint. Raise ValueError if the bracket is not ordered and positive, or if the predicted acceleration does not change sign across it.

```python
def equilibrium_width(power: float, gamma: float, alpha: float, sigma: float, lo: float = 1e-3, hi: float = 50.0) -> float:
    """Return the beam width at which the reduced description predicts no width acceleration, for the given power, local Kerr coefficient, confinement strength and nonlocal length. Locate it by bisection on the bracket from lo to hi, halving the bracket two hundred times and returning its midpoint. Raise ValueError if the bracket is not ordered and positive, or if the predicted acceleration does not change sign across it.

    Returns
    -------
    float: the width at which the predicted width acceleration vanishes.

    Raises
    ------
    ValueError
        If lo, hi do not satisfy 0 < lo < hi, or the predicted acceleration has the same sign at both ends.
    """
    return None
```

### Step 6

adiabatic_reference

Goal
----
Return the width a beam would sit at if the response length were ramped linearly from sigma_i to sigma_f over a distance zf and the beam were able to follow that change arbitrarily closely, evaluated at the distance z. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive or if the ramped response length reaches zero or below.

```python
def adiabatic_reference(z: float, zf: float, power: float, gamma: float, alpha: float, sigma_i: float, sigma_f: float) -> float:
    """Return the width a beam would sit at if the response length were ramped linearly from sigma_i to sigma_f over a distance zf and the beam were able to follow that change arbitrarily closely, evaluated at the distance z. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive or if the ramped response length reaches zero or below.

    Returns
    -------
    float: the equilibrium width at the response length the ramp has reached.

    Raises
    ------
    ValueError
        If zf is not strictly positive, or the ramped response length is not strictly positive.
    """
    return None
```

### Step 7

minimum_jerk_width

Goal
----
Return the prescribed beam width and its second derivative with respect to propagation distance at the given distance, for the smooth interpolation the source adopts between the initial and final widths over a total distance zf. The interpolation is the quintic in the normalized distance whose value matches the endpoints and whose first and second derivatives vanish at both of them. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive.

```python
def minimum_jerk_width(z: float, zf: float, a_i: float, a_f: float) -> 'np.ndarray':
    """Return the prescribed beam width and its second derivative with respect to propagation distance at the given distance, for the smooth interpolation the source adopts between the initial and final widths over a total distance zf. The interpolation is the quintic in the normalized distance whose value matches the endpoints and whose first and second derivatives vanish at both of them. Clamp the normalized distance to the closed unit interval. Raise ValueError if zf is not strictly positive.

    Returns
    -------
    ndarray of shape (2,), float64: the prescribed width and its second derivative.

    Raises
    ------
    ValueError
        If zf is not strictly positive.
    """
    return None
```

### Step 8

inverse_control

Goal
----
Return the value one medium parameter must take so that the reduced description produces exactly the given width and width acceleration, with the other two parameters held at their supplied values. The knob is named by the string 'sigma' for the nonlocal length, 'gamma' for the local Kerr coefficient, or 'alpha2' for the squared confinement strength. The supplied value of the parameter being solved for is ignored in every case. Raise ValueError if the width or the power is not strictly positive, if the knob name is not one of the three, or if no positive nonlocal length reproduces the requested pair.

```python
def inverse_control(knob: str, a: float, add: float, power: float, gamma: float, alpha: float, sigma: float) -> float:
    """Return the value one medium parameter must take so that the reduced description produces exactly the given width and width acceleration, with the other two parameters held at their supplied values. The knob is named by the string 'sigma' for the nonlocal length, 'gamma' for the local Kerr coefficient, or 'alpha2' for the squared confinement strength. The supplied value of the parameter being solved for is ignored in every case. Raise ValueError if the width or the power is not strictly positive, if the knob name is not one of the three, or if no positive nonlocal length reproduces the requested pair.

    Returns
    -------
    float: the required value of the named parameter, the squared strength when the knob is 'alpha2'.

    Raises
    ------
    ValueError
        If a or power is not strictly positive, if knob is not 'sigma', 'gamma' or 'alpha2', or if no positive nonlocal length reproduces the requested width and acceleration.
    """
    return None
```

### Step 9

propagate_field

Goal
----
March the complex field through the medium and return it after the last step. Use the symmetric split-step scheme in which each step applies half of the diffraction operator in the frequency domain, then the whole nonlinear and confinement phase in the coordinate domain, then the remaining half of diffraction. Element j of controls is the value the named knob takes on step j, with the other two parameters held at their supplied values. Raise ValueError if dz is not strictly positive, if controls is not a non-empty one-dimensional array, if u0 does not match the grid length, or if the knob name is not one of the three.

```python
def propagate_field(u0: 'np.ndarray', npts: int, half_width: float, controls: 'np.ndarray', dz: float, gamma: float, alpha: float, sigma: float, knob: str) -> 'np.ndarray':
    """March the complex field through the medium and return it after the last step. Use the symmetric split-step scheme in which each step applies half of the diffraction operator in the frequency domain, then the whole nonlinear and confinement phase in the coordinate domain, then the remaining half of diffraction. Element j of controls is the value the named knob takes on step j, with the other two parameters held at their supplied values. Raise ValueError if dz is not strictly positive, if controls is not a non-empty one-dimensional array, if u0 does not match the grid length, or if the knob name is not one of the three.

    Returns
    -------
    ndarray of shape (npts,), complex128: the field after the final step.

    Raises
    ------
    ValueError
        If dz is not strictly positive, controls is not a non-empty one-dimensional array, u0 does not match the grid length, or knob is not 'sigma', 'gamma' or 'alpha2'.
    """
    return None
```

### Step 10

stationary_soliton

Goal
----
Return the real, non-negative profile the medium relaxes to under the prescribed number of imaginary-distance steps, on the grid the earlier steps use. Start from the unit-width Gaussian rescaled to carry the given power, and take each step as half of the diffraction operator, then the full nonlinear and confinement factor, then the remaining half, rescaling to the given power after every step. Return the pointwise magnitude of the field after the final rescaling, so a sample whose real part is negative contributes its magnitude and not zero. Raise ValueError if the step size is not strictly positive, if the step count is not a positive integer, or if the power is not strictly positive.

```python
def stationary_soliton(npts: int, half_width: float, power: float, gamma: float, alpha: float, sigma: float, dtau: float, nsteps: int) -> 'np.ndarray':
    """Return the real, non-negative profile the medium relaxes to under the prescribed number of imaginary-distance steps, on the grid the earlier steps use. Start from the unit-width Gaussian rescaled to carry the given power, and take each step as half of the diffraction operator, then the full nonlinear and confinement factor, then the remaining half, rescaling to the given power after every step. Return the pointwise magnitude of the field after the final rescaling, so a sample whose real part is negative contributes its magnitude and not zero. Raise ValueError if the step size is not strictly positive, if the step count is not a positive integer, or if the power is not strictly positive.

    Returns
    -------
    ndarray of shape (npts,), float64: the relaxed profile, real and non-negative.

    Raises
    ------
    ValueError
        If dtau or power is not strictly positive, or nsteps is not a positive integer.
    """
    return None
```

### Step 11

beam_width

Goal
----
Return the beam width of the given field on the given coordinate grid, defined so that a Gaussian intensity profile of the form used by the reduced description returns exactly its own width parameter. Measure the spread about the field's own intensity centroid, not about the origin. Raise ValueError if the field and the grid have different shapes, if the spacing is not strictly positive, or if the field carries no power.

```python
def beam_width(u: 'np.ndarray', x: 'np.ndarray', dx: float) -> float:
    """Return the beam width of the given field on the given coordinate grid, defined so that a Gaussian intensity profile of the form used by the reduced description returns exactly its own width parameter. Measure the spread about the field's own intensity centroid, not about the origin. Raise ValueError if the field and the grid have different shapes, if the spacing is not strictly positive, or if the field carries no power.

    Returns
    -------
    float: the beam width of the field.

    Raises
    ------
    ValueError
        If u and x have different shapes, dx is not strictly positive, or the field carries no power.
    """
    return None
```

### Step 12

overlap_fidelity

Goal
----
Return the squared magnitude of the normalized overlap between a complex field and a target profile sampled on the same uniform grid of spacing dx: the squared magnitude of the integral of the conjugated target times the field, divided by the product of the two fields' powers. Raise ValueError if the two arrays have different shapes, if dx is not strictly positive, or if either field carries no power.

```python
def overlap_fidelity(u: 'np.ndarray', target: 'np.ndarray', dx: float) -> float:
    """Return the squared magnitude of the normalized overlap between a complex field and a target profile sampled on the same uniform grid of spacing dx: the squared magnitude of the integral of the conjugated target times the field, divided by the product of the two fields' powers. Raise ValueError if the two arrays have different shapes, if dx is not strictly positive, or if either field carries no power.

    Returns
    -------
    float: the normalized squared overlap, between zero and one.

    Raises
    ------
    ValueError
        If the shapes differ, dx is not strictly positive, or either field carries no power.
    """
    return None
```

### Step 13

confinement_floor

Goal
----
Return the smallest value the squared confinement strength takes anywhere along a compression protocol of length zf. The width is driven from the equilibrium width the medium supports at sigma_i to the equilibrium width it supports at sigma_f, following the minimum-jerk interpolation between them on the closed interval from zero to zf. At every propagation distance the squared confinement strength is the one value that reproduces that width together with its second derivative, with the nonlocal length held fixed at sigma_i and the Kerr coefficient held at gamma throughout: the confinement is the only parameter that moves. Search the closed interval including both endpoints and locate the minimum to an absolute accuracy of 1e-10 in the propagation distance. A negative result means the protocol demands a locally antiguiding profile somewhere along the way. Raise ValueError if zf is not strictly positive.

```python
def confinement_floor(zf: float, power: float, gamma: float, alpha: float, sigma_i: float, sigma_f: float) -> float:
    """Return the smallest value the squared confinement strength takes anywhere along a compression protocol of length zf. The width is driven from the equilibrium width the medium supports at sigma_i to the equilibrium width it supports at sigma_f, following the minimum-jerk interpolation between them on the closed interval from zero to zf. At every propagation distance the squared confinement strength is the one value that reproduces that width together with its second derivative, with the nonlocal length held fixed at sigma_i and the Kerr coefficient held at gamma throughout: the confinement is the only parameter that moves. Search the closed interval including both endpoints and locate the minimum to an absolute accuracy of 1e-10 in the propagation distance. A negative result means the protocol demands a locally antiguiding profile somewhere along the way. Raise ValueError if zf is not strictly positive.

    Returns
    -------
    float: the minimum squared confinement strength over the protocol, negative when the protocol turns antiguiding.

    Raises
    ------
    ValueError
        If zf is not strictly positive.
    """
    return None
```

### Step 14

antiguiding_onset

Goal
----
Return the shortest compression protocol that stays everywhere guiding: the protocol length in the closed interval from lo to hi at which the smallest squared confinement strength demanded anywhere along the protocol is exactly zero. Protocols shorter than this demand a locally antiguiding profile, longer ones do not. The protocol is the same one the previous quantity describes, built from the same two equilibrium widths and the same minimum-jerk interpolation. Locate the crossing to an absolute accuracy of 1e-10 in the protocol length. Raise ValueError unless lo and hi satisfy 0 < lo < hi, or if the interval does not bracket a change of sign.

```python
def antiguiding_onset(power: float, gamma: float, alpha: float, sigma_i: float, sigma_f: float, lo: float = 1e-2, hi: float = 20.0) -> float:
    """Return the shortest compression protocol that stays everywhere guiding: the protocol length in the closed interval from lo to hi at which the smallest squared confinement strength demanded anywhere along the protocol is exactly zero. Protocols shorter than this demand a locally antiguiding profile, longer ones do not. The protocol is the same one the previous quantity describes, built from the same two equilibrium widths and the same minimum-jerk interpolation. Locate the crossing to an absolute accuracy of 1e-10 in the protocol length. Raise ValueError unless lo and hi satisfy 0 < lo < hi, or if the interval does not bracket a change of sign.

    Returns
    -------
    float: the shortest protocol length that remains everywhere guiding.

    Raises
    ------
    ValueError
        If lo and hi do not satisfy 0 < lo < hi, or if the interval brackets no change of sign.
    """
    return None
```

### Step 15

shortcut_audit

Goal
----
Run the whole audit at the prescribed configuration and return the eleven reported quantities in the order given below. The refinement multiplies both the number of grid points and the number of propagation steps; the 2001 equally spaced distances at which the adiabatic reference is sampled for the adiabatic-following measure do not change with it. Raise ValueError if refine is not a positive integer.

```python
def shortcut_audit(refine: int) -> 'np.ndarray':
    """Run the whole audit at the prescribed configuration and return the eleven reported quantities in the order given below. The refinement multiplies both the number of grid points and the number of propagation steps; the 2001 equally spaced distances at which the adiabatic reference is sampled for the adiabatic-following measure do not change with it. Raise ValueError if refine is not a positive integer.

    Returns
    -------
    ndarray of shape (11,), float64: the final beam width under the nonlocal-length knob, under the Kerr knob and under the confinement knob; the initial and target equilibrium widths; the final width under a linear nonlocal-length ramp; the adiabatic reference width at the midpoint of that ramp; and the largest value, over the interior of those 2001 samples, of the magnitude of the centred three-point second difference of the adiabatic reference width divided by the product of that width with its squared oscillation frequency; the change in the effective potential from the initial to the target equilibrium width at the initial response length; and the accuracy of the nonlocal-length protocol against the medium's own self-trapped profile at the target response length, relaxed over 2000 imaginary-distance steps of size 0.01 on the same grid; and the shortest protocol length, found as in the previous step with its default bracket, for which the squared confinement strength demanded by the confinement-knob protocol stays non-negative everywhere.

    Raises
    ------
    ValueError
        If refine is not a positive integer.
    """
    return None
```
