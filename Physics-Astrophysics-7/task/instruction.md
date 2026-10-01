# Physics-Astrophysics-7

## Background

An ultralight axion behaves as a classical scalar field that rolls down its potential from
a misaligned initial value and then oscillates about the minimum. The oscillation period is
set by the inverse of the field's mass, and for the masses of cosmological interest that is
very much shorter than the expansion time. Integrating the exact equations of motion to the
present day is therefore prohibitive: the solver has to resolve every oscillation while the
quantities anyone wants, the background drift and the growth of structure, live underneath
them.

The usual response is to stop resolving the oscillation and track a cycle-averaged fluid
instead, with a density, a pressure and a velocity that vary only on the expansion
timescale. That works, but it throws the oscillation away, and the point at which one
switches from the exact description to the averaged one leaks into the answer even though
it is only a precision setting and should not affect the physics at all.

Two ingredients change the picture. Trading the real field for a complex wavefunction turns
the second-order equation of motion into a first-order one whose only awkwardness is an
explicit oscillatory factor, and a mode expansion of that wavefunction isolates a slow
component obeying equations with no explicit time dependence. The slow component plays the
role that the cycle average plays in the fluid picture, and exactly how it is defined is
what fixes the corrections that follow. The corrections are organised as a series in small
parameters, one built from the expansion rate and one from the comoving wavenumber.

The same construction carries over to linear perturbations, where the wavefunction
perturbation is coupled to the metric perturbations of the synchronous gauge, and the
gradient now supplies a second source of corrections alongside the expansion rate.

Switching between the two descriptions at the transition time is the step that has to be
got right, and it has to be done for the background and for the perturbations in turn.

## Problem

Ultralight axions behave as a coherently oscillating classical field whose oscillation
period is set by the inverse of the mass, far shorter than the expansion time, so neither
the background evolution nor the growth of perturbations can be carried to late times
directly. A recent line of work replaces the exact evolution with an effective description
that varies only on the expansion timescale and, unlike an effective fluid, can rebuild the
oscillation it discarded at both levels. Your task is to run that construction on one
prescribed configuration and report how faithfully the rebuilt perturbation tracks the
exact one.

Use the following configuration. Natural units with hbar = c = 1 are used throughout, and
the reduced Planck mass and the axion mass are both set to 1, so time is measured in units
of the inverse axion mass.

- Content: a spatially flat universe holding the axion plus two other components, one with
  energy density 0.126 a^-3 and no pressure, one with energy density 0.0432 a^-4 and
  pressure equal to one third of its energy density. a is the scale factor. Those two
  components carry no perturbations at all, so the metric is perturbed by the axion alone.
- Perturbations are treated in the synchronous gauge at the single comoving wavenumber
  k = 0.5, at linear order.
- Initial data at t = 0: a = 1, axion wavefunction psi = 0.312 + 0i, wavefunction
  perturbation dpsi = 0.001 + 0i, and curvature perturbation eta = 0.001. The axion energy
  density carried by the wavefunction is m times the squared modulus of psi, and its
  pressure is the quantity that oscillates at twice the mass.
- Transition time t* = 5. End time t_f = 65.
- The background matching at the transition time is refined for exactly 4 sweeps.
- Comparison times: the 400 equally spaced times t = t* + j (t_f - t*) / 400 for
  j = 1, 2, ..., 400.

Evolve the exact system, background and perturbation together, from t = 0 to the transition
time; match the effective description there, background first and then perturbation; evolve
the effective description to the end time; and at every comparison time rebuild the exact
wavefunction perturbation from it and compare against the exact perturbation carried forward
over the same interval. After the transition time the effective description carries its own
scale factor, seeded at the exact value there and expanding at the effective rate, and it is
that scale factor that sets the other components' density and pressure wherever the effective
description or the rebuilding needs them.

At the transition, initialize the slow curvature perturbation by inverting its reconstruction relation, rather than from the matched trace-metric rate through the effective time-time constraint.

In your reasoning, report the expansion rate at the initial time, the modulus of the matched
slow-mode wavefunction perturbation at the transition time, the axion density contrast of the slow mode at the end time, and the
slow-mode expansion rate there. Also state briefly, in a sentence or two each, how the slow
component of a variable is defined, what the effective description changes besides the
equation of motion, how the values handed to it at the transition time are obtained, and
what structure the rebuilding of the exact quantities has. Your final answer must be a single
number: the largest absolute difference between the rebuilt and the exact wavefunction
perturbation over the comparison times.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words): the few scalars that determine the final number, and the brief statements asked for above.

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

Implement **all 13 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

exact_hubble

Goal
----
Return the Hubble rate of a flat universe containing the axion described by the complex wavefunction psi together with all other species, whose combined energy density is rho_other. m is the axion mass and m_pl the reduced Planck mass. Raise ValueError if the total density comes out negative.

```python
def exact_hubble(psi: 'np.ndarray', rho_other: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    """Return the Hubble rate of a flat universe containing the axion described by the complex wavefunction psi together with all other species, whose combined energy density is rho_other. m is the axion mass and m_pl the reduced Planck mass. Raise ValueError if the total density comes out negative.

    Returns
    -------
    ndarray with the shape of psi, float64: the Hubble rate.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if the total density is negative.
    """
    return None
```

### Step 2

slow_hubble

Goal
----
Return the Hubble rate that the effective theory assigns to the slow-mode wavefunction psi_s, with rho_other_s the slow-mode energy density of all other species. Write A for abs(psi_s)**2. The effective total density is m*A + rho_other_s + (3*A/(32*m*m_pl**2)) * (m*A + 2*rho_other_s), and the slow-mode Hubble rate is sqrt of that total divided by 3*m_pl**2. Note that this is not the exact constraint of the previous step with slow variables substituted: the effective theory carries a relativistic correction of its own, the term with coefficient 3/32 above. Raise ValueError if the total comes out negative.

```python
def slow_hubble(psi_s: 'np.ndarray', rho_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    """Return the Hubble rate that the effective theory assigns to the slow-mode wavefunction psi_s, with rho_other_s the slow-mode energy density of all other species. Write A for abs(psi_s)**2. The effective total density is m*A + rho_other_s + (3*A/(32*m*m_pl**2)) * (m*A + 2*rho_other_s), and the slow-mode Hubble rate is sqrt of that total divided by 3*m_pl**2. Note that this is not the exact constraint of the previous step with slow variables substituted: the effective theory carries a relativistic correction of its own, the term with coefficient 3/32 above. Raise ValueError if the total comes out negative.

    Returns
    -------
    ndarray with the shape of psi_s, float64: the slow-mode Hubble rate.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if the total slow-mode density is negative.
    """
    return None
```

### Step 3

exact_rate

Goal
----
Return the proper-time derivative of the exact complex wavefunction psi at time t, given the Hubble rate hubble and the axion mass m. The result is complex and has the shape of psi. t may be a scalar or an array, and broadcasts against the other array inputs.

```python
def exact_rate(psi: 'np.ndarray', hubble: 'np.ndarray', m: float, t: 'np.ndarray') -> 'np.ndarray':
    """Return the proper-time derivative of the exact complex wavefunction psi at time t, given the Hubble rate hubble and the axion mass m. The result is complex and has the shape of psi. t may be a scalar or an array, and broadcasts against the other array inputs.

    Returns
    -------
    ndarray with the shape of psi, complex128: the time derivative of psi.

    Raises
    ------
    ValueError
        If m is not strictly positive.
    """
    return None
```

### Step 4

slow_rate

Goal
----
Return the proper-time derivative of the slow-mode wavefunction psi_s under the effective theory, given the slow-mode Hubble rate hubble_s and the slow-mode energy density and pressure of all other species. Write A for abs(psi_s)**2. The derivative is the sum of three terms: -1.5*hubble_s*psi_s, plus (3j/(16*m*m_pl**2)) * psi_s * (3*m*A + 2*rho_other_s), minus (9/(32*m**2*m_pl**2)) * hubble_s * psi_s * (m*A + rho_other_s + p_other_s). The result is complex and has the shape of psi_s. It carries no explicit dependence on time.

```python
def slow_rate(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', rho_other_s: 'np.ndarray', p_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    """Return the proper-time derivative of the slow-mode wavefunction psi_s under the effective theory, given the slow-mode Hubble rate hubble_s and the slow-mode energy density and pressure of all other species. Write A for abs(psi_s)**2. The derivative is the sum of three terms: -1.5*hubble_s*psi_s, plus (3j/(16*m*m_pl**2)) * psi_s * (3*m*A + 2*rho_other_s), minus (9/(32*m**2*m_pl**2)) * hubble_s * psi_s * (m*A + rho_other_s + p_other_s). The result is complex and has the shape of psi_s. It carries no explicit dependence on time.

    Returns
    -------
    ndarray with the shape of psi_s, complex128: the time derivative of psi_s.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive.
    """
    return None
```

### Step 5

matching_coefficients

Goal
----
Return the four coefficients that turn the reconstruction into a linear system for the real and imaginary parts of the slow-mode wavefunction at time t. The first two are the coefficients multiplying the unknown slow-mode parts; the last two collect everything that does not multiply an unknown linearly. Outputs are stacked along a new leading axis of length 4 in that order, all real. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, c4 and s4 for cos and sin of 4*m*t, and w for rho_other_s + p_other_s. The four are, in order: (3*hubble_s/(4*m))*s2; (3*hubble_s/(4*m))*c2; (3/(64*m*m_pl**2)) * (c4*R**3 + (4*s2 - s4)*I**3 - (4*s2 - 3*s4)*R**2*I + (8*c2 - 3*c4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(c2*R + s2*I); (3/(64*m*m_pl**2)) * ((4*s2 + s4)*R**3 + c4*I**3 - (8*c2 + 3*c4)*R**2*I - (4*s2 + 3*s4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(s2*R - c2*I). t may be a scalar or an array, and broadcasts against the other array inputs.

```python
def matching_coefficients(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', rho_other_s: 'np.ndarray', p_other_s: 'np.ndarray', m: float, m_pl: float, t: 'np.ndarray') -> 'np.ndarray':
    """Return the four coefficients that turn the reconstruction into a linear system for the real and imaginary parts of the slow-mode wavefunction at time t. The first two are the coefficients multiplying the unknown slow-mode parts; the last two collect everything that does not multiply an unknown linearly. Outputs are stacked along a new leading axis of length 4 in that order, all real. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, c4 and s4 for cos and sin of 4*m*t, and w for rho_other_s + p_other_s. The four are, in order: (3*hubble_s/(4*m))*s2; (3*hubble_s/(4*m))*c2; (3/(64*m*m_pl**2)) * (c4*R**3 + (4*s2 - s4)*I**3 - (4*s2 - 3*s4)*R**2*I + (8*c2 - 3*c4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(c2*R + s2*I); (3/(64*m*m_pl**2)) * ((4*s2 + s4)*R**3 + c4*I**3 - (8*c2 + 3*c4)*R**2*I - (4*s2 + 3*s4)*R*I**2) + (3/(16*m**2*m_pl**2))*w*(s2*R - c2*I). t may be a scalar or an array, and broadcasts against the other array inputs.

    Returns
    -------
    ndarray of shape (4,) + np.shape(psi_s), float64: the two linear coefficients followed by the two collected terms, for the real and imaginary parts in turn.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive.
    """
    return None
```

### Step 6

solve_matching_system

Goal
----
Return the real and imaginary parts of the slow-mode wavefunction implied by one sweep of the matching, given the known exact complex wavefunction psi and the four coefficients from the previous step in the order that step returns them. psi is a complex scalar and coeffs a length-4 real array. Raise ValueError if coeffs is not length 4, or if the system is singular.

```python
def solve_matching_system(psi: complex, coeffs: 'np.ndarray') -> 'np.ndarray':
    """Return the real and imaginary parts of the slow-mode wavefunction implied by one sweep of the matching, given the known exact complex wavefunction psi and the four coefficients from the previous step in the order that step returns them. psi is a complex scalar and coeffs a length-4 real array. Raise ValueError if coeffs is not length 4, or if the system is singular.

    Returns
    -------
    ndarray of shape (2,), float64: the real part of the slow-mode wavefunction followed by its imaginary part.

    Raises
    ------
    ValueError
        If coeffs does not hold exactly four entries, or if the system is singular.
    """
    return None
```

### Step 7

exact_perturbation_rate

Goal
----
Return the proper-time derivative of the exact complex wavefunction perturbation dpsi at comoving wavenumber k, in the synchronous gauge. psi is the exact background wavefunction, hubble the exact expansion rate, hdot the proper-time derivative of the synchronous metric perturbation, and a the scale factor. Writing q for 1j*k**2/(2*m*a**2), the derivative is -(1.5*hubble + q)*dpsi - 0.25*psi*hdot + exp(2j*m*t) * ((1.5*hubble - q)*conj(dpsi) + 0.25*conj(psi)*hdot). Note the sign of the gradient term reverses inside the oscillating bracket. The result is complex and has the shape of dpsi. t may be a scalar or an array, and broadcasts against the other array inputs.

```python
def exact_perturbation_rate(
    dpsi: "np.ndarray",
    psi: "np.ndarray",
    hubble: "np.ndarray",
    hdot: "np.ndarray",
    a: "np.ndarray",
    m: float,
    k: float,
    t: "np.ndarray",
) -> "np.ndarray":
    """Return the proper-time derivative of the exact complex wavefunction
    perturbation dpsi at comoving wavenumber k, in the synchronous gauge.
    psi is the exact background wavefunction, hubble the exact expansion
    rate, hdot the proper-time derivative of the synchronous metric
    perturbation, and a the scale factor. Writing q for 1j*k**2/(2*m*a**2),
    the derivative is -(1.5*hubble + q)*dpsi - 0.25*psi*hdot + exp(2j*m*t)
    * ((1.5*hubble - q)*conj(dpsi) + 0.25*conj(psi)*hdot). Note the sign of
    the gradient term reverses inside the oscillating bracket. The result
    is complex and has the shape of dpsi. t may be a scalar or an array,
    and broadcasts against the other array inputs.

    Returns
    -------
    ndarray with the shape of dpsi, complex128: the time derivative of
    dpsi.

    Raises
    ------
    ValueError
        If m is not strictly positive, or if any entry of a is not strictly
        positive.
    """
    return None
```

### Step 8

slow_perturbation_rate

Goal
----
Return the proper-time derivative of the slow-mode wavefunction perturbation dpsi_s at comoving wavenumber k under the effective theory. It carries no explicit dependence on time. The derivative is the sum of five terms: -(1.5*hubble_s + 1j*k**2/(2*m*a_s**2))*dpsi_s, minus 0.25*hdot_s*psi_s, plus (3j*hubble_s/(8*m) + k**2/(16*m**2*a_s**2))*psi_s*hdot_s, plus (3j*psi_s**2/(16*m_pl**2))*conj(dpsi_s), plus (9j*hubble_s**2/(8*m) + 3j*abs(psi_s)**2/(8*m_pl**2) + 1j*k**4/(8*m**3*a_s**4))*dpsi_s. The result is complex and has the shape of dpsi_s.

```python
def slow_perturbation_rate(
    dpsi_s: "np.ndarray",
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    hdot_s: "np.ndarray",
    a_s: "np.ndarray",
    m: float,
    m_pl: float,
    k: float,
) -> "np.ndarray":
    """Return the proper-time derivative of the slow-mode wavefunction
    perturbation dpsi_s at comoving wavenumber k under the effective
    theory. It carries no explicit dependence on time. The derivative is
    the sum of five terms: -(1.5*hubble_s + 1j*k**2/(2*m*a_s**2))*dpsi_s,
    minus 0.25*hdot_s*psi_s, plus (3j*hubble_s/(8*m) +
    k**2/(16*m**2*a_s**2))*psi_s*hdot_s, plus
    (3j*psi_s**2/(16*m_pl**2))*conj(dpsi_s), plus
    (9j*hubble_s**2/(8*m) + 3j*abs(psi_s)**2/(8*m_pl**2) +
    1j*k**4/(8*m**3*a_s**4))*dpsi_s. The result is complex and has the
    shape of dpsi_s.

    Returns
    -------
    ndarray with the shape of dpsi_s, complex128: the time derivative of
    dpsi_s.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if any entry of a_s is
        not strictly positive.
    """
    return None
```

### Step 9

metric_slow_rates

Goal
----
Return the proper-time derivatives of the two slow-mode synchronous metric perturbations at comoving wavenumber k, for a universe whose non-axion species carry no perturbations. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The first output is ((2*m_pl**2*k**2/(a_s**2*hubble_s))*eta_s + (m/hubble_s)*Re(P)) / m_pl**2. The second output is the real part of ((1j*m/4)*M - (1/16)*abs(psi_s)**2*hdot_s - (3/8)*hubble_s*P - (1j*k**2/(16*m*a_s**2))*M) / m_pl**2, where hdot_s is the first output. The two are stacked along a new leading axis of length 2 in that order, both real.

```python
def metric_slow_rates(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', eta_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, m_pl: float, k: float) -> 'np.ndarray':
    """Return the proper-time derivatives of the two slow-mode synchronous metric perturbations at comoving wavenumber k, for a universe whose non-axion species carry no perturbations. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The first output is ((2*m_pl**2*k**2/(a_s**2*hubble_s))*eta_s + (m/hubble_s)*Re(P)) / m_pl**2. The second output is the real part of ((1j*m/4)*M - (1/16)*abs(psi_s)**2*hdot_s - (3/8)*hubble_s*P - (1j*k**2/(16*m*a_s**2))*M) / m_pl**2, where hdot_s is the first output. The two are stacked along a new leading axis of length 2 in that order, both real.

    Returns
    -------
    ndarray of shape (2,) + np.shape(psi_s), float64: the time derivative of the trace metric perturbation followed by that of the curvature perturbation.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, or if any entry of a_s or hubble_s is not strictly positive.
    """
    return None
```

### Step 10

perturbation_matching

Goal
----
Return the slow-mode perturbation and the slow-mode trace metric rate implied by matching at time t, given the exact perturbation dpsi and the exact trace metric rate hdot there, and the slow-mode background psi_s with its expansion rate. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, and define Bp = (3*hubble_s/(4*m))*c2 + (k**2/(4*m**2*a_s**2))*s2, Bm = (3*hubble_s/(4*m))*s2 - (k**2/(4*m**2*a_s**2))*c2, Dp = (c2*R + s2*I)/(8*m), Dm = (c2*I - s2*R)/(8*m), Ep = (3/m_pl**2)*(I*s2 + R*c2), Em = (3/m_pl**2)*(I*c2 - R*s2). Solve the real three by three system whose matrix rows are [1+Bm, -Bp, -Dm], [-Bp, 1-Bm, -Dp] and [-Em, -Ep, 1] against the right-hand side [Re(dpsi), Im(dpsi), hdot]. The three outputs are the real part of the slow-mode perturbation, its imaginary part, and the slow-mode trace metric rate, in that order. Raise ValueError if the system is singular.

```python
def perturbation_matching(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, m_pl: float, k: float, t: float, dpsi: complex, hdot: float) -> 'np.ndarray':
    """Return the slow-mode perturbation and the slow-mode trace metric rate implied by matching at time t, given the exact perturbation dpsi and the exact trace metric rate hdot there, and the slow-mode background psi_s with its expansion rate. Write R and I for the real and imaginary parts of psi_s, c2 and s2 for cos and sin of 2*m*t, and define Bp = (3*hubble_s/(4*m))*c2 + (k**2/(4*m**2*a_s**2))*s2, Bm = (3*hubble_s/(4*m))*s2 - (k**2/(4*m**2*a_s**2))*c2, Dp = (c2*R + s2*I)/(8*m), Dm = (c2*I - s2*R)/(8*m), Ep = (3/m_pl**2)*(I*s2 + R*c2), Em = (3/m_pl**2)*(I*c2 - R*s2). Solve the real three by three system whose matrix rows are [1+Bm, -Bp, -Dm], [-Bp, 1-Bm, -Dp] and [-Em, -Ep, 1] against the right-hand side [Re(dpsi), Im(dpsi), hdot]. The three outputs are the real part of the slow-mode perturbation, its imaginary part, and the slow-mode trace metric rate, in that order. Raise ValueError if the system is singular.

    Returns
    -------
    ndarray of shape (3,), float64: the real and imaginary parts of the slow-mode perturbation followed by the slow-mode trace metric rate.

    Raises
    ------
    ValueError
        If m or m_pl is not strictly positive, if a_s is not strictly positive, or if the system is singular.
    """
    return None
```

### Step 11

reconstruct_perturbation

Goal
----
Return the exact complex wavefunction perturbation at proper time t and comoving wavenumber k, rebuilt from the slow-mode perturbation dpsi_s. Write E for exp(2j*m*t). The reconstruction is dpsi_s minus ((3j*hubble_s/(4*m) + k**2/(4*m**2*a_s**2))*conj(dpsi_s) + (1j*hdot_s/(8*m))*conj(psi_s)) * E. The result is complex and has the shape of dpsi_s. t may be a scalar or an array, and broadcasts against the other array inputs.

```python
def reconstruct_perturbation(dpsi_s: 'np.ndarray', psi_s: 'np.ndarray', hubble_s: 'np.ndarray', hdot_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float, t: 'np.ndarray') -> 'np.ndarray':
    """Return the exact complex wavefunction perturbation at proper time t and comoving wavenumber k, rebuilt from the slow-mode perturbation dpsi_s. Write E for exp(2j*m*t). The reconstruction is dpsi_s minus ((3j*hubble_s/(4*m) + k**2/(4*m**2*a_s**2))*conj(dpsi_s) + (1j*hdot_s/(8*m))*conj(psi_s)) * E. The result is complex and has the shape of dpsi_s. t may be a scalar or an array, and broadcasts against the other array inputs.

    Returns
    -------
    ndarray with the shape of dpsi_s, complex128: the reconstructed perturbation.

    Raises
    ------
    ValueError
        If m is not strictly positive, or if any entry of a_s is not strictly positive.
    """
    return None
```

### Step 12

slow_perturbation_observables

Goal
----
Return the slow-mode axion density, pressure and velocity perturbations at comoving wavenumber k. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The density perturbation is m*Re(P). The pressure perturbation is (k**2/(4*m*a_s**2))*Re(P); it is not the slow part of the exact pressure perturbation, which carries the oscillating factor instead, and it vanishes with the wavenumber. The velocity perturbation is the real part of -0.5j*M + (3*hubble_s/(4*m))*P + (1j*k**2/(8*m**2*a_s**2))*M + (1/(8*m))*abs(psi_s)**2*hdot_s. The three are stacked along a new leading axis of length 3 in that order, all real.

```python
def slow_perturbation_observables(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', hdot_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float) -> 'np.ndarray':
    """Return the slow-mode axion density, pressure and velocity perturbations at comoving wavenumber k. Write P for conj(psi_s)*dpsi_s + psi_s*conj(dpsi_s) and M for conj(psi_s)*dpsi_s - psi_s*conj(dpsi_s). The density perturbation is m*Re(P). The pressure perturbation is (k**2/(4*m*a_s**2))*Re(P); it is not the slow part of the exact pressure perturbation, which carries the oscillating factor instead, and it vanishes with the wavenumber. The velocity perturbation is the real part of -0.5j*M + (3*hubble_s/(4*m))*P + (1j*k**2/(8*m**2*a_s**2))*M + (1/(8*m))*abs(psi_s)**2*hdot_s. The three are stacked along a new leading axis of length 3 in that order, all real.

    Returns
    -------
    ndarray of shape (3,) + np.shape(psi_s), float64: the density, pressure and velocity perturbations of the axion slow mode.

    Raises
    ------
    ValueError
        If m is not strictly positive, or if any entry of a_s is not strictly positive.
    """
    return None
```

### Step 13

axion_reconstruction_audit

Goal
----
Run the whole audit at the given refinement by calling every earlier step function and using its output, and return the reported diagnostics. Use classical fourth-order Runge-Kutta. Set nominal counts n_exact=26000*refine and n_slow=2600*refine. Evolve the exact system from 0 to t*=5 with round(n_exact*5/65) uniform steps. Then compare at t_j=5+0.15*j for j=1,...,400, advancing both systems between successive comparison times with uniform steps: max(1, round(n_exact*0.15/65)) exact steps and max(1, round(n_slow*0.15/60)) effective steps per interval. Here round uses nearest-integer rounding with ties to even. Use four background-matching sweeps, starting the slow background guess at the exact transition value and computing its slow Hubble rate before each sweep. psi_init is the axion wavefunction at t=0 and defaults to the prescribed instance; the benchmark is axion_reconstruction_audit(1). For the returned density contrast use delta_a_s=delta_rho_a_s/rho_a_s, where rho_a_s=m*abs(psi_s)**2 + 3*(m*abs(psi_s)**2+rho_other_s)*abs(psi_s)**2/(16*m*m_pl**2). Here delta_rho_a_s is the density perturbation returned by slow_perturbation_observables, rho_other_s is the other-component density evaluated at a_s, and m and m_pl are the axion and reduced Planck masses. Evaluate these quantities at the end time.

```python
def axion_reconstruction_audit(
    refine: int, psi_init: complex = 0.312 + 0.0j
) -> "np.ndarray":
    """Run the whole audit at the given refinement by calling every earlier
    step function and using its output, and return the reported
    diagnostics. Use classical fourth-order Runge-Kutta. Set nominal counts
    n_exact=26000*refine and n_slow=2600*refine. Evolve the exact system
    from 0 to t*=5 with round(n_exact*5/65) uniform steps. Then compare at
    t_j=5+0.15*j for j=1,...,400, advancing both systems between successive
    comparison times with uniform steps: max(1, round(n_exact*0.15/65))
    exact steps and max(1, round(n_slow*0.15/60)) effective steps per
    interval. Here round uses nearest-integer rounding with ties to even.
    Use four background-matching sweeps, starting the slow background guess
    at the exact transition value and computing its slow Hubble rate before
    each sweep. psi_init is the axion wavefunction at t=0 and defaults to
    the prescribed instance; the benchmark is
    axion_reconstruction_audit(1). For the returned density contrast use
    delta_a_s=delta_rho_a_s/rho_a_s, where rho_a_s=m*abs(psi_s)**2 +
    3*(m*abs(psi_s)**2+rho_other_s)*abs(psi_s)**2/(16*m*m_pl**2). Here
    delta_rho_a_s is the density perturbation returned by
    slow_perturbation_observables, rho_other_s is the other-component
    density evaluated at a_s, and m and m_pl are the axion and reduced
    Planck masses. Evaluate these quantities at the end time.

    Returns
    -------
    ndarray of shape (9,), float64: the largest and the mean absolute
    difference between the exactly evolved and the rebuilt wavefunction
    perturbation over the comparison times, the modulus of the matched
    slow-mode perturbation at the transition time, the slow-mode trace
    metric rate there, the modulus of the matched slow-mode background
    wavefunction there, the slow-mode scale factor at the end time, the
    axion density contrast of the slow mode at the end time, the slow-mode
    curvature perturbation there, and the slow-mode expansion rate there.

    Raises
    ------
    ValueError
        If refine is not a positive integer.
    """
    return None
```
