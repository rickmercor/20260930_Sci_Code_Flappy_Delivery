# Chemistry-Computational_Chemistry-3

## Problem

Roaming reactions are organised not by saddle points on the potential energy surface but by unstable invariant objects in phase space. Recent work extends a standard barrierless ion-molecule dissociation model from two degrees of freedom to three, by breaking its cylindrical symmetry with an azimuthal coupling whose order is fixed by the point group of the polyatomic fragment left behind, and asks whether the newly opened out-of-plane direction is dynamically active. The answer is decided by the linear stability of one periodic orbit, the one lying in the flat, intermediate-range region of the potential where the roaming motion takes place, in the direction transverse to the reaction plane.

Work with that three-degree-of-freedom model at zero total angular momentum, taking every parameter of the potential and both moments of inertia at the published values, in unified atomic mass units, angstroms and kilocalories per mole. The reaction plane is invariant under the flow for every value of the symmetry-breaking strength.

Locate that orbit at a total energy of 0.5 kilocalories per mole with the symmetry-breaking strength switched off, searching starting separations between 3.55 and 3.75 angstroms on the section at which the polar angle vanishes. Follow that orbit by continuation in twelve equal increments, first in the symmetry-breaking strength from zero to 0.45 at that same energy, and then in the total energy from 0.5 to 1.10 kilocalories per mole at that strength. At the end point integrate the variational equations alongside the trajectory over exactly one period, starting from the identity, and separate the resulting one-period matrix into the block acting on the out-of-plane coordinate and its conjugate momentum and the block acting on the four in-plane components.

Report the trace of the out-of-plane block.

State the conventions you adopted at each point where the framework leaves a choice open, justifying each from the source literature, and in your reasoning also report four further quantities for the same calculation: the abbreviated action of the orbit at the end point, its period in the model time unit, the largest absolute in-plane multiplier there, and the abbreviated action of the same orbit at the anchor point where the continuation began.

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

extract_radial_parameters

Goal
----
Collect the four constant coefficients of the radial stretch term of the model potential from its four published parameters, so that the potential can afterwards be evaluated as a fixed linear combination of an exponential and two inverse powers. The radial term is the common prefactor de/(c1 - 6) multiplying a bracket of three contributions in the scaled separation: plus 2(3 - c2) times the exponential, minus (4c2 - c1c2 + c1) times the inverse sixth power, and minus (c1 - 6)c2 times the inverse fourth power. Return the prefactor first, then those three bracket coefficients in that order, each multiplied by the prefactor.

```python
def extract_radial_parameters(de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    """Collect the four constant coefficients of the radial stretch term of the model potential from its four published parameters, so that the potential can afterwards be evaluated as a fixed linear combination of an exponential and two inverse powers. The radial term is the common prefactor de/(c1 - 6) multiplying a bracket of three contributions in the scaled separation: plus 2(3 - c2) times the exponential, minus (4c2 - c1c2 + c1) times the inverse sixth power, and minus (c1 - 6)c2 times the inverse fourth power. Return the prefactor first, then those three bracket coefficients in that order, each multiplied by the prefactor.

    Parameters
    ----------
    de : float
        Positive well-depth scale of the radial term, in kcal/mol.
    re : float
        Positive reference separation, in angstroms.
    c1 : float
        Positive shape exponent of the repulsive block; must not equal six.
    c2 : float
        Positive weight of the inverse-fourth block.

    Returns
    -------
    coefficients : np.ndarray
        ndarray of shape (4,): de/(c1-6), then 2(3-c2)*de/(c1-6), then -(4c2-c1c2+c1)*de/(c1-6), then -(c1-6)c2*de/(c1-6).

    Raises
    ------
    ValueError
        if any parameter is not a positive finite number, or if c1 equals six, which makes the prefactor singular.
    """
    return coefficients
```

### Step 2

radial_potential

Goal
----
Evaluate the radial stretch term of the model potential at one or more internuclear separations, using the coefficients of the previous step. The separation enters only through its ratio to the reference length.

```python
def radial_potential(r: "np.ndarray", de: float, re: float, c1: float, c2: float) -> "np.ndarray":
    """Evaluate the radial stretch term of the model potential at one or more internuclear separations, using the coefficients of the previous step. The separation enters only through its ratio to the reference length.

    Parameters
    ----------
    r : np.ndarray
        Strictly positive internuclear separation(s), in angstroms.
    de : float
        Positive well-depth scale, in kcal/mol.
    re : float
        Positive reference separation, in angstroms.
    c1 : float
        Positive shape exponent; must not equal six.
    c2 : float
        Positive weight of the inverse-fourth block.

    Returns
    -------
    v_radial : np.ndarray
        ndarray with the same shape as r: the radial potential in kcal/mol.

    Raises
    ------
    ValueError
        if any separation is not strictly positive, or if the parameters are invalid for the previous step.
    """
    return v_radial
```

### Step 3

angular_potential

Goal
----
Evaluate the angular part of the model potential at a Cartesian body-frame position, for the symmetry-breaking strength b.

```python
def angular_potential(x: "np.ndarray", y: "np.ndarray", z: "np.ndarray", ve: float, alpha: float, re: float, b: float) -> "np.ndarray":
    """Evaluate the angular part of the model potential at a Cartesian body-frame position, for the symmetry-breaking strength b.

    Parameters
    ----------
    x : np.ndarray
        Cartesian body-frame x component(s), in angstroms.
    y : np.ndarray
        Cartesian body-frame y component(s), in angstroms.
    z : np.ndarray
        Cartesian body-frame z component(s), in angstroms.
    ve : float
        Finite amplitude of the radial envelope, in kcal/mol.
    alpha : float
        Finite Gaussian width parameter of the envelope, in inverse square angstroms.
    re : float
        Reference separation of the envelope, in angstroms.
    b : float
        Non-negative dimensionless strength of the symmetry-breaking term.

    Returns
    -------
    v_angular : np.ndarray
        ndarray with the broadcast shape of x, y and z: the angular potential in kcal/mol.

    Raises
    ------
    ValueError
        if ve, alpha or b is not finite, if b is negative, or if the origin is in the domain.
    """
    return v_angular
```

### Step 4

hamiltonian

Goal
----
Evaluate the total energy of the system at a six-component Cartesian phase-space point (three position components followed by their conjugate momenta), at zero total angular momentum: the rigid-body rotational energy of the fragment, the translational kinetic energy of the departing atom, and the radial and angular potentials.

```python
def hamiltonian(state: "np.ndarray", params: dict) -> float:
    """Evaluate the total energy of the system at a six-component Cartesian phase-space point (three position components followed by their conjugate momenta), at zero total angular momentum: the rigid-body rotational energy of the fragment, the translational kinetic energy of the departing atom, and the radial and angular potentials.

    Parameters
    ----------
    state : np.ndarray
        Six-component phase-space point (x, y, z, px, py, pz).
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    energy : float
        float: the total energy in kcal/mol.

    Raises
    ------
    ValueError
        if the state does not have six components, if any moment of inertia or the mass is not positive, or if a required parameter is missing.
    """
    return energy
```

### Step 5

vector_field

Goal
----
Return the time derivative of the six-component Cartesian phase-space state generated by the energy function of the previous step, in the same component order as the state.

```python
def vector_field(state: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the time derivative of the six-component Cartesian phase-space state generated by the energy function of the previous step, in the same component order as the state.

    Parameters
    ----------
    state : np.ndarray
        Six-component phase-space point (x, y, z, px, py, pz).
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    derivative : np.ndarray
        ndarray of shape (6,): the time derivative of the state.

    Raises
    ------
    ValueError
        if the state does not have six components.
    """
    return derivative
```

### Step 6

planar_return_map

Goal
----
The plane y = 0 with py = 0 is invariant under the flow. Working inside it, start from the point where the polar angle is zero (so the departing atom is on the positive z axis at separation r, with radial momentum pr), fix the remaining momentum from the requested total energy, and integrate until the polar angle has advanced by exactly two pi. Return the separation and radial momentum reached there, together with the elapsed time.

```python
def planar_return_map(r: float, pr: float, energy: float, params: dict) -> "np.ndarray":
    """The plane y = 0 with py = 0 is invariant under the flow. Working inside it, start from the point where the polar angle is zero (so the departing atom is on the positive z axis at separation r, with radial momentum pr), fix the remaining momentum from the requested total energy, and integrate until the polar angle has advanced by exactly two pi. Return the separation and radial momentum reached there, together with the elapsed time.

    Parameters
    ----------
    r : float
        Positive starting separation on the section, in angstroms.
    pr : float
        Finite starting radial momentum on the section.
    energy : float
        Total energy held fixed along the trajectory, in kcal/mol.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    section_point : np.ndarray
        ndarray of shape (3,): the separation and radial momentum at the return, and the time taken in the model time unit.

    Raises
    ------
    ValueError
        if r is not positive and finite, if pr is not finite, if the requested energy is unattainable at that point, or if no full turn occurs within the time cap.
    """
    return section_point
```

### Step 7

locate_periodic_orbit

Goal
----
Find a fixed point of the return map of the previous step, treating the separation and the radial momentum as independent unknowns so that closure is checked directly in both section coordinates. Return the converged separation and radial momentum together with the period.

```python
def locate_periodic_orbit(r_guess: float, pr_guess: float, energy: float, params: dict) -> "np.ndarray":
    """Find a fixed point of the return map of the previous step, treating the separation and the radial momentum as independent unknowns so that closure is checked directly in both section coordinates. Return the converged separation and radial momentum together with the period.

    Parameters
    ----------
    r_guess : float
        Starting separation for the iteration, in angstroms.
    pr_guess : float
        Starting radial momentum for the iteration.
    energy : float
        Total energy held fixed, in kcal/mol.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    orbit : np.ndarray
        ndarray of shape (3,): the separation and radial momentum of the fixed point, and the period in the model time unit.

    Raises
    ------
    ValueError
        if no periodic orbit is found from the supplied guess, or if any evaluation of the return map is invalid.
    """
    return orbit
```

### Step 8

orbit_action

Goal
----
Return the abbreviated action of the periodic orbit located from the supplied starting guess, that is the integral of the momenta against the coordinates taken once around the orbit. Evaluate it as the integral of twice the kinetic energy over one period.

```python
def orbit_action(r0: float, pr0: float, energy: float, params: dict) -> float:
    """Return the abbreviated action of the periodic orbit located from the supplied starting guess, that is the integral of the momenta against the coordinates taken once around the orbit. Evaluate it as the integral of twice the kinetic energy over one period.

    Parameters
    ----------
    r0 : float
        Positive starting separation for the iteration, in angstroms.
    pr0 : float
        Finite starting radial momentum for the iteration.
    energy : float
        Finite total energy held fixed, in kcal/mol.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    action : float
        float: the abbreviated action of the orbit.

    Raises
    ------
    ValueError
        if the orbit cannot be located from the supplied guess.
    """
    return action
```

### Step 9

monodromy_matrix

Goal
----
Integrate the variational equations of the six-dimensional Cartesian flow alongside the trajectory for exactly one period, starting from the identity, and return the resulting six by six matrix.

```python
def monodromy_matrix(state: "np.ndarray", params: dict, period: float) -> "np.ndarray":
    """Integrate the variational equations of the six-dimensional Cartesian flow alongside the trajectory for exactly one period, starting from the identity, and return the resulting six by six matrix.

    Parameters
    ----------
    state : np.ndarray
        Six-component phase-space point on the orbit.
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.
    period : float
        Positive period of the orbit, in the model time unit.

    Returns
    -------
    monodromy : np.ndarray
        ndarray of shape (6, 6): the matrix propagating an initial displacement over one period.

    Raises
    ------
    ValueError
        if the state does not have six components or the period is not positive and finite.
    """
    return monodromy
```

### Step 10

roaming_stability_report

Goal
----
Run the whole chain for the model at the requested total energy and symmetry-breaking strength. Use the published parameter set throughout: perpendicular moment of inertia 2.373409 and axial moment 4.746818 in unified atomic mass units times square angstroms, reduced mass 0.9445 in unified atomic mass units, and for the potential de = 47.0 kcal/mol, re = 1.1 angstroms, c1 = 7.37, c2 = 1.61, ve = 55.0 kcal/mol and alpha = 1.0 inverse square angstroms. Locate the orbit at zero symmetry-breaking and an energy of 0.5 by scanning starting separations between 3.55 and 3.75, taking the first separation from which the iteration converges, then follow it by continuation in n_steps equal increments first in the symmetry-breaking strength at that energy and then in the energy at the requested strength. At the end point compute the period, the abbreviated action, and the one-period matrix; split that matrix into the block acting on the out-of-plane pair (the second position component and its conjugate momentum) and the block acting on the four in-plane components. Return the trace of the out-of-plane block first, then the orbit data.

```python
def roaming_stability_report(energy: float, b: float, n_steps: int) -> "np.ndarray":
    """Run the whole chain for the model at the requested total energy and symmetry-breaking strength. Use the published parameter set throughout: perpendicular moment of inertia 2.373409 and axial moment 4.746818 in unified atomic mass units times square angstroms, reduced mass 0.9445 in unified atomic mass units, and for the potential de = 47.0 kcal/mol, re = 1.1 angstroms, c1 = 7.37, c2 = 1.61, ve = 55.0 kcal/mol and alpha = 1.0 inverse square angstroms. Locate the orbit at zero symmetry-breaking and an energy of 0.5 by scanning starting separations between 3.55 and 3.75, taking the first separation from which the iteration converges, then follow it by continuation in n_steps equal increments first in the symmetry-breaking strength at that energy and then in the energy at the requested strength. At the end point compute the period, the abbreviated action, and the one-period matrix; split that matrix into the block acting on the out-of-plane pair (the second position component and its conjugate momentum) and the block acting on the four in-plane components. Return the trace of the out-of-plane block first, then the orbit data.

    Parameters
    ----------
    energy : float
        Positive total energy at the end point, in kcal/mol.
    b : float
        Non-negative symmetry-breaking strength at the end point.
    n_steps : int
        Number of equal continuation increments, at least one.

    Returns
    -------
    report : np.ndarray
        ndarray of shape (6,): the trace of the out-of-plane block, the separation and radial momentum of the orbit, its period, its abbreviated action, and the largest absolute in-plane multiplier.

    Raises
    ------
    ValueError
        if the energy is not positive and finite, if b is negative or not finite, if n_steps is below one, if the anchor orbit cannot be located, or if any continuation increment fails to converge.
    """
    return report
```
