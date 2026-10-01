# Mathematics-Computational_Mechanics-34

## Background

Phase-field fracture inherits a limitation from the Griffith theory it is built on: it prescribes the energy dissipated once a crack propagates, but it does not prescribe the stress at which one nucleates, and it captures cohesive behaviour only by turning the regularisation into a material-modelling choice. Recent eigenstrain-based formulations fix both by adding fracture eigenstrains that decouple the strength surface from the fracture energy, but they solve for those eigenstrains as extra global field variables inside a symbolic energy-minimisation framework, which keeps them out of ordinary finite-element codes.

The source of this task removes that barrier. Because the fracture eigenstrains carry no spatial gradients, their evolution can be resolved entirely at an integration point, in exactly the way a plasticity model resolves a consistency condition, with no additional global degrees of freedom. The work supplies eigenstrain directions for each strength criterion, including a new pressure-sensitive cone, and closed-form consistent tangent operators for both a smooth and a non-smooth strength surface, including the switching between active and inactive failure modes and the conditioning problem that the near-singular tangent creates at the onset of fracture.

This task exercises that point-local machinery directly. It asks for the constitutive update at a single material point driven along a fixed strain path, under both of the source's strength criteria, with the staggered stress and phase-field updates the source uses, and it audits the result through the gap between the stress increment the consistent tangent predicts and the increment the return mapping actually delivers, which is the property the source claims for its tangent.

## Problem

A phase-field fracture model rooted in Griffith's theory prescribes how much energy a crack releases but never prescribes the stress at which one nucleates, and it represents a cohesive zone that still transfers traction only by turning the choice of regularisation into a material-modelling choice. A recent formulation removes both limitations by introducing fracture eigenstrains that decouple the material strength from the fracture energy, and then makes the idea usable inside an ordinary finite-element code by exploiting the fact that those eigenstrains require no spatial gradients: their evolution can be resolved point by point, exactly as a plasticity model resolves a consistency condition, with no additional global degrees of freedom. Your job is to implement that point-local constitutive machinery exactly as the source specifies it and to audit it along a fixed strain path.

The load-bearing choices are the source's, and you are expected to recover them from the paper: the six-component tensor layout the formulation is written in, the eigenstrain directions attached to each of the two strength criteria, which scalar of the current state selects the branch of the pressure-sensitive criterion, the magnitude and sign of the potential that holds the non-penetration condition together with whether the degradation is allowed to act on it, which quantity the degradation function degrades in this formulation, the phase-field distribution function the source selects and the reason it rejects the alternative, the irreversibility rule applied to the crack driving force, and the treatment an inactive eigenstrain direction receives when the consistent tangent is assembled. Most of these choices move the reported number by far more than the grading tolerance; the remaining ones are visible in the intermediate values you are asked to report.

Material and model data, with stress in GPa and length in m: Young's modulus E = 200, Poisson ratio nu = 0.3, tensile strength ft = 0.150, shear strength fs = 0.150, the reference strain of the pressure-sensitive criterion eps_ref = 0.01, the residual parameter carried by the degradation function kappa = 1.0e-3, the small stabilising modulus factor kappa_t = 1.0e-9, fracture energy release rate Gc = 1.0e-4, and phase-field length scale ell = 0.05. Tensor norms are the ones the source defines in its nomenclature, ||A||^2 = A : A.

The point starts undamaged and unstrained: total strain zero, phase-field parameter zero, crack driving force zero. Seven strain increments are then applied in order. Each is given as tensor components (eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy) and is multiplied by the single free parameter load_scale:

  1. ( 6.0e-4, -2.0e-4, 0.0, 0.0,    0.0,  6.0e-4)
  2. (-1.8e-3, -1.1e-3, 0.0, 0.0,    0.0,  8.0e-4)
  3. (-6.0e-4, -4.0e-4, 0.0, 0.0,    0.0,  5.0e-4)
  4. ( 4.0e-4,  3.0e-4, 0.0, 0.0,    0.0, -1.1e-3)
  5. ( 1.3e-3,  9.0e-4, 0.0, 2.0e-4, 0.0,  4.0e-4)
  6. ( 9.0e-4,  7.0e-4, 0.0, 0.0,    0.0,  5.0e-4)
  7. ( 1.1e-3,  8.0e-4, 0.0, 0.0,    0.0,  6.0e-4)

The Laplacian of the phase field is supplied as data at each increment, in the same order: 0.0, 10.0, 20.0, 15.0, 40.0, 25.0, 30.0.

Run the path twice: once under the smooth, pressure-sensitive strength criterion, and once under the non-smooth criterion that carries independent tensile and shear strengths. At each increment take a single staggered pass in the source's order. Add the increment to the total strain; resolve the point-local return mapping at the phase-field parameter carried in from the previous increment; assemble the consistent tangent from the eigenstrain directions and their activity; apply the irreversibility rule to the crack driving force; then update the phase-field parameter from the source's phase-field balance evaluated at that point with the supplied Laplacian, with no separate constraint imposed on the phase-field parameter itself. Eigenstrain multipliers are non-negative.

Collect one audit row per increment and per criterion, fourteen rows in all, the seven increments under the smooth criterion first and the same seven under the non-smooth criterion second. The columns are [multiplier 1, multiplier 2, stress norm, degradable potential value, crack driving force after the irreversibility rule, phase-field parameter, tangent norm, tangent increment error], where the tangent increment error is the norm of the difference between the stress increment actually delivered across that increment and the stress increment that the consistent tangent assembled in that same increment predicts from the applied strain increment. The second multiplier is zero throughout the smooth-criterion rows.

Evaluate the audit at load_scale = 1.0. All quantities are float64, finite and deterministic: two runs on identical input must agree exactly. As the final answer, report the sum of the tangent increment error over all fourteen audit rows, to six significant figures.

In your reasoning, state the conventions you adopted and justify each of them from the source. Report as numerical results the two eigenstrain multipliers of the last non-smooth row, the phase-field parameter reached at the end of each of the two runs, and the tangent-increment-error subtotals of the two criteria separately.

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

elastic_operator

Goal
----
Assemble the isotropic linear-elastic operator that the source uses for the bulk response, in the six-component tensor layout the source fixes in its discretisation section (Eq. 20). Return it together with the two moduli that Eq. (2) is written in. The layout the source fixes there is load bearing: it decides what the tensor norms appearing in the strength potentials evaluate to.

```python
def elastic_operator(E: float, nu: float) -> "np.ndarray":
    """Assemble the isotropic linear-elastic operator that the source uses for the bulk
    response, in the six-component tensor layout the source fixes in its discretisation
    section (Eq. 20), together with the two moduli that Eq. (2) is written in.

    Args:
        E: Young's modulus, in GPa.
        nu: Poisson ratio.

    Returns:
        A (7, 6) float64 array: rows 0 to 5 are the 6x6 elastic operator, row 6 is
        [bulk modulus, shear modulus, 0, 0, 0, 0].

    Raises:
        ValueError: If E is not a positive finite number or nu does not lie in (-1, 0.5).
    """
    return None
```

### Step 2

degradation_state

Goal
----
Evaluate the source's degradation function, Eq. (6), together with its derivative with respect to the phase-field parameter and the positive source coefficient that the phase-field balance, Eq. (22), carries. Note carefully which quantity this function degrades in this formulation, because it is not the quantity a standard phase-field model degrades, and the small floor parameter it carries is named accordingly.

```python
def degradation_state(phi: float, kappa: float) -> "np.ndarray":
    """Evaluate the source's degradation function, Eq. (6), together with its derivative with
    respect to the phase-field parameter and the positive source coefficient that the phase-
    field balance, Eq. (22), carries.

    Args:
        phi: Phase-field parameter, in [0, 1].
        kappa: The small floor parameter carried by the degradation function, in (0, 1).

    Returns:
        A (3,) float64 array [d, dd/dphi, source coefficient], where the source coefficient
        is the phase-field-independent positive factor multiplying the crack driving force in
        the source term of Eq. (22).

    Raises:
        ValueError: If phi is not a finite number in [0, 1] or kappa is not a finite number
            in (0, 1).
    """
    return None
```

### Step 3

faceted_directions

Goal
----
Build the two eigenstrain directions that the source attaches to the non-smooth strength criterion, Eq. (17b), from the current total strain. The first direction carries the volumetric response and the second the shape-changing response. Both are written in the source in a specific scaled form, stated in the sentence that follows Eq. (17); reproduce that form exactly as given, because it is what fixes the meaning of each multiplier.

```python
def faceted_directions(eps: "np.ndarray") -> "np.ndarray":
    """Build the two eigenstrain directions that the source attaches to the non-smooth strength
    criterion, Eq. (17b), from the current total strain, in the scaled form the source states
    in the sentence that follows Eq. (17).

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.

    Returns:
        A (2, 6) float64 array whose rows are the volumetric and the shape-changing direction.
        When the trace of eps is exactly zero the volumetric direction is the zero vector, and
        when the deviatoric part of eps vanishes the shape-changing direction is the zero vector.

    Raises:
        ValueError: If eps is not a finite (6,) array.
    """
    return None
```

### Step 4

faceted_potential_gradients

Goal
----
Differentiate both strength potentials of the non-smooth criterion, Eq. (17a), with respect to the fracture eigenstrain. The first potential is the degradable one that sets the tensile and shear capacity; the second is the potential the source adds to hold the non-penetration condition once the point has lost its cohesion. Reproduce the second potential's magnitude and sign exactly as the source states them, and note which of the two the degradation function is allowed to touch.

```python
def faceted_potential_gradients(eta: "np.ndarray", ft: float, fs: float) -> "np.ndarray":
    """Differentiate both strength potentials of the non-smooth criterion, Eq. (17a), with
    respect to the fracture eigenstrain: the degradable potential that sets the tensile and
    shear capacity, and the non-degradable potential that holds the non-penetration condition.

    Args:
        eta: Fracture eigenstrain as a (6,) array in the source's six-component tensor layout.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.

    Returns:
        A (2, 6) float64 array: row 0 is the gradient of the degradable potential, row 1 the
        gradient of the non-degradable one. At a vanishing trace of eta the positive-part
        bracket of the degradable potential is inactive and the negative-part bracket of the
        non-degradable potential is active, and at a vanishing deviatoric part of eta the
        deviatoric term of the degradable gradient is zero.

    Raises:
        ValueError: If eta is not a finite (6,) array, or if ft or fs is not positive.
    """
    return None
```

### Step 5

smooth_direction_gradient

Goal
----
Evaluate the source's smooth, pressure-sensitive strength criterion, Eq. (18): its single eigenstrain direction, the gradient of its strength potential with respect to the fracture eigenstrain, and the potential's own value. The criterion is written as two pieces. Which piece applies is decided by one specific scalar of the current state, and that scalar is not the one the potential is differentiated against; the source says explicitly why it has to be that one.

```python
def smooth_direction_gradient(eps: "np.ndarray", eta: "np.ndarray", ft: float, fs: float, eps_ref: float) -> "np.ndarray":
    """Evaluate the source's smooth, pressure-sensitive strength criterion, Eq. (18): its
    single eigenstrain direction, the gradient of its strength potential with respect to the
    fracture eigenstrain, and the potential's own value, on the piece of Eq. (18) that the
    source's own switch selects.

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.
        eta: Fracture eigenstrain as a (6,) array in the same layout.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.
        eps_ref: Reference strain of the pressure-sensitive criterion.

    Returns:
        A (3, 6) float64 array: row 0 the eigenstrain direction, row 1 the potential
        gradient, row 2 is [potential value, branch flag, 0, 0, 0, 0] with branch flag +1 on
        the first piece of Eq. (18) and -1 on the second. Where the potential value is zero
        its gradient is taken as the zero vector, and a direction whose defining strain
        vanishes is the zero vector.

    Raises:
        ValueError: If eps or eta is not a finite (6,) array, or if ft, fs or eps_ref is not
            positive.
    """
    return None
```

### Step 6

smooth_return_map

Goal
----
Resolve the point-local return mapping of Eq. (23) for the smooth criterion: find the single eigenstrain magnitude that brings the state back onto the degraded strength surface, and report the resulting eigenstrain, stress and strength-potential value. Eq. (23b) is a two-case residual; use the source's own test to decide which case applies, and include the extra term the source adds to the active case, whose purpose it states immediately below the equation. The magnitude is non-negative.

```python
def smooth_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, eps_ref: float, kappa: float, kappa_t: float) -> "np.ndarray":
    """Resolve the point-local return mapping of Eq. (23) for the smooth criterion: find the
    single non-negative eigenstrain magnitude that brings the state back onto the degraded
    strength surface, and report the resulting eigenstrain, stress and strength-potential
    value.

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.
        phi: Phase-field parameter carried into the increment, in [0, 1].
        E: Young's modulus, in GPa.
        nu: Poisson ratio.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.
        eps_ref: Reference strain of the pressure-sensitive criterion.
        kappa: The small floor parameter carried by the degradation function, in (0, 1).
        kappa_t: The small stabilising modulus factor of the return mapping, non-negative.

    Returns:
        A (15,) float64 array [magnitude, active flag, potential value, eigenstrain (6),
        stress (6)], with the active flag 1.0 when the eigenstrain magnitude is non-zero.

    Raises:
        ValueError: If eps is not a (6,) array, if kappa_t is negative or not finite, or if
            E, nu, phi, kappa, ft, fs or eps_ref lies outside the domain accepted by the
            earlier steps.
    """
    return None
```

### Step 7

faceted_return_map

Goal
----
Resolve the point-local return mapping of Eq. (27) for the non-smooth criterion, where the eigenstrain can grow along either facet independently. Each facet carries its own two-case residual and its own admissibility test, and a facet whose test says the state is already admissible contributes nothing. Sweep the facets until both magnitudes stop moving, then report them together with their activity, the eigenstrain, the stress, and the degradable potential's value.

```python
def faceted_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, kappa: float, kappa_t: float) -> "np.ndarray":
    """Resolve the point-local return mapping of Eq. (27) for the non-smooth criterion, where
    the eigenstrain can grow along either facet independently: sweep the two facets until
    both non-negative magnitudes stop moving, then report them together with their activity,
    the eigenstrain, the stress and the degradable potential's value.

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.
        phi: Phase-field parameter carried into the increment, in [0, 1].
        E: Young's modulus, in GPa.
        nu: Poisson ratio.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.
        kappa: The small floor parameter carried by the degradation function, in (0, 1).
        kappa_t: The small stabilising modulus factor of the return mapping, non-negative.

    Returns:
        A (17,) float64 array [magnitude 1, magnitude 2, active flag 1, active flag 2,
        degradable potential value, eigenstrain (6), stress (6)], with an active flag 1.0
        when the corresponding magnitude is non-zero.

    Raises:
        ValueError: If eps is not a (6,) array, if kappa_t is negative or not finite, or if
            E, nu, phi, kappa, ft or fs lies outside the domain accepted by the earlier
            steps.
    """
    return None
```

### Step 8

consistent_tangent

Goal
----
Assemble the consistent tangent operator of Eq. (30) from the eigenstrain directions and their activity. The source builds it through the reduction it names in the text under Eq. (30) rather than by inverting the full iteration matrix, and it gives that reason explicitly. A direction whose strength was not exceeded is handled by a specific, stated substitution in both factors of that reduction. The small stabilising modulus appears on the reduced diagonal.

```python
def consistent_tangent(dirs: "np.ndarray", active: "np.ndarray", E: float, nu: float, kappa_t: float) -> "np.ndarray":
    """Assemble the consistent tangent operator of Eq. (30) from the eigenstrain directions and
    their activity, through the reduction the source names in the text under Eq. (30), with
    an inactive direction handled by the substitution stated there.

    Args:
        dirs: An (m, 6) array whose rows are the eigenstrain directions, in the source's
            six-component tensor layout.
        active: An (m,) array of activity flags, 1.0 for a direction whose strength was
            exceeded and 0.0 otherwise.
        E: Young's modulus, in GPa.
        nu: Poisson ratio.
        kappa_t: The small stabilising modulus factor of the return mapping, non-negative.

    Returns:
        A (6, 6) float64 consistent tangent operator in the same layout as the elastic
        operator.

    Raises:
        ValueError: If dirs is not a two-dimensional array with six columns, if active does
            not carry exactly one flag per direction, if kappa_t is negative or not finite,
            or if E or nu lies outside the domain accepted by the elastic operator step.
    """
    return None
```

### Step 9

phase_field_update

Goal
----
Solve the source's phase-field balance at a point for the updated phase-field value, given the current crack driving force, the source coefficient produced by the degradation step, the supplied Laplacian of the phase field, and the fracture properties. Use the phase-field distribution function the source selects in Section 2, together with the source coefficient its balance carries; the source argues at length why the other common choice of distribution function would be wrong for this formulation, and that argument fixes which one to use.

```python
def phase_field_update(F_hist: float, Gc: float, ell: float, source_coeff: float, lap_phi: float) -> float:
    """Solve the source's phase-field balance at a point for the updated phase-field value,
    given the current crack driving force, the source coefficient produced by the
    degradation step, the supplied Laplacian of the phase field, and the fracture
    properties, using the distribution function the source selects in Section 2.

    Args:
        F_hist: Crack driving force after the irreversibility rule, in GPa, non-negative.
        Gc: Fracture energy release rate.
        ell: Phase-field length scale, in m.
        source_coeff: The positive, phase-field-independent source coefficient returned by
            the degradation step.
        lap_phi: Laplacian of the phase field supplied as data at this increment.

    Returns:
        A Python float: the updated phase-field value.

    Raises:
        ValueError: If F_hist is negative or not finite, if Gc or ell is not positive, if
            source_coeff is not a positive finite number, or if lap_phi is not finite.
    """
    return None
```

### Step 10

cohesive_audit

Goal
----
Drive the fixed strain path of the configuration twice, once under each strength criterion, running the source's staggered update at every increment: resolve the point-local return mapping at the phase-field value carried in from the previous increment, build the consistent tangent, apply the source's irreversibility rule to the crack driving force, then update the phase field. Record one row per increment and per criterion. The last column is the norm of the difference between the actual stress increment and the increment the consistent tangent predicts.

```python
def cohesive_audit(load_scale: float) -> "np.ndarray":
    """Drive the fixed strain path of the configuration twice, once under each strength
    criterion, running the source's staggered update at every increment: resolve the point-
    local return mapping at the phase-field value carried in from the previous increment,
    build the consistent tangent, apply the source's irreversibility rule to the crack
    driving force, then update the phase field, and record one row per increment and per
    criterion.

    Args:
        load_scale: The single free parameter multiplying every strain increment of the
            configuration's fixed path, positive.

    Returns:
        A (14, 8) float64 array. Rows 0 to 6 are the seven increments under the smooth
        criterion and rows 7 to 13 the same increments under the non-smooth criterion. The
        columns are [magnitude 1, magnitude 2, stress norm, potential value, driving force
        after irreversibility, phase field, tangent norm, tangent increment error].

    Raises:
        ValueError: If load_scale is not a positive finite number.
    """
    return None
```
