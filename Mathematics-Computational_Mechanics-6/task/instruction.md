# Mathematics-Computational_Mechanics-6

## Background

Dynamic fragmentation couples fracture with dense simultaneous contacts. Penalty treatments of unilateral contact become unstable in that regime, so nonsmooth integrators enforce the contact condition as a set-valued constraint at the velocity level. This task exercises one semi-explicit member of that family on a one-dimensional bar.

## Problem

Dynamic fragmentation couples fracture with dense, simultaneously active contacts, and penalty-based contact treatments become unstable in that regime, so a family of nonsmooth integrators has been developed that enforces unilateral contact as a set-valued condition at the velocity level rather than through a stiff restoring force. This problem concerns one such semi-explicit scheme, in which the non-impulsive bulk dynamics are advanced explicitly with second-order accuracy while the contact impulse is obtained implicitly from a nonnegative quadratic program posed over the constraints judged active for the step; the impulsive velocity is then folded back into both the end-of-step velocity and the end-of-step configuration. Consider a uniform axial bar of length L = 0.254 m, cross-sectional area A = 6.45e-4 m^2, Young's modulus E = 211 GPa and density rho = 7847 kg/m^3, discretised into 50 equal two-node linear elements with a diagonal inertia operator, and let a single unilateral constraint act on the node at the leading end of the bar against a fixed rigid obstacle. The 26th element counted from the leading end, the one on the far side of the bar's midpoint node from the obstacle, is replaced by a degradable interface of cohesive strength 6.0e8 Pa and critical opening 6e-4 m, carrying a fixed tangential offset of 2e-5 m, with mixed-mode weight beta = 1.4 and regularisation parameter alpha = 4.0; its state variable starts at zero, it carries the same secant traction in compression as in tension, and the traction evaluated at the end-of-step configuration acts on the two nodes it separates through the end-of-step acceleration. The bar is launched as a rigid body at the obstacle with every node at velocity 5 m/s directed toward it, starting from a clearance of 3.17 times the distance the bar travels in one time step; contact is governed by a coefficient of restitution e = 0.85. Take the time step to be 0.40 times the explicit stability limit set by the element transit time of the material wave speed, and integrate for 5.0 multiples of the bar's round-trip wave transit time 2L/c, rounding the resulting step count to the nearest integer. Report the axial velocity of the trailing node, the node furthest from the obstacle, at the end of the final step. State the conventions you adopted at each point where the scheme leaves a choice open, and justify each one from the source literature. In your reasoning also report four quantities from the same run: the number of contact onsets, the accumulated contact impulse, the kinetic energy of the bar at the end of the final step, and the final value of the interface state variable.

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

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

bar_system_matrices

Goal
----
Assemble the discrete operators for a uniform axial bar discretised with linear elements: the elastic stiffness operator and the diagonal inertia used by an explicit integrator. Pack both into one array so the pair can be compared exactly.

```python
import numpy as np


def bar_system_matrices(n_elements, length, area, youngs, density):
    """Assemble the discrete operators for a uniform axial bar discretised with linear elements: the elastic stiffness operator and the diagonal inertia used by an explicit integrator. Pack both into one array so the pair can be compared exactly.

    Returns
    -------
    ndarray of shape (n_elements+2, n_elements+1): the stiffness operator in the first n_elements+1 rows, the diagonal inertia in the last row.

    Raises
    ------
    ValueError: if n_elements is below 1, or any of length, area, youngs or density is not positive.
    """
    return np.zeros(1)  # placeholder
```

### Step 2

critical_time_step

Goal
----
Return the stability limit on the time step for explicit integration of the discretised bar, from the element size and the material wave speed.

```python
import numpy as np


def critical_time_step(n_elements, length, youngs, density):
    """Return the stability limit on the time step for explicit integration of the discretised bar, from the element size and the material wave speed.

    Returns
    -------
    float, the critical time step in seconds.

    Raises
    ------
    ValueError: if n_elements is below 1, or any of length, youngs or density is not positive.
    """
    return 0.0  # placeholder
```

### Step 3

smooth_predictor

Goal
----
Advance the state one step using only the non-impulsive part of the dynamics, producing the configuration the contact treatment is built on.

```python
import numpy as np


def smooth_predictor(u, v, a, dt):
    """Advance the state one step using only the non-impulsive part of the dynamics, producing the configuration the contact treatment is built on.

    Returns
    -------
    ndarray, the smooth configuration at the end of the step, same shape as u.

    Raises
    ------
    ValueError: if u, v and a do not share a shape, or dt is not positive.
    """
    return np.zeros(1)  # placeholder
```

### Step 4

active_contact_set

Goal
----
Decide which unilateral constraints must carry an impulse over the coming step. The source fixes both the configuration this decision is taken on and whether the gap test is inclusive; follow it exactly.

```python
import numpy as np


def active_contact_set(u, v, a, dt, contact_rows):
    """Decide which unilateral constraints must carry an impulse over the coming step. The source fixes both the configuration this decision is taken on and whether the gap test is inclusive; follow it exactly.

    Returns
    -------
    ndarray of shape (n_constraints,), 1.0 where the constraint is active and 0.0 otherwise.

    Raises
    ------
    ValueError: if contact_rows has a column count that does not match the number of degrees of freedom.
    """
    return np.zeros(1)  # placeholder
```

### Step 5

contact_response_operator

Goal
----
Build the operator that maps impulses on the active constraints to the constraint velocities they produce. It must be consistent with how this scheme feeds the impulse back into the configuration, so it is not the operator a purely kinematic argument gives; take its form from the source.

```python
import numpy as np


def contact_response_operator(stiffness, lumped_mass, contact_rows, active, dt):
    """Build the operator that maps impulses on the active constraints to the constraint velocities they produce. It must be consistent with how this scheme feeds the impulse back into the configuration, so it is not the operator a purely kinematic argument gives; take its form from the source.

    Returns
    -------
    ndarray of shape (n_active, n_active); a (0, 0) array when no constraint is active.

    Raises
    ------
    ValueError: if dt is not positive, or any lumped_mass entry is not positive.
    """
    return np.zeros((1, 1))  # placeholder
```

### Step 6

contact_impulse

Goal
----
Given the response operator from the previous step, solve for the impulses on the active constraints. The right-hand side combines the incoming constraint velocity, the smooth acceleration contribution and the elastic term; how the coefficient of restitution enters it is fixed by the source. Impulses are nonnegative.

```python
import numpy as np


def contact_impulse(response_operator, stiffness, lumped_mass, contact_rows, active, smooth_state, v, a, dt, restitution):
    """Given the response operator from the previous step, solve for the impulses on the active constraints. The right-hand side combines the incoming constraint velocity, the smooth acceleration contribution and the elastic term; how the coefficient of restitution enters it is fixed by the source. Impulses are nonnegative.

    Returns
    -------
    ndarray of shape (n_active,) of nonnegative impulses; empty when nothing is active.

    Raises
    ------
    ValueError: if restitution lies outside [0, 1], or response_operator is not square of size n_active.
    """
    return np.zeros(1)  # placeholder
```

### Step 7

nonsmooth_state_update

Goal
----
Given the smooth state and the impulse, close the step and return the end-of-step configuration, velocity and acceleration. The weight the impulsive velocity carries into the configuration is set by the source and is not the same as its weight into the velocity.

```python
import numpy as np


def nonsmooth_state_update(stiffness, lumped_mass, contact_rows, active, smooth_state, v, a, dt, impulse):
    """Given the smooth state and the impulse, close the step and return the end-of-step configuration, velocity and acceleration. The weight the impulsive velocity carries into the configuration is set by the source and is not the same as its weight into the velocity.

    Returns
    -------
    ndarray of shape (3, n_dof): row 0 configuration, row 1 velocity, row 2 acceleration.

    Raises
    ------
    ValueError: if dt is not positive, any lumped_mass entry is not positive, or smooth_state and v do not share a shape.
    """
    return np.zeros((3, 1))  # placeholder
```

### Step 8

cohesive_effective_opening

Goal
----
Combine the normal and tangential parts of an interface opening into the single scalar measure the interface law is written in terms of. The source fixes how the tangential part is weighted; follow it exactly.

```python
import numpy as np


def cohesive_effective_opening(delta_n, delta_t, beta):
    """Combine the normal and tangential parts of an interface opening into the single scalar measure the interface law is written in terms of. The source fixes how the tangential part is weighted; follow it exactly.

    Returns
    -------
    ndarray of shape (n_interfaces,), the scalar opening measure per interface.

    Raises
    ------
    ValueError: if beta is negative, or delta_t does not have one row per interface.
    """
    return np.zeros(1)  # placeholder
```

### Step 9

cohesive_damage_update

Goal
----
Advance the interface state variable given the current opening measure and its previous value. The source states whether the state may decrease when the interface closes again; follow it.

```python
import numpy as np


def cohesive_damage_update(effective_opening, critical_opening, damage_previous):
    """Advance the interface state variable given the current opening measure and its previous value. The source states whether the state may decrease when the interface closes again; follow it.

    Returns
    -------
    ndarray of shape (n_interfaces,), the updated state variable in [0, 1].

    Raises
    ------
    ValueError: if critical_opening is not positive, damage_previous lies outside [0, 1], or the two arrays do not share a shape.
    """
    return np.zeros(1)  # placeholder
```

### Step 10

cohesive_stiffness_cap

Goal
----
Return the interface stiffness ceiling used to keep the explicit time step usable, together with the state-variable threshold that corresponds to it. Both come from the source; the quantities the ceiling is built from are what matter.

```python
import numpy as np


def cohesive_stiffness_cap(youngs, element_size, alpha, cohesive_strength, critical_opening):
    """Return the interface stiffness ceiling used to keep the explicit time step usable, together with the state-variable threshold that corresponds to it. Both come from the source; the quantities the ceiling is built from are what matter.

    Returns
    -------
    ndarray of shape (2,): the stiffness ceiling, then the state-variable threshold.

    Raises
    ------
    ValueError: if any of youngs, element_size, alpha, cohesive_strength or critical_opening is not positive.
    """
    return np.zeros(1)  # placeholder
```

### Step 11

cohesive_traction

Goal
----
Return the traction the interface carries for the given opening and state. The source splits the law at the threshold from the previous step and gives a different expression on each side; reproduce both, and note the tangential weighting used here is not the one used to form the scalar opening measure.

```python
import numpy as np


def cohesive_traction(delta_n, delta_t, beta, damage, cohesive_strength, critical_opening, damage_threshold):
    """Return the traction the interface carries for the given opening and state. The source splits the law at the threshold from the previous step and gives a different expression on each side; reproduce both, and note the tangential weighting used here is not the one used to form the scalar opening measure.

    Returns
    -------
    ndarray of shape (n_interfaces, 1 + n_tangential): the traction vector per interface.

    Raises
    ------
    ValueError: if damage lies outside [0, 1], or cohesive_strength or critical_opening is not positive.
    """
    return np.zeros(1)  # placeholder
```

### Step 12

nsn_bar_impact

Goal
----
Run the whole scheme for a bar launched at a rigid obstacle, with a degradable interface at its midpoint, and report the end-of-run diagnostics. Call the earlier step functions in sequence; do not re-implement them. Conventions this driver fixes: the constrained node is index 0 and the bar moves toward the obstacle at negative velocity; the released element has zero-based index n_elements // 2 counted from the constrained end, so the interface opening is u[mid + 1] - u[mid]; the interface carries the same secant traction in compression as in tension; the traction evaluated at the end-of-step configuration, times the cross-sectional area, enters the end-of-step acceleration of the two interface nodes and hence this step's trapezoidal velocity update; a contact onset is a step whose active set is nonempty after a step whose active set was empty.

```python
import numpy as np


def nsn_bar_impact(n_elements, length, area, youngs, density, impact_speed, restitution, time_step_fraction, initial_gap_fraction, bounce_cycles, cohesive_strength, critical_opening, beta, alpha, tangential_offset):
    """Run the whole scheme for a bar launched at a rigid obstacle, with a degradable interface at its midpoint, and report the end-of-run diagnostics. Call the earlier step functions in sequence; do not re-implement them.

    Returns
    -------
    ndarray of shape (5,): far-end nodal velocity, accumulated impulse, kinetic energy, number of contact onsets, final interface damage.

    Raises
    ------
    ValueError: if bounce_cycles is not positive, or any value it forwards to an earlier step is invalid.
    """
    return np.zeros(1)  # placeholder
```
