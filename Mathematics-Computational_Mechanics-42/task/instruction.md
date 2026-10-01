# Mathematics-Computational_Mechanics-42

## Background

Elastic waves travelling through a fractured rock mass are attenuated and scattered by the fractures they cross, and how much depends entirely on how the fracture is allowed to deform. The simplest treatment makes the fracture a two-sided spring: traction is transmitted in proportion to the displacement jump across it, in both the normal and the tangential direction. That model is linear and cheap, but it is only valid for small wave amplitudes, because it lets the fracture carry tension across an opening and lets the shear traction grow without bound.

Richer treatments replace the spring by contact mechanics. In the normal direction a unilateral non-penetration condition allows the fracture to open and carry nothing, while forbidding the two surfaces from interpenetrating; when they are in contact, the elastic closure may follow either a linear relation of constant stiffness or a nonlinear one that stiffens as the fracture closes toward a maximum admissible closure. In the tangential direction Coulomb friction caps the shear traction at a bound proportional to the normal compression, so that a point sticks while it is inside the friction cone and slips on its boundary. Because these are inequalities rather than equations, they are recast as equalities by a radial-return map so that a Newton solver can handle them, exactly as is done for plasticity. The resulting family of models, from linear spring to frictional contact with nonlinear normal closure, spans the physical complexity a wave-propagation code has to choose between, and the choice shows up directly in which parts of a fracture open, stick and slide.

## Problem

A fracture in a rock mass is not a crack that simply reflects everything that hits it: it deforms, it can close under compression until it stiffens, it can slip once the shear traction it carries exceeds what friction allows, and it can open and carry nothing at all. How a wave-propagation code represents that deformation changes the answer categorically. A recent computational framework unifies four fracture deformation models of increasing physical complexity, from linear spring-type formulations to full contact mechanics with Coulomb friction and unilateral non-penetration. Your job is to implement the source's four models exactly as specified and audit them on the fixed fracture configuration below. The load-bearing choices - the nonlinear normal stiffness relation and its inversion, the sign convention of the Coulomb friction bound, the radial-return map that enforces the friction conditions, the non-penetration condition that distinguishes a contact model from a spring, the time-integration update, and the two distinct wave impedances in the absorbing boundary condition - are the source's; recover them from the paper. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

The rock is a soft rock with density rho = 2600 kg/m^3 and Lame parameters lambda = mu = 4.0e9 Pa. A single fracture is sampled at six points and has normal and tangential fracture stiffnesses K_n = K_tau = 2.0e11 Pa/m, maximum allowed fracture closure du_max = 5.0e-5 m, and coefficient of friction F = 1.0. The radial-return map uses the numerical parameter c = 3.1e6. Time integration uses dt = 1.0e-7 s with the standard Newmark parameters stated in the paper. All prescribed jumps are multiplied by load_scale: the normal displacement jump is load_scale times (1.8e-5, -1.2e-5, -2.4e-5, 0.6e-5, -3.0e-5, -0.9e-5) m, the tangential displacement jump is load_scale times (0.8e-5, -1.1e-5, 1.6e-5, -0.4e-5, 2.2e-5, 0.5e-5) m, and the tangential velocity jump is load_scale times (0.8, -1.1, 1.6, -0.4, 2.2, 0.5) m/s. The previous tangential velocity and acceleration are (0.2, -0.4, 0.6, 0.1, -0.3, 0.5) m/s and (1.0e6, 0.0, -1.0e6, 2.0e6, 5.0e5, -5.0e5) m/s^2, and the same two vectors are also the previous normal velocity and acceleration, so the time integrator is advanced from the same previous state in both directions.

Evaluate the source's model over its four variants in the order S-Lin-Lin, S-Lin-BB, C-Coul-Lin, C-Coul-BB, where the first tag selects the spring or the contact family and the last selects the linear or the nonlinear normal relation. For each variant compute the normal contact traction from the prescribed normal displacement jump under that variant's normal relation and model family. Then compute the tangential traction: for the two spring variants apply the source's linear tangential relation to the prescribed tangential displacement jump, whereas for the two contact variants start from zero tangential traction and apply the source's radial-return map driven by the tangential velocity jump. Classify every fracture point by its opening, its slip tendency and its contact state. Advance both jumps by one step of the source's time integrator: the new tangential jump is the prescribed tangential jump plus the tangential traction divided by the tangential stiffness, and the new normal jump is the prescribed normal jump plus the normal traction divided by the normal stiffness. Build the source's absorbing boundary coefficient matrix for the declared material and the outward unit normal (1, 0), apply it at each fracture point to the two-component velocity formed from the updated normal and tangential velocity jumps, and sum the magnitudes of the resulting boundary tractions. Also evaluate the elastic normal closure that the source's normal relation returns for the computed traction. Collect each variant into the audit row [sum of normal tractions, sum of tangential traction magnitudes, sum of fracture openings, largest slip tendency, number of open points, number of sticking points, number of sliding points, norm of the updated tangential velocity jump, summed absorbing-boundary traction magnitude, sum of the elastic normal closure], so that the full audit is a float64 array of shape (4, 10) whose rows follow the variant order above.

Evaluate the audit with load_scale = 1.0. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. As the final answer, report the sum over the four variants of the summed absorbing-boundary traction magnitude, which is the ninth audit column, expressed in gigapascals, to six significant figures. That column is reached only through the whole chain: it applies the boundary coefficient matrix to the velocities the time integrator produces from the normal traction and from the tangential traction that the friction treatment of each variant returns, so every model choice above changes it.

In your reasoning give the four per-variant absorbing-boundary traction magnitude sums, which are the scalars whose sum is the final number, together with the four per-variant summed normal tractions, the number of open, sticking and sliding points of each contact variant, and the largest slip tendency of each spring variant.

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

normal_stiffness

Goal
----
Return the source's normal fracture deformation relation g_n(q_n), which maps the normal fracture contact traction to the normal deformation of the fracture (the paper's Eq. 15). Two variants are used: the linear (Schoenberg) relation selected by model='Lin', and the nonlinear Barton-Bandis relation selected by model='BB', which additionally depends on the maximum allowed fracture closure du_max and stiffens as the fracture closes toward it. Recover both exact expressions from the paper; the Barton-Bandis relation is NOT the linear one and its denominator is the source's.

```python
import numpy as np


def normal_stiffness(q_n, K_n, du_max, model):
    """q_n: normal fracture contact traction (<= 0 in compression); K_n: normal fracture
    stiffness; du_max: maximum allowed fracture closure; model: 'Lin' or 'BB'.
    Returns a float64 array shaped like q_n with the source's normal deformation relation
    g_n(q_n) (paper Eq. 15). Raises ValueError on an unknown model or a non-positive
    Barton-Bandis denominator."""
    return np.zeros_like(np.asarray(q_n, dtype=np.float64))
```

### Step 2

normal_traction

Goal
----
Return the normal fracture contact traction produced by a prescribed normal displacement jump, for both families of fracture deformation model. For the spring model (contact=False) the traction is whatever the source's normal relation of Eq. 15 gives when the deformation equals the displacement jump (the paper's Eq. 9), so the fracture can carry traction of either sign. For the contact mechanics model (contact=True) the source instead imposes the unilateral non-penetration conditions of Eq. 12, under which the traction is never tensile and vanishes wherever the fracture is open. Invert the source's relation exactly; the Barton-Bandis inversion is not the linear one.

```python
import numpy as np


def normal_traction(jump_n, K_n, du_max, model, contact):
    """jump_n: normal displacement jump across the fracture (negative in closure); K_n:
    normal fracture stiffness; du_max: maximum allowed closure; model: 'Lin' or 'BB';
    contact: False for the spring model (paper Eq. 9), True for the contact mechanics
    model with non-penetration (paper Eq. 12). Returns a float64 array shaped like jump_n
    with the normal contact traction q_n. Raises ValueError on an unknown model or a
    closure at or beyond du_max."""
    return np.zeros_like(np.asarray(jump_n, dtype=np.float64))
```

### Step 3

friction_bound

Goal
----
Return the Coulomb friction bound of the source's contact model from the normal contact traction and the coefficient of friction (the paper's Eq. 10). The source's sign convention has the normal contact traction non-positive in compression, and the bound must come out non-negative so that it can bound a magnitude. Recover the exact expression, including its sign, from the paper.

```python
import numpy as np


def friction_bound(q_n, F):
    """q_n: normal contact traction (<= 0 in compression); F: coefficient of friction.
    Returns a float64 array shaped like q_n with the source's Coulomb friction bound
    (paper Eq. 10), non-negative for a fracture in contact."""
    return np.zeros_like(np.asarray(q_n, dtype=np.float64))
```

### Step 4

coulomb_return

Goal
----
Apply the source's radial-return map to the tangential fracture traction (the inner projection of the paper's Eq. 13, the Alart-Curnier reformulation of the Coulomb conditions of Eq. 11). A trial tangential traction is formed from the current tangential traction and the tangential velocity jump scaled by the numerical parameter c, and is then scaled back onto the friction cone when it exceeds the bound and left untouched when it does not. The bound is clipped at zero before the comparison so that an open fracture returns no tangential traction. Recover the exact form of the map from the paper. Two-dimensional setting, so the tangential direction has a single component and the norm is the absolute value.

```python
import numpy as np


def coulomb_return(q_tau, jump_vel_tau, b, c):
    """q_tau: current tangential traction; jump_vel_tau: tangential velocity jump; b:
    Coulomb friction bound; c: positive numerical parameter of the return map. Returns a
    float64 array shaped like q_tau with the tangential traction after the source's
    radial-return projection (paper Eq. 13). Raises ValueError if c <= 0."""
    return np.zeros_like(np.asarray(q_tau, dtype=np.float64))
```

### Step 5

contact_state

Goal
----
Classify the mechanical state of each point of a fracture in the source's contact model. Row 0 is the fracture opening as the source defines it in Section 2.3.3, that is the part of the normal displacement jump not accounted for by the elastic normal deformation of Eq. 15. Row 1 is the source's slip tendency, the ratio of the tangential traction magnitude to the magnitude of the friction bound; report it as 0.0 wherever the friction bound vanishes, since row 2 already records that the point is open. Row 2 encodes the source's three contact states as 0 for open, 1 for sticking and 2 for sliding, using the source's own criteria: a point with no contact traction is open, a point in contact whose slip tendency is below one is sticking, and a point in contact whose slip tendency has reached one is sliding. Recover the definitions of the opening and the slip tendency from the paper.

```python
import numpy as np


def contact_state(jump_n, q_n, q_tau, K_n, du_max, model, F):
    """jump_n: (N,) normal displacement jump; q_n: (N,) normal contact traction; q_tau:
    (N,) tangential traction; K_n: normal fracture stiffness; du_max: maximum allowed
    closure; model: 'Lin' or 'BB'; F: coefficient of friction. Returns a float64 array
    (3, N): the fracture opening, the slip tendency (0.0 where the friction bound
    vanishes) and the state code (0 open, 1 sticking, 2 sliding)."""
    return None
```

### Step 6

newmark_update

Goal
----
Advance velocity and acceleration by one step of the time integrator the source uses for the elastic wave equation and for the tangential velocity jump (the paper's Eqs. 20-21), given the newly computed displacement and the previous displacement, velocity and acceleration. The source uses the standard parameter choice stated in the paper. Recover both update formulas exactly; they are the Newmark relations written with the new displacement as the known quantity, not a simple finite difference of the displacement.

```python
import numpy as np


def newmark_update(u_new, u_old, v_old, a_old, dt, beta, gamma):
    """u_new, u_old, v_old, a_old: displacement, previous displacement, velocity and
    acceleration; dt: time step; beta, gamma: Newmark parameters. Returns a float64 array
    (2, N) with the updated velocity and acceleration (paper Eqs. 20-21). Raises
    ValueError if dt <= 0 or beta <= 0."""
    return None
```

### Step 7

absorbing_matrix

Goal
----
Return the coefficient matrix of the source's absorbing boundary condition, which relates the boundary traction to the boundary velocity so that outgoing waves leave the domain without reflecting (the paper's Eq. 19). The matrix is built from the density and the two Lame parameters and is NOT isotropic: it applies one wave impedance along the boundary normal and a different one in the directions tangential to it, because pressure and shear waves travel at different speeds. Recover the exact matrix, including which impedance goes with which direction, from the paper.

```python
import numpy as np


def absorbing_matrix(rho, lam, mu, n):
    """rho: density; lam, mu: first and second Lame parameters; n: (d,) outward unit
    normal of the boundary. Returns a float64 array (d, d) with the source's absorbing
    boundary coefficient matrix (paper Eq. 19). Raises ValueError if n is not a unit
    vector."""
    return None
```

### Step 8

fracture_audit

Goal
----
Run the full fracture deformation audit over the source's four model variants, in the order S-Lin-Lin, S-Lin-BB, C-Coul-Lin, C-Coul-BB, on the declared fixed fracture configuration with all prescribed jumps scaled by load_scale. For each variant compute the normal contact traction from the prescribed normal displacement jump under that variant's normal relation and model family, then the tangential traction: for the two spring variants the tangential relation of Eq. 16 applies, whereas for the two contact variants the tangential traction follows from the source's radial-return map starting from zero traction with the declared numerical parameter. Then classify the fracture points and advance the tangential jump by one step of the source's time integrator, taking the new tangential jump to be the prescribed tangential jump plus the tangential traction divided by the tangential stiffness. Also advance the normal jump by the same integrator, taking the new normal jump to be the prescribed normal jump plus the normal traction divided by the normal stiffness, and build the absorbing boundary coefficient matrix for the declared material and the outward normal (1, 0); at each fracture point apply that matrix to the two-component velocity (updated normal velocity jump, updated tangential velocity jump) and sum the magnitudes of the resulting boundary tractions. Report the row [sum of normal tractions, sum of tangential traction magnitudes, sum of fracture openings, largest slip tendency, number of open points, number of sticking points, number of sliding points, norm of the updated tangential velocity jump, summed absorbing-boundary traction magnitude, sum of the elastic normal closure g_n(q_n)]. Assemble by calling the earlier sub-problem functions; every earlier step must be reached.

```python
import numpy as np


def fracture_audit(load_scale):
    """load_scale: positive multiplier on the declared prescribed displacement jumps.
    Builds the declared fracture configuration and returns a float64 array (4, 10) whose
    rows are the audit of the four model variants S-Lin-Lin, S-Lin-BB, C-Coul-Lin and
    C-Coul-BB, with columns [sum q_n, sum |q_tau|, sum opening, max slip tendency,
    n_open, n_sticking, n_sliding, |v_tau|, summed absorbing-boundary traction,
    sum of elastic normal closure]. Assembled by calling the earlier sub-problem functions.
    Raises ValueError on a non-positive or nonfinite load_scale."""
    return np.zeros((4, 10))
```
