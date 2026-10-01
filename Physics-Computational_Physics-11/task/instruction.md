# Physics-Computational_Physics-11

## Problem

Consider a closed planar curve evolving by unit-mobility surface diffusion in the finite model below, and determine its percentage length loss over one time slab of duration 0.001. Its twelve initial nodes are X_j = r_j(cos(theta_j), sin(theta_j)), where theta_j = j*pi/6 and r_j = 1 + 0.16*cos(2*theta_j) + 0.07*sin(3*theta_j) + 0.04*cos(theta_j), for j = 0,...,11; six periodic quadratic Lagrange elements connect nodes (2e, 2e+1, (2e+2) mod 12), with local nodal coordinates (-1,0,1), for e = 0,...,5.

All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Select the smooth solution branch continued from the zero-duration limit through regular curves and nonsingular discrete equations. For curved-element length L(t), return Q = 100*[L(0)-L(0.001)]/L(0) to an absolute tolerance of 0.0000001 percentage points. Justify the result with the endpoint curved lengths, enclosed-area check, and the selected velocity's integrated normal H-minus-one norm squared and surface H1 deformation seminorm squared.

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

01_normal_diffusion_cost

Goal
----
Measure diffusion dissipation for a prescribed weak normal motion.

```python
def normal_diffusion_cost(history: "np.ndarray", velocity: "np.ndarray", dt: float) -> float:
    """Return the squared normal H-minus-one norm in the stated curve model.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2), with coefficients of (1,tau,tau**2).
    velocity : np.ndarray
        Shape (2,12,2), with coefficients of (1,2*tau-1).
    dt : float
        Positive physical slab length, where tau=t/dt.

    Returns
    -------
    float
        A finite, nonnegative squared normal H-minus-one norm.

    Raises
    ------
    ValueError
        If shapes or values are invalid, dt is not positive, the history
        is outside the stated domain, or the weak normal action is incompatible.

    Notes
    -----
    Elements e connect nodes (2e,2e+1,(2e+2)%12) at local coordinates
    (-1,0,1), using quadratic interpolation. Include the physical time
    measure, and use outward normals for counterclockwise node ordering.
    Each element tangent must have positive projection onto its initial
    endpoint chord direction throughout the entire element and time slab.

    The normal norm is dual to the square root of the physical
    space-time integral of squared scalar surface gradient on the
    whole degree-one temporal, continuous quadratic spatial scalar
    test space. Require the velocity's weak normal action to annihilate
    every test field constant in space.
    The supplied velocity need not be the history's derivative or a
    flow solution.
    """
    return
```

### Step 2

02_minimum_deformation_lift

Goal
----
Choose a vector representative for prescribed weak normal motion.

```python
def minimum_deformation_lift(history: "np.ndarray", normal_action: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the model's minimum-H1 representative of a normal action.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2) in temporal powers (1,tau,tau**2).
    normal_action : np.ndarray
        Shape (2,12), containing covector entries on the scalar nodal
        basis times (1,2*tau-1), including physical time and curve measures.
    dt : float
        Positive physical slab length, where tau=t/dt.

    Returns
    -------
    np.ndarray
        Vector coefficients of shape (2,12,2) in (1,2*tau-1).

    Raises
    ------
    ValueError
        If data are invalid, dt is not positive, the history is outside
        the stated domain, or the constrained minimizer does not exist uniquely.

    Notes
    -----
    Elements connect (2e,2e+1,(2e+2)%12) at local (-1,0,1) with
    quadratic interpolation. Use outward normals for counterclockwise
    node ordering. Each element tangent must have positive projection
    onto its initial endpoint chord direction over the entire element
    and slab.

    The returned coefficients minimize the physical space-time integral of the squared
    surface gradient of the full vector field while preserving the
    supplied weak normal action.
    Hold the geometry fixed. Area-changing actions are allowed here;
    do not impose the diffusion-compatible restriction.
    """
    return
```

### Step 3

03_normal_length_force

Goal
----
Express geometric length work in weak normal coordinates.

```python
def normal_length_force(history: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the scalar normal representation of the model's length work.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2) in powers (1,tau,tau**2), where tau=t/dt.
    dt : float
        Positive physical slab length.

    Returns
    -------
    np.ndarray
        Scalar coefficients of shape (2,12) in (1,2*tau-1).

    Raises
    ------
    ValueError
        If data are invalid, dt is not positive, the history is outside
        the stated domain, or the force representation is not unique.

    Notes
    -----
    Use six periodic quadratic elements with connectivity
    (2e,2e+1,(2e+2)%12) and local nodes (-1,0,1), and outward normals
    for counterclockwise node ordering. Each element tangent must have
    positive projection onto its initial endpoint chord direction over
    the entire element and slab.

    The field represents length work on the full weak-normal action
    space through its minimum-deformation representatives, including
    area-changing actions, in the stated exact-dual model.
    Retain the spatially constant components. The sign is increasing
    length work. Geometry is held fixed while this force is evaluated.
    """
    return
```

### Step 4

04_fixed_geometry_response

Goal
----
Find a dynamical response while keeping the prescribed geometry fixed.

```python
def fixed_geometry_response(history: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the velocity selected by the fixed-geometry curve model.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2) in powers (1,tau,tau**2), where tau=t/dt.
    dt : float
        Positive physical slab length.

    Returns
    -------
    np.ndarray
        Vector coefficients of shape (2,12,2) in (1,2*tau-1).

    Raises
    ------
    ValueError
        If inputs are invalid, dt is not positive, the history is outside
        the stated domain, or the response is undefined.

    Notes
    -----
    Use periodic quadratic elements (2e,2e+1,(2e+2)%12) with local
    nodes (-1,0,1), include the physical time measure, and use outward
    normals for counterclockwise node ordering. Each element tangent
    must have positive projection onto its initial endpoint chord
    direction over the entire element and slab.

    Return the velocity selected by the stated whole-slab exact-dual
    surface-diffusion model on this prescribed geometry path. Keep
    the prescribed geometry fixed while evaluating this response.
    The supplied history need not be a flow solution.
    """
    return
```

### Step 5

05_discrete_curve_trajectory

Goal
----
Resolve the regular polynomial trajectory of the finite curve model.

```python
def discrete_curve_trajectory(initial_nodes: "np.ndarray", dt: float) -> "np.ndarray":
    """Return the self-consistent curve path on the regular zero-slab branch.

    Parameters
    ----------
    initial_nodes : np.ndarray
        Finite array of shape (12,2) defining a closed counterclockwise curve.
    dt : float
        Finite positive slab length.

    Returns
    -------
    np.ndarray
        Coefficients of shape (3,12,2) in (1,tau,tau**2), where tau=t/dt.

    Raises
    ------
    ValueError
        If inputs are invalid, dt is not positive, the path is outside
        the stated domain, or the selected regular branch fails.

    Notes
    -----
    The curve uses quadratic elements (2e,2e+1,(2e+2)%12) at local
    nodes (-1,0,1), with outward normals. Restrict the trajectory domain
    to paths for which each element tangent has positive projection onto
    its initial endpoint chord direction over the entire element and slab.

    The path belongs to the model stated in the task: its physical
    derivative is its exact-dual surface-diffusion velocity. Select
    the regular continuation from the zero-slab limit, where geometry
    is the initial curve and velocity is its fixed-geometry response.
    Preserve node labels and connectivity.
    """
    return
```

### Step 6

06_relative_curve_length_loss

Goal
----
Measure endpoint lengths of curved finite-element histories.

```python
def relative_curve_length_loss(history: "np.ndarray") -> float:
    """Return the relative endpoint curve-length loss in percent.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2), with coefficients of (1,tau,tau**2) for tau in [0,1].

    Returns
    -------
    float
        One finite relative endpoint length loss in percent.

    Raises
    ------
    ValueError
        If the shape or coefficients are invalid or an endpoint geometry
        is degenerate.

    Notes
    -----
    Elements connect nodes (2e,2e+1,(2e+2)%12) at local (-1,0,1),
    using quadratic rather than polygonal curves. Use initial curved-element
    length as the denominator. A constant path returns zero; an expanding
    path can return a negative loss. Do not assume the history solves a flow.
    """
    return
```

### Step 7

07_curve_diffusion_length_loss

Goal
----
FINAL ORCHESTRATOR: the finite curve-diffusion length-loss observable.

```python
def curve_diffusion_length_loss(initial_nodes: "np.ndarray", dt: float) -> float:
    """Compute the specified finite curve model's length loss in percent.

    Parameters
    ----------
    initial_nodes : np.ndarray
        Shape (12,2), defining six periodic quadratic elements.
    dt : float
        Finite positive slab length.

    Returns
    -------
    float
        One finite endpoint curve-length loss in percent.

    Raises
    ------
    ValueError
        If inputs are invalid or a finite flow is not well defined on
        the selected branch.

    Notes
    -----
    Elements connect (2e,2e+1,(2e+2)%12) at local coordinates (-1,0,1).
    Use the stated degree-two geometry, degree-one test fields,
    exact-integral dual surface-diffusion model, and regular branch continued
    from the zero-slab limit. Measure endpoint curved lengths with initial
    length as denominator. Use outward normals for counterclockwise ordering
    and require every element tangent to have positive projection on its
    initial endpoint chord direction throughout the slab. Use the earlier
    subproblem functions for their stated operations.
    """
    return
```
