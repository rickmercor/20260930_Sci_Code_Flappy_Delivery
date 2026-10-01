# Log-stiffness sensitivity of discrete contact impulse

## Background

# Scientific background

Contact forces are usually imposed through surface pairing, closest-point projection, or complementarity constraints. Those constructions become cumbersome when two bodies have unrelated meshes, sharp features, large deformation, or changing contact topology. The cited work instead develops a geometry-driven contact formulation that is compatible with unrelated discretizations.

Reliable numerical contact response requires an estimator whose variation remains meaningful as the bodies move and their contact topology changes. The source supplies a construction that avoids surface correspondence and projection iterations while retaining a tractable mechanical response.

The formulation can be incorporated into either quasistatic or dynamic solvers. The reduced experiments in this task isolate the source's estimator and dynamic procedure from finite-element assembly, material constitutive laws, and mesh-generation cost.

This task extends that setting to parameter sensitivity of a discrete contact trajectory. Such sensitivities distinguish a force response evaluated at fixed geometry from the accumulated response when the geometry itself changes with the parameter. The extension is derived from the source dynamics; it is not presented as a separate result established in the source paper.

## Problem

Three frictionless planar contact experiments use the source's contact estimator to advance coupled translational and rotational degrees of freedom, and the required result is the log-stiffness sensitivity of their combined normalized contact impulse. In experiment $r$, body $j$ is the ellipse $g_j(x;q)=(x-c_j-\delta_{j2}\theta d)^TA_j(q)(x-c_j-\delta_{j2}\theta d)-1$, where $q=(\theta,\psi)$ and $A_j(q)=R(\phi_j+\delta_{j2}\psi)\operatorname{diag}(a_j^{-2},b_j^{-2})R(\phi_j+\delta_{j2}\psi)^T$, points with $g_j\leq0$ are inside, body 1 is fixed, and body 2 translates in the stated unit direction $d$ and rotates by $\psi$ about its translated center. Each rectangular sampling box contains the stated number of horizontal fibers followed by vertical fibers, placed at cell midpoints and directed from its lower-coordinate boundary to its upper-coordinate boundary; apply the source's sampling-density normalization to this deterministic two-family quadrature. All geometry is in metres, and no random seed is used. The generalized mass is $D_r=\operatorname{diag}(M_r,I_r)$, initially $\psi=0$, with inertias $I_r=(0.18,0.21,0.095)\ \mathrm{kg\,m^2}$ and initial angular velocities $(0.15,-0.12,0.08)\ \mathrm{rad\,s^{-1}}$ for the following three experiments:

| $r$ | box $(x_{\min},x_{\max},y_{\min},y_{\max})$ | $(n_h,n_v)$ | $c_1$; $(a_1,b_1)$; $\phi_1$ | $c_2$; $(a_2,b_2)$; $\phi_2$ | $d$ | $(\theta_0,v_0,M,k,\Delta t,N_t)$ |
|---:|---|---|---|---|---|---|
| 1 | $(-1.4,1.4,-1.1,1.1)$ | $(64,64)$ | $(-0.22,0)$; $(0.82,0.55)$; $0.20$ | $(0.38,0.07)$; $(0.70,0.46)$; $-0.32$ | $(1,0)$ | $(0,-0.35,1.7,480,0.002,50)$ |
| 2 | $(-1.6,1.6,-1.2,1.2)$ | $(48,48)$ | $(-0.30,-0.04)$; $(0.94,0.48)$; $-0.28$ | $(0.46,0.12)$; $(0.63,0.51)$; $0.41$ | $(0.9659258263,0.2588190451)$ | $(0,-0.29,2.1,620,0.0015,60)$ |
| 3 | $(-1.2,1.2,-1.2,1.2)$ | $(80,80)$ | $(-0.18,0)$; $(0.68,0.52)$; $0.15$ | $(0.69,0.03)$; $(0.44,0.36)$; $-0.20$ | $(1,0)$ | $(0,-0.22,1.3,700,0.001,80)$ |

For each experiment, replace $k_r$ by $e^\eta k_r$ and apply the source's estimator and dynamic procedure exactly with the same fixed fibers, using float64 arithmetic with root tolerance $10^{-12}$ and transversality threshold $10^{-10}$. The task extends the source by asking for the local derivative of this fully discrete map, with all initial data, sampling weights, time steps, and step counts held fixed; use the unique active endpoint at each force evaluation, without continuous-time event corrections. In the reasoning, identify the source-dependent choices, derive the root curvature and force Jacobian needed for the trajectory sensitivity, state how the discrete sensitivity evolves, and report the first and third initial overlap areas, each experiment's signed sensitivity contribution, and the second experiment's terminal generalized velocity derivative with respect to $\eta$. Compare the source's treatment with the baseline estimator it replaces and quantify the baseline's consequence for the audit. Return the single finite decimal $S=\left.dJ/d\eta\right|_{\eta=0}$, where $J(\eta)=\sum_{r=1}^3\|z_r(\eta)\|_2$, $z_r=(M_r(v_{T,r}-v_{0,r})/(1\ \mathrm{kg\,m\,s^{-1}}),I_r(\omega_{T,r}-\omega_{0,r})/(1\ \mathrm{kg\,m^2\,s^{-1}}))$; $S$ is signed and is not $J(0)$ or the partial stiffness derivative at a frozen trajectory.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words), while including the requested
source-dependent choices, scalar checks, and baseline comparison.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

01_evaluate_ellipse_field

Goal
----
Evaluate a parametrically translated and rotated implicit ellipse.



The state may contain several generalized coordinates. Each coordinate has

its own center-translation direction and angular rate, so the returned field

table contains the spatial gradient, complete state gradient, and the packed

upper triangle of the state Hessian.

```python
import numpy as np


def evaluate_ellipse_field(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
) -> np.ndarray:
    """Evaluate an implicit ellipse and its first- and second-order derivatives.

    Parameters
    ----------
    points : np.ndarray
        Finite array of shape ``(n_points, 2)`` in metres.
    center : np.ndarray
        Undeformed ellipse center with shape ``(2,)`` in metres.
    axes : np.ndarray
        Strictly positive semiaxes ``(a, b)`` with shape ``(2,)`` in metres.
    angle : float
        Counterclockwise reference rotation in radians.
    state : np.ndarray
        Finite generalized-coordinate vector with shape ``(p,)``.
    translation_directions : np.ndarray
        Shape ``(p, 2)``. Each row is either zero or unit length and maps one
        state coordinate to center translation.
    rotation_rates : np.ndarray
        Finite shape ``(p,)`` rates mapping the state to additional rotation.
        The deformed center and angle are ``center + state @ directions`` and
        ``angle + state @ rotation_rates``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_points, 3 + p + p*(p+1)//2)``. Columns
        contain ``g``, the two spatial-gradient entries, the ``p`` state-
        gradient entries, then the upper triangle of the symmetric state
        Hessian in NumPy ``triu_indices(p)`` order.

    Raises
    ------
    ValueError
        If an array has an incompatible shape, any input is non-finite, a
        semiaxis is non-positive, or a translation-direction row is neither
        zero nor unit length.
    """
    return result
```

### Step 2

02_compute_fiber_roots

Goal
----
Compute the two boundary parameters of an implicit ellipse on each fiber.



Apply the same multi-coordinate translation and rotation map used by the

field-evaluation step before intersecting the ellipse.

Roots are returned on the infinite supporting line; clipping to

`$0 <= h <= 1$` belongs to the contact-interval step.  Preserve distinct

crossings when the fiber coordinates are much larger than the ellipse:

forming the polynomial discriminant by subtracting nearly equal terms is

not sufficiently accurate.  A line with no real crossing receives two

``NaN`` values.

```python
import numpy as np


def compute_fiber_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    discriminant_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return ordered ellipse crossings for each fiber-supporting line.

    Parameters
    ----------
    starts, ends : np.ndarray
        Matching finite arrays of shape ``(n_fibers, 2)`` in metres. Every
        fiber must have positive length.
    center, axes : np.ndarray
        Shape ``(2,)`` reference ellipse parameters; axes are positive.
    angle : float
        Reference rotation in radians.
    state : np.ndarray
        Finite generalized-coordinate vector with shape ``(p,)``.
    translation_directions : np.ndarray
        Shape ``(p, 2)`` with zero or unit rows.
    rotation_rates : np.ndarray
        Finite shape ``(p,)`` angular rates. Motion is applied using the same
        center and angle maps as in ``evaluate_ellipse_field``.
    discriminant_tolerance : float, optional
        Positive absolute tolerance used to recognize a repeated root.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_fibers, 2)`` containing ascending roots;
        a nonintersecting line has ``[NaN, NaN]``. Root separation must be
        retained for finite inputs whose coordinate and ellipse scales differ
        by up to eight orders of magnitude.

    Raises
    ------
    ValueError
        If a fiber has zero length or ``discriminant_tolerance`` is not
        positive.
    """
    return result
```

### Step 3

03_intersect_inside_intervals

Goal
----
Intersect all body-interior intervals with the finite fiber.



The input stacks the ordered root pairs for two or more convex bodies.

Contact is their common intersection with ``[0, 1]``. Owner code 0 denotes

a fiber boundary and code `$j + 1$` denotes body index `$j$`.

```python
import numpy as np


def intersect_inside_intervals(
    root_sets: np.ndarray, tie_tolerance: float = 1e-12
) -> np.ndarray:
    """Return contact-interval bounds and active endpoint owners.

    Parameters
    ----------
    root_sets : np.ndarray
        Float array of shape ``(n_bodies, n_fibers, 2)`` for at least two
        bodies. Each row contains ascending roots or paired ``NaN`` values.
    tie_tolerance : float, optional
        Positive tolerance for detecting nondifferentiable owner ties.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_fibers, 4)`` with columns
        ``[h_lower, h_upper, lower_owner, upper_owner]``. Owner codes range
        from 0 through ``n_bodies``. A fiber with no positive-length common
        contact interval has four ``NaN`` values.

    Raises
    ------
    ValueError
        If fewer than two bodies are supplied, a root row is unpaired or
        descending, or an active endpoint has a nondifferentiable owner tie.
    """
    return result
```

### Step 4

04_estimate_overlap_area

Goal
----
Aggregate contact-interval lengths into a weighted overlap area.



If a fiber has parameter interval `$[h_lower, h_upper]$`, its contact length

is ``||end - start|| * (h_upper - h_lower)``.  Multiplication by a line weight

with units of length gives an area contribution.  Rows containing no contact

are represented by ``NaN`` interval bounds and contribute zero. Compute the

two-dimensional line norm without overflowing for finite, widely scaled

coordinates.

```python
import numpy as np


def estimate_overlap_area(
    interval_state: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Estimate contact area from weighted fiber-intersection lengths.

    Parameters
    ----------
    interval_state : np.ndarray
        Shape ``(n_fibers, 4)`` from the contact-interval step.
    starts, ends : np.ndarray
        Matching finite arrays of shape ``(n_fibers, 2)`` in metres.
    weights : np.ndarray
        Finite nonnegative line weights of shape ``(n_fibers,)`` in metres,
        with at least one positive value.

    Returns
    -------
    float
        Finite nonnegative overlap estimate in square metres. Finite products
        must be retained when a segment component is near ``1e200`` and its
        corresponding line weight is near ``1e-200``.

    Raises
    ------
    ValueError
        If any weight is negative, all weights are zero, or an active interval
        has a bound outside ``[0, 1]``.
    """
    return result
```

### Step 5

05_differentiate_boundary_roots

Goal
----
Differentiate moving fiber-boundary roots with respect to every state coordinate.



Translation and rotation may be coupled through a vector state. At each root,

apply implicit differentiation twice to obtain the full state gradient and

Hessian. The common transversality denominator vanishes at a tangent crossing,

where both derivatives are undefined and the input is rejected.

```python
import numpy as np


def differentiate_boundary_roots(
    starts: np.ndarray,
    ends: np.ndarray,
    roots: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    angle: float,
    state: np.ndarray,
    translation_directions: np.ndarray,
    rotation_rates: np.ndarray,
    transversality_tolerance: float = 1e-10,
) -> np.ndarray:
    """Return the state gradient and Hessian of every finite ellipse root.

    Parameters
    ----------
    starts, ends : np.ndarray
        Matching arrays of shape ``(n_fibers, 2)`` in metres.
    roots : np.ndarray
        Shape ``(n_fibers, 2)`` with finite ordered roots or paired ``NaN``.
    center, axes : np.ndarray
        Shape ``(2,)`` reference ellipse data.
    angle : float
        Reference rotation in radians.
    state : np.ndarray
        Generalized coordinates with shape ``(p,)``.
    translation_directions : np.ndarray
        Shape ``(p, 2)`` with zero or unit rows.
    rotation_rates : np.ndarray
        Finite angular rates with shape ``(p,)``.
    transversality_tolerance : float, optional
        Positive lower bound for the absolute implicit-root denominator.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_fibers, 2, p + 1, p)``. Slice
        ``result[:, :, 0, :]`` contains each root gradient and
        ``result[:, :, 1:, :]`` contains its symmetric Hessian. Noncrossing
        lines retain ``NaN`` in every derivative entry.

    Raises
    ------
    ValueError
        If a root row is unpaired or a finite root is non-transversal at the
        stated tolerance.
    """
    return result
```

### Step 6

06_differentiate_overlap_area

Goal
----
Differentiate the weighted overlap area for a multi-coordinate state.



The endpoint-owner columns select either a fixed fiber endpoint or one of the

stacked bodies. Each selected root carries a gradient and Hessian. Accumulate

entry and exit contributions with stable line norms and cancellation-resistant

summation in every derivative channel.

```python
import math
import numpy as np


def differentiate_overlap_area(
    interval_state: np.ndarray,
    root_sensitivities: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the generalized gradient and Hessian of the contact area.

    Parameters
    ----------
    interval_state : np.ndarray
        Shape ``(n_fibers, 4)`` with bounds and owner codes.
    root_sensitivities : np.ndarray
        Shape ``(n_bodies, n_fibers, 2, p + 1, p)``. Derivative row zero is
        the root gradient and rows 1 through ``p`` are its Hessian. Owner code
        ``j + 1`` selects body index ``j``.
    starts, ends : np.ndarray
        Matching fiber endpoint arrays of shape ``(n_fibers, 2)``.
    weights : np.ndarray
        Nonnegative line weights of shape ``(n_fibers,)`` in metres.

    Returns
    -------
    np.ndarray
        Finite float64 array with shape ``(p + 1, p)``. Row zero is the overlap
        gradient and rows 1 through ``p`` are its symmetric Hessian. Every
        entry is summed accurately when large contributions nearly cancel.

    Raises
    ------
    ValueError
        If an active owner code is outside ``{0, ..., n_bodies}`` or an active
        endpoint has no finite derivative, or an active root Hessian is not
        symmetric to numerical precision.
    """
    return result
```

### Step 7

07_compute_contact_energy_force

Goal
----
Convert overlap measure and its derivative into contact energy and force.



For $w(V_c)=kV_c^m$, the generalized contact force is the negative state

gradient of this energy. Return the energy, force, and consistent force

Jacobian assembled from the overlap gradient and Hessian. The no-contact state

is defined to have zero energy, force, and Jacobian. Evaluate every power-law

coefficient without overflowing an avoidable intermediate.

```python
import numpy as np

def compute_contact_energy_force(
    overlap_area: float,
    overlap_sensitivities: np.ndarray,
    stiffness: float,
    exponent: float = 1.0,
) -> np.ndarray:
    """Return overlap energy, generalized force, and force Jacobian.

    Parameters
    ----------
    overlap_area : float
        Nonnegative contact area in square metres.
    overlap_sensitivities : np.ndarray
        Finite array with shape ``(p + 1, p)``. Row zero is the overlap
        gradient and rows 1 through ``p`` are its symmetric Hessian.
    stiffness : float
        Strictly positive energy coefficient in units compatible with the
        selected exponent.
    exponent : float, optional
        Finite exponent at least one.

    Returns
    -------
    np.ndarray
        Symmetric float64 array of shape ``(p + 1, p + 1)``. Entry ``[0, 0]``
        is the energy, row and column zero contain the generalized force, and
        the lower-right block is its symmetric Jacobian with respect to state.
        Finite mathematical results must not overflow merely because a raw
        power of ``overlap_area`` does so as an intermediate.

    Raises
    ------
    ValueError
        If any input is non-finite, the overlap Hessian is not symmetric,
        ``overlap_area`` is negative, ``stiffness`` is not positive, or
        ``exponent`` is below one.
    """
    return result
```

### Step 8

08_run_contact_impulse_audit

Goal
----
Differentiate the fully discrete contact impulse with respect to log stiffness.



For the three fixed experiments, replace every stiffness by $e^\eta k_r$.

The returned scalar is $S(\lambda)=dJ(\eta)/d\eta$ at

$\eta=\log\lambda$, where $J$ sums the norms of normalized final generalized momentum changes.

Differentiate the source's finite-step kick-drift-kick map, with initial

positions and velocities held fixed, and rebuild the active contact geometry

at each force evaluation. This is a sensitivity of the discrete trajectory,

not an integral of partial force derivatives at frozen positions.

The moving state is $q=(\theta,\psi)$: body 2 has center

$c_2+\theta d$ and angle $\phi_2+\psi$, and body 1 stays fixed.

The generalized mass is $\operatorname{diag}(M,I)$, initial $\psi=0$,

and initial angular velocities are $0.15,-0.12,0.08$ radians per second.

The corresponding inertias are $0.18,0.21,0.095$ kilograms square metres.

Normalize linear and angular momentum changes by one SI unit of their

respective dimensions before taking their Euclidean norm.

The sample fibers remain fixed. Differentiate locally through each unique

active endpoint; do not differentiate the integer owner labels or insert

continuous-time event corrections. At exact owner ties or tangencies the

underlying derivative contract rejects the state.



Fixed experiment data (SI units; angles in radians):



| $r$ | box $(x_{\min},x_{\max},y_{\min},y_{\max})$ | $(n_h,n_v)$ | $c_1$; $(a_1,b_1)$; $\phi_1$ | $c_2$; $(a_2,b_2)$; $\phi_2$ | $d$ | $(\theta_0,v_0,M,k,\Delta t,N_t)$ |

|---:|---|---|---|---|---|---|

| 1 | $(-1.4,1.4,-1.1,1.1)$ | $(64,64)$ | $(-0.22,0)$; $(0.82,0.55)$; $0.20$ | $(0.38,0.07)$; $(0.70,0.46)$; $-0.32$ | $(1,0)$ | $(0,-0.35,1.7,480,0.002,50)$ |

| 2 | $(-1.6,1.6,-1.2,1.2)$ | $(48,48)$ | $(-0.30,-0.04)$; $(0.94,0.48)$; $-0.28$ | $(0.46,0.12)$; $(0.63,0.51)$; $0.41$ | $(0.9659258263,0.2588190451)$ | $(0,-0.29,2.1,620,0.0015,60)$ |

| 3 | $(-1.2,1.2,-1.2,1.2)$ | $(80,80)$ | $(-0.18,0)$; $(0.68,0.52)$; $0.15$ | $(0.69,0.03)$; $(0.44,0.36)$; $-0.20$ | $(1,0)$ | $(0,-0.22,1.3,700,0.001,80)$ |



Fibers are midpoint horizontal lines followed by midpoint vertical lines.

The two families each carry half of the sampling-density weight; use the

stated root and transversality tolerances $10^{-12}$ and $10^{-10}$.

```python
import numpy as np

def run_contact_impulse_audit(force_scale: float = 1.0) -> float:
    r"""Return the log-stiffness derivative of the discrete impulse audit.

    Parameters
    ----------
    force_scale : float, optional
        Positive finite multiplier applied to every stated contact stiffness.
        The task instance uses $\lambda=1$. The derivative is evaluated at
        $\eta=\log\lambda$, not at zero for every input scale.

    Returns
    -------
    float
        Signed finite dimensionless derivative $S(\lambda)$. Momentum is
        components normalized by $1\,\mathrm{kg\,m\,s^{-1}}$ for translation
        and $1\,\mathrm{kg\,m^2\,s^{-1}}$ for rotation. Use the Euclidean
        norm of these two dimensionless components in each experiment. Differentiate both kicks
        and the drift, including the force dependence on the evolving state.
        Use the current matrix contract of compute_contact_energy_force;
        its state Jacobian is required. Numerical differentiation is allowed
        if it resolves this same local derivative to the test tolerance.

    Raises
    ------
    ValueError
        If ``force_scale`` is non-positive or non-finite, or a terminal
        generalized momentum change is exactly zero (undefined norm derivative).
    """
    return result
```
