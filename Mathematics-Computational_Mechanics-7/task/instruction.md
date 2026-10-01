# Mathematics-Computational_Mechanics-7

## Background

Interactive deformation requires local constitutive updates that preserve residual deformation without an expensive nonlinear solve at each material point. Smoothing an instantaneous yield activation alone does not make a response irreversible: a separate stored history is needed to prevent the internal state from decreasing during unloading. A residual tensor also carries directional information that a scalar history cannot retain.

For proportional or nearly proportional isotropic loading, a radial update can provide a compact graphics-oriented model with distinct active and frozen branches. Its consistent material tangent must differentiate the active history and attenuation maps while freezing stored variables on unloading, so off-axis perturbations test more than the scalar loading curve. A finite-element Jacobian then also requires differentiating polar kinematics and every factor of the finite-geometry stress transformation before shared-node assembly. Attenuation of inelastic work and loss of elastic unloading stiffness are different modeling choices, and converting a corotational response into nodal forces requires an explicit stress-measure convention. These distinctions matter when a finite deformation path revisits an earlier strain magnitude. This setting does not establish a general non-proportional plastic flow law, a mesh-objective fracture model, or thermodynamic admissibility for arbitrary finite-strain histories.

Path sensitivity measures how a small change in imposed nodal motion alters the full loading response. A plastic tensor can remain constant across unloading frames while still changing with the earlier loading path. Consequently, the tangent at fixed previous state is only one part of the derivative of a complete history-dependent replay. Differentiating the residual-state transport and the finite-geometry force map exposes that distinction without introducing an equilibrium solver or a new constitutive law.

## Problem

Evaluate the explicit isotropic elastoplastic graphics response based on normalized smooth activation and maximum equivalent plastic history, using its August 2026 finite-geometry corotated-Cauchy convention with the optional unloading-stiffness-loss variable fixed at $d=0$.
Use the following reference coordinates $X$ in mm and zero-based tetrahedral connectivity $T$, with material rows corresponding to the rows of $T$ and columns $(E,\nu,\sigma_y,H,\beta,C)$, where $E,\sigma_y,H$ are in MPa and the remaining entries are dimensionless:

$$
X=\begin{pmatrix}0&0&0\\1&0.1&0\\0.2&1.2&0.1\\0.1&0.2&1.1\\1.1&1&1.3\end{pmatrix},\qquad
T=\begin{pmatrix}0&1&2&3\\1&2&3&4\end{pmatrix},\qquad
M=\begin{pmatrix}20&0.3&2&0.5&12&2.2\\30&0.25&1.2&2.4&8&4.4\end{pmatrix}.
$$

For frames $n=0,\ldots,8$, prescribe $x_{n,i}=F_nX_i$ using column vectors, $B=\operatorname{diag}(1,-1/2,-1/2)$, the following amplitudes and angles in radians, and right-handed active rotations:

$$
a=(0,0.03,0.09,0.16,0.07,0.13,0.18,0.04,0.12),\quad
\theta=(0,0.1,0.2,0.3,0.2,0.1,-0.1,0.15,0.25),\quad \phi_n=0.01n,
$$

$$
R_z(t)=\begin{pmatrix}\cos t&-\sin t&0\\\sin t&\cos t&0\\0&0&1\end{pmatrix},\qquad
R_y(t)=\begin{pmatrix}\cos t&0&\sin t\\0&1&0\\-\sin t&0&\cos t\end{pmatrix},
$$

$$
F_n=R_z(\theta_n)\left\{I+a_n\left[R_y(\phi_n)BR_y(\phi_n)^T+0.1I\right]\right\}.
$$

All nodal positions are imposed, so evaluate exactly these configurations in order without an equilibrium solve, intermediate time steps, inertia, or contact; initialize each element's scalar maximum history and residual deviatoric plastic tensor to zero, retain both through unloading and sub-maximum reloading, and use the inactive branch when the instantaneous candidate equals the previous maximum.
Use the activation whose value and slope vanish at zero equivalent strain, with $q=\sqrt{2(e:e)/3}$ for the deviator $e$ of the polar-stretch strain, and retain its exact first and second derivatives together with both derivatives of the attenuated inelastic-work term.
Before force assembly, form the branch-consistent symmetric-strain tangent in Mandel order $(xx,yy,zz,yz,xz,xy)$ with $\sqrt{2}$ shear scaling, holding the previous state fixed and selecting the inactive derivative at equality; also differentiate the polar stretch and every factor in the finite-geometry stress map to assemble the consistent node-major Jacobian of resisting force.
Interpret the resulting corotational stress as Cauchy stress and assemble linear-tetrahedron internal nodal forces with reference volumes and the resisting-force sign, taking the initial force to be exactly zero.
For the non-affine nodal perturbation below, set $x_{n,i}(\lambda)=F_nX_i+\lambda b_nZ_i$ with dimensionless $\lambda$, fixed reference geometry and materials, and replay all internal states from zero at each $\lambda$; return the derivative at $\lambda=0$ of the discrete functional $W(\lambda)$ in N mm, explaining in the short reasoning why frozen state can have nonzero path sensitivity and reporting the two final unperturbed maximum histories:

$$
W(\lambda)=-\sum_{n=1}^{8}\sum_{i=0}^{4}\frac{f_{n,i}(\lambda)+f_{n-1,i}(\lambda)}{2}\cdot(x_{n,i}(\lambda)-x_{n-1,i}(\lambda)).
$$

$$
Z=\begin{pmatrix}0.03&-0.02&0.01\\-0.01&0.04&0.02\\0.02&0.01&-0.03\\-0.04&0.02&0.01\\0.01&-0.03&0.04\end{pmatrix}\ \mathrm{mm},\qquad
b=(0,0.2,-0.3,0.4,-0.1,0.15,-0.2,0.3,-0.4),\qquad
\mathcal S=\left.\frac{dW(\lambda)}{d\lambda}\right|_{\lambda=0}.
$$

Use float64 arithmetic and determine $\mathcal S$ within absolute tolerance $10^{-9}$ N mm plus relative tolerance $10^{-8}$; on any exact branch tie, the requested linearization keeps the inactive branch selected rather than asserting a two-sided derivative.

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
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
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

01_reference_geometry

Goal
----
Compute the reference volume and barycentric shape-function gradients for every tetrahedral element.

```python
import numpy as np


def reference_geometry(
    reference: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    r"""Construct reference volumes and barycentric gradients for a tetrahedral patch.

    Reference coordinates must be finite with shape (n_nodes, 3), n_nodes >= 4.
    Connectivity must be a nonempty integer (n_elements, 4) array with distinct, in-
    range vertices per cell. A cell is rejected if its absolute edge determinant is at
    most 1e-12 times the product of its edge lengths. Invalid inputs raise ValueError.

    Parameters
    ----------
    reference : np.ndarray
        Reference coordinates, shape (n_nodes, 3), in mm.
    cells : np.ndarray
        Zero-based cell indices, shape (n_elements, 4), in local vertex order.

    Raises
    ------
    ValueError
        If coordinates are nonfinite or have the wrong shape; connectivity is empty,
        nonintegral, repeated, or out of range; or any reference tetrahedron is
        degenerate at the stated relative tolerance.

    Returns
    -------
    tuple
        (volumes, gradients): float arrays of shapes (n_elements,) and (n_elements, 4,
        3), in mm cubed and inverse mm.
    """
    return result
```

### Step 2

02_corotational_kinematics

Goal
----
Recover polar kinematics, equivalent strain, and the exact Mandel-strain Fréchet derivative for each element.

```python
import numpy as np


def corotational_kinematics(
    reference: np.ndarray,
    deformed: np.ndarray,
    cells: np.ndarray,
) -> tuple:
    r"""Recover the corotational strain and equivalent deviatoric strain from nodal positions.

    Reference coordinates must be finite with shape (n_nodes, 3), n_nodes >= 4.
    Connectivity must be a nonempty integer (n_elements, 4) array with distinct, in-
    range vertices per cell. A cell is rejected if its absolute edge determinant is at
    most 1e-12 times the product of its edge lengths. Deformed coordinates must match
    reference shape and be finite; every deformation determinant must be positive.
    Equivalent strain below 1e-12 is treated as exactly zero, together with its
    deviator, so that a rigid motion produces an exact branch tie in later steps.
    Invalid inputs raise ValueError.

    Parameters
    ----------
    reference : np.ndarray
        Reference coordinates, shape (n_nodes, 3), in mm.
    deformed : np.ndarray
        Deformed coordinates, same shape and units as reference.
    cells : np.ndarray
        Zero-based connectivity, shape (n_elements, 4).

    Raises
    ------
    ValueError
        If either coordinate array is nonfinite or malformed; connectivity is empty,
        nonintegral, repeated, or out of range; a reference cell is degenerate; or a
        deformation reverses orientation or has zero determinant.

    Returns
    -------
    tuple
        (F, R, S, strain, deviator, q, strain_jacobian): five dimensionless arrays
        of shape (n_elements, 3, 3), one of shape (n_elements,), and the Fréchet
        derivative of Mandel strain with respect to row-major F of shape
        (n_elements, 6, 9).
    """
    return result
```

### Step 3

03_normalized_activation

Goal
----
Evaluate the normalized smooth activation candidate and its first two equivalent-strain derivatives for every element.

```python
import numpy as np


def normalized_activation(
    equivalent_strain: np.ndarray,
    shear_modulus: np.ndarray,
    yield_stress: np.ndarray,
    sharpness: np.ndarray,
) -> tuple:
    r"""Evaluate a normalized smooth candidate and its first two derivatives.

    Inputs must be finite, nonempty one-dimensional arrays of identical shape with q >=
    0 and all material parameters positive. Invalid inputs raise ValueError.

    Parameters
    ----------
    equivalent_strain : np.ndarray
        Nonnegative dimensionless q, shape (n_elements,).
    shear_modulus : np.ndarray
        Positive shear modulus in MPa, same shape.
    yield_stress : np.ndarray
        Positive yield scale in MPa, same shape; zero yield is outside the domain.
    sharpness : np.ndarray
        Positive dimensionless transition sharpness, same shape.

    Raises
    ------
    ValueError
        If an input is nonfinite; arrays are empty, not one-dimensional, or have
        different shapes; equivalent strain is negative; a material parameter is
        nonpositive; or the computed activation exceeds finite arithmetic range.

    Returns
    -------
    tuple
        (candidate, derivative, curvature): dimensionless float arrays of shape
        (n_elements,), where the derivatives are with respect to equivalent strain.
    """
    return result
```

### Step 4

04_irreversible_state

Goal
----
Advance the maximum scalar history and residual deviatoric plastic tensor using the active or frozen branch.

```python
import numpy as np


def irreversible_state(
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    candidate: np.ndarray,
    derivative: np.ndarray,
    previous_history: np.ndarray,
    previous_plastic: np.ndarray,
) -> tuple:
    r"""Advance scalar maximum history and the residual deviatoric tensor.

    All inputs must be finite with matching element counts. Tensors must be symmetric
    and traceless within absolute tolerance 1e-10. Scalar strains/history must be
    nonnegative, derivatives in [0, 1], and active cells must have q > 0. Tensor norms
    must agree with q and previous_history within atol=1e-10, rtol=1e-8. Invalid inputs
    raise ValueError.

    Parameters
    ----------
    deviator : np.ndarray
        Symmetric traceless e, shape (n_elements, 3, 3).
    equivalent_strain : np.ndarray
        q >= 0, shape (n_elements,).
    candidate : np.ndarray
        Nonnegative instantaneous candidate, shape (n_elements,).
    derivative : np.ndarray
        Candidate derivative in [0, 1], shape (n_elements,).
    previous_history : np.ndarray
        Nonnegative stored scalar history, shape (n_elements,).
    previous_plastic : np.ndarray
        Symmetric traceless residual tensor, shape (n_elements, 3, 3).

    Raises
    ------
    ValueError
        If inputs are nonfinite or have inconsistent shapes; scalar strain, candidate,
        or history is negative; a derivative lies outside [0, 1]; a tensor is not
        symmetric and traceless; a tensor norm disagrees with its scalar measure; or
        an active state has zero equivalent strain.

    Returns
    -------
    tuple
        (history, plastic, branch_derivative, active): dimensionless arrays of shapes
        (n_elements,), (n_elements, 3, 3), (n_elements,), (n_elements,); active contains
        float 0.0 or 1.0.
    """
    return result
```

### Step 5

05_inelastic_work

Goal
----
Compute the inelastic work, attenuation factor, and first two derivatives of their product with respect to history.

```python
import numpy as np


def inelastic_work(
    history: np.ndarray,
    yield_stress: np.ndarray,
    hardening: np.ndarray,
    attenuation_rate: np.ndarray,
) -> tuple:
    r"""Differentiate the attenuated inelastic-work contribution twice with respect to history.

    Inputs must be finite, nonempty matched vectors with h >= 0, yield_stress > 0, H >=
    0, C >= 0. Invalid inputs raise ValueError.

    Parameters
    ----------
    history : np.ndarray
        Nonnegative dimensionless h, shape (n_elements,).
    yield_stress : np.ndarray
        Positive yield scale in MPa, same shape.
    hardening : np.ndarray
        Nonnegative hardening scale H in MPa, same shape.
    attenuation_rate : np.ndarray
        Nonnegative dimensionless C, same shape.

    Raises
    ------
    ValueError
        If inputs are nonfinite; arrays are empty, not one-dimensional, or have
        different shapes; history, hardening, or attenuation rate is negative; yield
        stress is nonpositive; or a computed result is nonfinite.

    Returns
    -------
    tuple
        (work, survival, derivative, second_derivative): arrays of shape
        (n_elements,), respectively MPa, dimensionless, MPa, and MPa.
    """
    return result
```

### Step 6

06_branch_stress

Goal
----
Evaluate the branchwise corotational Cauchy stress and its consistent Mandel tangent.

```python
import numpy as np


def branch_stress(
    strain: np.ndarray,
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    plastic: np.ndarray,
    history: np.ndarray,
    branch_derivative: np.ndarray,
    active: np.ndarray,
    bulk_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    attenuated_work_derivative: np.ndarray,
    activation_curvature: np.ndarray,
    attenuated_work_second_derivative: np.ndarray,
) -> tuple:
    r"""Evaluate branchwise stress and the consistent symmetric-strain tangent.

    All inputs must be finite with matching shapes. Tensors must be symmetric within
    absolute tolerance 1e-10; q,h >= 0; K,mu > 0; g_p in [0,1]; flags exactly 0 or 1.
    Inactive g_p and activation curvature must be zero, while active q must be
    positive. Deviator and plastic tensors must be traceless and their equivalent
    norms must agree with q and h within atol=1e-10 and rtol=1e-8. The tangent uses
    Mandel order (xx, yy, zz, yz, xz, xy), with square-root-of-two scaling on shear
    entries, and differentiates the selected branch while holding the previous state
    fixed. Invalid inputs raise ValueError.

    Parameters
    ----------
    strain : np.ndarray
        Symmetric polar-stretch strain, shape (n_elements, 3, 3).
    deviator : np.ndarray
        Symmetric deviator of strain, same shape.
    equivalent_strain : np.ndarray
        Nonnegative q, shape (n_elements,).
    plastic : np.ndarray
        Updated symmetric residual tensor, shape (n_elements, 3, 3).
    history : np.ndarray
        Updated nonnegative h, shape (n_elements,).
    branch_derivative : np.ndarray
        g_p in [0, 1], zero on inactive cells, shape (n_elements,).
    active : np.ndarray
        Numerical branch flags 0 or 1, shape (n_elements,).
    bulk_modulus : np.ndarray
        Positive K in MPa, shape (n_elements,).
    shear_modulus : np.ndarray
        Positive mu in MPa, shape (n_elements,).
    attenuated_work_derivative : np.ndarray
        G in MPa, shape (n_elements,); may be negative.
    activation_curvature : np.ndarray
        Second candidate derivative with respect to q, shape (n_elements,); it is
        supplied as zero on inactive cells.
    attenuated_work_second_derivative : np.ndarray
        Derivative dG/dh in MPa, shape (n_elements,); may be negative.

    Raises
    ------
    ValueError
        If inputs are nonfinite or have inconsistent shapes; a tensor is nonsymmetric
        or nondeviatoric where required; scalar and tensor equivalent measures
        disagree; strain or history is negative; a modulus is nonpositive; a branch
        flag is not 0 or 1; an inactive derivative or curvature is nonzero; or an
        active state has zero equivalent strain.

    Returns
    -------
    tuple
        (stress, tangent): float arrays of shapes (n_elements, 3, 3) and
        (n_elements, 6, 6), in MPa. The tangent maps Mandel strain increments to
        Mandel stress increments.
    """
    return result
```

### Step 7

07_reference_forces

Goal
----
Assemble resisting forces and their consistent node-major Jacobian under finite deformation.

```python
import numpy as np


def reference_forces(
    deformation: np.ndarray,
    rotation: np.ndarray,
    stretch: np.ndarray,
    stress: np.ndarray,
    material_tangent: np.ndarray,
    strain_jacobian: np.ndarray,
    volumes: np.ndarray,
    gradients: np.ndarray,
    cells: np.ndarray,
    node_count: int,
) -> tuple:
    r"""Map stress to nodal forces and consistently linearize with respect to positions.

    Inputs must be finite with the stated shapes and positive volumes and det(F).
    Connectivity must have distinct, in-range integer vertices. R must be proper
    orthogonal, S positive definite and symmetric, stress symmetric, F=RS, and local
    gradients sum to zero, all tensor consistency checks within atol=1e-10,
    rtol=1e-8. The material tangent is symmetric in Mandel order
    (xx, yy, zz, yz, xz, xy), and strain_jacobian maps row-major F increments to
    Mandel strain increments. Invalid inputs raise ValueError.

    Parameters
    ----------
    deformation : np.ndarray
        F, shape (n_elements, 3, 3), dimensionless.
    rotation : np.ndarray
        Proper R from polar decomposition, same shape.
    stretch : np.ndarray
        Symmetric positive-definite S, same shape.
    stress : np.ndarray
        Symmetric corotational Cauchy stress, same shape, MPa.
    material_tangent : np.ndarray
        Consistent corotational Mandel tangent, shape (n_elements, 6, 6), MPa.
    strain_jacobian : np.ndarray
        Mandel-strain derivative with respect to row-major F, shape
        (n_elements, 6, 9).
    volumes : np.ndarray
        Positive reference volumes, shape (n_elements,), mm cubed.
    gradients : np.ndarray
        Reference gradients, shape (n_elements, 4, 3), inverse mm.
    cells : np.ndarray
        Integer zero-based connectivity, shape (n_elements, 4).
    node_count : int
        Number of global nodes, at least four.

    Raises
    ------
    ValueError
        If inputs are nonfinite or malformed; volumes are nonpositive; connectivity is
        nonintegral, repeated, or out of range; node_count is invalid; F reverses
        orientation; R is not proper orthogonal; S is not symmetric positive definite;
        stress is nonsymmetric; F differs from RS; or local gradients do not sum to
        zero within the stated tolerances.

    Returns
    -------
    tuple
        (forces, stiffness): arrays of shapes (node_count, 3) and
        (3*node_count, 3*node_count), in N and N/mm. Stiffness is the Jacobian of
        resisting force with node-major displacement degrees of freedom.
    """
    return result
```

### Step 8

08_prescribed_path_work

Goal
----
Differentiate discrete path work while transporting the residual-state sensitivity through loading and unloading.

```python
import numpy as np


def prescribed_path_work(
    reference: np.ndarray,
    cells: np.ndarray,
    materials: np.ndarray,
    amplitudes: np.ndarray,
    rotations: np.ndarray,
    direction_angles: np.ndarray,
    perturbations: np.ndarray,
) -> float:
    r"""Return the directional sensitivity of discrete resisting-force work along a nodal path.

    Reference coordinates must be finite with shape (n_nodes, 3), n_nodes >= 4.
    Connectivity must be a nonempty integer (n_elements, 4) array with distinct, in-
    range vertices per cell. A cell is rejected if its absolute edge determinant is at
    most 1e-12 times the product of its edge lengths. Materials must satisfy E > 0, -1 <
    nu < 0.5, sigma_y > 0, H >= 0, beta > 0, C >= 0. All arrays are finite. Frame
    vectors have matching nonempty shapes, at least two frames, and initial amplitude
    and angles exactly zero. Each amplitude must satisfy 1 + 1.1*a > 0 and 1 - 0.4*a >
    0. No random seed is used. Invalid inputs raise ValueError.

    At each frame use x(lambda) = x(0) + lambda * perturbations, with lambda
    dimensionless and reference geometry and materials fixed. Differentiate the
    complete replay, including the state carried from earlier frames. At a branch
    tie return the selected inactive-branch linearization, without differentiating
    the activity flag; this is not asserted to be a two-sided derivative at a tie.
    Initial force, state, and their rates are exactly zero. Perturbations must have
    shape (n_frames, n_nodes, 3), be finite, and vanish exactly at frame zero.

    Parameters
    ----------
    reference : np.ndarray
        Reference points, shape (n_nodes, 3), in mm.
    cells : np.ndarray
        Zero-based connectivity, shape (n_elements, 4).
    materials : np.ndarray
        Rows (E, nu, sigma_y, H, beta, C), shape (n_elements, 6); E, sigma_y, H in MPa.
    amplitudes : np.ndarray
        Dimensionless amplitude vector, shape (n_frames,), in replay order.
    rotations : np.ndarray
        Right-handed z-axis angles theta in radians, same shape.
    direction_angles : np.ndarray
        Right-handed y-axis material-direction angles phi in radians, same shape.

    perturbations : np.ndarray
        Nodal path direction in mm per dimensionless lambda, shape
        (n_frames, n_nodes, 3); first frame must be exactly zero.

    Raises
    ------
    ValueError
        If mesh data are nonfinite, malformed, disconnected from valid indices, or
        degenerate; material rows have the wrong shape or violate E > 0, -1 < nu <
        0.5, sigma_y > 0, H >= 0, beta > 0, or C >= 0; frame arrays are nonfinite,
        mismatched, or shorter than two frames; the initial amplitude or angles are
        nonzero; a prescribed stretch is not positive definite; or the resulting work
        is nonfinite; or perturbations are nonfinite, incorrectly shaped, or nonzero
        at the initial frame.

    Returns
    -------
    float
        One finite Python float: selected-branch work sensitivity dW/dlambda in N mm.
    """
    return result
```
