# Physics-Computational_Physics-10

## Background

Thermal metamaterials steer diffusive heat flow through spatially structured conductivities. Designs that match an external field using isotropic materials can be easier to fabricate, while anisotropic transformation media can operate across applied field directions. A physical design's local fields can connect these two descriptions.

Material variability creates a second question beyond nominal performance: how the device responds when an insulating region changes, its geometry deforms, and a compensating region is retuned. Local field directions and the most disruptive external drive can both move, so robustness involves the changing physical solution as well as the material values.

## Problem

I have a freeform thermal-invisibility device made only of isotropic materials, designed for one applied heat-flow direction. I want to convert it through the recently reported duality in which the design's own physical fields prescribe a coordinate transformation, realize the result as layers of isotropic materials, and assess how its gain changes when the insulating material and physical shape change together.

Work in the dimensionless square $-1\le x,y\le 1$ with background conductivity $\kappa_0=1$; a drive at angle $\varphi$ holds every boundary node at $T_0=x\cos\varphi+y\sin\varphi$. Steady fields are computed with linear (P1) finite elements on the uniform $121\times121$ node grid, each grid square split into two triangles by its diagonal from the lower-left to the upper-right corner, each triangle taking the conductivity of the region that contains its centroid. In polar coordinates $(r,\theta)$ about the origin the regions are the core, $r<0.20+0.04\cos2\theta$ (conductivity $\kappa_0$); the inner shell, out to $r=0.30+0.06\cos2\theta+0.015\cos4\theta$ (conductivity $0.1$); the outer shell, out to $r=0.45+0.09\cos2\theta+0.02\cos3\theta$ (isotropic conductivity $\kappa_{\mathrm{out}}$); and the background.

Write the inner-shell conductivity as $0.1\exp(\eta)$, with $\eta$ dimensionless; the core and background remain at $1$. At every $\eta$ near zero, retune the outer-shell conductivity $q(\eta)$ so that the mean of $x(T-x)$ under the $x$-drive vanishes over exterior nodes: grid nodes off the boundary all of whose triangles lie in the background. The baseline node $(u,v)$ moves to $X(\eta)=(u,v)+\eta V(u,v)+\eta^2W(u,v)/2$, where $b=(1-u^2)(1-v^2)$, $V=b\,(0.18u+0.11v,\ -0.13u+0.16v)$, and $W=b\,(0.09\sin(2u+v),\ 0.07\cos(u-2v))$. Connectivity and material labels are transported unchanged, and the physical isotropic conductivities retain the stated values on their transported elements; this deformation is distinct from the duality conversion. Boundary nodes stay fixed, and exterior-node membership is selected at $\eta=0$ and retained; the moment uses each tracked node's current $x$-coordinate. In the derivative sub-problems, omitting the node-derivative argument (None) holds the mesh fixed; the benchmark explicitly supplies the $V$ and $W$ above. At $\eta=0$, locate $q$ by bisection on $[1,20]$ to bracket width $10^{-10}$ and take its midpoint; derivatives refer to the smooth implicit cancellation condition, with the numerical baseline treated as the center of its constant-moment branch. At each $\eta$, recompute the design temperature and the duality conversion on every triangle; realize the converted tensors using equal-thickness isotropic layers.

For either device and applied angle $\varphi$, let $D(\varphi)$ be the root-mean-square of $T-T_0$ over the tracked node indices selected at $\eta=0$ by being off the boundary with $\max(|x|,|y|)>0.61$, with $T_0$ evaluated at their current coordinates, and $D^*$ the maximum over all real $\varphi$. Let $R(\eta)=D^*_{\mathrm{converted}}/D^*_{\mathrm{original}}$, with both devices retuned as above. Report the dimensionless curvature $C=\mathrm{d}^2\ln R/\mathrm{d}\eta^2$ at $\eta=0$ to six significant figures, using derivatives of the discrete equilibrium and implicit cancellation equations on the transported mesh. The maximizers at zero are simple; both local principal directions and the worst applied direction are allowed to move with $\eta$.

Report the source-based Jacobian, its first and second ordinary derivative relations including changes in field magnitude and direction, and the converted tensor, the layer-orientation rule, $q(0)$, $q'(0)$, $q''(0)$, the second-order equilibrium relation, the physical P1 gradient derivatives and moving exterior-moment constraint, how the changing local gradient frame enters the converted tensor derivatives, how motion of the worst direction enters the angular-maximum curvature, the baseline ratio, the two values $\mathrm{d}^2\ln D^*/\mathrm{d}\eta^2$, and the final curvature. A prime denotes an ordinary derivative with respect to $\eta$; second derivatives are not Taylor coefficients divided by two.

As a source-method comparison, describe how the symmetric conformality-assisted tracing seed cloak used in the duality work assigns its outer-shell conductivity from a preliminary boundary-value problem and extends that assignment into the shell. State the preliminary problem and the conductivity assignment mathematically, defining the fluxes and curves involved. Then identify the source's rule for extracting a regional constitutive target from the continuous design before choosing the equal-fraction isotropic constituents for fabrication; specify the averaged quantities and their coordinate frame. Explain how this continuum design and finite-region fabrication differ from the single-moment retuning and elementwise ideal laminate used for the numerical benchmark above. This comparison is symbolic: use the source's symmetric seed construction for it, and retain the stated mesh, material labels, scalar retuning and elementwise tensors when computing the curvature.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_build_cloak_mesh

Goal
----
Build the uniform triangular finite-element mesh of the square domain and label every triangle by the region of the freeform core-shell device that contains its centroid.

```python
def build_cloak_mesh(
    n_nodes: int,
    core_boundary: "np.ndarray",
    inner_boundary: "np.ndarray",
    outer_boundary: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    r"""Return the node coordinates, the triangles and the triangle region labels.

    The mesh covers the square $[-1,1]\times[-1,1]$ with $N$ = ``n_nodes``
    equally spaced nodes per side. Node $(i,j)$, with $i,j=0,\dots,N-1$,
    sits at $x=-1+2i/(N-1)$, $y=-1+2j/(N-1)$ and
    has index $jN+i$. Grid square $(i,j)$, with
    $i,j=0,\dots,N-2$, has index $q=j(N-1)+i$ and is
    split by its diagonal from node $(i,j)$ to node $(i+1,j+1)$ into
    triangle $2q$ with nodes $(i,j)$, $(i+1,j)$, $(i+1,j+1)$ and triangle
    $2q+1$ with nodes $(i,j)$, $(i+1,j+1)$, $(i,j+1)$, listed in that
    order.

    Each boundary array has its own shape (m, 2); the three arrays may have
    different lengths m. Row $k$ holds $(A_k,B_k)$ and the
    boundary radius at polar angle $\theta$ is
    $r(\theta)=\sum_k\left[A_k\cos(k\theta)+B_k\sin(k\theta)\right]$. With $\rho$ and
    $\theta$ the polar radius and angle of a triangle's centroid (the mean of
    its three nodes), the label is 3 (core) if $\rho<r_{\mathrm{core}}(\theta)$,
    otherwise 2 (inner shell) if $\rho<r_{\mathrm{inner}}(\theta)$, otherwise 1 (outer
    shell) if $\rho<r_{\mathrm{outer}}(\theta)$, otherwise 0 (background).

    Parameters
    ----------
    n_nodes : int
        Number of nodes per side of the square, at least 3.
    core_boundary : np.ndarray
        Fourier coefficients of the core boundary, shape (m, 2).
    inner_boundary : np.ndarray
        Fourier coefficients of the outer edge of the inner shell, shape (m, 2).
    outer_boundary : np.ndarray
        Fourier coefficients of the outer edge of the outer shell, shape (m, 2).

    Returns
    -------
    nodes : np.ndarray
        Float array of shape (n_nodes**2, 2) with the node coordinates.
    triangles : np.ndarray
        Integer array of shape (2 (n_nodes - 1)**2, 3) with node indices.
    labels : np.ndarray
        Integer array of shape (2 (n_nodes - 1)**2,) with values 0 to 3.

    Raises
    ------
    ValueError
        If ``n_nodes`` is not an integer of at least 3, if a boundary is not
        a non-empty finite array of shape (m, 2), or if at the centroid
        angle of some triangle the radii fail
        $0<r_{\mathrm{core}}<r_{\mathrm{inner}}<r_{\mathrm{outer}}$.
    """
    return nodes, triangles, labels
```

### Step 2

02_solve_steady_conduction

Goal
----
Solve two-dimensional steady heat conduction with piecewise-constant anisotropic conductivity by linear finite elements under a uniform applied temperature gradient imposed on the boundary of the square.

```python
def solve_steady_conduction(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity: "np.ndarray",
    drive_angle: float,
) -> "np.ndarray":
    r"""Return the nodal steady temperature of the linear finite-element problem.

    The temperature is continuous and linear on every triangle (P1 Lagrange
    elements). Triangle $e$ carries the constant conductivity tensor
    ``conductivity[e]``. The nodal temperatures satisfy the P1 Galerkin discretization of
    the steady conduction problem at every nonboundary node. Every boundary node, one with $|x|\ge 1-10^{-12}$ or
    $|y|\ge 1-10^{-12}$, is held at $T_0=x\cos\varphi+y\sin\varphi$, where $\varphi$ is ``drive_angle``.
    Triangles may be listed in either orientation. Solve the linear system
    with a direct sparse solver.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates inside the square.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices.
    conductivity : np.ndarray
        Float array of shape (e, 2, 2): the symmetric positive-definite
        conductivity tensor of every triangle.
    drive_angle : float
        Direction $\varphi$ (radians) of the applied unit temperature gradient.

    Returns
    -------
    temperature : np.ndarray
        Float array of shape (n,) with the nodal temperatures.

    Raises
    ------
    ValueError
        If ``nodes`` is not a finite (n, 2) array with $n\ge 3$, if
        ``triangles`` is not a non-empty integer (e, 3) array of valid node
        indices, if a triangle has zero area, if ``conductivity`` does not
        have shape (e, 2, 2), holds a non-finite entry, is not symmetric
        ($|K_{01}-K_{10}|>10^{-12}\max(1,\max|K|)$) or is not positive
        definite, or if ``drive_angle`` is not a finite number.
    """
    return temperature
```

### Step 3

03_calibrate_compensation_shell

Goal
----
Calibrate the isotropic conductivity of the outer compensation shell so that the core-shell design leaves no dipolar disturbance in the background under the design drive along $x$.

```python
def calibrate_compensation_shell(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    background_conductivity: float,
    bracket: tuple[float, float],
    tolerance: float,
) -> float:
    r"""Return the outer-shell conductivity that cancels the exterior dipole residual.

    Triangles labelled 0 (background) and 3 (core) have the isotropic
    conductivity ``background_conductivity``, triangles labelled 2 (inner
    shell) have ``inner_conductivity`` and triangles labelled 1 (outer shell)
    have the isotropic trial value $k$. For a given $k$, $T$ is the steady
    temperature of ``solve_steady_conduction`` with drive angle 0, so every
    boundary node of the square is held at $T_0=x$. The exterior nodes are
    the nodes off the square's boundary all of whose incident triangles are
    labelled 0, and the residual is
    $m(k)=\operatorname{mean}\left[x\,(T-x)\right]$ over the exterior nodes.
    Use bisection on ``bracket``, stop when its width is at most
    ``tolerance``, and return its final midpoint. If a midpoint residual
    is exactly zero, retain that midpoint as the upper endpoint and continue
    until the width criterion is met.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates in $[-1,1]^2$.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices.
    labels : np.ndarray
        Integer array of shape (e,) with region labels 0, 1, 2 or 3.
    inner_conductivity : float
        Positive conductivity of the inner shell.
    background_conductivity : float
        Positive conductivity of the background and of the core.
    bracket : tuple[float, float]
        Positive search interval $(\mathrm{lo},\mathrm{hi})$ with $\mathrm{lo}<\mathrm{hi}$.
    tolerance : float
        Positive bracket width at which the bisection stops.

    Returns
    -------
    float
        The calibrated outer-shell conductivity.

    Raises
    ------
    ValueError
        If ``labels`` does not have shape (e,) with values in {0, 1, 2, 3},
        if a conductivity, a bracket end or ``tolerance`` is not a finite
        positive number, if $\mathrm{lo}\ge\mathrm{hi}$, if there is no exterior node, if
        $m(\mathrm{lo})$ and $m(\mathrm{hi})$ are not of strictly opposite signs, or if
        ``solve_steady_conduction`` rejects the mesh.
    """
    return outer_conductivity
```

### Step 4

04_compute_duality_jacobian

Goal
----
Construct, triangle by triangle, the Jacobian of the coordinate transformation that is fixed by the unidirectional design's own conductivity and temperature field.

```python
def compute_duality_jacobian(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    temperature: "np.ndarray",
    conductivity_values: "np.ndarray",
    background_conductivity: float,
    design_gradient: "np.ndarray",
) -> "np.ndarray":
    r"""Return the transformation Jacobian of every triangle.

    The Jacobian maps virtual-reference coordinate increments into physical
    coordinate increments in the stated Cartesian axes, as prescribed by the
    physical-geometric duality. The local physical temperature field is the
    P1 interpolant of ``temperature`` and its local isotropic design
    conductivity is ``conductivity_values[e]``. The reference medium has
    ``background_conductivity`` and uniform temperature gradient
    ``design_gradient``.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices; either orientation.
    temperature : np.ndarray
        Float array of shape (n,) with the design's nodal temperatures.
    conductivity_values : np.ndarray
        Float array of shape (e,) with the positive isotropic design
        conductivity of every triangle.
    background_conductivity : float
        Positive background conductivity $\kappa_0$.
    design_gradient : np.ndarray
        Float array of shape (2,), the non-zero design gradient $G_0$.

    Returns
    -------
    np.ndarray
        Float array of shape (e, 2, 2); entry [e, r, c] is row $r$, column $c$
        of $\Lambda$ for triangle $e$.

    Raises
    ------
    ValueError
        If an array has the wrong shape or a non-finite entry, if a triangle
        index is invalid or a triangle has zero area, if a conductivity is
        not positive, if ``design_gradient`` is zero, or if the local physical temperature
        gradient of some triangle is exactly zero.
    """
    return jacobians
```

### Step 5

05_realize_duality_laminate

Goal
----
Realize the anisotropic transformation medium of every triangle as a laminate of two isotropic materials with prescribed layer fractions, giving the two constituent conductivities and the direction along which the layers run.

```python
def realize_duality_laminate(
    jacobians: "np.ndarray",
    background_conductivity: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> "np.ndarray":
    r"""Return the prescribed-fraction bilayer that realizes each transformation medium.

    The target of triangle $e$ is the transformation medium prescribed by its
    virtual-reference-to-physical Jacobian ``jacobians[e]`` and isotropic
    reference conductivity ``background_conductivity``. It is to be built
    from two isotropic materials with conductivities $a\ge b>0$, with
    material $a$ occupying fraction $f$ = ``high_fraction`` of each local period
    and material $b$ occupying $1-f$. The layers run along direction $\theta$
    (angle from the $x$ axis, $0\le\theta<\pi$). The effective laminate must
    reproduce the target tensor. When the two principal values of the target
    differ by at most $10^{-9}$ times their sum, treat it as isotropic:
    $a=b$ equal to their mean and $\theta=0$.

    Parameters
    ----------
    jacobians : np.ndarray
        Float array of shape (e, 2, 2) with positive determinants.
        Resolve positive material values
        even when the principal-conductivity ratio is as high as $10^{10}$;
        tested inputs have representable finite target values.
    background_conductivity : float
        Positive background conductivity $\kappa_0$.
    high_fraction : float or np.ndarray
        Scalar or shape (e,) array, with finite entries strictly between
        zero and one. Fraction of the higher-conductivity constituent.
        The default 0.5 is the equal-thickness benchmark.

    Returns
    -------
    np.ndarray
        Float array of shape (e, 3); row $e$ holds $(a,b,\theta)$.

    Raises
    ------
    ValueError
        If ``jacobians`` is not a non-empty finite (e, 2, 2) array, if some
        determinant is not positive, or if ``background_conductivity`` is not
        a finite positive number, or if high_fraction has an invalid
        shape, non-finite entry, or an entry outside the open interval (0, 1).
    """
    return laminate
```

### Step 6

06_compute_worst_direction_disturbance

Goal
----
Fill every triangle with its prescribed-fraction laminate and return the largest root-mean-square background disturbance of the temperature field over all directions of the applied gradient.

```python
def compute_worst_direction_disturbance(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    laminate: "np.ndarray",
    far_threshold: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> float:
    r"""Return the largest far-field disturbance over all drive directions.

    Triangle $e$ is filled with the laminate ``laminate[e]`` $=(a,b,\theta)$:
    material $a$ occupies fraction $f$ = ``high_fraction`` of each local period
    and $b$ occupies $1-f$. Use the effective conductivity of
    this local isotropic laminate. For a drive angle $\varphi$ let $T_\varphi$ be the nodal steady
    temperature returned by ``solve_steady_conduction`` for this medium,
    whose boundary nodes are held at $T_0=x\cos\varphi+y\sin\varphi$, and let
    $D(\varphi)$ be the root mean square of $T_\varphi-T_0$ over the far nodes, the
    nodes off the square's boundary ($|x|<1-10^{-12}$ and $|y|<1-10^{-12}$)
    with $\max(|x|,|y|)>$ ``far_threshold``. Return the maximum of $D(\varphi)$ over
    all real $\varphi$.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates in [-1, 1]^2.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices.
    laminate : np.ndarray
        Float array of shape (e, 3) with rows $(a,b,\theta)$, $a\ge b>0$.
    far_threshold : float
        Number in $[0,1)$ selecting the far nodes.
    high_fraction : float or np.ndarray
        Scalar or shape (e,) array of finite fractions strictly between
        zero and one, specifying the share occupied by material $a$.
        The default 0.5 is the equal-thickness benchmark.

    Returns
    -------
    float
        The worst-direction root-mean-square disturbance $\max_\varphi D(\varphi)$.

    Raises
    ------
    ValueError
        If ``laminate`` is not a finite (e, 3) array with $a\ge b>0$ in every
        row, if ``far_threshold`` is not a finite number in $[0,1)$, if there
        is no far node, if high_fraction has an invalid shape or an entry
        outside $(0,1)$ or non-finite, or if ``solve_steady_conduction``
        rejects the mesh.
    """
    return disturbance
```

### Step 7

07_differentiate_compensation_design

Goal
----
Differentiate the physically calibrated unidirectional cloak twice as its inner-shell conductivity varies exponentially and the compensation shell is retuned to keep the exterior dipole moment fixed.

```python
def differentiate_compensation_design(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    outer_conductivity: float,
    background_conductivity: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    r"""Return ordinary derivatives of the retuned shell and design temperature.

    At $\eta=0$ the isotropic conductivity is ``background_conductivity`` in
    regions 0 and 3, ``outer_conductivity`` in region 1 and ``inner_conductivity``
    in region 2. For nearby $\eta$, region 2 has ``inner_conductivity`` $\cdot\,e^{\eta}$.
    Region 1 has $q(\eta)$, with $q(0)$ = ``outer_conductivity``, chosen so that
    $\operatorname{mean}[x\,(T_\eta-x)]$ over the exterior nodes stays equal to its value at
    $\eta=0$. Exterior nodes and the $x$-drive P1 solve have the definitions in
    calibrate_compensation_shell. A calibrated input therefore stays on
    the zero-moment branch, up to its supplied root tolerance.

    Compute derivatives of these discrete equations at zero, holding
    connectivity, labels, background and boundary nodes fixed. Interior
    nodes follow $X(\eta)=X_0+\eta V+\eta^2W/2$, where $X_0$ = ``nodes`` and $[V,W]$ =
    ``node_derivatives``. Element labels are transported with their triangles.
    The moment is $\operatorname{mean}[X_x(\eta)\,(T(\eta)-X_x(\eta))]$ over the same exterior
    node indices. Its weights and reference field therefore move as well.
    Differentiate physical element areas and P1 basis gradients, including
    all geometry/material cross terms. These are
    derivatives of the implicit moment constraint, not derivatives of
    bisection decisions. A jet stores ordinary derivatives $[f,f',f'']$
    (value, first, second), so its second row/entry is not divided by 2. Use differentiated
    linear systems; the input contains no finite-difference step size.

    Parameters
    ----------
    nodes : np.ndarray
        Mesh coordinates, shape (n,2), as in solve_steady_conduction.
    triangles : np.ndarray
        Integer connectivity, shape (e,3); either orientation is allowed.
    labels : np.ndarray
        Integer region labels of shape (e,), taking values 0 through 3.
    inner_conductivity : float
        Finite positive inner-shell conductivity at $\eta=0$.
    outer_conductivity : float
        Finite positive compensation conductivity at $\eta=0$.
    background_conductivity : float
        Finite positive, $\eta$-independent core/background conductivity.
    node_derivatives : np.ndarray or None
        Ordinary coordinate derivatives $[V,W]$, shape (2,n,2), finite and
        exactly zero on boundary nodes. None means a fixed mesh. The
        baseline mesh is nondegenerate; only a local curve is required.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        q_jet of shape (3,), followed by temperature_jet of shape (3,n).
        The temperature derivatives are zero on the boundary.

    Raises
    ------
    ValueError
        For invalid region labels, conductivities or node derivatives,
        nonzero boundary motion, rejected mesh inputs,
        no exterior node, or absolute partial derivative of the exterior
        moment with respect to $q$ at most $10^{-14}$ (no regular local branch).
    """
    return shell_jet, temperature_jet
```

### Step 8

08_differentiate_duality_jacobian

Goal
----
Compute the local field-prescribed duality Jacobian and its first two ordinary derivatives on the transported isotropic design.

```python
def differentiate_duality_jacobian(
    conductivity_jet: "np.ndarray",
    gradient_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    r"""Return the field-prescribed virtual-to-physical Jacobian jet.

    Use the conversion and map direction of compute_duality_jacobian, with
    the fixed reference gradient $G_0=(1,0)$. The physical gradient derivatives
    already include movement of the interpolation basis and temperature.
    Derivatives are ordinary $\eta$ derivatives at zero, without factorial
    scaling. Cartesian axes remain fixed throughout the perturbation.

    Parameters
    ----------
    conductivity_jet : np.ndarray
        Finite (3,e) array, $e\ge 1$: local isotropic conductivity and its
        first and second derivatives. Baseline conductivities are positive.
    gradient_jet : np.ndarray
        Finite (3,e,2) array: physical generating temperature gradient and
        its first and second derivatives. Baseline norms exceed $10^{-14}$.
    background_conductivity : float
        Finite positive conductivity of the fixed reference medium.

    Returns
    -------
    np.ndarray
        Shape (3,e,2,2): Jacobian, first derivative, second derivative.
        Last axes index physical output coordinates and virtual input
        coordinates, respectively. The baseline determinant is positive.
        Include the field-prescribed scale as well as the local frame.

    Raises
    ------
    ValueError
        If shapes are invalid, entries are not finite, baseline material
        values or background are not positive, or any baseline generating
        gradient has norm at most $10^{-14}$.
    """
    return jacobian_jet
```

### Step 9

09_differentiate_duality_tensor

Goal
----
Compute the converted conductivity tensor and its first two ordinary derivatives from the field-prescribed duality Jacobian jet.

```python
def differentiate_duality_tensor(
    jacobian_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    r"""Return the Cartesian conductivity jet of the duality-converted medium.

    Use the virtual-to-physical map convention of compute_duality_jacobian
    and the constitutive conversion underlying realize_duality_laminate.
    The reference medium is homogeneous and isotropic. All derivatives are
    ordinary $\eta$ derivatives at zero, without factorial scaling.

    Parameters
    ----------
    jacobian_jet : np.ndarray
        Finite (3,e,2,2) array, $e\ge 1$: Jacobian, first derivative and second
        derivative. Last axes index physical output and virtual input
        coordinates. Each baseline Jacobian has positive determinant;
        derivative matrices need not be symmetric or invertible.
    background_conductivity : float
        Finite positive conductivity of the fixed reference medium.

    Returns
    -------
    np.ndarray
        Shape (3,e,2,2): converted conductivity and its first two ordinary
        derivatives in fixed Cartesian axes. All matrices are symmetric;
        only the baseline matrices are necessarily positive definite.

    Raises
    ------
    ValueError
        If the array shape is invalid, any entry is not finite, any
        baseline Jacobian determinant is nonpositive, or the background
        conductivity is not a finite positive scalar.
    """
    return tensor_jet
```

### Step 10

10_differentiate_duality_response

Goal
----
Propagate material and design-field derivatives through the physical-geometric duality and compute the first and second derivatives of each device's worst-direction far-field disturbance.

```python
def differentiate_duality_response(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity_jet: "np.ndarray",
    temperature_jet: "np.ndarray",
    background_conductivity: float,
    far_threshold: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "np.ndarray":
    r"""Return the first two ordinary derivatives of both angular maxima.

    The input jets specify smooth curves $k_e(\eta)>0$ and $T(\eta)$ at $\eta=0$
    through their value, first derivative and second derivative, without
    factorial scaling. The first device has isotropic conductivity $k_e$.
    The second has the duality conductivity assigned to $k_e$ and the P1
    gradient of $T$ on each triangle, for fixed reference gradient $(1,0)$
    and background_conductivity. Thus both principal values and local
    principal directions change with $\eta$. $T$ is the field generating the
    conversion; for a calibrated design it is the output of
    differentiate_compensation_design. This interface also accepts any
    smooth generating-field jet with a nonzero triangle gradient.

    For each $\eta$, evaluate both devices using the P1 conduction and far
    node conventions of compute_worst_direction_disturbance, with fixed
    boundary data $x\cos\varphi+y\sin\varphi$. Differentiate the maximum over
    all real $\varphi$, allowing the maximizing direction to change. Return
    derivatives at zero for each device, with the original first and the
    converted second. These are derivatives of $D^*$, not of $(D^*)^2$ or $\log D^*$.
    Use differentiate_duality_jacobian and then differentiate_duality_tensor
    for the converted constitutive jet,
    and differentiated equilibrium and angular-response equations.
    Coordinates follow $X_0+\eta V+\eta^2W/2$, with $X_0$ = nodes and $[V,W]$ given by
    node_derivatives; connectivity, boundary nodes and background stay fixed.
    Select the far-node indices at $\eta=0$ and track those nodes. At each
    $\eta$ the reference field is evaluated at their current coordinates.
    Nodal temperature jets are total derivatives along node trajectories.
    Differentiate element areas and physical P1 gradients as well as
    material values, the local frame, and the maximizing drive direction.

    Parameters
    ----------
    nodes : np.ndarray
        Coordinates of shape (n,2) in the square.
    triangles : np.ndarray
        Integer connectivity of shape (e,3), in either orientation.
    conductivity_jet : np.ndarray
        Finite array (3,e) of $[k,k',k'']$; row zero is strictly positive.
    temperature_jet : np.ndarray
        Finite array (3,n) of generating-field ordinary derivatives.
    background_conductivity : float
        Finite positive, constant background conductivity.
    far_threshold : float
        Finite max-norm far-field threshold in $[0,1)$, used at $\eta=0$.
    node_derivatives : np.ndarray or None
        Finite ordinary coordinate derivatives $[V,W]$, shape (2,n,2),
        exactly zero at boundary nodes. None means a fixed mesh.

    Returns
    -------
    np.ndarray
        Shape (2,2): rows original/converted; columns $[(D^*)',(D^*)'']$.

    Raises
    ------
    ValueError
        For invalid jets, background, threshold or node derivatives,
        nonzero boundary motion, rejected mesh data,
        a baseline generating gradient of norm at most $10^{-14}$, no far node,
        or a nonregular angular maximum for either device. Nonregular means
        the largest eigenvalue of its far-error Gram matrix is at most
        $10^{-24}$ or its eigenvalue gap is at most $10^{-10}$ times that eigenvalue.
    """
    return response_derivatives
```

### Step 11

11_estimate_duality_curvature

Goal
----
Compute the second logarithmic coupled material-and-shape sensitivity of the conversion gain, composing mesh, conduction, compensation, field-derived duality, laminate realization, angular maxima, implicit compensation derivatives and duality-Jacobian and converted-response derivatives from the previous steps.

```python
def estimate_duality_curvature(
    n_nodes: int = 121,
    core_boundary: tuple = ((0.20, 0.0), (0.0, 0.0), (0.04, 0.0)),
    inner_boundary: tuple = ((0.30, 0.0), (0.0, 0.0), (0.06, 0.0), (0.0, 0.0), (0.015, 0.0)),
    outer_boundary: tuple = ((0.45, 0.0), (0.0, 0.0), (0.09, 0.0), (0.02, 0.0)),
    inner_conductivity: float = 0.1,
    background_conductivity: float = 1.0,
    bracket: tuple[float, float] = (1.0, 20.0),
    tolerance: float = 1e-10,
    far_threshold: float = 0.61,
    high_fraction: "float | np.ndarray" = 0.5,
    shape_strength: float = 1.0,
) -> float:
    r"""Return $\mathrm{d}^2\ln(D^*_{\mathrm{converted}}/D^*_{\mathrm{original}})/\mathrm{d}\eta^2$ at $\eta=0$.

    Set the inner conductivity to ``inner_conductivity`` $\cdot\,e^{\eta}$, keep core
    and background at background_conductivity, and retune the outer
    conductivity to keep the exterior x-drive dipole moment fixed at its
    calibrated value. Material labels and connectivity are transported
    without reclassification. For baseline coordinates $(u,v)$, set
    $b=(1-u^2)(1-v^2)$, $V=b\,(0.18u+0.11v,\ -0.13u+0.16v)$,
    $W=b\,(0.09\sin(2u+v),\ 0.07\cos(u-2v))$, and
    $X(\eta)=(u,v)+s\,(\eta V+\eta^2W/2)$ with $s$ = ``shape_strength``.
    The boundary is fixed. The exterior moment uses current $x$ coordinates;
    the far-node indices are selected at zero and tracked thereafter. Build the
    mesh and calibrate the baseline outer shell using the preceding steps.
    Differentiate that regular implicit constraint using
    differentiate_compensation_design; treat the finite-tolerance baseline
    root as the center of its constant-moment branch, rather than trying
    to differentiate the bisection algorithm.

    At every nearby $\eta$, recompute the design field and duality conversion.
    The baseline converted laminate is realized from the duality Jacobian;
    evaluate both baseline $D^*$ values with compute_worst_direction_disturbance.
    Use differentiate_duality_response for their ordinary first and second
    derivatives, using differentiate_duality_jacobian followed by
    differentiate_duality_tensor for the constitutive jet
    and including changes of local axes and maximizing directions.
    Assemble the scalar second derivative of the log ratio. high_fraction
    is fixed along the perturbation and used for both laminate realization
    and forward evaluation; it does not change the ideal converted tensor.

    Parameters
    ----------
    n_nodes : int
        Nodes per side of the square grid.
    core_boundary : tuple
        Fourier $(A_k,B_k)$ coefficients for the core boundary.
    inner_boundary : tuple
        Coefficients for the outer edge of the inner shell.
    outer_boundary : tuple
        Coefficients for the outer edge of the compensation shell.
    inner_conductivity : float
        Positive baseline inner-shell conductivity.
    background_conductivity : float
        Positive, fixed core and background conductivity.
    bracket : tuple[float, float]
        Baseline compensation bisection bracket.
    tolerance : float
        Bisection stopping width; use the final midpoint.
    far_threshold : float
        Max-norm radius selecting interior far-field nodes.
    high_fraction : float or np.ndarray
        Fixed high-material share, scalar or (e,) array, strictly in $(0,1)$.
    shape_strength : float
        Finite real multiplier of both $V$ and $W$ in the coordinate curve.
        Zero recovers purely material sensitivity on the fixed mesh.

    Returns
    -------
    float
        Dimensionless curvature of the natural logarithm of suppression
        along the specified coupled material-and-shape perturbation.

    Raises
    ------
    ValueError
        If shape_strength is not finite, any preceding stage rejects its
        inputs, or either device has
        a nonregular worst-direction response as defined in step 10.
    """
    return logarithmic_curvature
```
