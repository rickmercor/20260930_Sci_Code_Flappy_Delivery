# Physics-Computational_Physics-23

## Background

Thermal barrier coatings, microelectronic packaging films and wave-detection layers are sub-micrometre to millimetre skins bonded to a comparatively massive substrate, and their transient response has to be predicted at aspect ratios that domain methods cannot mesh economically. A finite element model must fill the coating with elements whose in-plane size is set by the layer thickness, so the element count and the conditioning both degrade as the layer thins, and for the thinnest layers the model stops converging at all. A boundary element model stores only the outline and so is indifferent to the aspect ratio in its unknown count, which is what makes it attractive here.

Two things stand in the way. The first is time. A boundary element treatment of a transient problem either carries time-domain fundamental solutions, which force a convolution over the whole history and a fresh derivation of the kernel for every governing operator, or it moves the time derivative to the right-hand side and treats it as an equivalent body force, which leaves an operator whose fundamental solution is the elementary one but produces a domain integral over a field that is not yet known. A spectral treatment of the time direction replaces the low-order finite difference that would otherwise be used with an expansion in orthogonal polynomials evaluated at quadrature nodes inside the step, so that the derivative at those nodes is a fixed matrix acting on the values at those nodes; the payoff is that the step length is no longer bounded by stability or by low-order truncation.

The second is that both domain integrals have to go back to the boundary if the method is to keep its boundary-only character, and that the geometry then fights back. A change of variables that scales the boundary towards the collocation point sweeps the whole region with rays and converts an area integral into an integral along the same outline, and it does so exactly rather than by interpolation. But in a slender layer every interior point sits a small fraction of an element length from a face, so the fundamental solution and its normal derivative vary over a stretch of the element far shorter than the spacing of any fixed quadrature rule, and both the boundary integrals and the swept-ray integrals collapse. Recovering them is a question of where the quadrature points are put rather than how many are used, and it is what decides whether a boundary-only transient model of an ultra-thin layer is worth anything at all.

## Problem

A coating of thermal conductivity $k_1 = 2\ \mathrm{W\,m^{-1}\,^{\circ}C^{-1}}$ occupying $0 \le x \le L$ and $0 \le y \le h$ lies on a substrate of thermal conductivity $k_2 = 15\ \mathrm{W\,m^{-1}\,^{\circ}C^{-1}}$ occupying $0 \le x \le L$ and $-1\ \mathrm{m} \le y \le 0$, with $L = 4\ \mathrm{m}$, and in each layer the temperature obeys $\nabla^2 u = k^{-1}\,\partial u/\partial t + f$ with the heat source density $f$ fixed by requiring that $u(x, y, t) = (a\,y + 3 - \sin x)\,e^{t}$ be an exact solution in that layer, the coefficient $a$ being unity in the substrate and, in the coating, whatever continuity of temperature and of normal heat flux across $y = 0$ then demands. I evaluate the temperature at interior sample points of each layer from the interior boundary integral representation of the Laplace operator, evaluating the analytic temperature and its analytic outward normal derivative directly at every boundary quadrature point, using the three-node shape functions only to represent element geometry, and I keep the model boundary-only: the transient and source contributions are carried by a domain integral that a scaled coordinate transformation centred on the sample point turns into an integral over the same boundary discretisation, and the time derivative at the Gauss nodes of a single step $\Delta t = 1\ \mathrm{s}$ is supplied by a Legendre spectral integration on $P = 5$ nodes acting on the nodal temperatures of that step. Each layer outline is traversed anticlockwise from its lower left corner and every face is divided into equal-length straight three-node quadratic elements whose midside geometry node is the midpoint of its two end nodes: ten elements on each face of length $L$, and two in the coating or three in the substrate on each short face. The interior sample points are the cell centres of a uniform grid, $10 \times 2$ in the coating and $8 \times 3$ in the substrate. Every element integral uses forty Gauss points in the element local coordinate, the two boundary kernels and the circumferential direction of that element's swept domain sector alike, and the radial direction of each swept sector uses sixteen Gauss points. With $h = 10^{-4}\ \mathrm{m}$, run the whole evaluation twice, once with the element quadrature regularised by the hyperbolic-sine coordinate change that thin-layer boundary element work uses for nearly singular integrals and once with a plain Gauss-Legendre rule, measure in each case the Euclidean-norm relative error of the reconstructed temperatures pooled over the interior sample points of both layers and all five Gauss node times, and report the ratio of the unregularised error to the regularised one. Then tell me what that ratio implies for boundary-only modelling of ultra-thin layers as the thickness is pushed further down, what limits the accuracy that survives the regularisation, and how the comparison would change if the coating were as thick as the substrate; report compactly, but do report the closures you took, the intermediate values the number rests on and the checks you made on it.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_time_spectral_operator

Goal
----
Build the Gauss node times of one time step together with the matrix that turns nodal values of a field into nodal values of its first time derivative.

```python
def time_spectral_operator(num_nodes: int, time_step: float) -> tuple:
    """Return the Gauss node times of one time step and the nodal derivative operator.

    Parameters
    ----------
    num_nodes : int
        Number of Gauss-Legendre nodes ``P`` placed inside the time step
        (``P >= 2``). The Legendre expansion is truncated at ``P`` terms, that
        is at polynomial degrees 0 to ``P - 1``.
    time_step : float
        Length of the time step (``time_step > 0``). The step is taken to run
        from time zero to ``time_step``.

    Returns
    -------
    result : tuple
        ``(times, operator)`` where ``times`` is a float array of shape ``(P,)``
        holding the Gauss node times inside the step in increasing order, and
        ``operator`` is a float array of shape ``(P, P)`` such that the nodal
        time derivatives equal ``operator`` applied to the nodal values minus
        the value carried at the start of the step, the time derivative being
        expanded over the step in Legendre polynomials of degrees 0 to
        ``P - 1``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is not an integer >= 2, or if ``time_step`` is not a
        finite real number > 0.

    """
    return (times, operator)  # placeholder
```

### Step 2

02_rectangular_layer_mesh

Goal
----
Lay out the quadratic boundary elements of one rectangular layer together with the interior points at which that layer is sampled.

```python
def rectangular_layer_mesh(length: float, y_bottom: float, y_top: float,
                           n_long: int, n_short: int, n_x: int, n_y: int) -> tuple:
    """Discretise the outline of one rectangular layer and sample its interior.

    Parameters
    ----------
    length : float
        Extent of the layer along x (``length > 0``); the layer occupies
        ``0 <= x <= length``.
    y_bottom : float
        Lower bound of the layer along y.
    y_top : float
        Upper bound of the layer along y (``y_top > y_bottom``).
    n_long : int
        Number of boundary elements on each of the two faces of length
        ``length`` (``n_long >= 1``).
    n_short : int
        Number of boundary elements on each of the two faces of height
        ``y_top - y_bottom`` (``n_short >= 1``).
    n_x : int
        Number of interior sample columns along x (``n_x >= 1``).
    n_y : int
        Number of interior sample rows along y (``n_y >= 1``).

    Returns
    -------
    result : tuple
        ``(elements, interior)``. ``elements`` is a float array of shape
        ``(2 * n_long + 2 * n_short, 3, 2)`` holding, for each boundary element
        in anticlockwise order starting from the corner ``(0, y_bottom)`` and
        running first along the face at ``y_bottom``, the ``(x, y)``
        coordinates of its first end node, its middle node and its second end
        node. ``interior`` is a float array of shape ``(n_x * n_y, 2)`` holding
        the interior sample points, which are the cell centres of a uniform
        ``n_x`` by ``n_y`` grid over the layer, ordered with the x index
        varying slowest.

    Raises
    ------
    ValueError
        If ``length`` is not a finite real number > 0, if ``y_bottom`` or
        ``y_top`` is not finite, if ``y_top`` is not greater than ``y_bottom``,
        or if any of ``n_long``, ``n_short``, ``n_x``, ``n_y`` is not an
        integer >= 1.

    """
    return (elements, interior)  # placeholder
```

### Step 3

03_layer_reference_fields

Goal
----
Evaluate the benchmark temperature field of one layer, its two spatial derivatives and the heat source density that makes it an exact solution.

```python
def layer_reference_fields(points: "np.ndarray", time: float, slope: float,
                           conductivity: float) -> "np.ndarray":
    """Sample the benchmark field of one layer and the source density it implies.

    The benchmark temperature of a layer is
    ``u(x, y, t) = (slope * y + 3 - sin(x)) * exp(t)``, where ``slope`` is the
    coefficient that the interface continuity of temperature and of normal heat
    flux assigns to that layer.

    Parameters
    ----------
    points : np.ndarray
        Float array of shape ``(n, 2)`` holding the ``(x, y)`` coordinates at
        which the fields are wanted.
    time : float
        Time at which the fields are wanted (finite).
    slope : float
        Coefficient multiplying ``y`` in the benchmark field of this layer
        (finite).
    conductivity : float
        Thermal conductivity of this layer (``conductivity > 0``).

    Returns
    -------
    fields : np.ndarray
        Float array of shape ``(n, 4)`` whose columns are, in order, the
        temperature, its derivative with respect to x, its derivative with
        respect to y, and the heat source density of the transient conduction
        balance at the sample points.

    Raises
    ------
    ValueError
        If ``points`` is not a finite real array of shape ``(n, 2)`` with
        ``n >= 1``, if ``time`` or ``slope`` is not a finite real number, or if
        ``conductivity`` is not a finite real number > 0.

    """
    return fields  # placeholder
```

### Step 4

04_near_singular_rule

Goal
----
Build the quadrature rule used to integrate one boundary element as seen from a source point that may lie very close to it.

```python
def near_singular_rule(source_point: "np.ndarray", element_nodes: "np.ndarray",
                       num_gauss: int, use_sinh: bool) -> "np.ndarray":
    """Return the quadrature nodes and weights for one element and source point.

    The rule integrates a function of the element local coordinate over
    ``[-1, 1]``. With ``use_sinh`` set, the local coordinate is written as
    ``xi(t) = xi0 + d * sinh(k1 * t - k2)`` for a new variable ``t`` running
    over ``[-1, 1]``, where ``xi0`` is the local coordinate of the foot of the
    perpendicular dropped from the source point onto the straight line carrying
    the element, ``d`` is the distance from the source point to that foot
    divided by half the element length, and the two constants are fixed by
    requiring that ``t = -1`` and ``t = 1`` map onto the two ends of the
    element. The Gauss-Legendre rule is then applied in ``t``.

    Parameters
    ----------
    source_point : np.ndarray
        Float array of shape ``(2,)`` holding the ``(x, y)`` coordinates of the
        source point.
    element_nodes : np.ndarray
        Float array of shape ``(3, 2)`` holding the first end node, the middle
        node and the second end node of a straight boundary element whose
        middle node is its midpoint.
    num_gauss : int
        Number of Gauss-Legendre points of the rule (``num_gauss >= 1``).
    use_sinh : bool
        If true, apply the hyperbolic-sine change of variable; if false, return
        the plain Gauss-Legendre rule of the element local coordinate.

    Returns
    -------
    rule : np.ndarray
        Float array of shape ``(num_gauss, 2)`` whose first column holds the
        element local coordinates of the quadrature points and whose second
        column holds the matching weights, already carrying the derivative of
        the change of variable, so that the integral of a function over the
        element local coordinate is the weighted sum of its values.

    Raises
    ------
    ValueError
        If ``source_point`` is not a finite real array of shape ``(2,)``, if
        ``element_nodes`` is not a finite real array of shape ``(3, 2)``, if
        the element has zero length or its middle node is not its midpoint, if
        the source point lies on the straight line carrying the element, if
        ``num_gauss`` is not an integer >= 1, or if ``use_sinh`` is not a
        boolean.

    """
    return rule  # placeholder
```

### Step 5

05_element_boundary_terms

Goal
----
Turn one boundary element and one source point into the quadrature points, kernel weights and outward normals that carry the boundary integral of the representation formula.

```python
def element_boundary_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                           rule: "np.ndarray") -> "np.ndarray":
    """Assemble the boundary quadrature data of one element for one source point.

    The element is interpolated with the three quadratic shape functions of its
    local coordinate, whose nodes lie at ``-1``, ``0`` and ``1``, and its
    outward normal follows from the anticlockwise traversal of the layer
    outline.

    Parameters
    ----------
    source_point : np.ndarray
        Float array of shape ``(2,)`` holding the ``(x, y)`` coordinates of the
        source point, which must not lie on the element.
    element_nodes : np.ndarray
        Float array of shape ``(3, 2)`` holding the first end node, the middle
        node and the second end node of the element.
    rule : np.ndarray
        Float array of shape ``(m, 2)`` holding the element local coordinates
        of the quadrature points and their weights.

    Returns
    -------
    terms : np.ndarray
        Float array of shape ``(m, 6)`` whose columns are, in order, the x and
        y coordinates of the quadrature points, the weight that multiplies the
        normal derivative of the field there, the weight that multiplies the
        field there, and the two components of the outward unit normal. Summing
        the third column against the normal derivative minus the fourth column
        against the field gives this element's contribution to the boundary
        integral of the representation formula.

    Raises
    ------
    ValueError
        If ``source_point`` is not a finite real array of shape ``(2,)``, if
        ``element_nodes`` is not a finite real array of shape ``(3, 2)``, if
        ``rule`` is not a finite real array of shape ``(m, 2)`` with
        ``m >= 1``, if the element has a vanishing tangent at a quadrature
        point, or if the source point coincides with a quadrature point.

    """
    return terms  # placeholder
```

### Step 6

06_sct_domain_terms

Goal
----
Convert the part of one layer swept by rays from a source point through one boundary element into quadrature points and weights for the domain integral of the fundamental solution against a known density.

```python
def sct_domain_terms(source_point: "np.ndarray", element_nodes: "np.ndarray",
                     rule: "np.ndarray", num_radial: int) -> "np.ndarray":
    """Build the domain quadrature of one angular sector of a layer.

    The sector is the set of points swept by the segments joining the source
    point to the boundary element. Its points are written as the source point
    plus a radial fraction, running over ``[0, 1]``, of the vector from the
    source point to the boundary, and the element is interpolated with the
    three quadratic shape functions of its local coordinate, whose nodes lie at
    ``-1``, ``0`` and ``1``. The radial direction is integrated with a
    Gauss-Legendre rule of ``num_radial`` points mapped onto ``[0, 1]``.

    Parameters
    ----------
    source_point : np.ndarray
        Float array of shape ``(2,)`` holding the ``(x, y)`` coordinates of the
        source point, which must lie strictly inside the layer.
    element_nodes : np.ndarray
        Float array of shape ``(3, 2)`` holding the first end node, the middle
        node and the second end node of the element, in the anticlockwise
        traversal of the layer outline.
    rule : np.ndarray
        Float array of shape ``(m, 2)`` holding the element local coordinates
        of the circumferential quadrature points and their weights.
    num_radial : int
        Number of Gauss-Legendre points in the radial direction
        (``num_radial >= 1``).

    Returns
    -------
    terms : np.ndarray
        Float array of shape ``(num_radial * m, 3)`` whose columns are the x
        and y coordinates of the quadrature points and the weight to be applied
        there, so that the weighted sum of a density over all sectors of the
        layer is the domain integral of the fundamental solution of the
        two-dimensional Laplace operator, normalised so that its Laplacian is
        minus the Dirac distribution at the source point, against that density.
        The rows are ordered with the radial index varying slowest.

    Raises
    ------
    ValueError
        If ``source_point`` is not a finite real array of shape ``(2,)``, if
        ``element_nodes`` is not a finite real array of shape ``(3, 2)``, if
        ``rule`` is not a finite real array of shape ``(m, 2)`` with
        ``m >= 1``, if ``num_radial`` is not an integer >= 1, or if the source
        point coincides with a circumferential quadrature point.

    """
    return terms  # placeholder
```

### Step 7

07_domain_integrand_values

Goal
----
Combine the spectral time derivative of the temperature with the heat source density into the density that the domain integral of the representation formula must carry at every Gauss node of the time step.

```python
def domain_integrand_values(operator: "np.ndarray", nodal_values: "np.ndarray",
                            initial_values: "np.ndarray", source_values: "np.ndarray",
                            conductivity: float) -> "np.ndarray":
    """Assemble the density of the domain integral at every Gauss node time.

    Parameters
    ----------
    operator : np.ndarray
        Float array of shape ``(p, p)`` mapping nodal temperatures, measured
        from the value carried into the step, onto nodal time derivatives.
    nodal_values : np.ndarray
        Float array of shape ``(p, n)`` holding the temperature at the ``n``
        sample points at each of the ``p`` Gauss node times.
    initial_values : np.ndarray
        Float array of shape ``(n,)`` holding the temperature at the ``n``
        sample points at the start of the step.
    source_values : np.ndarray
        Float array of shape ``(p, n)`` holding the heat source density at the
        ``n`` sample points at each of the ``p`` Gauss node times.
    conductivity : float
        Thermal conductivity of the layer (``conductivity > 0``).

    Returns
    -------
    density : np.ndarray
        Float array of shape ``(p, n)`` holding the density of the domain
        integral of the representation formula at each sample point and each
        Gauss node time.

    Raises
    ------
    ValueError
        If ``operator`` is not a finite real array of shape ``(p, p)`` with
        ``p >= 1``, if ``nodal_values`` or ``source_values`` is not a finite
        real array of shape ``(p, n)`` with ``n >= 1``, if ``initial_values``
        is not a finite real array of shape ``(n,)``, or if ``conductivity`` is
        not a finite real number > 0.

    """
    return density  # placeholder
```

### Step 8

08_reconstruct_interior_value

Goal
----
Close the representation formula at one interior point by combining the assembled boundary quadrature with the assembled domain quadrature.

```python
def reconstruct_interior_value(boundary_terms: "np.ndarray", boundary_fields: "np.ndarray",
                               domain_terms: "np.ndarray", domain_values: "np.ndarray") -> float:
    """Evaluate the representation formula at one interior point.

    Parameters
    ----------
    boundary_terms : np.ndarray
        Float array of shape ``(m, 6)`` gathered over every boundary element of
        the layer, whose columns are the x and y coordinates of the boundary
        quadrature points, the weight that multiplies the normal derivative of
        the temperature there, the weight that multiplies the temperature
        there, and the two components of the outward unit normal.
    boundary_fields : np.ndarray
        Float array of shape ``(m, 3)`` holding the temperature and its two
        spatial derivatives at the same boundary quadrature points.
    domain_terms : np.ndarray
        Float array of shape ``(q, 3)`` gathered over every angular sector of
        the layer, whose columns are the x and y coordinates of the domain
        quadrature points and the weight of the fundamental solution there.
    domain_values : np.ndarray
        Float array of shape ``(q,)`` holding the density of the domain
        integral at the same domain quadrature points.

    Returns
    -------
    value : float
        Temperature at the interior source point implied by the representation
        formula, as a native Python float.

    Raises
    ------
    ValueError
        If ``boundary_terms`` is not a finite real array of shape ``(m, 6)``
        with ``m >= 1``, if ``boundary_fields`` is not a finite real array of
        shape ``(m, 3)``, if ``domain_terms`` is not a finite real array of
        shape ``(q, 3)`` with ``q >= 1``, or if ``domain_values`` is not a
        finite real array of shape ``(q,)``.

    """
    return 0.0  # placeholder
```

### Step 9

09_run_coating_benchmark

Goal
----
Chain the sub-problem functions 01-08 over the coating and the substrate and return the factor by which the near-singular treatment improves the accuracy of the reconstructed temperature.

```python
def run_coating_benchmark(length: float, thickness: float, substrate_depth: float,
                          conductivities: tuple, coating_mesh: tuple, substrate_mesh: tuple,
                          num_nodes: int, time_step: float, num_gauss: int,
                          num_radial: int) -> float:
    """Run the whole two-layer benchmark and return the accuracy gain factor.

    Parameters
    ----------
    length : float
        Extent of both layers along x (``length > 0``).
    thickness : float
        Thickness of the coating, which occupies ``0 <= y <= thickness``
        (``thickness > 0``).
    substrate_depth : float
        Depth of the substrate, which occupies ``-substrate_depth <= y <= 0``
        (``substrate_depth > 0``).
    conductivities : tuple
        Two positive floats, the thermal conductivity of the coating and of the
        substrate.
    coating_mesh : tuple
        Four positive integers for the coating: the number of boundary elements
        on each long face, the number on each short face, the number of
        interior sample columns and the number of interior sample rows.
    substrate_mesh : tuple
        The same four positive integers for the substrate.
    num_nodes : int
        Number of Gauss nodes of the time step (``num_nodes >= 2``).
    time_step : float
        Length of the time step, which starts at time zero (``time_step > 0``).
    num_gauss : int
        Number of Gauss points used on every boundary element, both for the
        boundary integrals and for the circumferential direction of the domain
        integrals (``num_gauss >= 1``).
    num_radial : int
        Number of Gauss points used in the radial direction of the domain
        integrals (``num_radial >= 1``).

    Returns
    -------
    gain : float
        Ratio of the pooled relative error obtained without the near-singular
        treatment to the pooled relative error obtained with it, as a native
        Python float.

    Raises
    ------
    ValueError
        If ``length``, ``thickness``, ``substrate_depth`` or ``time_step`` is
        not a finite real number > 0, if ``conductivities`` is not a pair of
        finite real numbers > 0, if ``coating_mesh`` or ``substrate_mesh`` is
        not a quadruple of integers >= 1, if ``num_nodes`` is not an integer
        >= 2, or if ``num_gauss`` or ``num_radial`` is not an integer >= 1.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-08 (``time_spectral_operator``, ``rectangular_layer_mesh``,
    ``layer_reference_fields``, ``near_singular_rule``,
    ``element_boundary_terms``, ``sct_domain_terms``,
    ``domain_integrand_values``, ``reconstruct_interior_value``) and feed each
    returned value into the next, rather than reimplementing them. The same
    element quadrature rule is used for the boundary integrals and for the
    circumferential direction of the domain integrals of that element. The
    coefficient of y in the benchmark field of each layer is whatever
    continuity of temperature and of normal heat flux across the interface
    at y = 0 demands, normalised so that the substrate carries unity. The
    pooled relative error is the Euclidean norm of the difference between
    the reconstructed and the benchmark temperatures
    over the interior sample points of both layers and all Gauss node times,
    divided by the Euclidean norm of the benchmark temperatures over the same
    set. Include every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder
```
