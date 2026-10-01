# Constraint dominance in a transiently heated layered half-space

## Background

Heat that enters the ground stays there for a long time. A dam pouring hydration heat into its foundation, an energy pile cycling a building's waste heat into the soil around it, a canister of spent fuel warming the rock of a repository, a road embankment thawing the permafrost beneath it - all of these are transient conduction problems in a medium that is layered, that extends without bound, and that responds mechanically to every degree of temperature change. The mechanical response is what engineers actually fear. Free thermal expansion generates no stress at all; stress appears only where the expansion is prevented, and in a layered half-space the prevention comes partly from the surrounding cold material and partly from the neighbouring strata, which expand by different amounts because they have different expansion coefficients and are held back by different stiffnesses.

Modelling this numerically runs into the unbounded domain. Truncating it with a fictitious wall reflects heat that should have escaped and produces spurious restraint against the deformation that should have been free, and pushing the wall far enough away to be harmless multiplies the number of unknowns in a region nobody cares about. The scaled boundary finite element method removes the wall instead of moving it. It discretises only a surface, treats the remaining coordinate semi-analytically, and condenses the entire exterior into a small dense operator acting on that surface - a matrix that relates the temperatures there to the heat flowing out to infinity, with no elements beyond it at all. Classical formulations scale the discretised surface about a single point, which forces the exterior to be a cone through that point and forces the geometric Jacobian to factor into a radial part times a surface part. Scaling instead between two similar surfaces removes that factorisation, and the coefficient matrices that were constants become functions of the radial coordinate, which is the price paid for being able to represent exteriors whose shape is not star-like about any one point.

Whichever scaling is used, the exterior operator is obtained the same way: the semi-analytical radial equation collapses into a matrix quadratic for the surface operator, whose roots are read off an eigen-decomposition of the associated Hamiltonian matrix. That quadratic has more than one root and they are not equally physical, so the decomposition settles nothing on its own. Adding that operator to the finite element matrices of the near field leaves a system of ordinary differential equations in time whose order is that of the near field alone. Advancing it is then a matter of the matrix exponential, and the precise integration method computes that exponential essentially to machine precision by repeated squaring of a very finely subdivided step, an arrangement whose numerical success turns on details of how the squaring is arranged. The exponential propagates the homogeneous solution alone; the particular integral against a time-varying load has to be obtained separately.

The physical question the layered half-space poses is which property of the stratification controls what. Thermal conductivity and volumetric heat capacity govern the thermal side: together they set the diffusivity and so the depth the thermal front reaches, while the conductivity on its own is what relates a temperature gradient to a flux, and which of the two the near-surface gradient ends up following is a question the calculation has to settle rather than assume. Young's modulus, Poisson's ratio and the expansion coefficient set how much stress a given temperature rise generates in a restrained layer. These two groups are independent, so the ordering of stratifications by thermal gradient and the ordering by peak stress need not agree, and if they disagree the common engineering shortcut of reading stress severity off a temperature contour plot is unsafe. Deciding the matter requires running the whole coupled calculation for several stratifications and comparing the spread of one quantity against the spread of the other.

## Problem

Sequentially coupled thermo-elastic analyses of layered semi-infinite media are usually reported as contour plots, and the accompanying claim - that the stress field is not proportional to the thermal gradient the layering establishes - is asserted rather than measured. Measure it on the testbed below by permuting the stratification and comparing how far the peak thermal stress moves against how far the near-surface thermal gradient moves. Four isotropic materials are available, each given as (thermal conductivity in W m$^{-1}$ K$^{-1}$, density in kg m$^{-3}$, Young's modulus in Pa, Poisson's ratio, coefficient of thermal expansion in K$^{-1}$, specific heat capacity in J kg$^{-1}$ K$^{-1}$): material 1 is $(50,\,1000,\,10\times10^{6},\,0.30,\,1\times10^{-5},\,10)$, material 2 is $(66,\,1500,\,15\times10^{6},\,0.35,\,5\times10^{-5},\,15)$, material 3 is $(89,\,2000,\,20\times10^{6},\,0.40,\,1\times10^{-6},\,20)$ and material 4 is $(100,\,2500,\,25\times10^{6},\,0.45,\,5\times10^{-6},\,25)$. The three stratification orders to be compared, listed from the free surface downwards, are $(1,2,3,4)$, $(2,3,1,4)$ and $(4,3,2,1)$.

The near field is the box $0 \le x, y \le 120$ m and $0 \le z \le 40$ m with $z$ measured downwards from the free surface, split into four strata 10 m thick and meshed with $4\times4\times8$ trilinear eight-node hexahedra integrated at $2\times2\times2$ Gauss points, giving consistent conductance and capacity operators; nodes are numbered $i + 5(j + 5k)$ over the $x$, $y$ and $z$ grid indices and the mechanical degrees of freedom at node $a$ are $3a$, $3a+1$, $3a+2$. The central $60\ \mathrm{m}\times60\ \mathrm{m}$ of the free surface exchanges heat with an ambient temperature $80\left(1 - e^{-t/10^{5}}\right)\,^{\circ}$C through a film coefficient of 25 W m$^{-2}$ K$^{-1}$, the surface terms being integrated at $2\times2$ Gauss points on the square faces the mesh induces; the rest of the free surface and the four vertical faces are adiabatic and the medium starts at $0\,^{\circ}$C throughout.

The base $z = 40$ m carries the steady-state conduction matrix of the unbounded far field below it, built by the framework's scaling-surface formulation: the boundary surface is the base itself with the $4\times4$ bilinear four-node elements the mesh induces there, the scaling surface is that same square contracted by one half about the vertical axis through the box centre and laid on the plane $z=0$, the far-field medium is the bottom stratum, and the coefficient matrices are frozen at radial coordinate $\xi = 1$; this steady operator is the zero-frequency limit of the framework's transient far-field operator. That far-field operator is a matrix in W K$^{-1}$ over the 25 base nodes, mapping the interface temperatures to the heat the near field loses downwards through the base, and it is added to the near-field conductance on those degrees of freedom. The temperature field is advanced to $t = 2\times10^{5}$ s by the framework's precise time-step integration using a step of 2500 s, $2^{20}$ subdivisions per step and a six-point Gauss-Legendre rule for the load term. The mechanical response is one-way coupled to that temperature field on the same mesh with the base fully restrained, the four vertical faces on normal rollers and the free surface unloaded, the stress-free reference state being $0\,^{\circ}$C. For each stratification order, record the largest von Mises stress over the element centroids at $t = 2\times10^{5}$ s, and the largest magnitude of the depth derivative of temperature over the centroids of the elements of the uppermost stratum. Your final answer must be a single number: the ratio of the largest to the smallest peak stress across the three orders divided by the ratio of the largest to the smallest of those gradients; report compactly, but do report which method choices the calculation turned on and the intermediate quantities the ratio rests on, not the final number alone.

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

01_assemble_near_field_operators

Goal
----
Assemble the conductance and capacity operators of the layered near field, including the convective surface term of the heated patch.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_near_field_operators(order, width: float, depth: float, patch_width: float,
                                  n_lateral: int, n_depth: int,
                                  film_coefficient: float) -> np.ndarray:
    """Assemble the thermal operators of the layered near field.

    Parameters
    ----------
    order : sequence of int
        The four material numbers, from the free surface downwards; entries lie
        between 1 and 4 and may repeat.
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    patch_width : float
        Side length of the centred square heated patch in metres
        (patch_width >= 0). The patch is resolved by the mesh: a top-surface
        element face carries the convective term when the whole face lies
        inside the patch and carries none otherwise, so a face is never
        partially heated.
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    film_coefficient : float
        Convective film coefficient over the patch in W m^-2 K^-1
        (film_coefficient >= 0).

    Returns
    -------
    operators : np.ndarray
        Array of shape (2, n_nodes, n_nodes); entry 0 is the conductance
        operator including the convective surface term and entry 1 is the
        consistent capacity operator, both in SI units.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return operators  # placeholder
```

### Step 2

02_assemble_patch_load_vector

Goal
----
Assemble the consistent nodal vector through which the convective patch drives the near field, before it is scaled by the ambient temperature history.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_patch_load_vector(width: float, depth: float, patch_width: float,
                               n_lateral: int, n_depth: int,
                               film_coefficient: float) -> np.ndarray:
    """Assemble the spatial part of the convective load over the heated patch.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    patch_width : float
        Side length of the centred square heated patch in metres
        (patch_width >= 0). The patch is resolved by the mesh: a top-surface
        element face carries the convective term when the whole face lies
        inside the patch and carries none otherwise, so a face is never
        partially heated.
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    film_coefficient : float
        Convective film coefficient over the patch in W m^-2 K^-1
        (film_coefficient >= 0).

    Returns
    -------
    load : np.ndarray
        Vector of length n_nodes holding the film coefficient times the
        integral of each shape function over the heated patch, in W K^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return load  # placeholder
```

### Step 3

03_assemble_scaling_surface_coefficients

Goal
----
Assemble the three radial coefficient matrices of the unbounded far field from the scaling-surface coordinate map, frozen at a prescribed radial coordinate.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_scaling_surface_coefficients(width: float, depth: float, n_lateral: int,
                                          far_conductivity: float, contraction: float,
                                          radial_coordinate: float) -> np.ndarray:
    """Assemble the radial coefficient matrices of the scaling-surface far field.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth at which the boundary surface sits, in metres (depth > 0).
    n_lateral : int
        Number of boundary-surface elements along each lateral direction
        (n_lateral >= 1).
    far_conductivity : float
        Isotropic thermal conductivity of the far-field medium in
        W m^-1 K^-1 (far_conductivity > 0).
    contraction : float
        Ratio by which the boundary square is contracted about the vertical
        centre axis to form the scaling surface, 0 <= contraction < 1.
    radial_coordinate : float
        Radial coordinate at which the coefficient matrices are frozen
        (radial_coordinate > 0).

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (3, m, m) with m = (n_lateral + 1) ** 2, holding the
        rescaled radial-derivative matrix, the cross matrix and the
        circumferential matrix in that order, in SI units.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the two
        surfaces do not define a diverging exterior.
    """
    return coefficients  # placeholder
```

### Step 4

04_solve_far_field_conduction_matrix

Goal
----
Solve the matrix quadratic of the scaling-surface far field for the steady-state conduction matrix that condenses the whole exterior onto the interface.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_far_field_conduction_matrix(radial_matrix: np.ndarray, cross_matrix: np.ndarray,
                                      circumferential_matrix: np.ndarray) -> np.ndarray:
    """Solve for the steady-state conduction matrix of the unbounded far field.

    Parameters
    ----------
    radial_matrix : np.ndarray
        Rescaled radial-derivative coefficient matrix of shape (m, m),
        symmetric positive definite.
    cross_matrix : np.ndarray
        Cross coefficient matrix of shape (m, m); it is not symmetric.
    circumferential_matrix : np.ndarray
        Rescaled circumferential coefficient matrix of shape (m, m),
        symmetric positive semi-definite.

    Returns
    -------
    conduction : np.ndarray
        Symmetric positive definite matrix of shape (m, m) relating the
        interface temperatures to the steady heat flow into the far field,
        in W K^-1.

    Raises
    ------
    ValueError
        Admissibility of the coefficients is the routine's duty to check
        rather than a property it may assume. The argument descriptions
        above are therefore requirements, and this is raised if any matrix
        is not square, if the three do not share one order, if any entry is
        not finite, if ``radial_matrix`` is singular or is not positive
        definite, or if the coefficients admit no decaying half-spectrum.
    """
    return conduction  # placeholder
```

### Step 5

05_compute_precise_propagator

Goal
----
Build the propagator of the semi-discrete thermal system over one interval by the precise integration algorithm, with the increment held apart from the identity.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_precise_propagator(state_matrix: np.ndarray, interval: float,
                               n_levels: int) -> np.ndarray:
    """Build the propagator of a linear first-order system over one interval.

    Parameters
    ----------
    state_matrix : np.ndarray
        Square array of shape (n, n) holding the state matrix of the
        semi-discrete system, in s^-1.
    interval : float
        Length of the interval to propagate over, in seconds
        (interval >= 0).
    n_levels : int
        Number of subdivision levels, so the interval is split into
        2 ** n_levels parts (n_levels >= 0).

    Returns
    -------
    propagator : np.ndarray
        Array of shape (n, n) holding the propagator of the system over the
        given interval.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return propagator  # placeholder
```

### Step 6

06_integrate_temperature_field

Goal
----
March the semi-discrete thermal system from a uniform initial state to the end of the analysis, combining the exact propagators supplied by sub-problem 05 with a quadratured particular integral.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def integrate_temperature_field(step_propagator: np.ndarray, tail_propagators: list,
                                load_influence: np.ndarray, ambient_amplitude: float,
                                ramp_time: float, step: float, n_steps: int) -> np.ndarray:
    """March the semi-discrete thermal system to the end of the analysis.

    The load acting on the system at time t is ``load_influence`` scaled by the
    ambient history ``ambient_amplitude * (1 - exp(-t / ramp_time))``, and the
    field starts from zero everywhere at t = 0.

    Every propagator the march needs is supplied by the caller, which obtains it
    from ``compute_precise_propagator`` of sub-problem 05; none is built here.
    The order of the Gauss-Legendre rule is the length of ``tail_propagators``,
    and its abscissae and weights are those of
    ``numpy.polynomial.legendre.leggauss(len(tail_propagators))``, so entry i of
    ``tail_propagators`` belongs to abscissa i of that rule in the order the
    routine returns them.

    Parameters
    ----------
    step_propagator : np.ndarray
        Square array of shape (n, n) holding the propagator of the system over
        a whole step.
    tail_propagators : list
        Sequence of at least one array, each of shape (n, n), holding the
        propagator over the time remaining from a quadrature abscissa of the
        step to the end of that step, one entry per abscissa in the order
        described above.
    load_influence : np.ndarray
        Vector of length n giving the response rate per unit ambient
        temperature, in K s^-1 K^-1.
    ambient_amplitude : float
        Final ambient temperature of the saturating history, in degrees
        Celsius.
    ramp_time : float
        Time constant of the ambient history in seconds (ramp_time > 0).
    step : float
        Time step in seconds (step > 0), the same step the supplied
        propagators were built over.
    n_steps : int
        Number of steps to march (n_steps >= 0).

    Returns
    -------
    temperature : np.ndarray
        Vector of length n holding the nodal temperatures at the end of the
        analysis, in degrees Celsius.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return temperature  # placeholder
```

### Step 7

07_solve_thermoelastic_displacement

Goal
----
Solve the one-way coupled elastic problem driven by a given temperature field, with the base restrained and the vertical faces on normal rollers.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_thermoelastic_displacement(order, width: float, depth: float, n_lateral: int,
                                     n_depth: int, temperature: np.ndarray) -> np.ndarray:
    """Solve the restrained elastic problem driven by a temperature field.

    Parameters
    ----------
    order : sequence of int
        The four material numbers, from the free surface downwards; entries lie
        between 1 and 4 and may repeat.
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    temperature : np.ndarray
        Vector of length n_nodes holding the nodal temperature rise above the
        stress-free reference state, in kelvin.

    Returns
    -------
    displacement : np.ndarray
        Vector of length 3 * n_nodes holding the nodal displacements in
        metres, ordered x, y, z at each node.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return displacement  # placeholder
```

### Step 8

08_compute_peak_stress_and_heave

Goal
----
Recover the element-centroid stresses from the displacement and temperature fields and report the peak equivalent stress together with the peak heave of the free surface.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_peak_stress_and_heave(order, width: float, depth: float, n_lateral: int,
                                  n_depth: int, temperature: np.ndarray,
                                  displacement: np.ndarray) -> np.ndarray:
    """Report the peak equivalent stress and the peak heave of the free surface.

    Parameters
    ----------
    order : sequence of int
        The four material numbers, from the free surface downwards; entries lie
        between 1 and 4 and may repeat.
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    temperature : np.ndarray
        Vector of length n_nodes holding the nodal temperature rise above the
        stress-free reference state, in kelvin.
    displacement : np.ndarray
        Vector of length 3 * n_nodes holding the nodal displacements in metres,
        ordered x, y, z at each node.

    Returns
    -------
    measures : np.ndarray
        Array of two floats: the largest von Mises stress over the element
        centroids in pascals, and the largest magnitude of the vertical
        displacement over the free-surface nodes in metres.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return measures  # placeholder
```

### Step 9

09_compute_surface_gradient

Goal
----
Measure the steepness of the thermal front in the uppermost stratum as the largest magnitude of the depth derivative of temperature over the centroids of that stratum's elements.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_surface_gradient(width: float, depth: float, n_lateral: int, n_depth: int,
                             temperature: np.ndarray) -> float:
    """Measure the steepest depth gradient of temperature in the uppermost stratum.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    temperature : np.ndarray
        Vector of length n_nodes holding the nodal temperatures.

    Returns
    -------
    gradient : float
        Largest magnitude of the depth derivative of temperature over the
        centroids of the uppermost stratum's elements, in K m^-1, as a native
        Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return gradient  # placeholder
```

### Step 10

10_run_constraint_dominance_pipeline

Goal
----
Chain the sub-problem functions 01-09 end-to-end over the three stratification orders and return the constraint dominance index of the layered half-space. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (assemble_near_field_operators, assemble_patch_load_vector, assemble_scaling_surface_coefficients, solve_far_field_conduction_matrix, compute_precise_propagator, integrate_temperature_field, solve_thermoelastic_displacement, compute_peak_stress_and_heave, compute_surface_gradient) rather than reimplementing them.

```python
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_constraint_dominance_pipeline(width: float = 120.0, depth: float = 40.0,
                                      patch_width: float = 60.0, n_lateral: int = 4,
                                      n_depth: int = 8, film_coefficient: float = 25.0,
                                      ambient_amplitude: float = 80.0,
                                      ramp_time: float = 1.0e5, step: float = 2500.0,
                                      n_steps: int = 80, n_gauss: int = 6,
                                      n_levels: int = 20, contraction: float = 0.5,
                                      radial_coordinate: float = 1.0) -> float:
    """Run the constraint dominance measurement over the three stratification orders.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    patch_width : float
        Side length of the centred square heated patch in metres
        (patch_width >= 0). The patch is resolved by the mesh, so a
        top-surface element face is either wholly heated or not heated.
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    film_coefficient : float
        Convective film coefficient over the patch in W m^-2 K^-1
        (film_coefficient >= 0).
    ambient_amplitude : float
        Final ambient temperature of the saturating history, in degrees
        Celsius.
    ramp_time : float
        Time constant of the ambient history in seconds (ramp_time > 0).
    step : float
        Time step in seconds (step > 0).
    n_steps : int
        Number of steps to march (n_steps >= 0).
    n_gauss : int
        Number of Gauss-Legendre points used for the load integral over each
        step (n_gauss >= 1).
    n_levels : int
        Number of subdivision levels used to build each propagator
        (n_levels >= 0).
    contraction : float
        Ratio by which the boundary square is contracted to form the scaling
        surface, 0 <= contraction < 1.
    radial_coordinate : float
        Radial coordinate at which the far-field coefficient matrices are
        frozen (radial_coordinate > 0).

    Returns
    -------
    index : float
        The spread of the peak von Mises stress across the three
        stratification orders divided by the spread of the near-surface
        thermal gradient, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return index  # placeholder
```
