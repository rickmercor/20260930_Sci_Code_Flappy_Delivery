# Physics-Computational_Physics-6

## Background

Thermal metamaterials are engineered materials that control heat flow through spatially tailored thermal properties. In transient heat conduction, both thermal conductivity and volumetric heat capacity influence how the temperature field evolves with time. Designs optimized for a particular imposed thermal field may reproduce a desired response under that loading while behaving differently for other incident directions. Anisotropic material realizations provide additional directional control, while local material orientation and thermal-storage properties can strongly influence transient performance.

## Problem

A two-dimensional thermal metamaterial is constructed from a solved isotropic unidirectional diffusion design. Use the field-derived construction in the source paper to obtain the local ideal omnidirectional material properties required by the solved design data; any source-specific constitutive or transient relations needed for the calculation must be inferred from that construction.

The homogeneous background has thermal conductivity

$$
\kappa_0=1.000\ \mathrm{W\,m^{-1}K^{-1}},
$$

and design-state temperature gradient

$$
\mathbf G_0=
\begin{pmatrix}
1\\
0
\end{pmatrix}
\ \mathrm{K\,m^{-1}}.
$$

The background volumetric heat capacity $\rho_0c_0$ was not recorded. At three representative cells, the solved unidirectional design provides

$$
\begin{array}{c|c|c|c|c}
i&w_i&\kappa_{P,i}&G_{x,i}&G_{y,i}\\
\hline
1&0.25&0.22&0.34&0.79\\
2&0.35&1.85&0.72&0.31\\
3&0.40&4.60&0.46&-0.88
\end{array}.
$$

Fabrication changes only the orientation of the paper-derived local conductivity tensor: in cell $i$, its material basis is rotated by an unknown signed offset $\delta_i$ relative to the ideal orientation, while its principal conductivity values remain unchanged. A positive $\delta_i$ denotes an active counterclockwise rotation of the material basis, and all stated angular values are in degrees. The unknown parameters satisfy

$$
-20^\circ\le\delta_1\le0^\circ,\qquad 3^\circ\le\delta_2\le22^\circ,\qquad -28^\circ\le\delta_3\le-8^\circ,
$$

and

$$
1000\le\rho_0c_0\le1400\ \mathrm{J\,m^{-3}K^{-1}}.
$$

Four transient calibration experiments give the local temperature-Hessian components

$$
\begin{array}{c|c|r|r|r}
m&i&T_{xx,i}^{(m)}&T_{xy,i}^{(m)}&T_{yy,i}^{(m)}\\
\hline
A&1&120&-250&-70\\
A&2&-180&90&260\\
A&3&75&310&-140\\
B&1&-210&160&95\\
B&2&140&-280&-130\\
B&3&260&85&-190\\
C&1&55&330&-180\\
C&2&-90&-210&240\\
C&3&-175&270&130\\
D&1&-130&-280&210\\
D&2&190&240&-160\\
D&3&-220&-150&280
\end{array},
$$

in $\mathrm{K\,m^{-2}}$, and the measured area-weighted instantaneous temperature-rate responses are

$$
\mathcal R_A=1.588634201628,\qquad \mathcal R_B=-1.692631704136,
$$

$$
\mathcal R_C=-1.727994022560,\qquad \mathcal R_D=1.377882682405
\quad\mathrm{K\,s^{-1}}.
$$

Over the stated bounds, the transient calibration measurements can admit more than one parameter set. An independent microscopy measurement of cell 2 shows that the physical laminate strips make an angle $\alpha_2$ satisfying

$$
26^\circ<\alpha_2<30^\circ
$$

counterclockwise from the positive $x$ direction. Use the microscopy observation as an independent physical admissibility constraint on the bounded calibration solutions, interpreting the observed laminate orientation consistently with the source paper.

Treat each cell as locally uniform and use the transient heat-conduction law with the material properties obtained from the source-paper construction. Determine the bounded parameter sets that reproduce all four calibration responses, use the microscopy observation to select the physically admissible branch, and then hold that branch fixed when evaluating the independent prediction state

$$
\begin{array}{c|r|r|r}
i&T_{xx,i}^{(\mathrm{pred})}&T_{xy,i}^{(\mathrm{pred})}&T_{yy,i}^{(\mathrm{pred})}\\
\hline
1&-331&-49&437\\
2&403&230&240\\
3&370&-192&488
\end{array}.
$$

In the reasoning, report the inferred fabrication offsets and background volumetric heat capacity before calculating the predicted area-weighted response $\mathcal R_{\mathrm{pred}}$.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_recover_field_geometry.py

Goal
----
Recover the local geometric information used by the field-derived transformation. For each nonzero design-state temperature gradient, calculate its magnitude, the reference-to-local gradient scale factor, and the rotation matrix whose first column is aligned with the local gradient direction.

```python
def recover_field_geometry(
    g0: "np.ndarray",
    gradients: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Recover local field geometry from reference and design-state gradients.

    Parameters
    ----------
    g0 : np.ndarray
        Reference temperature-gradient vector with shape (2,).
    gradients : np.ndarray
        Local temperature-gradient vectors with shape (N, 2).

    Returns
    -------
    magnitudes : np.ndarray
        Euclidean magnitudes of the local gradients with shape (N,).
    scale_factors : np.ndarray
        Reference-to-local gradient scale factors with shape (N,).
    rotations : np.ndarray
        Gradient-aligned rotation matrices with shape (N, 2, 2).

    Raises
    ------
    ValueError
        If the reference gradient or any local gradient contains a non-finite
        component, or if the reference gradient or any local gradient has
        zero magnitude.
    """
    return magnitudes, scale_factors, rotations
```

### Step 2

02_build_dual_jacobian.py

Goal
----
Construct the local geometric transformation used by the field-derived thermal-metamaterial method. For each cell, combine the recovered scale factor and gradient-aligned rotation with the conductivity-dependent transverse stretch, then calculate the determinant of the local Jacobian.

```python
def build_dual_jacobian(
    scale_factors: "np.ndarray",
    rotations: "np.ndarray",
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Construct the local stretching matrices and field-derived Jacobians.

    Parameters
    ----------
    scale_factors : np.ndarray
        Positive local geometric scale factors with shape (N,).
    rotations : np.ndarray
        Gradient-aligned rotation matrices with shape (N, 2, 2).
    kappa_0 : float
        Positive background thermal conductivity.
    kappa_p : np.ndarray
        Positive local isotropic conductivities with shape (N,).

    Returns
    -------
    stretches : np.ndarray
        Local stretching matrices with shape (N, 2, 2).
    jacobians : np.ndarray
        Local Jacobian matrices with shape (N, 2, 2).
    determinants : np.ndarray
        Local Jacobian determinants with shape (N,).

    Raises
    ------
    ValueError
        If `scale_factors`, `kappa_0`, or `kappa_p` contains a non-finite or
        non-positive value, or if any constructed Jacobian has a non-finite
        or non-positive determinant.
    """
    return stretches, jacobians, determinants
```

### Step 3

03_compute_principal_conductivities.py

Goal
----
Calculate the two principal thermal conductivities required by the omnidirectional counterpart of each local unidirectional design. Keep the conductivity along the design-state temperature-gradient direction equal to the physical-equivalence conductivity and calculate the orthogonal principal conductivity from the field-derived construction.

```python
def compute_principal_conductivities(
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "np.ndarray":
    """
    Calculate the principal conductivities of the omnidirectional cells.

    Parameters
    ----------
    kappa_0 : float
        Positive background thermal conductivity.
    kappa_p : np.ndarray
        Positive local physical-equivalence conductivities with shape (N,).

    Returns
    -------
    principal_conductivities : np.ndarray
        Principal conductivity pairs with shape (N, 2), where column 0
        contains the conductivity parallel to the local design-state
        gradient and column 1 contains the orthogonal conductivity.

    Raises
    ------
    ValueError
        If `kappa_0` is non-finite or non-positive, or if `kappa_p` contains
        any non-finite or non-positive value.
    """
    return principal_conductivities
```

### Step 4

04_construct_conductivity_tensors.py

Goal
----
Construct the anisotropic thermal-conductivity tensor of each omnidirectional cell. Rotate the two principal conductivities from the local gradient-aligned basis into the common coordinate basis using the rotation recovered from the design-state temperature gradient.

```python
def construct_conductivity_tensors(
    rotations: "np.ndarray",
    principal_conductivities: "np.ndarray",
) -> "np.ndarray":
    """
    Rotate local principal conductivities into the common coordinate basis.

    Parameters
    ----------
    rotations : np.ndarray
        Gradient-aligned rotation matrices with shape (N, 2, 2).
    principal_conductivities : np.ndarray
        Positive principal conductivity pairs with shape (N, 2).
        Column 0 contains the conductivity parallel to the local
        design-state gradient and column 1 contains the orthogonal
        conductivity.

    Returns
    -------
    conductivity_tensors : np.ndarray
        Rotated anisotropic conductivity tensors with shape (N, 2, 2).

    Raises
    ------
    ValueError
        If any principal conductivity is non-finite or non-positive, or if
        any supplied rotation matrix is not orthogonal with determinant +1.
    """
    return conductivity_tensors
```

### Step 5

05_transform_heat_capacity.py

Goal
----
Calculate the transformed volumetric heat capacity required for transient conduction in each omnidirectional cell. Use the determinant of the recovered local Jacobian to transform the background thermal-storage property.

```python
def transform_heat_capacity(
    rho_c0: float,
    determinants: "np.ndarray",
) -> "np.ndarray":
    """
    Transform the background volumetric heat capacity in each cell.

    Parameters
    ----------
    rho_c0 : float
        Positive background volumetric heat capacity.
    determinants : np.ndarray
        Positive local Jacobian determinants with shape (N,).

    Returns
    -------
    heat_capacities : np.ndarray
        Transformed volumetric heat capacities with shape (N,).

    Raises
    ------
    ValueError
        If `rho_c0` is non-finite or non-positive, or if `determinants`
        contains any non-finite or non-positive value.
    """
    return heat_capacities
```

### Step 6

06_evaluate_offset_response.py

Goal
----
Apply the specified fabrication-axis offsets to the ideal local anisotropic conductivity tensors and evaluate the transient response for one or more Hessian states. Rotate each ideal tensor using the signed counterclockwise offset, contract the realized tensor with the local temperature Hessian including the mixed-derivative term, divide by the transformed volumetric heat capacity, and combine the local rates using the prescribed area weights.

```python
def evaluate_offset_response(
    ideal_conductivity_tensors: "np.ndarray",
    heat_capacities: "np.ndarray",
    weights: "np.ndarray",
    hessian_components: "np.ndarray",
    offsets_deg: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Apply fabrication offsets and evaluate transient responses.

    Parameters
    ----------
    ideal_conductivity_tensors : np.ndarray
        Symmetric ideal conductivity tensors with shape (N, 2, 2).
    heat_capacities : np.ndarray
        Positive transformed volumetric heat capacities with shape (N,).
    weights : np.ndarray
        Nonnegative cell weights with shape (N,) that sum to one.
    hessian_components : np.ndarray
        Hessian components with shape (M, N, 3), ordered as
        T_xx, T_xy, and T_yy for M transient states.
    offsets_deg : np.ndarray
        Signed counterclockwise fabrication offsets in degrees with shape (N,).

    Returns
    -------
    realized_tensors : np.ndarray
        Fabrication-adjusted conductivity tensors with shape (N, 2, 2).
    temperature_rates : np.ndarray
        Local transient temperature rates with shape (M, N).
    responses : np.ndarray
        Area-weighted transient responses with shape (M,).
    """
    return realized_tensors, temperature_rates, responses
```

### Step 7

07_infer_calibration_parameters.py

Goal
----
Recover all distinct fabrication-offset and background volumetric heat-capacity parameter sets that reproduce the measured transient calibration responses within the prescribed bounds. Solve the coupled bounded nonlinear inverse problem using one shared parameter vector across all calibration experiments, retain every distinct solution whose maximum absolute response residual is below the numerical tolerance, and return the solutions in deterministic lexicographic order.

```python
def infer_calibration_parameters(
    ideal_conductivity_tensors: "np.ndarray",
    determinants: "np.ndarray",
    weights: "np.ndarray",
    calibration_hessians: "np.ndarray",
    measured_responses: "np.ndarray",
    offset_bounds_deg: "np.ndarray",
    rho_c0_bounds: "np.ndarray",
) -> "np.ndarray":
    """
    Recover all distinct bounded calibration parameter sets.

    Parameters
    ----------
    ideal_conductivity_tensors : np.ndarray
        Symmetric ideal conductivity tensors with shape (N, 2, 2).
    determinants : np.ndarray
        Positive local Jacobian determinants with shape (N,).
    weights : np.ndarray
        Nonnegative cell weights with shape (N,) that sum to one.
    calibration_hessians : np.ndarray
        Calibration Hessian components with shape (M, N, 3), ordered as
        T_xx, T_xy, and T_yy.
    measured_responses : np.ndarray
        Measured area-weighted calibration responses with shape (M,).
    offset_bounds_deg : np.ndarray
        Lower and upper fabrication-offset bounds in degrees with shape
        (N, 2).
    rho_c0_bounds : np.ndarray
        Lower and upper bounds for the background volumetric heat capacity
        with shape (2,).

    Returns
    -------
    parameter_sets : np.ndarray
        Distinct bounded calibration solutions with shape (K, N + 1).
        Each row contains the N fabrication offsets followed by rho_c0.
        The rows are returned in deterministic lexicographic order.

    Raises
    ------
    ValueError
        If the inputs are invalid or if no bounded parameter set reproduces
        the calibration measurements.
    """
    return parameter_sets
```

### Step 8

08_run_inverse_duality_pipeline.py

Goal
----
Run the complete inverse physical-geometric duality pipeline. Recover the ideal field-derived geometry and conductivity tensors, recover all bounded calibration parameter sets, use the independent laminate-orientation measurement together with the source-paper realization convention to select the physically admissible branch, construct the transformed heat capacities for that branch, and evaluate the independent prediction state without refitting.

```python
def run_inverse_duality_pipeline(
    gradients: "np.ndarray",
    reference_gradient: "np.ndarray",
    kappa_p: "np.ndarray",
    kappa_0: float,
    weights: "np.ndarray",
    calibration_hessians: "np.ndarray",
    measured_responses: "np.ndarray",
    offset_bounds_deg: "np.ndarray",
    rho_c0_bounds: "np.ndarray",
    microscopy_cell_index: int,
    laminate_angle_bounds_deg: "np.ndarray",
    prediction_hessians: "np.ndarray",
) -> float:
    """
    Run the complete inverse duality calibration and prediction pipeline.

    Parameters
    ----------
    gradients : np.ndarray
        Local design-state temperature gradients with shape (N, 2).
    reference_gradient : np.ndarray
        Background design-state temperature gradient with shape (2,).
    kappa_p : np.ndarray
        Positive isotropic unidirectional conductivities with shape (N,).
    kappa_0 : float
        Positive background thermal conductivity.
    weights : np.ndarray
        Nonnegative cell weights with shape (N,) that sum to one.
    calibration_hessians : np.ndarray
        Calibration Hessian components with shape (M, N, 3), ordered as
        T_xx, T_xy, and T_yy.
    measured_responses : np.ndarray
        Measured area-weighted calibration responses with shape (M,).
    offset_bounds_deg : np.ndarray
        Lower and upper fabrication-offset bounds in degrees with shape
        (N, 2).
    rho_c0_bounds : np.ndarray
        Lower and upper background volumetric heat-capacity bounds with
        shape (2,).
    microscopy_cell_index : int
        Zero-based index of the cell containing the independent laminate
        orientation observation.
    laminate_angle_bounds_deg : np.ndarray
        Open lower and upper bounds of the measured absolute laminate
        orientation in degrees, with shape (2,).
    prediction_hessians : np.ndarray
        Prediction-state Hessian components with shape (N, 3), ordered as
        T_xx, T_xy, and T_yy.

    Returns
    -------
    response : float
        Area-weighted instantaneous temperature-rate response for the
        independent prediction state.
    """
    return response
```
