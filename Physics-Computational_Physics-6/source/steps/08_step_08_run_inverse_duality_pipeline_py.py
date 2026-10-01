"""
Run the complete inverse physical-geometric duality pipeline. Recover the ideal field-derived geometry and conductivity tensors, recover all bounded calibration parameter sets, use the independent laminate-orientation measurement together with the source-paper realization convention to select the physically admissible branch, construct the transformed heat capacities for that branch, and evaluate the independent prediction state without refitting.

The final computation combines the field-derived transformation with the task-specific inverse calibration and physical branch selection. The calibration step returns all distinct bounded parameter vectors



$$

\mathbf p^{(k)}=

\begin{pmatrix}

\delta_1^{(k)}\\

\vdots\\

\delta_N^{(k)}\\

\rho_0c_0^{(k)}

\end{pmatrix},

$$



that reproduce the measured transient responses.



The independent microscopy observation is then used to select the physical branch. For the observed cell $j$, the ideal field direction is



$$

\gamma_j=\operatorname{atan2}(G_{y,j},G_{x,j}).

$$



For the source-paper realization considered here, when



$$

\kappa_{P,j}>\kappa_0,

$$



the laminate direction is parallel to the local streamline direction. The realized laminate angle associated with calibration branch $k$ is therefore



$$

\alpha_j^{(k)}=\operatorname{mod}\left(\gamma_j+\delta_j^{(k)},180^\circ\right).

$$



The physically admissible branch is the unique branch satisfying the independent microscopy interval



$$

\alpha_{\min}<\alpha_j^{(k)}<\alpha_{\max}.

$$



After selecting that branch, its background volumetric heat capacity determines the transformed local heat capacities through



$$

(\rho c)_i=\frac{\rho_0c_0}{\det\boldsymbol{\Lambda}_i},

$$



while its fabrication offsets determine the realized conductivity tensors



$$

\widetilde{\boldsymbol{\kappa}}_i=\mathbf Q(\delta_i)\boldsymbol{\kappa}_{D,i}\mathbf Q(\delta_i)^T.

$$



For the independent prediction Hessian, the local instantaneous temperature rate is



$$

\dot T_i^{(\mathrm{pred})}=\frac{\widetilde{\kappa}_{xx,i}T_{xx,i}^{(\mathrm{pred})}+2\widetilde{\kappa}_{xy,i}T_{xy,i}^{(\mathrm{pred})}+\widetilde{\kappa}_{yy,i}T_{yy,i}^{(\mathrm{pred})}}{(\rho c)_i}.

$$



The final area-weighted prediction is



$$

\mathcal R_{\mathrm{pred}}=\sum_{i=1}^{N} w_i\dot T_i^{(\mathrm{pred})}.

$$

Returns
-------
float, the area-weighted instantaneous temperature-rate response for the independent prediction state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_inverse_duality_pipeline(
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
    """Reference implementation of the complete inverse duality pipeline."""
    gradients = np.asarray(
        gradients,
        dtype=float,
    )
    reference_gradient = np.asarray(
        reference_gradient,
        dtype=float,
    )
    kappa_p = np.asarray(
        kappa_p,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )
    calibration_hessians = np.asarray(
        calibration_hessians,
        dtype=float,
    )
    measured_responses = np.asarray(
        measured_responses,
        dtype=float,
    )
    offset_bounds_deg = np.asarray(
        offset_bounds_deg,
        dtype=float,
    )
    rho_c0_bounds = np.asarray(
        rho_c0_bounds,
        dtype=float,
    )
    laminate_angle_bounds_deg = np.asarray(
        laminate_angle_bounds_deg,
        dtype=float,
    )
    prediction_hessians = np.asarray(
        prediction_hessians,
        dtype=float,
    )

    if reference_gradient.shape != (2,):
        raise ValueError(
            "reference_gradient must have shape (2,)."
        )

    if not np.all(np.isfinite(reference_gradient)):
        raise ValueError(
            "reference_gradient must contain only finite values."
        )

    if np.linalg.norm(reference_gradient) <= 0.0:
        raise ValueError(
            "reference_gradient must have positive magnitude."
        )

    magnitudes, scale_factors, rotations = (
        _oracle_recover_field_geometry(
            g0=reference_gradient,
            gradients=gradients,
        )
    )

    stretching_matrices, jacobians, determinants = (
        _oracle_build_dual_jacobian(
            scale_factors=scale_factors,
            rotations=rotations,
            kappa_0=kappa_0,
            kappa_p=kappa_p,
        )
    )

    principal_conductivities = (
        _oracle_compute_principal_conductivities(
            kappa_0=kappa_0,
            kappa_p=kappa_p,
        )
    )

    ideal_conductivity_tensors = (
        _oracle_construct_conductivity_tensors(
            rotations=rotations,
            principal_conductivities=principal_conductivities,
        )
    )

    parameter_sets = (
        _oracle_infer_calibration_parameters(
            ideal_conductivity_tensors=ideal_conductivity_tensors,
            determinants=determinants,
            weights=weights,
            calibration_hessians=calibration_hessians,
            measured_responses=measured_responses,
            offset_bounds_deg=offset_bounds_deg,
            rho_c0_bounds=rho_c0_bounds,
        )
    )

    n_cells = gradients.shape[0]

    if (
        not isinstance(microscopy_cell_index, (int, np.integer))
        or microscopy_cell_index < 0
        or microscopy_cell_index >= n_cells
    ):
        raise ValueError(
            "microscopy_cell_index must identify a valid cell."
        )

    if laminate_angle_bounds_deg.shape != (2,):
        raise ValueError(
            "laminate_angle_bounds_deg must have shape (2,)."
        )

    if not np.all(np.isfinite(laminate_angle_bounds_deg)):
        raise ValueError(
            "laminate_angle_bounds_deg must contain only finite values."
        )

    if (
        laminate_angle_bounds_deg[0] < 0.0
        or laminate_angle_bounds_deg[1] > 180.0
        or laminate_angle_bounds_deg[0]
        >= laminate_angle_bounds_deg[1]
    ):
        raise ValueError(
            "laminate_angle_bounds_deg must define an increasing interval "
            "within [0, 180] degrees."
        )

    if (
        parameter_sets.ndim != 2
        or parameter_sets.shape[1] != n_cells + 1
    ):
        raise ValueError(
            "Calibration must return parameter sets with shape (K, N + 1)."
        )

    if kappa_p[microscopy_cell_index] <= kappa_0:
        raise ValueError(
            "The microscopy cell must correspond to the field-aligned "
            "laminate case with kappa_p greater than kappa_0."
        )

    gradient_angle_deg = np.rad2deg(
        np.arctan2(
            gradients[microscopy_cell_index, 1],
            gradients[microscopy_cell_index, 0],
        )
    )

    candidate_angles_deg = np.mod(
        gradient_angle_deg
        + parameter_sets[:, microscopy_cell_index],
        180.0,
    )

    lower_angle = laminate_angle_bounds_deg[0]
    upper_angle = laminate_angle_bounds_deg[1]

    admissible = (
        (candidate_angles_deg > lower_angle)
        & (candidate_angles_deg < upper_angle)
    )

    admissible_indices = np.flatnonzero(admissible)

    if admissible_indices.size != 1:
        raise ValueError(
            "The microscopy observation must select exactly one "
            "calibration branch."
        )

    parameters = parameter_sets[admissible_indices[0]]

    offsets_deg = parameters[:-1]
    rho_c0 = float(parameters[-1])

    heat_capacities = (
        _oracle_transform_heat_capacity(
            rho_c0=rho_c0,
            determinants=determinants,
        )
    )

    if prediction_hessians.shape != (n_cells, 3):
        raise ValueError(
            "prediction_hessians must have shape (N, 3)."
        )

    if not np.all(np.isfinite(prediction_hessians)):
        raise ValueError(
            "prediction_hessians must contain only finite values."
        )

    prediction_states = prediction_hessians[None, :, :]

    realized_tensors, temperature_rates, responses = (
        _oracle_evaluate_offset_response(
            ideal_conductivity_tensors=ideal_conductivity_tensors,
            heat_capacities=heat_capacities,
            weights=weights,
            hessian_components=prediction_states,
            offsets_deg=offsets_deg,
        )
    )

    response = float(responses[0])

    return response

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential whole-pipeline tests for run_inverse_duality_pipeline."""

    return [
        {
            "setup": """import numpy as np

gradients = np.array([
    [0.34, 0.79],
    [0.72, 0.31],
    [0.46, -0.88],
], dtype=float)

reference_gradient = np.array([
    1.0,
    0.0,
], dtype=float)

kappa_p = np.array([
    0.22,
    1.85,
    4.60,
], dtype=float)

kappa_0 = 1.0

weights = np.array([
    0.25,
    0.35,
    0.40,
], dtype=float)

calibration_hessians = np.array([
    [
        [120.0, -250.0, -70.0],
        [-180.0, 90.0, 260.0],
        [75.0, 310.0, -140.0],
    ],
    [
        [-210.0, 160.0, 95.0],
        [140.0, -280.0, -130.0],
        [260.0, 85.0, -190.0],
    ],
    [
        [55.0, 330.0, -180.0],
        [-90.0, -210.0, 240.0],
        [-175.0, 270.0, 130.0],
    ],
    [
        [-130.0, -280.0, 210.0],
        [190.0, 240.0, -160.0],
        [-220.0, -150.0, 280.0],
    ],
], dtype=float)

measured_responses = np.array([
    1.588634201628,
    -1.692631704136,
    -1.727994022560,
    1.377882682405,
], dtype=float)

offset_bounds_deg = np.array([
    [-20.0, 0.0],
    [3.0, 22.0],
    [-28.0, -8.0],
], dtype=float)

rho_c0_bounds = np.array([
    1000.0,
    1400.0,
], dtype=float)

microscopy_cell_index = 1

laminate_angle_bounds_deg = np.array([
    26.0,
    30.0,
], dtype=float)

prediction_hessians = np.array([
    [-331.0, -49.0, 437.0],
    [403.0, 230.0, 240.0],
    [370.0, -192.0, 488.0],
], dtype=float)
""",
            "call": "run_inverse_duality_pipeline(gradients.copy(), reference_gradient.copy(), kappa_p.copy(), kappa_0, weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy(), microscopy_cell_index, laminate_angle_bounds_deg.copy(), prediction_hessians.copy())",
            "gold_call": "_oracle_run_inverse_duality_pipeline(gradients.copy(), reference_gradient.copy(), kappa_p.copy(), kappa_0, weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy(), microscopy_cell_index, laminate_angle_bounds_deg.copy(), prediction_hessians.copy())",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np

gradients = np.array([
    [1.0, 0.0],
], dtype=float)

reference_gradient = np.array([
    1.0,
    0.0,
], dtype=float)

kappa_p = np.array([
    2.0,
], dtype=float)

kappa_0 = 1.0

weights = np.array([
    1.0,
], dtype=float)

calibration_hessians = np.array([
    [
        [4.0, -2.0, 1.0],
    ],
    [
        [-3.0, 1.5, 6.0],
    ],
], dtype=float)

measured_responses = np.array([
    0.0037484487003512316,
    -0.00028748783362402404,
], dtype=float)

offset_bounds_deg = np.array([
    [5.0, 30.0],
], dtype=float)

rho_c0_bounds = np.array([
    700.0,
    1000.0,
], dtype=float)

microscopy_cell_index = 0

laminate_angle_bounds_deg = np.array([
    17.0,
    18.0,
], dtype=float)

prediction_hessians = np.array([
    [2.0, 1.0, -4.0],
], dtype=float)
""",
            "call": "run_inverse_duality_pipeline(gradients.copy(), reference_gradient.copy(), kappa_p.copy(), kappa_0, weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy(), microscopy_cell_index, laminate_angle_bounds_deg.copy(), prediction_hessians.copy())",
            "gold_call": "_oracle_run_inverse_duality_pipeline(gradients.copy(), reference_gradient.copy(), kappa_p.copy(), kappa_0, weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy(), microscopy_cell_index, laminate_angle_bounds_deg.copy(), prediction_hessians.copy())",
            "tol": 1e-9,
        },
        {
            "setup": """import numpy as np

gradients = np.array([
    [0.34, 0.79],
    [0.72, 0.31],
    [0.46, -0.88],
], dtype=float)

reference_gradient = np.array([
    1.0,
    0.0,
], dtype=float)

kappa_p = np.array([
    0.22,
    1.85,
    4.60,
], dtype=float)

kappa_0 = 1.0

weights = np.array([
    0.25,
    0.35,
    0.40,
], dtype=float)

calibration_hessians = np.array([
    [
        [120.0, -250.0, -70.0],
        [-180.0, 90.0, 260.0],
        [75.0, 310.0, -140.0],
    ],
    [
        [-210.0, 160.0, 95.0],
        [140.0, -280.0, -130.0],
        [260.0, 85.0, -190.0],
    ],
    [
        [55.0, 330.0, -180.0],
        [-90.0, -210.0, 240.0],
        [-175.0, 270.0, 130.0],
    ],
    [
        [-130.0, -280.0, 210.0],
        [190.0, 240.0, -160.0],
        [-220.0, -150.0, 280.0],
    ],
], dtype=float)

measured_responses = np.array([
    1.588634201628,
    -1.692631704136,
    -1.727994022560,
    1.377882682405,
], dtype=float)

offset_bounds_deg = np.array([
    [-20.0, 0.0],
    [3.0, 22.0],
    [-28.0, -8.0],
], dtype=float)

rho_c0_bounds = np.array([
    1000.0,
    1170.347027773097,
], dtype=float)

microscopy_cell_index = 1

laminate_angle_bounds_deg = np.array([
    26.0,
    30.0,
], dtype=float)

prediction_hessians = np.array([
    [-331.0, -49.0, 437.0],
    [403.0, 230.0, 240.0],
    [370.0, -192.0, 488.0],
], dtype=float)
""",
            "call": "run_inverse_duality_pipeline(gradients.copy(), reference_gradient.copy(), kappa_p.copy(), kappa_0, weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy(), microscopy_cell_index, laminate_angle_bounds_deg.copy(), prediction_hessians.copy())",
            "gold_call": "_oracle_run_inverse_duality_pipeline(gradients.copy(), reference_gradient.copy(), kappa_p.copy(), kappa_0, weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy(), microscopy_cell_index, laminate_angle_bounds_deg.copy(), prediction_hessians.copy())",
            "tol": 1e-9,
        },
    ]
