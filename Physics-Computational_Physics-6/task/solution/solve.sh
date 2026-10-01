#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def recover_field_geometry(
    g0: "np.ndarray",
    gradients: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Reference implementation of the field-geometry recovery."""
    g0 = np.asarray(g0, dtype=float)
    gradients = np.asarray(gradients, dtype=float)

    if g0.shape != (2,):
        raise ValueError("g0 must have shape (2,).")

    if gradients.ndim != 2 or gradients.shape[1] != 2:
        raise ValueError("gradients must have shape (N, 2).")

    if gradients.shape[0] < 1:
        raise ValueError("gradients must contain at least one local gradient.")

    if not np.all(np.isfinite(g0)) or not np.all(np.isfinite(gradients)):
        raise ValueError("All gradient values must be finite.")

    g0_norm = float(np.linalg.norm(g0))

    if g0_norm <= 0.0:
        raise ValueError("The reference gradient must be nonzero.")

    magnitudes = np.linalg.norm(gradients, axis=1)

    if np.any(magnitudes <= 0.0):
        raise ValueError("Every local gradient must be nonzero.")

    scale_factors = g0_norm / magnitudes

    rotations = np.empty((gradients.shape[0], 2, 2), dtype=float)

    rotations[:, 0, 0] = gradients[:, 0] / magnitudes
    rotations[:, 0, 1] = -gradients[:, 1] / magnitudes
    rotations[:, 1, 0] = gradients[:, 1] / magnitudes
    rotations[:, 1, 1] = gradients[:, 0] / magnitudes

    return magnitudes, scale_factors, rotations

import numpy as np
def build_dual_jacobian(
    scale_factors: "np.ndarray",
    rotations: "np.ndarray",
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Reference implementation of the local Jacobian construction."""
    scale_factors = np.asarray(scale_factors, dtype=float)
    rotations = np.asarray(rotations, dtype=float)
    kappa_p = np.asarray(kappa_p, dtype=float)

    if scale_factors.ndim != 1 or scale_factors.size < 1:
        raise ValueError("scale_factors must have shape (N,) with N >= 1.")

    n_cells = scale_factors.size

    if rotations.shape != (n_cells, 2, 2):
        raise ValueError("rotations must have shape (N, 2, 2).")

    if kappa_p.shape != (n_cells,):
        raise ValueError("kappa_p must have shape (N,).")

    if not np.isscalar(kappa_0) or not np.isfinite(float(kappa_0)):
        raise ValueError("kappa_0 must be a finite scalar.")

    kappa_0 = float(kappa_0)

    if kappa_0 <= 0.0:
        raise ValueError("kappa_0 must be positive.")

    if not np.all(np.isfinite(scale_factors)):
        raise ValueError("scale_factors must contain only finite values.")

    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotations must contain only finite values.")

    if not np.all(np.isfinite(kappa_p)):
        raise ValueError("kappa_p must contain only finite values.")

    if np.any(scale_factors <= 0.0):
        raise ValueError("scale_factors must be positive.")

    if np.any(kappa_p <= 0.0):
        raise ValueError("kappa_p must be positive.")

    identity = np.eye(2, dtype=float)

    for rotation in rotations:
        if not np.allclose(
            rotation.T @ rotation,
            identity,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("Each rotation matrix must be orthogonal.")

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "Each rotation matrix must have determinant +1."
            )

    stretches = np.zeros((n_cells, 2, 2), dtype=float)
    stretches[:, 0, 0] = 1.0
    stretches[:, 1, 1] = kappa_0 / kappa_p

    jacobians = (
        scale_factors[:, None, None]
        * np.matmul(rotations, stretches)
    )

    determinants = np.linalg.det(jacobians)

    if np.any(determinants <= 0.0):
        raise ValueError(
            "Every local Jacobian must have positive determinant."
        )

    return stretches, jacobians, determinants

import numpy as np
def compute_principal_conductivities(
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the principal-conductivity calculation."""
    kappa_p = np.asarray(kappa_p, dtype=float)

    if not np.isscalar(kappa_0) or not np.isfinite(float(kappa_0)):
        raise ValueError("kappa_0 must be a finite scalar.")

    kappa_0 = float(kappa_0)

    if kappa_0 <= 0.0:
        raise ValueError("kappa_0 must be positive.")

    if kappa_p.ndim != 1 or kappa_p.size < 1:
        raise ValueError("kappa_p must have shape (N,) with N >= 1.")

    if not np.all(np.isfinite(kappa_p)):
        raise ValueError("kappa_p must contain only finite values.")

    if np.any(kappa_p <= 0.0):
        raise ValueError("kappa_p must be positive.")

    principal_conductivities = np.empty((kappa_p.size, 2), dtype=float)

    principal_conductivities[:, 0] = kappa_p
    principal_conductivities[:, 1] = (kappa_0**2) / kappa_p

    return principal_conductivities

import numpy as np
def construct_conductivity_tensors(
    rotations: "np.ndarray",
    principal_conductivities: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the conductivity-tensor rotation."""
    rotations = np.asarray(rotations, dtype=float)
    principal_conductivities = np.asarray(
        principal_conductivities,
        dtype=float,
    )

    if rotations.ndim != 3 or rotations.shape[1:] != (2, 2):
        raise ValueError("rotations must have shape (N, 2, 2).")

    n_cells = rotations.shape[0]

    if n_cells < 1:
        raise ValueError("rotations must contain at least one cell.")

    if principal_conductivities.shape != (n_cells, 2):
        raise ValueError(
            "principal_conductivities must have shape (N, 2)."
        )

    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotations must contain only finite values.")

    if not np.all(np.isfinite(principal_conductivities)):
        raise ValueError(
            "principal_conductivities must contain only finite values."
        )

    if np.any(principal_conductivities <= 0.0):
        raise ValueError(
            "principal conductivities must be positive."
        )

    identity = np.eye(2, dtype=float)

    for rotation in rotations:
        if not np.allclose(
            rotation.T @ rotation,
            identity,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("Each rotation matrix must be orthogonal.")

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "Each rotation matrix must have determinant +1."
            )

    conductivity_tensors = np.empty(
        (n_cells, 2, 2),
        dtype=float,
    )

    for i in range(n_cells):
        principal_tensor = np.diag(
            principal_conductivities[i]
        )

        conductivity_tensors[i] = (
            rotations[i]
            @ principal_tensor
            @ rotations[i].T
        )

    return conductivity_tensors

import numpy as np
def transform_heat_capacity(
    rho_c0: float,
    determinants: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the transient heat-capacity transformation."""
    determinants = np.asarray(determinants, dtype=float)

    if not np.isscalar(rho_c0) or not np.isfinite(float(rho_c0)):
        raise ValueError("rho_c0 must be a finite scalar.")

    rho_c0 = float(rho_c0)

    if rho_c0 <= 0.0:
        raise ValueError("rho_c0 must be positive.")

    if determinants.ndim != 1 or determinants.size < 1:
        raise ValueError(
            "determinants must have shape (N,) with N >= 1."
        )

    if not np.all(np.isfinite(determinants)):
        raise ValueError(
            "determinants must contain only finite values."
        )

    if np.any(determinants <= 0.0):
        raise ValueError(
            "Every Jacobian determinant must be positive."
        )

    heat_capacities = rho_c0 / determinants

    return heat_capacities

import numpy as np


def evaluate_offset_response(
    ideal_conductivity_tensors: "np.ndarray",
    heat_capacities: "np.ndarray",
    weights: "np.ndarray",
    hessian_components: "np.ndarray",
    offsets_deg: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Reference implementation of the fabrication-offset transient response."""
    ideal_conductivity_tensors = np.asarray(
        ideal_conductivity_tensors,
        dtype=float,
    )
    heat_capacities = np.asarray(
        heat_capacities,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )
    hessian_components = np.asarray(
        hessian_components,
        dtype=float,
    )
    offsets_deg = np.asarray(
        offsets_deg,
        dtype=float,
    )

    if (
        ideal_conductivity_tensors.ndim != 3
        or ideal_conductivity_tensors.shape[1:] != (2, 2)
        or ideal_conductivity_tensors.shape[0] < 1
    ):
        raise ValueError(
            "ideal_conductivity_tensors must have shape (N, 2, 2) with N >= 1."
        )

    n_cells = ideal_conductivity_tensors.shape[0]

    if heat_capacities.shape != (n_cells,):
        raise ValueError(
            "heat_capacities must have shape (N,)."
        )

    if weights.shape != (n_cells,):
        raise ValueError(
            "weights must have shape (N,)."
        )

    if offsets_deg.shape != (n_cells,):
        raise ValueError(
            "offsets_deg must have shape (N,)."
        )

    if (
        hessian_components.ndim != 3
        or hessian_components.shape[0] < 1
        or hessian_components.shape[1:] != (n_cells, 3)
    ):
        raise ValueError(
            "hessian_components must have shape (M, N, 3) with M >= 1."
        )

    if not np.all(np.isfinite(ideal_conductivity_tensors)):
        raise ValueError(
            "ideal_conductivity_tensors must contain only finite values."
        )

    if not np.all(np.isfinite(heat_capacities)):
        raise ValueError(
            "heat_capacities must contain only finite values."
        )

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "weights must contain only finite values."
        )

    if not np.all(np.isfinite(hessian_components)):
        raise ValueError(
            "hessian_components must contain only finite values."
        )

    if not np.all(np.isfinite(offsets_deg)):
        raise ValueError(
            "offsets_deg must contain only finite values."
        )

    if np.any(heat_capacities <= 0.0):
        raise ValueError(
            "heat_capacities must be positive."
        )

    if np.any(weights < 0.0):
        raise ValueError(
            "weights must be nonnegative."
        )

    if not np.isclose(
        np.sum(weights),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "weights must sum to one."
        )

    if not np.allclose(
        ideal_conductivity_tensors,
        np.swapaxes(ideal_conductivity_tensors, 1, 2),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Each ideal conductivity tensor must be symmetric."
        )

    angles_rad = np.deg2rad(offsets_deg)
    realized_tensors = np.empty_like(
        ideal_conductivity_tensors,
        dtype=float,
    )

    for i in range(n_cells):
        cos_delta = np.cos(angles_rad[i])
        sin_delta = np.sin(angles_rad[i])

        rotation_offset = np.array(
            [
                [cos_delta, -sin_delta],
                [sin_delta, cos_delta],
            ],
            dtype=float,
        )

        realized_tensors[i] = (
            rotation_offset
            @ ideal_conductivity_tensors[i]
            @ rotation_offset.T
        )

    t_xx = hessian_components[:, :, 0]
    t_xy = hessian_components[:, :, 1]
    t_yy = hessian_components[:, :, 2]

    contractions = (
        realized_tensors[None, :, 0, 0] * t_xx
        + 2.0 * realized_tensors[None, :, 0, 1] * t_xy
        + realized_tensors[None, :, 1, 1] * t_yy
    )

    temperature_rates = contractions / heat_capacities[None, :]
    responses = temperature_rates @ weights

    return realized_tensors, temperature_rates, responses

import numpy as np
from itertools import product
from scipy.optimize import least_squares


def infer_calibration_parameters(
    ideal_conductivity_tensors: "np.ndarray",
    determinants: "np.ndarray",
    weights: "np.ndarray",
    calibration_hessians: "np.ndarray",
    measured_responses: "np.ndarray",
    offset_bounds_deg: "np.ndarray",
    rho_c0_bounds: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation for recovery of all bounded calibration roots."""
    ideal_conductivity_tensors = np.asarray(
        ideal_conductivity_tensors,
        dtype=float,
    )
    determinants = np.asarray(
        determinants,
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

    if (
        ideal_conductivity_tensors.ndim != 3
        or ideal_conductivity_tensors.shape[1:] != (2, 2)
        or ideal_conductivity_tensors.shape[0] < 1
    ):
        raise ValueError(
            "ideal_conductivity_tensors must have shape (N, 2, 2) with N >= 1."
        )

    n_cells = ideal_conductivity_tensors.shape[0]

    if determinants.shape != (n_cells,):
        raise ValueError(
            "determinants must have shape (N,)."
        )

    if weights.shape != (n_cells,):
        raise ValueError(
            "weights must have shape (N,)."
        )

    if offset_bounds_deg.shape != (n_cells, 2):
        raise ValueError(
            "offset_bounds_deg must have shape (N, 2)."
        )

    if rho_c0_bounds.shape != (2,):
        raise ValueError(
            "rho_c0_bounds must have shape (2,)."
        )

    if (
        calibration_hessians.ndim != 3
        or calibration_hessians.shape[1:] != (n_cells, 3)
    ):
        raise ValueError(
            "calibration_hessians must have shape (M, N, 3)."
        )

    n_experiments = calibration_hessians.shape[0]

    if n_experiments < n_cells + 1:
        raise ValueError(
            "At least N + 1 calibration experiments are required."
        )

    if measured_responses.shape != (n_experiments,):
        raise ValueError(
            "measured_responses must have shape (M,)."
        )

    arrays = [
        ideal_conductivity_tensors,
        determinants,
        weights,
        calibration_hessians,
        measured_responses,
        offset_bounds_deg,
        rho_c0_bounds,
    ]

    if not all(
        np.all(np.isfinite(array))
        for array in arrays
    ):
        raise ValueError(
            "All inputs must contain only finite values."
        )

    if np.any(determinants <= 0.0):
        raise ValueError(
            "determinants must be positive."
        )

    if np.any(weights < 0.0):
        raise ValueError(
            "weights must be nonnegative."
        )

    if not np.isclose(
        np.sum(weights),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "weights must sum to one."
        )

    if not np.allclose(
        ideal_conductivity_tensors,
        np.swapaxes(
            ideal_conductivity_tensors,
            1,
            2,
        ),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Each ideal conductivity tensor must be symmetric."
        )

    if np.any(
        offset_bounds_deg[:, 0]
        >= offset_bounds_deg[:, 1]
    ):
        raise ValueError(
            "Each offset lower bound must be less than its upper bound."
        )

    if (
        rho_c0_bounds[0] <= 0.0
        or rho_c0_bounds[0] >= rho_c0_bounds[1]
    ):
        raise ValueError(
            "rho_c0_bounds must define a positive increasing interval."
        )

    lower_bounds = np.concatenate(
        (
            offset_bounds_deg[:, 0],
            [rho_c0_bounds[0]],
        )
    )

    upper_bounds = np.concatenate(
        (
            offset_bounds_deg[:, 1],
            [rho_c0_bounds[1]],
        )
    )

    parameter_span = upper_bounds - lower_bounds

    def predicted_responses(
        parameters: np.ndarray,
    ) -> np.ndarray:
        offsets_deg = parameters[:n_cells]
        rho_c0 = parameters[-1]

        angles_rad = np.deg2rad(offsets_deg)
        cos_delta = np.cos(angles_rad)
        sin_delta = np.sin(angles_rad)

        offset_rotations = np.empty(
            (n_cells, 2, 2),
            dtype=float,
        )

        offset_rotations[:, 0, 0] = cos_delta
        offset_rotations[:, 0, 1] = -sin_delta
        offset_rotations[:, 1, 0] = sin_delta
        offset_rotations[:, 1, 1] = cos_delta

        realized_tensors = (
            offset_rotations
            @ ideal_conductivity_tensors
            @ np.swapaxes(
                offset_rotations,
                1,
                2,
            )
        )

        heat_capacities = rho_c0 / determinants

        t_xx = calibration_hessians[:, :, 0]
        t_xy = calibration_hessians[:, :, 1]
        t_yy = calibration_hessians[:, :, 2]

        contractions = (
            realized_tensors[None, :, 0, 0] * t_xx
            + 2.0
            * realized_tensors[None, :, 0, 1]
            * t_xy
            + realized_tensors[None, :, 1, 1]
            * t_yy
        )

        temperature_rates = (
            contractions
            / heat_capacities[None, :]
        )

        return temperature_rates @ weights

    def residuals(
        parameters: np.ndarray,
    ) -> np.ndarray:
        return (
            predicted_responses(parameters)
            - measured_responses
        )

    def add_solution(
        collection,
        candidate,
        residual_norm,
    ):
        candidate = np.asarray(
            candidate,
            dtype=float,
        )

        duplicate_index = None

        for index, (solution, _) in enumerate(
            collection
        ):
            normalized_distance = np.max(
                np.abs(
                    (candidate - solution)
                    / parameter_span
                )
            )

            if normalized_distance <= 1e-5:
                duplicate_index = index
                break

        if duplicate_index is None:
            collection.append(
                (
                    candidate.copy(),
                    float(residual_norm),
                )
            )
        elif (
            residual_norm
            < collection[duplicate_index][1]
        ):
            collection[duplicate_index] = (
                candidate.copy(),
                float(residual_norm),
            )

    start_fractions = (
        0.15,
        0.50,
        0.85,
    )

    numerically_found_solutions = []

    for fractions in product(
        start_fractions,
        repeat=n_cells + 1,
    ):
        starting_point = (
            lower_bounds
            + np.asarray(
                fractions,
                dtype=float,
            )
            * parameter_span
        )

        result = least_squares(
            residuals,
            starting_point,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            x_scale="jac",
            max_nfev=20000,
        )

        residual_norm = np.linalg.norm(
            result.fun,
            ord=np.inf,
        )

        if residual_norm > 1e-8:
            continue

        add_solution(
            numerically_found_solutions,
            result.x,
            residual_norm,
        )

    if not numerically_found_solutions:
        raise ValueError(
            "No bounded parameter set reproduces the calibration responses."
        )

    expanded_solutions = []

    for base_solution, _ in numerically_found_solutions:
        angle_options = []

        for cell_index in range(n_cells):
            angle = float(
                base_solution[cell_index]
            )

            lower_angle = float(
                offset_bounds_deg[
                    cell_index,
                    0,
                ]
            )

            upper_angle = float(
                offset_bounds_deg[
                    cell_index,
                    1,
                ]
            )

            k_min = int(
                np.ceil(
                    (
                        lower_angle
                        - angle
                        - 1e-10
                    )
                    / 180.0
                )
            )

            k_max = int(
                np.floor(
                    (
                        upper_angle
                        - angle
                        + 1e-10
                    )
                    / 180.0
                )
            )

            equivalents = []

            for k in range(
                k_min,
                k_max + 1,
            ):
                equivalent_angle = (
                    angle
                    + 180.0 * k
                )

                if (
                    equivalent_angle
                    < lower_angle - 1e-9
                    or equivalent_angle
                    > upper_angle + 1e-9
                ):
                    continue

                equivalents.append(
                    equivalent_angle
                )

            if not equivalents:
                equivalents = [angle]

            angle_options.append(
                equivalents
            )

        for equivalent_offsets in product(
            *angle_options
        ):
            candidate = np.concatenate(
                (
                    np.asarray(
                        equivalent_offsets,
                        dtype=float,
                    ),
                    [
                        float(
                            base_solution[-1]
                        )
                    ],
                )
            )

            residual_norm = np.linalg.norm(
                residuals(candidate),
                ord=np.inf,
            )

            if residual_norm > 1e-8:
                continue

            if np.any(
                candidate
                < lower_bounds - 1e-9
            ):
                continue

            if np.any(
                candidate
                > upper_bounds + 1e-9
            ):
                continue

            add_solution(
                expanded_solutions,
                candidate,
                residual_norm,
            )

    if not expanded_solutions:
        raise ValueError(
            "No bounded parameter set reproduces the calibration responses."
        )

    solutions = [
        solution
        for solution, _ in expanded_solutions
    ]

    solutions.sort(
        key=lambda solution: tuple(
            solution.tolist()
        )
    )

    return np.asarray(
        solutions,
        dtype=float,
    )

import numpy as np


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
        recover_field_geometry(
            g0=reference_gradient,
            gradients=gradients,
        )
    )

    stretching_matrices, jacobians, determinants = (
        build_dual_jacobian(
            scale_factors=scale_factors,
            rotations=rotations,
            kappa_0=kappa_0,
            kappa_p=kappa_p,
        )
    )

    principal_conductivities = (
        compute_principal_conductivities(
            kappa_0=kappa_0,
            kappa_p=kappa_p,
        )
    )

    ideal_conductivity_tensors = (
        construct_conductivity_tensors(
            rotations=rotations,
            principal_conductivities=principal_conductivities,
        )
    )

    parameter_sets = (
        infer_calibration_parameters(
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
        transform_heat_capacity(
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
        evaluate_offset_response(
            ideal_conductivity_tensors=ideal_conductivity_tensors,
            heat_capacities=heat_capacities,
            weights=weights,
            hessian_components=prediction_states,
            offsets_deg=offsets_deg,
        )
    )

    response = float(responses[0])

    return response
SCICODE_GOLD_EOF
