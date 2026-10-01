"""
Recover all distinct fabrication-offset and background volumetric heat-capacity parameter sets that reproduce the measured transient calibration responses within the prescribed bounds. Solve the coupled bounded nonlinear inverse problem using one shared parameter vector across all calibration experiments, retain every distinct solution whose maximum absolute response residual is below the numerical tolerance, and return the solutions in deterministic lexicographic order.

The unknown calibration parameter vector is

$$

\mathbf p=\begin{pmatrix}

\delta_1\\

\vdots\\

\delta_N\\

\rho_0c_0

\end{pmatrix}.

$$

Because a symmetric second-rank conductivity tensor is unchanged by a $180^\circ$ rotation,

$$

\mathbf Q(\delta_i+180^\circ)\boldsymbol{\kappa}_{D,i}\mathbf Q(\delta_i+180^\circ)^T=\mathbf Q(\delta_i)\boldsymbol{\kappa}_{D,i}\mathbf Q(\delta_i)^T.

$$

Therefore, numerically distinct offset vectors related by admissible $180^\circ$ shifts are separate bounded parameter sets and must all be returned when they lie within the supplied bounds.

Returns
-------
np.ndarray with shape (K, N + 1), containing every distinct bounded calibration parameter set in deterministic lexicographic order; the first N columns are fabrication offsets in degrees and the final column is rho_c0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from itertools import product
from scipy.optimize import least_squares


def _oracle_infer_calibration_parameters(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test cases for infer_calibration_parameters."""

    validation_helpers = r"""import numpy as np
from functools import cmp_to_key

PARAMETER_TOL = 1e-6
RESPONSE_TOL = 1e-8
BOUND_TOL = 1e-9


def _predict_calibration_responses(root):
    root = np.asarray(
        root,
        dtype=float,
    )

    offsets_deg = root[:-1]
    rho_c0 = float(root[-1])

    n_cells = ideal_conductivity_tensors.shape[0]

    angles_rad = np.deg2rad(
        offsets_deg
    )

    cos_delta = np.cos(
        angles_rad
    )

    sin_delta = np.sin(
        angles_rad
    )

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

    heat_capacities = (
        rho_c0
        / determinants
    )

    t_xx = calibration_hessians[:, :, 0]
    t_xy = calibration_hessians[:, :, 1]
    t_yy = calibration_hessians[:, :, 2]

    contractions = (
        realized_tensors[
            None,
            :,
            0,
            0,
        ]
        * t_xx
        + 2.0
        * realized_tensors[
            None,
            :,
            0,
            1,
        ]
        * t_xy
        + realized_tensors[
            None,
            :,
            1,
            1,
        ]
        * t_yy
    )

    temperature_rates = (
        contractions
        / heat_capacities[
            None,
            :,
        ]
    )

    return (
        temperature_rates
        @ weights
    )


def _root_set_signature(roots):
    roots = np.asarray(
        roots,
        dtype=float,
    )

    n_cells = (
        ideal_conductivity_tensors.shape[0]
    )

    expected_columns = (
        n_cells + 1
    )

    if (
        roots.ndim != 2
        or roots.shape[0] < 1
        or roots.shape[1] != expected_columns
    ):
        raise AssertionError(
            "Calibration roots must have shape (K, N + 1) with K >= 1."
        )

    if not np.all(
        np.isfinite(roots)
    ):
        raise AssertionError(
            "Calibration roots must contain only finite values."
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

    spans = (
        upper_bounds
        - lower_bounds
    )

    if np.any(
        roots
        < lower_bounds - BOUND_TOL
    ):
        raise AssertionError(
            "Calibration root set contains a parameter below its lower bound."
        )

    if np.any(
        roots
        > upper_bounds + BOUND_TOL
    ):
        raise AssertionError(
            "Calibration root set contains a parameter above its upper bound."
        )

    normalized_roots = (
        roots
        - lower_bounds[
            None,
            :,
        ]
    ) / spans[
        None,
        :,
    ]

    # Check the public output contract using ordinary
    # deterministic lexicographic ordering.
    lexsort_keys = tuple(
        roots[:, column]
        for column in range(
            expected_columns - 1,
            -1,
            -1,
        )
    )

    lexicographic_order = np.lexsort(
        lexsort_keys
    )

    if not np.array_equal(
        lexicographic_order,
        np.arange(
            roots.shape[0]
        ),
    ):
        raise AssertionError(
            "Calibration roots are not in deterministic lexicographic order."
        )

    # Reject numerically duplicate roots after normalizing
    # by the admissible parameter spans.
    for i in range(
        roots.shape[0]
    ):
        for j in range(
            i + 1,
            roots.shape[0],
        ):
            normalized_separation = np.max(
                np.abs(
                    normalized_roots[i]
                    - normalized_roots[j]
                )
            )

            if (
                normalized_separation
                <= PARAMETER_TOL
            ):
                raise AssertionError(
                    "Calibration root set contains duplicate roots."
                )

    # Every returned root must independently reproduce
    # all measured calibration responses.
    for root in roots:
        residuals = (
            _predict_calibration_responses(
                root
            )
            - measured_responses
        )

        if not np.all(
            np.isfinite(
                residuals
            )
        ):
            raise AssertionError(
                "A calibration root produces non-finite response residuals."
            )

        if np.max(
            np.abs(
                residuals
            )
        ) > RESPONSE_TOL:
            raise AssertionError(
                "A calibration root exceeds the allowed response residual."
            )

    # Candidate and oracle can differ by harmless numerical
    # noise. Canonicalize each set with a tolerance-aware
    # comparator before differential comparison.
    def _compare_indices(
        left,
        right,
    ):
        for column in range(
            expected_columns
        ):
            difference = (
                normalized_roots[
                    left,
                    column,
                ]
                - normalized_roots[
                    right,
                    column,
                ]
            )

            if (
                difference
                < -PARAMETER_TOL
            ):
                return -1

            if (
                difference
                > PARAMETER_TOL
            ):
                return 1

        return 0

    canonical_indices = sorted(
        range(
            roots.shape[0]
        ),
        key=cmp_to_key(
            _compare_indices
        ),
    )

    return normalized_roots[
        np.asarray(
            canonical_indices,
            dtype=int,
        )
    ]
"""

    return [
        {
            "setup": validation_helpers + r"""
ideal_conductivity_tensors = np.array([
    [
        [3.869474356926, -1.570659849816],
        [-1.570659849816, 0.895980188529],
    ],
    [
        [1.645217161832, 0.475624656390],
        [0.475624656390, 0.745323378708],
    ],
    [
        [1.157918687715, -1.799269776876],
        [-1.799269776876, 3.659472616633],
    ],
], dtype=float)

determinants = np.array([
    6.144997357651,
    0.879642864997,
    0.220477996296,
], dtype=float)

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
""",
            "call": "_root_set_signature(infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "gold_call": "_root_set_signature(_oracle_infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": validation_helpers + r"""
ideal_conductivity_tensors = np.array([
    [
        [2.0, 0.3],
        [0.3, 5.0],
    ],
], dtype=float)

determinants = np.array([
    0.8,
], dtype=float)

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
    0.014829245339241,
    0.020012566449251,
], dtype=float)

offset_bounds_deg = np.array([
    [5.0, 30.0],
], dtype=float)

rho_c0_bounds = np.array([
    700.0,
    1000.0,
], dtype=float)
""",
            "call": "_root_set_signature(infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "gold_call": "_root_set_signature(_oracle_infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": validation_helpers + r"""
ideal_conductivity_tensors = np.array([
    [
        [4.0, -0.8],
        [-0.8, 1.2],
    ],
    [
        [1.5, 0.4],
        [0.4, 3.2],
    ],
], dtype=float)

determinants = np.array([
    1.4,
    0.55,
], dtype=float)

weights = np.array([
    0.45,
    0.55,
], dtype=float)

calibration_hessians = np.array([
    [
        [2.0, -4.0, 1.0],
        [-3.0, 2.0, 5.0],
    ],
    [
        [-5.0, 1.0, 4.0],
        [6.0, -3.0, -2.0],
    ],
    [
        [3.0, 2.0, -6.0],
        [1.0, 4.0, -5.0],
    ],
], dtype=float)

measured_responses = np.array([
    0.013581658683664,
    -0.006570890521614,
    -0.007168380046757,
], dtype=float)

offset_bounds_deg = np.array([
    [-20.0, -5.0],
    [10.0, 30.0],
], dtype=float)

rho_c0_bounds = np.array([
    900.0,
    1300.0,
], dtype=float)
""",
            "call": "_root_set_signature(infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "gold_call": "_root_set_signature(_oracle_infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": validation_helpers + r"""
ideal_conductivity_tensors = np.array([
    [
        [3.0, 0.6],
        [0.6, 1.0],
    ],
], dtype=float)

determinants = np.array([
    1.25,
], dtype=float)

weights = np.array([
    1.0,
], dtype=float)

calibration_hessians = np.array([
    [
        [5.0, -2.0, -1.0],
    ],
    [
        [-4.0, 3.0, 2.0],
    ],
    [
        [1.0, 5.0, -3.0],
    ],
], dtype=float)

measured_responses = np.array([
    0.018449233904943,
    -0.013374701059079,
    0.001320452683485,
], dtype=float)

offset_bounds_deg = np.array([
    [-15.0, 0.0],
], dtype=float)

rho_c0_bounds = np.array([
    1000.0,
    1300.0,
], dtype=float)
""",
            "call": "_root_set_signature(infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "gold_call": "_root_set_signature(_oracle_infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "tol": 1e-6,
        },
        {
            "setup": validation_helpers + r"""
ideal_conductivity_tensors = np.array([
    [
        [3.869474356926, -1.570659849816],
        [-1.570659849816, 0.895980188529],
    ],
    [
        [1.645217161832, 0.475624656390],
        [0.475624656390, 0.745323378708],
    ],
    [
        [1.157918687715, -1.799269776876],
        [-1.799269776876, 3.659472616633],
    ],
], dtype=float)

determinants = np.array([
    6.144997357651,
    0.879642864997,
    0.220477996296,
], dtype=float)

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
    [-200.0, 180.0],
    [-180.0, 200.0],
    [-200.0, 180.0],
], dtype=float)

rho_c0_bounds = np.array([
    1000.0,
    1400.0,
], dtype=float)
""",
            "call": "_root_set_signature(infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "gold_call": "_root_set_signature(_oracle_infer_calibration_parameters(ideal_conductivity_tensors.copy(), determinants.copy(), weights.copy(), calibration_hessians.copy(), measured_responses.copy(), offset_bounds_deg.copy(), rho_c0_bounds.copy()))",
            "tol": 1e-6,
        },
    ]
