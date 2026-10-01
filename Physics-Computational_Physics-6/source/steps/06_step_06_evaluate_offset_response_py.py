"""
Apply the specified fabrication-axis offsets to the ideal local anisotropic conductivity tensors and evaluate the transient response for one or more Hessian states. Rotate each ideal tensor using the signed counterclockwise offset, contract the realized tensor with the local temperature Hessian including the mixed-derivative term, divide by the transformed volumetric heat capacity, and combine the local rates using the prescribed area weights.

For cell $i$, the task-specific fabrication offset rotates the ideal field-derived conductivity tensor according to



$$

\widetilde{\boldsymbol{\kappa}}_i=\mathbf Q(\delta_i)\boldsymbol{\kappa}_{D,i}\mathbf Q(\delta_i)^T,

$$



where



$$

\mathbf Q(\delta_i)=

\begin{pmatrix}

\cos\delta_i & -\sin\delta_i\\

\sin\delta_i & \cos\delta_i

\end{pmatrix}.

$$



The offset $\delta_i$ is supplied in degrees and must be converted to radians when evaluating the trigonometric functions. The rotation changes the orientation of the conductivity tensor but not its principal conductivity values.



For transient state $m$, the locally uniform heat equation gives



$$

(\rho c)_i\dot T_i^{(m)}=\widetilde{\kappa}_{xx,i}T_{xx,i}^{(m)}+2\widetilde{\kappa}_{xy,i}T_{xy,i}^{(m)}+\widetilde{\kappa}_{yy,i}T_{yy,i}^{(m)}.

$$



Therefore,



$$

\dot T_i^{(m)}=\frac{\widetilde{\kappa}_{xx,i}T_{xx,i}^{(m)}+2\widetilde{\kappa}_{xy,i}T_{xy,i}^{(m)}+\widetilde{\kappa}_{yy,i}T_{yy,i}^{(m)}}{(\rho c)_i},

$$



and the area-weighted response is



$$

\mathcal R_m=\sum_{i=1}^{N}w_i\dot T_i^{(m)}.

$$

Returns
-------
return realized_tensors, temperature_rates, responses
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_offset_response(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test cases for evaluate_offset_response."""
    return [
        {
            "setup": """import numpy as np

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

rho_c0 = 1182.700
heat_capacities = rho_c0 / determinants

weights = np.array([
    0.25,
    0.35,
    0.40,
], dtype=float)

offsets_deg = np.array([
    -7.318,
    11.463,
    -16.827,
], dtype=float)

hessian_components = np.array([
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
""",
            "call": "evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
            "gold_call": "_oracle_evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
        },
        {
            "setup": """import numpy as np

ideal_conductivity_tensors = np.array([
    [
        [2.0, 0.0],
        [0.0, 3.0],
    ],
    [
        [1.0, 0.25],
        [0.25, 4.0],
    ],
], dtype=float)

heat_capacities = np.array([
    10.0,
    20.0,
], dtype=float)

weights = np.array([
    0.4,
    0.6,
], dtype=float)

offsets_deg = np.array([
    0.0,
    0.0,
], dtype=float)

hessian_components = np.array([
    [
        [4.0, 1.0, -2.0],
        [3.0, -4.0, 2.0],
    ],
    [
        [-1.0, 2.0, 5.0],
        [6.0, 1.0, -3.0],
    ],
], dtype=float)
""",
            "call": "evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
            "gold_call": "_oracle_evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
        },
        {
            "setup": """import numpy as np

ideal_conductivity_tensors = np.array([
    [
        [2.0, 0.0],
        [0.0, 5.0],
    ],
], dtype=float)

heat_capacities = np.array([
    4.0,
], dtype=float)

weights = np.array([
    1.0,
], dtype=float)

offsets_deg = np.array([
    45.0,
], dtype=float)

hessian_components = np.array([
    [
        [3.0, -2.0, 1.0],
    ],
    [
        [-4.0, 1.0, 6.0],
    ],
], dtype=float)
""",
            "call": "evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
            "gold_call": "_oracle_evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
        },
        {
            "setup": """import numpy as np

ideal_conductivity_tensors = np.array([
    [
        [2.5, 0.0],
        [0.0, 2.5],
    ],
], dtype=float)

heat_capacities = np.array([
    12.5,
], dtype=float)

weights = np.array([
    1.0,
], dtype=float)

offsets_deg = np.array([
    137.0,
], dtype=float)

hessian_components = np.array([
    [
        [3.0, -7.0, -2.0],
    ],
], dtype=float)
""",
            "call": "evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
            "gold_call": "_oracle_evaluate_offset_response(ideal_conductivity_tensors.copy(), heat_capacities.copy(), weights.copy(), hessian_components.copy(), offsets_deg.copy())",
        },
    ]
