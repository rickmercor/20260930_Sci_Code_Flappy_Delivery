"""
After a local ALS update, adapting the regularization to the current residual keeps the ridge penalty scaled to the fit, and shifting the optimized core through the tensor train with a gauge-fixed SVD lets the next local system be solved for the neighboring core while preserving the value-function norm as the active core's Frobenius norm. The regularization magnitude is reset to gamma times the squared residual divided by the squared Frobenius norm of the updated core. Shifting the core one position to the right uses a reduced singular value decomposition: the left singular vectors remain in the middle TT component, while the singular values and right singular vectors are absorbed into the last component. The bond gauge is right anchored: absorbed-right rows are ordered by decreasing squared norm with stable original-index ties, then each row is oriented by its first largest-magnitude coefficient and the same sign is applied to the matching left singular vector. A zero absorbed row falls back to the corresponding left-vector pivot. The next local system sums one value channel and every complete derivative-channel block before solving with the adapted regularization.

Inputs
------
weighted_features: Three-coordinate BSDE feature channels of shape (1 + 3q, K, 3, m).
middle_design: Local matrix used for the middle-core update.
targets: BSDE target vector.
left_core: Left-orthonormal first TT core.
updated_middle_core: Result of the preceding local solve.
right_core: Right TT component before the shift.
gamma: Positive relative regularization weight.

Returns
-------
adaptive_state: One-dimensional array containing tau_next, the shifted middle core, and the updated last core in C order.

Returns
-------
np.ndarray of length 1 + r_1*m*r_2 + r_2*m, the adaptive sweep state in C order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_adaptive_sweep(
    weighted_features: np.ndarray,
    middle_design: np.ndarray,
    targets: np.ndarray,
    left_core: np.ndarray,
    updated_middle_core: np.ndarray,
    right_core: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Adapt regularization, shift the TT core, and update the last core.

    Parameters
    ----------
    weighted_features : np.ndarray
        One value channel followed by q positive three-coordinate derivative
        blocks, with shape (1 + 3*q, K, 3, m).
    middle_design : np.ndarray
        Local matrix from the preceding middle-core micro-step.
    targets : np.ndarray
        BSDE target vector with one value per sample.
    left_core : np.ndarray
        First TT core with shape (1, m, r_1).
    updated_middle_core : np.ndarray
        Updated middle core with shape (r_1, m, r_2).
    right_core : np.ndarray
        Last TT core with shape (r_2, m, 1).
    gamma : float
        Finite positive relative regularization weight.

    Raises
    ------
    ValueError
        If features, matrix, targets, or cores have incompatible shapes or
        non-finite values, if the TT ranks cannot be preserved by the SVD shift,
        if `gamma` is not positive, or if the adaptive magnitude is not positive.

    Returns
    -------
    adaptive_state : np.ndarray
        Packed tau, right-anchored shifted middle core, and updated last core
        as float64. The bond gauge is right anchored: absorbed-right rows are
        ordered by decreasing squared norm (stable ties), each row is oriented
        so its first largest-magnitude coefficient is positive, and the same
        sign is applied to the matching left singular vector. A zero right row
        uses the corresponding left-vector pivot.
    """
    return adaptive_state  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _assemble_last_system(features, cores):
    """Assemble the core-index-two local system for a three-core TT."""
    sample_count = features.shape[1]
    local_shape = cores[2].shape
    matrix = np.zeros((sample_count, int(np.prod(local_shape))), dtype=float)
    for k in range(sample_count):
        row = np.zeros(local_shape, dtype=float)
        for channel in range(features.shape[0]):
            left = np.ones(1, dtype=float)
            for coordinate in range(2):
                evaluated = np.tensordot(
                    cores[coordinate], features[channel, k, coordinate], axes=(1, 0)
                )
                left = left @ evaluated
            row += np.einsum("a,q,b->aqb", left, features[channel, k, 2], np.ones(1))
        matrix[k] = row.reshape(-1)
    return matrix


def _oracle_advance_adaptive_sweep(
    weighted_features: np.ndarray,
    middle_design: np.ndarray,
    targets: np.ndarray,
    left_core: np.ndarray,
    updated_middle_core: np.ndarray,
    right_core: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Reference implementation."""
    features = np.asarray(weighted_features, dtype=float)
    matrix = np.asarray(middle_design, dtype=float)
    y = np.asarray(targets, dtype=float)
    first = np.asarray(left_core, dtype=float)
    middle = np.asarray(updated_middle_core, dtype=float)
    last = np.asarray(right_core, dtype=float)
    if (
        features.ndim != 4
        or features.shape[2] != 3
        or features.shape[0] <= 1
        or (features.shape[0] - 1) % 3 != 0
    ):
        raise ValueError("weighted_features must have shape (1 + 3*q, K, 3, m) for q >= 1")
    sample_count, basis_count = features.shape[1], features.shape[3]
    if sample_count == 0 or basis_count == 0:
        raise ValueError("weighted_features must be non-empty")
    if matrix.shape != (sample_count, middle.size):
        raise ValueError("middle_design must match the samples and middle core size")
    if y.shape != (sample_count,):
        raise ValueError("targets must have one value per sample")
    if first.ndim != 3 or first.shape[0] != 1 or first.shape[1] != basis_count:
        raise ValueError("left_core must have shape (1, m, r_1)")
    if middle.ndim != 3 or middle.shape[:2] != (first.shape[2], basis_count):
        raise ValueError("updated_middle_core must have shape (r_1, m, r_2)")
    if last.shape != (middle.shape[2], basis_count, 1):
        raise ValueError("right_core must have shape (r_2, m, 1)")
    arrays = (features, matrix, y, first, middle, last)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("all array inputs must be finite")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if middle.shape[0] * basis_count < middle.shape[2]:
        raise ValueError("the SVD shift cannot preserve the right rank")

    residual_sq = float(np.dot(matrix @ middle.reshape(-1) - y, matrix @ middle.reshape(-1) - y))
    core_norm_sq = float(np.dot(middle.reshape(-1), middle.reshape(-1)))
    if core_norm_sq == 0.0:
        raise ValueError("the updated middle core must have positive norm")
    tau_next = float(gamma) * residual_sq / core_norm_sq
    if not np.isfinite(tau_next) or tau_next <= 0.0:
        raise ValueError("the adaptive regularization magnitude must be positive")

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        middle.reshape(middle.shape[0] * basis_count, middle.shape[2]),
        full_matrices=False,
    )
    transfer = singular_values[:, None] * right_vectors
    absorbed_matrix = transfer @ last.reshape(last.shape[0], basis_count)
    row_energy = np.sum(absorbed_matrix * absorbed_matrix, axis=1)
    bond_order = np.lexsort((np.arange(row_energy.size), -row_energy))
    left_vectors = left_vectors[:, bond_order]
    absorbed_matrix = absorbed_matrix[bond_order]
    for bond in range(absorbed_matrix.shape[0]):
        right_pivot = int(np.argmax(np.abs(absorbed_matrix[bond])))
        sign = float(np.sign(absorbed_matrix[bond, right_pivot]))
        if sign == 0.0:
            left_pivot = int(np.argmax(np.abs(left_vectors[:, bond])))
            sign = float(np.sign(left_vectors[left_pivot, bond]))
            if sign == 0.0:
                sign = 1.0
        left_vectors[:, bond] *= sign
        absorbed_matrix[bond] *= sign
    shifted_middle = left_vectors.reshape(middle.shape)
    absorbed_last = absorbed_matrix.reshape(last.shape)
    last_design = _assemble_last_system(features, [first, shifted_middle, absorbed_last])
    normal_matrix = last_design.T @ last_design + tau_next * np.eye(last_design.shape[1])
    updated_last = np.linalg.solve(normal_matrix, last_design.T @ y).reshape(last.shape)
    return np.concatenate(([tau_next], shifted_middle.reshape(-1), updated_last.reshape(-1)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
sample_count, basis_count, block_count = 4, 4, 2
channel_count = 1 + 3 * block_count
raw_features = np.arange(channel_count * sample_count * 3 * basis_count, dtype=float)
features = ((5.0 * raw_features) % 31.0 - 15.0).reshape(channel_count, sample_count, 3, basis_count) / 13.0
raw_first = np.arange(1, 1 + basis_count * 2, dtype=float).reshape(1, basis_count, 2)
raw_middle = np.arange(1, 1 + 2 * basis_count * 3, dtype=float).reshape(2, basis_count, 3)
raw_last = np.arange(1, 1 + 3 * basis_count, dtype=float).reshape(3, basis_count, 1)
first = ((3.0 * raw_first) % 11.0 - 5.0) / 7.0
middle = ((7.0 * raw_middle) % 19.0 - 9.0) / 8.0
last = ((11.0 * raw_last) % 23.0 - 11.0) / 9.0
A = ((13.0 * np.arange(sample_count * middle.size, dtype=float)) % 37.0 - 18.0).reshape(sample_count, middle.size) / 17.0
y = np.array([0.3, -0.2, 0.7, -0.45])
""",
            "call": "np.round(advance_adaptive_sweep(features, A, y, first, middle, last, 0.2), 12).tolist()",
            "gold_call": "np.round(_oracle_advance_adaptive_sweep(features, A, y, first, middle, last, 0.2), 12).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3]])
features = np.zeros((4, 2, 3, 3))
features[0, :, :, 0] = 1.0
first = np.array([[[1.0], [0.0], [0.0]]])
middle = np.array([[[0.5], [0.1], [-0.2]]])
last = np.array([[[1.0], [0.0], [0.0]]])
A = np.array([[1.0, 0.0, 0.0], [1.0, 0.2, -0.1]])
y = np.array([0.4, 0.7])
""",
            "call": "np.round(advance_adaptive_sweep(features, A, y, first, middle, last, 0.1), 12).tolist()",
            "gold_call": "np.round(_oracle_advance_adaptive_sweep(features, A, y, first, middle, last, 0.1), 12).tolist()",
        },
        {
            "setup": """import numpy as np
features = np.zeros((4, 2, 3, 3))
first = np.zeros((1, 3, 1))
middle = np.ones((1, 3, 1))
last = np.zeros((1, 3, 1))
A = np.ones((2, 3))
y = np.ones(2)
def run_model():
    try:
        advance_adaptive_sweep(features, A, y, first, middle, last, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_advance_adaptive_sweep(features, A, y, first, middle, last, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        {
            "setup": """import numpy as np
features = np.zeros((5, 2, 3, 3))
first = np.ones((1, 3, 1))
middle = np.ones((1, 3, 1))
last = np.ones((1, 3, 1))
A = np.ones((2, 3))
y = np.array([0.5, -0.2])
def run_model():
    try:
        advance_adaptive_sweep(features, A, y, first, middle, last, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_advance_adaptive_sweep(features, A, y, first, middle, last, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
