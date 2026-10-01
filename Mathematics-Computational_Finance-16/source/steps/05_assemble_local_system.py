"""
Fixing all tensor-train cores except core j makes the explicit BSDE loss linear in that core, turning the nonlinear tensor-train fit into a sequence of local linear regressions solved one core at a time in an alternating least-squares sweep. The feature tensor contains one value channel followed by q complete coordinate-ordered derivative blocks, for 1 + qd channels. Contracting the fixed cores to the left and right gives channel-specific environment vectors. Their outer product with the local feature vector is summed over every block and vectorized in C order to yield one row of the local regression matrix.

Inputs
------
weighted_features: Array of shape (1 + qd, K, d, m) for a positive integer q.
cores: Sequence of d compatible order-three TT cores.
core_index: Zero-based position of the core being optimized.

Returns
-------
design_matrix: Float array of shape (K, r_{j-1} m r_j).

Returns
-------
np.ndarray of shape (K, r_left * m * r_right), the C-order local BSDE matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_local_system(
    weighted_features: np.ndarray,
    cores: list[np.ndarray],
    core_index: int,
) -> np.ndarray:
    """Assemble the derivative-aware local BSDE regression matrix.

    Parameters
    ----------
    weighted_features : np.ndarray
        One value channel followed by q positive coordinate-ordered derivative
        blocks, with shape (1 + q*d, K, d, m).
    cores : list[np.ndarray]
        Compatible TT cores with basis mode m and exterior ranks one.
    core_index : int
        Zero-based index of the core treated as the local unknown.

    Raises
    ------
    ValueError
        If the feature tensor is malformed, does not contain a whole positive
        number of derivative blocks, or is non-finite, if the cores are not a
        finite compatible tensor train, or if `core_index` is outside [0, d).

    Returns
    -------
    design_matrix : np.ndarray
        Local matrix with shape (K, r_{j-1} m r_j).
    """
    return design_matrix  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_assemble_local_system(
    weighted_features: np.ndarray,
    cores: list[np.ndarray],
    core_index: int,
) -> np.ndarray:
    """Reference implementation."""
    features = np.asarray(weighted_features, dtype=float)
    if features.ndim != 4:
        raise ValueError("weighted_features must have four dimensions")
    channel_count, sample_count, dimension, basis_count = features.shape
    if (
        dimension == 0
        or sample_count == 0
        or basis_count == 0
        or channel_count <= 1
        or (channel_count - 1) % dimension != 0
    ):
        raise ValueError("weighted_features must have shape (1 + q*d, K, d, m) for q >= 1")
    if not np.all(np.isfinite(features)):
        raise ValueError("weighted_features must be finite")
    if not isinstance(core_index, (int, np.integer)) or not 0 <= int(core_index) < dimension:
        raise ValueError("core_index must identify one TT core")
    if not isinstance(cores, (list, tuple)) or len(cores) != dimension:
        raise ValueError("cores must contain one tensor per coordinate")
    parsed = [np.asarray(core, dtype=float) for core in cores]
    if any(core.ndim != 3 or core.shape[1] != basis_count for core in parsed):
        raise ValueError("every core must have the feature basis mode")
    if parsed[0].shape[0] != 1 or parsed[-1].shape[2] != 1:
        raise ValueError("the exterior TT ranks must equal one")
    if any(parsed[i].shape[2] != parsed[i + 1].shape[0] for i in range(dimension - 1)):
        raise ValueError("adjacent TT ranks must agree")
    if any(not np.all(np.isfinite(core)) for core in parsed):
        raise ValueError("cores must be finite")

    position = int(core_index)
    local_shape = parsed[position].shape
    matrix = np.zeros((sample_count, int(np.prod(local_shape))), dtype=float)
    for k in range(sample_count):
        local_row = np.zeros(local_shape, dtype=float)
        for channel in range(channel_count):
            left = np.ones(1, dtype=float)
            for coordinate in range(position):
                evaluated = np.tensordot(
                    parsed[coordinate], features[channel, k, coordinate], axes=(1, 0)
                )
                left = left @ evaluated
            right = np.ones(1, dtype=float)
            for coordinate in range(dimension - 1, position, -1):
                evaluated = np.tensordot(
                    parsed[coordinate], features[channel, k, coordinate], axes=(1, 0)
                )
                right = evaluated @ right
            local_row += np.einsum(
                "a,q,b->aqb", left, features[channel, k, position], right
            )
        matrix[k] = local_row.reshape(-1)
    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
x = np.array([[-0.8, -0.5, 0.2], [0.4, 0.5, -0.1]])
xi = np.array([[0.2, -1.1, 0.5], [-0.2, -0.7, 0.9]])
sigma = np.array([0.35, 0.22, 0.18])
basis = np.stack((np.ones_like(x), x, x * x), axis=2)
derivative = np.stack((np.zeros_like(x), np.ones_like(x), 2.0 * x), axis=2)
features = np.broadcast_to(basis, (4, 2, 3, 3)).copy()
sn = xi * sigma[None, :] * np.sqrt(0.15)
for i in range(3):
    features[i + 1, :, i, :] = sn[:, i, None] * derivative[:, i, :]
c1 = np.array([[[0.7071067811865476, 0.0], [0.0, 1.0], [0.7071067811865476, 0.0]]])
c2 = np.zeros((2, 3, 2))
c3 = np.array([[[0.7071067811865476], [0.0], [0.7071067811865476]], [[0.0], [1.0], [0.0]]])
cores = [c1, c2, c3]
""",
            "call": "np.round(assemble_local_system(features, cores, 1), 12).tolist()",
            "gold_call": "np.round(_oracle_assemble_local_system(features, cores, 1), 12).tolist()",
        },
        {
            "setup": """import numpy as np
features = np.array([[[[1.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0]]]])
cores = [np.zeros((1, 3, 1))]
""",
            "call": "assemble_local_system(features, cores, 0).tolist()",
            "gold_call": "_oracle_assemble_local_system(features, cores, 0).tolist()",
        },
        {
            "setup": """import numpy as np
sample_count, dimension, basis_count, block_count = 2, 4, 4, 2
channel_count = 1 + block_count * dimension
raw_features = np.arange(channel_count * sample_count * dimension * basis_count, dtype=float)
features = ((7.0 * raw_features) % 29.0 - 14.0).reshape(channel_count, sample_count, dimension, basis_count) / 11.0
ranks = [1, 2, 3, 2, 1]
cores = []
for coordinate in range(dimension):
    size = ranks[coordinate] * basis_count * ranks[coordinate + 1]
    raw = np.arange(1, size + 1, dtype=float).reshape(ranks[coordinate], basis_count, ranks[coordinate + 1])
    cores.append(((raw * (coordinate + 3)) % 17.0 - 8.0) / (coordinate + 4.0))
indices = (0, 2, 3)
""",
            "call": "[np.round(assemble_local_system(features, cores, j), 12).tolist() for j in indices]",
            "gold_call": "[np.round(_oracle_assemble_local_system(features, cores, j), 12).tolist() for j in indices]",
        },
        {
            "setup": """import numpy as np
features = np.zeros((4, 1, 2, 3))
cores = [np.zeros((1, 3, 2)), np.zeros((2, 3, 1))]
def run_model():
    try:
        assemble_local_system(features, cores, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_assemble_local_system(features, cores, 0)
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
features = np.zeros((4, 1, 3, 3))
features[2, 0, 1, 2] = np.nan
cores = [np.ones((1, 3, 2)), np.ones((2, 3, 2)), np.ones((2, 3, 1))]
def run_model():
    try:
        assemble_local_system(features, cores, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_assemble_local_system(features, cores, 1)
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
