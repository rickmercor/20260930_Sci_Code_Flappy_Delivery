"""
The value-function gradient determines the reverse-diffusion feedback for diagonal diffusion, connecting the fitted tensor-train continuation value back to the sampler's drift correction. u*(x) = -s * grad V(x). The adaptive sweep state stores the final two TT components after the middle-to-right core shift. The shared basis mode m is inferred from the first core, and each coordinate uses monomials [1, z, ..., z^(m-1)]. Reconstructing the packed components and replacing one basis vector at a time by its derivative gives all three gradient components at the query state.

Inputs
------
query: Three-factor state with shape (3,).
left_core: First TT core with shape (1, m, r_1), with m at least two.
adaptive_state: Packed tau, shifted middle core, and updated last core.
sigma_diag: Positive diffusion diagonal with shape (3,).

Returns
-------
feedback: Float array of shape (3,).

Returns
-------
np.ndarray of shape (3,), the feedback -sigma_diag * gradient V at query
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_feedback_control(
    query: np.ndarray,
    left_core: np.ndarray,
    adaptive_state: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    """Recover the three-factor monomial-basis feedback from the adapted TT.

    Parameters
    ----------
    query : np.ndarray
        Evaluation state with shape (3,).
    left_core : np.ndarray
        First TT core with shape (1, m, r_1), with m at least two. All three
        coordinates use monomials of degrees 0 through m-1.
    adaptive_state : np.ndarray
        Packed tau, shifted middle core, and updated last core.
    sigma_diag : np.ndarray
        Positive diagonal entries of the diffusion matrix.

    Raises
    ------
    ValueError
        If an input is non-finite, if array shapes or the packed length are
        incompatible with a three-core shared-monomial-basis TT, if the basis
        mode is less than two, if the packed rank is not positive, or if
        `sigma_diag` or the stored tau is not positive.

    Returns
    -------
    feedback : np.ndarray
        Gradient feedback vector with shape (3,).
    """
    return feedback  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_recover_feedback_control(
    query: np.ndarray,
    left_core: np.ndarray,
    adaptive_state: np.ndarray,
    sigma_diag: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    point = np.asarray(query, dtype=float)
    first = np.asarray(left_core, dtype=float)
    packed = np.asarray(adaptive_state, dtype=float)
    sigma = np.asarray(sigma_diag, dtype=float)
    if point.shape != (3,):
        raise ValueError("query must have shape (3,)")
    if first.ndim != 3 or first.shape[0] != 1 or first.shape[1] < 2 or first.shape[2] == 0:
        raise ValueError("left_core must have shape (1, m, r_1) with m >= 2")
    if packed.ndim != 1 or packed.size < 2:
        raise ValueError("adaptive_state must be a non-empty vector")
    if sigma.shape != (3,):
        raise ValueError("sigma_diag must have shape (3,)")
    if not all(np.all(np.isfinite(array)) for array in (point, first, packed, sigma)):
        raise ValueError("all inputs must be finite")
    if packed[0] <= 0.0:
        raise ValueError("the stored adaptive regularization must be positive")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_diag must be positive")
    basis_count = first.shape[1]
    rank_left = first.shape[2]
    denominator = basis_count * (rank_left + 1)
    if (packed.size - 1) % denominator != 0:
        raise ValueError("adaptive_state length is incompatible with left_core")
    rank_right = (packed.size - 1) // denominator
    if rank_right <= 0:
        raise ValueError("the packed right rank must be positive")

    middle_size = rank_left * basis_count * rank_right
    middle = packed[1 : 1 + middle_size].reshape(rank_left, basis_count, rank_right)
    last = packed[1 + middle_size :].reshape(rank_right, basis_count, 1)
    cores = [first, middle, last]
    degree = np.arange(basis_count, dtype=int)
    basis = [np.power(z, degree) for z in point]
    derivative = []
    for z in point:
        differentiated = np.zeros(basis_count, dtype=float)
        differentiated[1:] = degree[1:] * np.power(z, degree[:-1])
        derivative.append(differentiated)
    gradient = np.empty(3, dtype=float)
    for differentiated in range(3):
        product = np.ones((1, 1), dtype=float)
        for coordinate, core in enumerate(cores):
            vector = derivative[coordinate] if coordinate == differentiated else basis[coordinate]
            product = product @ np.tensordot(core, vector, axes=(1, 0))
        gradient[differentiated] = product.item()
    return -sigma * gradient

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
query = np.array([0.35, -0.15, 0.55])
first = np.array([[[0.7071067811865476, 0.0], [0.0, 1.0], [0.7071067811865476, 0.0]]])
middle = np.array([[[0.8, -0.1], [0.2, 0.3], [-0.05, 0.15]], [[0.1, 0.6], [-0.25, 0.2], [0.12, -0.08]]])
last = np.array([[[0.7], [0.1], [-0.04]], [[-0.2], [0.5], [0.09]]])
packed = np.concatenate(([0.02], middle.ravel(), last.ravel()))
sigma = np.array([0.35, 0.22, 0.18])
""",
            "call": "np.round(recover_feedback_control(query, first, packed, sigma), 12).tolist()",
            "gold_call": "np.round(_oracle_recover_feedback_control(query, first, packed, sigma), 12).tolist()",
        },
        {
            "setup": """import numpy as np
query = np.zeros(3)
first = np.array([[[1.0], [0.0], [0.0]]])
middle = np.array([[[1.0], [0.0], [0.0]]])
last = np.array([[[1.0], [0.0], [0.0]]])
packed = np.concatenate(([0.1], middle.ravel(), last.ravel()))
sigma = np.ones(3)
""",
            "call": "recover_feedback_control(query, first, packed, sigma).tolist()",
            "gold_call": "_oracle_recover_feedback_control(query, first, packed, sigma).tolist()",
        },
        {
            "setup": """import numpy as np
query = np.array([-0.65, 0.25, 0.8])
rank_left, rank_right, basis_count = 3, 4, 5
raw_first = np.arange(1, 1 + basis_count * rank_left, dtype=float).reshape(1, basis_count, rank_left)
raw_middle = np.arange(1, 1 + rank_left * basis_count * rank_right, dtype=float).reshape(rank_left, basis_count, rank_right)
raw_last = np.arange(1, 1 + rank_right * basis_count, dtype=float).reshape(rank_right, basis_count, 1)
first = ((5.0 * raw_first) % 17.0 - 8.0) / 7.0
middle = ((7.0 * raw_middle) % 23.0 - 11.0) / 9.0
last = ((11.0 * raw_last) % 19.0 - 9.0) / 8.0
packed = np.concatenate(([0.035], middle.ravel(), last.ravel()))
sigma = np.array([0.17, 0.43, 0.29])
""",
            "call": "np.round(recover_feedback_control(query, first, packed, sigma), 12).tolist()",
            "gold_call": "np.round(_oracle_recover_feedback_control(query, first, packed, sigma), 12).tolist()",
        },
        {
            "setup": """import numpy as np
query = np.zeros(3)
first = np.zeros((1, 1, 2))
packed = np.ones(7)
sigma = np.ones(3)
def run_model():
    try:
        recover_feedback_control(query, first, packed, sigma)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_recover_feedback_control(query, first, packed, sigma)
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
query = np.array([0.0, np.nan, 0.0])
first = np.ones((1, 3, 2))
middle = np.ones((2, 3, 3))
last = np.ones((3, 3, 1))
packed = np.concatenate(([0.1], middle.ravel(), last.ravel()))
sigma = np.ones(3)
def run_model():
    try:
        recover_feedback_control(query, first, packed, sigma)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_recover_feedback_control(query, first, packed, sigma)
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
