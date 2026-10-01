"""
An extended tensor train represents a scalar function by contracting one matrix-valued core per coordinate against a univariate basis. Returning the value beside every partial derivative supplies the continuation data required by a BSDE time step. A core with basis mode m_i uses the coordinate-specific monomial vector phi_i(z) = [1, z, ..., z^(m_i-1)]; a partial derivative replaces only that vector by phi_i'(z). The prescribed quadratic instance is recovered when every m_i equals three.

Inputs
------
points: Float array of shape (K, d).
cores: Sequence of d compatible order-three TT cores with basis modes at least two.

Returns
-------
continuation: Float array of shape (K, d + 1), with values in column zero and gradients in the remaining columns.

Returns
-------
np.ndarray of shape (K, d + 1), value in column 0 and gradient in columns 1:d+1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_tt_continuation(points: np.ndarray, cores: list[np.ndarray]) -> np.ndarray:
    """Evaluate a coordinate-wise monomial tensor train and its gradient.

    Parameters
    ----------
    points : np.ndarray
        Evaluation points with shape (K, d).
    cores : list[np.ndarray]
        Compatible TT cores with shapes (r_{i-1}, m_i, r_i), r_0 = r_d = 1,
        and m_i at least two. Core i uses monomials of degrees 0 through m_i-1.

    Raises
    ------
    ValueError
        If `points` is empty or non-finite, if the number or shape of the cores
        is incompatible, if a basis mode is less than two, or if a core is non-finite.

    Returns
    -------
    continuation : np.ndarray
        Values and gradients with shape (K, d + 1).
    """
    return continuation  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_evaluate_tt_continuation(
    points: np.ndarray, cores: list[np.ndarray]
) -> np.ndarray:
    """Reference implementation."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("points must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("points must be finite")
    if not isinstance(cores, (list, tuple)) or len(cores) != x.shape[1]:
        raise ValueError("cores must contain one tensor for each coordinate")
    parsed = [np.asarray(core, dtype=float) for core in cores]
    if any(core.ndim != 3 or core.shape[1] < 2 for core in parsed):
        raise ValueError("every core must have shape (r_left, m_i, r_right) with m_i >= 2")
    if parsed[0].shape[0] != 1 or parsed[-1].shape[2] != 1:
        raise ValueError("the exterior TT ranks must equal one")
    if any(parsed[i].shape[2] != parsed[i + 1].shape[0] for i in range(len(parsed) - 1)):
        raise ValueError("adjacent TT ranks must agree")
    if any(not np.all(np.isfinite(core)) for core in parsed):
        raise ValueError("cores must be finite")

    sample_count, dimension = x.shape
    result = np.empty((sample_count, dimension + 1), dtype=float)
    for k in range(sample_count):
        basis = []
        derivative = []
        for z, core in zip(x[k], parsed):
            degree = np.arange(core.shape[1], dtype=int)
            basis.append(np.power(z, degree))
            differentiated = np.zeros(core.shape[1], dtype=float)
            differentiated[1:] = degree[1:] * np.power(z, degree[:-1])
            derivative.append(differentiated)
        value_product = np.ones((1, 1), dtype=float)
        for coordinate, core in enumerate(parsed):
            value_product = value_product @ np.tensordot(
                core, basis[coordinate], axes=(1, 0)
            )
        result[k, 0] = value_product.item()
        for differentiated in range(dimension):
            gradient_product = np.ones((1, 1), dtype=float)
            for coordinate, core in enumerate(parsed):
                vector = derivative[coordinate] if coordinate == differentiated else basis[coordinate]
                gradient_product = gradient_product @ np.tensordot(core, vector, axes=(1, 0))
            result[k, differentiated + 1] = gradient_product.item()
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
x = np.array([[-0.2, 0.4, 0.6], [0.7, -0.3, 0.1]])
c1 = np.array([[[0.75, -0.20], [0.10, 0.35], [-0.08, 0.12]]])
c2 = np.array([[[0.90, -0.15], [0.20, 0.25], [-0.10, 0.18]], [[0.05, 0.70], [-0.30, 0.12], [0.22, -0.08]]])
c3 = np.array([[[0.80], [0.15], [-0.05]], [[-0.20], [0.60], [0.10]]])
cores = [c1, c2, c3]
""",
            "call": "np.round(evaluate_tt_continuation(x, cores), 12).tolist()",
            "gold_call": "np.round(_oracle_evaluate_tt_continuation(x, cores), 12).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.array([[0.0], [1.0]])
cores = [np.array([[[2.0], [-1.0], [0.5]]])]
""",
            "call": "evaluate_tt_continuation(x, cores).tolist()",
            "gold_call": "_oracle_evaluate_tt_continuation(x, cores).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.array([[-0.7, 0.0, 0.45, -0.2, 0.9], [0.3, -0.8, 0.1, 0.65, -0.4]])
ranks = [1, 3, 2, 4, 2, 1]
modes = [2, 4, 3, 5, 2]
cores = []
for coordinate in range(5):
    size = ranks[coordinate] * modes[coordinate] * ranks[coordinate + 1]
    raw = np.arange(1, size + 1, dtype=float).reshape(ranks[coordinate], modes[coordinate], ranks[coordinate + 1])
    cores.append(((raw * (coordinate + 2)) % 13.0 - 6.0) / (coordinate + 3.0))
""",
            "call": "np.round(evaluate_tt_continuation(x, cores), 12).tolist()",
            "gold_call": "np.round(_oracle_evaluate_tt_continuation(x, cores), 12).tolist()",
        },
        {
            "setup": """import numpy as np
x = np.zeros((1, 2))
cores = [np.zeros((1, 1, 2)), np.zeros((2, 3, 1))]
def run_model():
    try:
        evaluate_tt_continuation(x, cores)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_evaluate_tt_continuation(x, cores)
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
x = np.array([[0.2, np.inf]])
cores = [np.ones((1, 3, 2)), np.ones((2, 3, 1))]
def run_model():
    try:
        evaluate_tt_continuation(x, cores)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_evaluate_tt_continuation(x, cores)
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
