"""
A quadratic full-order term has the matrix form F(x kron x), where column i * N + j multiplies x_i x_j under row-major Kronecker ordering. A sparse monomial table stores each nonzero as (output_index, first_state_index, second_state_index, coefficient); repeated entries add to the same tensor coefficient.

Returns
-------
np.ndarray of shape (n_state, n_state**2), the float64 quadratic operator F
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_quadratic_operator(n_state: int, terms: np.ndarray) -> np.ndarray:
    """Assemble the full quadratic operator from sparse monomial terms.

    Parameters
    ----------
    n_state : int
        Positive dimension N of the full-order state.
    terms : np.ndarray
        Array of shape (n_terms, 4) containing output, first input, second input,
        and coefficient columns. The first three columns must contain
        integer-valued indices in the half-open range [0, N).

    Returns
    -------
    quadratic : np.ndarray
        Quadratic operator F of shape (N, N * N).

    Raises
    ------
    ValueError
        If n_state is not a positive integer, if terms is not a finite array
        with four columns, or if a term index is nonintegral or outside [0, N).
    """
    return quadratic

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_quadratic_operator(n_state: int, terms: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(n_state, (int, np.integer)) or int(n_state) < 1:
        raise ValueError("n_state must be a positive integer")
    n_state = int(n_state)
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or terms.shape[1] != 4:
        raise ValueError("terms must have shape (n_terms, 4)")
    if not np.all(np.isfinite(terms)):
        raise ValueError("terms must contain finite values")
    indices = terms[:, :3]
    if not np.all(indices == np.floor(indices)):
        raise ValueError("term indices must be integers")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= n_state):
        raise ValueError("term indices are outside the state range")

    quadratic = np.zeros((n_state, n_state * n_state), dtype=float)
    for (output, first, second), coefficient in zip(indices, terms[:, 3]):
        quadratic[output, first * n_state + second] += coefficient
    return quadratic

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def _pack(value):
    """Flatten a returned value into native numbers for comparison.

    Container sizes and array shapes are packed alongside the numbers, so a
    result carrying the right values in the wrong structure cannot compare equal.
    """
    if isinstance(value, dict):
        packed = [len(value)]
        for key in sorted(value):
            packed.extend(_pack(value[key]))
        return packed
    if isinstance(value, np.ndarray):
        packed = list(value.shape)
        packed.extend(round(float(entry), 12) for entry in value.ravel().tolist())
        return packed
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [round(float(value), 12)]


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
n_state = 3
terms = np.array([
    [0, 0, 1, 0.5],
    [0, 0, 1, 0.2],
    [1, 2, 2, -0.4],
    [2, 1, 0, 0.3],
], dtype=float)
""",
            "call": "_pack(assemble_quadratic_operator(n_state, terms))",
            "gold_call": "_pack(_oracle_assemble_quadratic_operator(n_state, terms))",
        },
        {
            "setup": """import numpy as np
n_state = 1
terms = np.array([[0, 0, 0, -0.25]], dtype=float)
""",
            "call": "_pack(assemble_quadratic_operator(n_state, terms))",
            "gold_call": "_pack(_oracle_assemble_quadratic_operator(n_state, terms))",
        },
        {
            "setup": """import numpy as np
n_state = 2
terms = np.array([[0, 0, 2, 1.0]], dtype=float)
def run_model():
    try:
        assemble_quadratic_operator(n_state, terms)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_assemble_quadratic_operator(n_state, terms)
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
