"""
Stack the eigenvalue-weighted coupled residuals as the rows of one matrix.

The shared residual compression works on a single matrix that holds every mode's coupled remainder. Row `$n$` of the stack is the residual of mode `$n$$multiplied by its coherent-mode eigenvalue$`eigenvalues[n]`$and flattened in row-major (C) order, so the stack has shape$$(n_modes, ny * nx)$`. The eigenvalues are positive and finite; they need not sum to one.

Returns
-------
np.ndarray, stack: Complex128 array of shape ``(n_modes, ny * nx)``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def residual_stack(
    residuals: "np.ndarray",
    eigenvalues: "np.ndarray",
) -> "np.ndarray":
    """Return one row-major row for each eigenvalue-weighted residual.

    Parameters
    ----------
    residuals : np.ndarray
        Complex array of shape ``(n_modes, ny, nx)``.
    eigenvalues : np.ndarray
        Positive finite coherent-mode eigenvalues, shape ``(n_modes,)``.

    Returns
    -------
    stack : np.ndarray
        Complex128 array of shape ``(n_modes, ny * nx)``.

    Raises
    ------
    ValueError
        If the arrays do not match or an eigenvalue is not positive.
    """
    return stack

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_residual_stack(
    residuals: "np.ndarray",
    eigenvalues: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(residuals)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    if fields.ndim != 3 or weights.ndim != 1 or weights.shape[0] != fields.shape[0]:
        raise ValueError("residuals and eigenvalues must describe the same modes.")
    if fields.shape[0] < 1 or min(fields.shape[1:]) < 2:
        raise ValueError("residuals must have shape (n_modes, ny, nx) with ny, nx >= 2.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("eigenvalues must be positive and finite.")
    rows = [weights[index] * fields[index].ravel() for index in range(fields.shape[0])]
    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original cases with independent candidate/reference input graphs."""
    return [{'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'residuals = np.arange(2 * 3 * 4, dtype=np.complex128).reshape(2, 3, 4)\n'
               'residuals = residuals + 0.25j * residuals\n'
               'eigenvalues = np.array([0.7, 0.3])',
      'call': '_review_independent(residual_stack, residuals, eigenvalues)',
      'gold_call': '_review_independent(_oracle_residual_stack, residuals, eigenvalues)',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'rng = np.random.default_rng(4)\n'
               'residuals = rng.standard_normal((4, 5, 6)) + 1j * rng.standard_normal((4, 5, 6))\n'
               'eigenvalues = np.array([0.4, 0.3, 0.2, 0.1])',
      'call': '_review_independent(residual_stack, residuals, eigenvalues)',
      'gold_call': '_review_independent(_oracle_residual_stack, residuals, eigenvalues)',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               '\n'
               'def invalid(function, args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'residuals = np.ones((2, 3, 4), dtype=np.complex128)\n'
               'eigenvalues = np.array([0.5, 0.0])',
      'call': '_review_independent(invalid, residual_stack, (residuals, eigenvalues))',
      'gold_call': '_review_independent(invalid, _oracle_residual_stack, (residuals, eigenvalues))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'r = (np.arange(12).reshape(1, 3, 4) + 0.2j).astype(complex)\n'
               'w = np.array([2.0])',
      'call': '_review_independent(residual_stack, r, w)',
      'gold_call': '_review_independent(_oracle_residual_stack, r, w)',
      'tol': 1e-12}]
