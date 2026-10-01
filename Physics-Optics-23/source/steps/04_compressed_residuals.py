"""
Return the shared low-rank compression of the residual stack at a requested component count, as one field per mode.

The compression keeps a requested number of shared components of the residual stack and discards the rest; the kept part is returned row by row, each row reshaped in row-major order to the field shape ``(ny, nx)``.



Conventions: `$n_keep$` ranges from zero through the stack rank `$min(n_modes, ny * nx)$`; zero components returns an all-zero array; the returned fields still carry the eigenvalue weights applied when the stack was built, and the rescaling happens in a later step.

Returns
-------
np.ndarray, compressed: Complex128 array of shape ``(n_modes, ny, nx)``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compressed_residuals(
    stack: "np.ndarray",
    n_keep: int,
    field_shape: tuple,
) -> "np.ndarray":
    """Return the compressed residual fields for the requested component count.

    A request for zero components returns an array of zeros. Each row is
    reshaped in row-major order to ``field_shape``.

    Parameters
    ----------
    stack : np.ndarray
        Complex array of shape ``(n_modes, ny * nx)``.
    n_keep : int
        Number of residual components, from zero through the stack rank.
    field_shape : tuple
        ``(ny, nx)`` matching the row length of ``stack``.

    Returns
    -------
    compressed : np.ndarray
        Complex128 array of shape ``(n_modes, ny, nx)``.

    Raises
    ------
    ValueError
        If ``field_shape`` disagrees with ``stack`` or ``n_keep`` is
        outside the stack rank.
    """
    return compressed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compressed_residuals(
    stack: "np.ndarray",
    n_keep: int,
    field_shape: tuple,
) -> "np.ndarray":
    """Reference implementation."""
    matrix = np.asarray(stack)
    field_shape = tuple(int(value) for value in field_shape)
    if matrix.ndim != 2 or len(field_shape) != 2 or min(field_shape) < 2:
        raise ValueError("stack and field_shape must describe residual fields.")
    if matrix.shape[1] != field_shape[0] * field_shape[1]:
        raise ValueError("field_shape does not match the stacked row length.")
    keep = int(n_keep)
    rank = min(matrix.shape)
    if keep < 0 or keep > rank:
        raise ValueError("n_keep must lie between zero and the stack rank.")
    n_modes = matrix.shape[0]
    if keep == 0:
        return np.zeros((n_modes, field_shape[0], field_shape[1]), dtype=np.complex128)
    left, values, right = np.linalg.svd(matrix, full_matrices=False)
    approx = (left[:, :keep] * values[:keep]) @ right[:keep]
    return approx.reshape((n_modes, field_shape[0], field_shape[1]))

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
               'rng = np.random.default_rng(9)\n'
               'stack = rng.standard_normal((3, 20)) + 1j * rng.standard_normal((3, 20))\n'
               'field_shape = (4, 5)',
      'call': '_review_independent(compressed_residuals, stack, 2, field_shape)',
      'gold_call': '_review_independent(_oracle_compressed_residuals, stack, 2, field_shape)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'stack = np.array([[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 3.0, 0.0]], dtype=np.complex128)\n'
               'field_shape = (2, 2)',
      'call': '_review_independent(compressed_residuals, stack, 0, field_shape)',
      'gold_call': '_review_independent(_oracle_compressed_residuals, stack, 0, field_shape)',
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
               'stack = np.ones((2, 6), dtype=np.complex128)\n'
               'field_shape = (2, 3)',
      'call': '_review_independent(invalid, compressed_residuals, (stack, -1, field_shape))',
      'gold_call': '_review_independent(invalid, _oracle_compressed_residuals, (stack, -1, field_shape))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'r = np.array([[1, 2j, 3, 4, 5, 6], [-1j, 3, 2, -4j, 5, 1]], dtype=complex)',
      'call': '_review_independent(compressed_residuals, r, 2, (2, 3))',
      'gold_call': '_review_independent(_oracle_compressed_residuals, r, 2, (2, 3))',
      'tol': 1e-10}]
