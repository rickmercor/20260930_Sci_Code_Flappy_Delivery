"""
Remove the leading separable component of each coherent mode and return the coupled remainder.

In the journal compression a two-dimensional mode is split into a dominant part that factorizes into one horizontal and one vertical function and a smaller coupled remainder that does not. The remainder is the field that the shared residual compression of the later steps acts on.



Conventions: the leading separable component is removed together with its scale factor, so the remainder is exactly the mode minus that component; the returned field has the shape of the input mode; both spatial dimensions must be at least 2.

Returns
-------
np.ndarray, residuals: Complex128 array with the same shape as ``modes``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupled_residuals(modes: "np.ndarray") -> "np.ndarray":
    """Return the coupled residual of every coherent mode.

    Parameters
    ----------
    modes : np.ndarray
        Complex array of shape ``(n_modes, ny, nx)`` with both spatial
        sides at least 2.

    Returns
    -------
    residuals : np.ndarray
        Complex128 array with the same shape as ``modes``.

    Raises
    ------
    ValueError
        If ``modes`` does not have that shape.
    """
    return residuals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coupled_residuals(modes: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(modes)
    if fields.ndim != 3 or min(fields.shape[1:]) < 2:
        raise ValueError("modes must have shape (n_modes, ny, nx) with ny, nx >= 2.")
    residuals = np.empty(fields.shape, dtype=np.complex128)
    for index in range(fields.shape[0]):
        left, values, right = np.linalg.svd(fields[index], full_matrices=False)
        separable = values[0] * np.outer(left[:, 0], right[0])
        residuals[index] = fields[index] - separable
    return residuals

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
               'rng = np.random.default_rng(11)\n'
               'modes = rng.standard_normal((3, 6, 7)) + 1j * rng.standard_normal((3, 6, 7))',
      'call': '_review_independent(coupled_residuals, modes)',
      'gold_call': '_review_independent(_oracle_coupled_residuals, modes)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'modes = np.zeros((2, 4, 5), dtype=np.complex128)\n'
               'modes[0] = np.outer(np.array([1.0, 0.2, 0.0, -0.4]), np.array([0.5, -0.3, 0.7, 0.1, 0.0]))\n'
               'modes[0, 0, 0] += 0.8\n'
               'modes[1] = 1j * modes[0]\n'
               'modes[1, 2, 3] += 0.5 - 0.25j',
      'call': '_review_independent(coupled_residuals, modes)',
      'gold_call': '_review_independent(_oracle_coupled_residuals, modes)',
      'tol': 1e-10},
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
               'modes = np.ones((4, 4), dtype=np.complex128)',
      'call': '_review_independent(invalid, coupled_residuals, (modes,))',
      'gold_call': '_review_independent(invalid, _oracle_coupled_residuals, (modes,))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'modes = np.outer(np.array([1.0, 0.5, -0.2]), np.array([1.0, 0.3j, -0.4, 2.0])).reshape(1, 3, 4)',
      'call': '_review_independent(coupled_residuals, modes)',
      'gold_call': '_review_independent(_oracle_coupled_residuals, modes)',
      'tol': 1e-12}]
