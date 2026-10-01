"""
Decouple the element transmission into the separable terms that carry a requested fraction of its occupation.

The journal method transports separable mode components with one-dimensional operators, which requires the two-dimensional element to be expressed as a sum of terms that each factorize into a horizontal function and a vertical function. Conventions: the terms are ordered by decreasing singular value; each returned term already includes its singular value, so the terms sum to the transmission when all are kept; the retained count $K$ is the smallest number of leading terms whose cumulative share of the sum of squared singular values is at least ``occupation``; an ``occupation`` of one keeps every term of the decomposition. The result has shape ``(K, ny, nx)`` and does not depend on the phase conventions of the underlying vectors.

Returns
-------
np.ndarray, terms: Complex128 array of shape ``(K, ny, nx)``; term ``k`` is the ``k``-th separable term including its singular value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def element_terms(
    transmission: "np.ndarray",
    occupation: float,
) -> "np.ndarray":
    """Return the leading separable terms of the element transmission.

    Parameters
    ----------
    transmission : np.ndarray
        Complex array of shape ``(ny, nx)`` with both sides at least 2.
    occupation : float
        Target cumulative share of the sum of squared singular values,
        in ``(0, 1]``.

    Returns
    -------
    terms : np.ndarray
        Complex128 array of shape ``(K, ny, nx)``; term ``k`` is the
        ``k``-th separable term including its singular value.

    Raises
    ------
    ValueError
        If ``transmission`` is not a two-dimensional field or
        ``occupation`` is outside ``(0, 1]``.
    """
    return terms

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_element_terms(
    transmission: "np.ndarray",
    occupation: float,
) -> "np.ndarray":
    """Reference implementation."""
    field = np.asarray(transmission)
    if field.ndim != 2 or min(field.shape) < 2:
        raise ValueError("transmission must be a two-dimensional field.")
    target = float(occupation)
    if not np.isfinite(target) or target <= 0.0 or target > 1.0:
        raise ValueError("occupation must lie in (0, 1].")
    left, values, right = np.linalg.svd(field, full_matrices=False)
    total = float(np.sum(values ** 2))
    if total == 0.0:
        raise ValueError("transmission must not vanish identically.")
    shares = np.cumsum(values ** 2) / total
    reached = np.nonzero(shares >= target)[0]
    if target == 1.0:
        count = int(values.shape[0])
    else:
        count = int(reached[0]) + 1 if reached.size else int(values.shape[0])
    terms = np.empty((count, field.shape[0], field.shape[1]), dtype=np.complex128)
    for k in range(count):
        terms[k] = values[k] * np.outer(left[:, k], right[k])
    return terms

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
               'element_inputs = ((6, 8), (2.5, 1.5), 0.3, 0.04)\n'
               'model_transmission = _review_independent(element_transmission, *element_inputs)\n'
               'reference_transmission = _review_independent(_oracle_element_transmission, *element_inputs)',
      'call': '_review_independent(element_terms, model_transmission, 0.99)',
      'gold_call': '_review_independent(_oracle_element_terms, reference_transmission, 0.99)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'element_inputs = ((5, 7), (3.0, 2.0), -0.4, 0.0)\n'
               'model_transmission = _review_independent(element_transmission, *element_inputs)\n'
               'reference_transmission = _review_independent(_oracle_element_transmission, *element_inputs)',
      'call': '_review_independent(element_terms, model_transmission, 1.0)',
      'gold_call': '_review_independent(_oracle_element_terms, reference_transmission, 1.0)',
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
               'element_inputs = ((5, 7), (3.0, 2.0), -0.4, 0.0)\n'
               'model_transmission = _review_independent(element_transmission, *element_inputs)\n'
               'reference_transmission = _review_independent(_oracle_element_transmission, *element_inputs)',
      'call': '_review_independent(invalid, element_terms, (model_transmission, 1.2))',
      'gold_call': '_review_independent(invalid, _oracle_element_terms, (reference_transmission, 1.2))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'model = _review_independent(element_transmission, (5, 7), (3.0, 2.0), 0.0, 0.0)\n'
               'reference = _review_independent(_oracle_element_transmission, (5, 7), (3.0, 2.0), 0.0, 0.0)',
      'call': '_review_independent(element_terms, model, 1.0)',
      'gold_call': '_review_independent(_oracle_element_terms, reference, 1.0)',
      'tol': 1e-10}]
