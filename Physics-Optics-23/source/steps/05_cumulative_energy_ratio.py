"""
Evaluate the nominal cumulative compression score at a requested number of shared residual components.

The score applies the journal's cumulative-energy expression to the eigenvalue-weighted residual stack prescribed for this benchmark. It is an operational rank-selection score, not the physical power fraction of the restored fields. Conventions: the modes have unit Euclidean norm and positive finite eigenvalues that need not sum to one; $n_keep$ ranges from zero through the residual-stack rank; zero components gives the eigenvalue-weighted separable-energy ratio; the ratio is nondecreasing in $n_keep$, equals one at full rank, and is returned as a native Python float.

Returns
-------
float, ratio: Native Python float in [0, 1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cumulative_energy_ratio(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
) -> float:
    """Return the nominal cumulative compression score at ``n_keep``.

    Every mode must have unit Euclidean norm. Zero retained components
    returns the eigenvalue-weighted separable-energy ratio.

    Parameters
    ----------
    modes : np.ndarray
        Complex unit-norm modes of shape ``(n_modes, ny, nx)``.
    eigenvalues : np.ndarray
        Positive finite eigenvalues, shape ``(n_modes,)``.
    n_keep : int
        Number of compressed residual components, from zero through the
        residual-stack rank.

    Returns
    -------
    ratio : float
        Native Python float in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the eigenvalues do not match, a mode is not unit-norm, or
        ``n_keep`` is outside the residual-stack rank.
    """
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cumulative_energy_ratio(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
) -> float:
    """Reference implementation."""
    fields = np.asarray(modes)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    if fields.ndim != 3 or weights.ndim != 1 or weights.shape[0] != fields.shape[0]:
        raise ValueError("modes and eigenvalues must describe the same set of modes.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("eigenvalues must be positive and finite.")
    norms = np.linalg.norm(fields.reshape(fields.shape[0], -1), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("every mode must have unit Euclidean norm.")
    keep = int(n_keep)
    if keep < 0:
        raise ValueError("n_keep must be nonnegative.")
    leading = np.array(
        [np.linalg.svd(field, compute_uv=False)[0] for field in fields], dtype=np.float64
    )
    separable = float(np.sum(weights * leading ** 2) / np.sum(weights))
    if keep == 0:
        return separable
    residuals = _oracle_coupled_residuals(fields)
    stack = _oracle_residual_stack(residuals, weights)
    beta = np.linalg.svd(stack, compute_uv=False)
    if keep > beta.shape[0]:
        raise ValueError("n_keep exceeds the residual-stack rank.")
    total = float(np.sum(beta ** 2))
    if total == 0.0:
        return separable
    fraction = float(np.sum(beta[:keep] ** 2) / total)
    return float(separable + fraction * (1.0 - separable))

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
               'mode_inputs = ((6, 8), np.array([1.4, 2.2, 1.1]), np.array([[0.2, -0.1], [0.0, 0.15], [-0.2, '
               '0.05]]), np.array([0.03, -0.05, 0.08]))\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'eigenvalues = np.array([0.5, 0.3, 0.2])',
      'call': '_review_independent(cumulative_energy_ratio, model_modes, eigenvalues, 2)',
      'gold_call': '_review_independent(_oracle_cumulative_energy_ratio, reference_modes, eigenvalues, 2)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mode_inputs = ((5, 5), np.array([1.0, 1.7]), np.array([[0.1, 0.1], [-0.3, 0.2]]), np.array([0.0, '
               '0.12]))\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'eigenvalues = np.array([0.8, 0.2])',
      'call': '_review_independent(cumulative_energy_ratio, model_modes, eigenvalues, 0)',
      'gold_call': '_review_independent(_oracle_cumulative_energy_ratio, reference_modes, eigenvalues, 0)',
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
               'mode_inputs = ((5, 5), np.array([1.0, 1.7]), np.array([[0.1, 0.1], [-0.3, 0.2]]), np.array([0.0, '
               '0.12]))\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'eigenvalues = np.array([0.8, 0.2])',
      'call': '_review_independent(invalid, cumulative_energy_ratio, (2.0 * model_modes, eigenvalues, 1))',
      'gold_call': '_review_independent(invalid, _oracle_cumulative_energy_ratio, (2.0 * reference_modes, '
                   'eigenvalues, 1))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'modes = np.zeros((1, 2, 2), dtype=complex)\n'
               'modes[0, 0, 0] = 1\n'
               'w = np.array([1.0])',
      'call': '_review_independent(cumulative_energy_ratio, modes, w, 1)',
      'gold_call': '_review_independent(_oracle_cumulative_energy_ratio, modes, w, 1)',
      'tol': 1e-12}]
