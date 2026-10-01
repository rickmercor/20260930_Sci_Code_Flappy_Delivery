"""
Transport the restored coupled residual of every mode through the full element to the output plane.

The coupled remainders are the part of the compression that stays two dimensional. For each mode the compressed residual is restored, multiplied by the complete element transmission, and propagated as a two-dimensional field. Conventions from the problem statement: the compressed residual of mode `$n$` is divided by the eigenvalue ``eigenvalues[n]`` that weighted it when the stack was built; `$n_keep$` ranges from zero through the residual-stack rank and zero components gives an all-zero result; the modes have unit Euclidean norm and positive finite eigenvalues; propagation follows the same paraxial transfer function and unpadded discrete Fourier frequencies as the separable transport. The result has the shape of ``modes``.

Returns
-------
np.ndarray, transported: Complex128 array with the same shape as ``modes``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupled_transport(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
    transmission: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Return the transported restored coupled residual of each mode.

    Parameters
    ----------
    modes : np.ndarray
        Complex unit-norm modes of shape ``(n_modes, ny, nx)``.
    eigenvalues : np.ndarray
        Positive finite eigenvalues, shape ``(n_modes,)``.
    n_keep : int
        Number of compressed residual components, from zero through the
        residual-stack rank.
    transmission : np.ndarray
        Complex element transmission of shape ``(ny, nx)``.
    fresnel_parameter : float
        Finite propagation parameter in units of the squared pixel pitch.

    Returns
    -------
    transported : np.ndarray
        Complex128 array with the same shape as ``modes``.

    Raises
    ------
    ValueError
        If the eigenvalues or the transmission do not match the modes,
        ``n_keep`` is outside the residual-stack rank, or the propagation
        parameter is not finite.
    """
    return transported

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coupled_transport(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
    transmission: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(modes)
    weights = np.asarray(eigenvalues, dtype=np.float64)
    element = np.asarray(transmission)
    if fields.ndim != 3 or weights.shape != (fields.shape[0],):
        raise ValueError("modes and eigenvalues must describe the same set of modes.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("eigenvalues must be positive and finite.")
    if element.shape != fields.shape[1:]:
        raise ValueError("transmission must match the spatial shape of the modes.")
    parameter = float(fresnel_parameter)
    if not np.isfinite(parameter):
        raise ValueError("fresnel_parameter must be finite.")
    residuals = _oracle_coupled_residuals(fields)
    stack = _oracle_residual_stack(residuals, weights)
    compressed = _oracle_compressed_residuals(stack, n_keep, fields.shape[1:])
    restored = compressed / weights[:, None, None]
    transported = np.zeros(fields.shape, dtype=np.complex128)
    for index in range(fields.shape[0]):
        transported[index] = _propagate_field(restored[index] * element, parameter)
    return transported

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
               'element_inputs = ((6, 8), (2.5, 1.5), 0.3, 0.04)\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'model_transmission = _review_independent(element_transmission, *element_inputs)\n'
               'reference_transmission = _review_independent(_oracle_element_transmission, *element_inputs)\n'
               'eigenvalues = np.array([0.5, 0.3, 0.2])',
      'call': '_review_independent(coupled_transport, model_modes, eigenvalues, 2, model_transmission, 3.5)',
      'gold_call': '_review_independent(_oracle_coupled_transport, reference_modes, eigenvalues, 2, '
                   'reference_transmission, 3.5)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mode_inputs = ((7, 6), np.array([1.3, 2.0, 1.6]), np.array([[0.3, 0.1], [-0.2, 0.25], [0.1, '
               '-0.3]]), np.array([0.15, -0.1, 0.2]))\n'
               'element_inputs = ((7, 6), (2.0, 1.2), -0.5, -0.06)\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'model_transmission = _review_independent(element_transmission, *element_inputs)\n'
               'reference_transmission = _review_independent(_oracle_element_transmission, *element_inputs)\n'
               'eigenvalues = np.array([0.6, 0.3, 0.1])',
      'call': '_review_independent(coupled_transport, model_modes, eigenvalues, 1, model_transmission, 0.0)',
      'gold_call': '_review_independent(_oracle_coupled_transport, reference_modes, eigenvalues, 1, '
                   'reference_transmission, 0.0)',
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
               'element_inputs = ((5, 5), (2.0, 1.2), -0.5, -0.06)\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'model_transmission = _review_independent(element_transmission, *element_inputs)\n'
               'reference_transmission = _review_independent(_oracle_element_transmission, *element_inputs)\n'
               'eigenvalues = np.array([0.8, 0.2])',
      'call': '_review_independent(invalid, coupled_transport, (model_modes, eigenvalues, 3, model_transmission, '
              '1.0))',
      'gold_call': '_review_independent(invalid, _oracle_coupled_transport, (reference_modes, eigenvalues, 3, '
                   'reference_transmission, 1.0))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'a = ((5, 7), np.array([1.2]), np.array([[0.2, -0.1]]), np.array([0.08]))\n'
               'model = _review_independent(orthonormal_source_modes, *a)\n'
               'reference = _review_independent(_oracle_orthonormal_source_modes, *a)\n'
               'w = np.array([1.0])\n'
               't = np.ones((5, 7), dtype=complex)',
      'call': '_review_independent(coupled_transport, model, w, 0, t, 3.5)',
      'gold_call': '_review_independent(_oracle_coupled_transport, reference, w, 0, t, 3.5)',
      'tol': 1e-12}]
