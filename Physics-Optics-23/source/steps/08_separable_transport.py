"""
Transport the leading separable component of every mode through the retained element terms to the output plane.

In the journal method the separable part of a mode is not propagated as a full two-dimensional wavefront: it is multiplied by each retained separable element term and each product is transported with one-dimensional operators along the two axes. The result requested here is the transported field of the leading separable component of every mode, summed over the retained element terms.



Conventions: the leading separable component of a mode is its leading singular term including the singular value; each element term already includes its singular value; propagation over the parameter `$fresnel_parameter$` (the product of wavelength and distance in units of the squared pixel pitch) multiplies the discrete Fourier transform of an unpadded field by `$exp(-1j * pi * fresnel_parameter * (fx**2 + fy**2))$`, with `$fx$` and `$fy$` the ``numpy.fft.fftfreq`` frequencies of the two axes at unit pitch, so a separable field propagates into the outer product of its one-dimensionally propagated factors; a zero parameter leaves a field unchanged. The transported field equals the propagation of the product of the separable component with the sum of the retained terms. The result has the shape of ``modes``.

Returns
-------
np.ndarray, transported: Complex128 array with the same shape as ``modes``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def separable_transport(
    modes: "np.ndarray",
    terms: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Return the transported leading separable component of each mode.

    Parameters
    ----------
    modes : np.ndarray
        Complex modes of shape ``(n_modes, ny, nx)``.
    terms : np.ndarray
        Complex separable element terms of shape ``(K, ny, nx)``, each
        including its singular value.
    fresnel_parameter : float
        Finite propagation parameter in units of the squared pixel pitch.

    Returns
    -------
    transported : np.ndarray
        Complex128 array with the same shape as ``modes``.

    Raises
    ------
    ValueError
        If the spatial shapes of ``modes`` and ``terms`` differ or the
        propagation parameter is not finite.
    """
    return transported

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _paraxial_factor(length: int, fresnel_parameter: float) -> "np.ndarray":
    """One-dimensional paraxial transfer function on the DFT frequencies."""
    freq = np.fft.fftfreq(int(length), d=1.0)
    return np.exp(-1j * np.pi * float(fresnel_parameter) * freq * freq)


def _propagate_line(line: "np.ndarray", fresnel_parameter: float) -> "np.ndarray":
    """Propagate a one-dimensional complex profile."""
    spectrum = np.fft.fft(np.asarray(line, dtype=np.complex128))
    return np.fft.ifft(spectrum * _paraxial_factor(line.shape[0], fresnel_parameter))


def _propagate_field(field: "np.ndarray", fresnel_parameter: float) -> "np.ndarray":
    """Propagate a two-dimensional complex field."""
    data = np.asarray(field, dtype=np.complex128)
    kernel = np.outer(
        _paraxial_factor(data.shape[0], fresnel_parameter),
        _paraxial_factor(data.shape[1], fresnel_parameter),
    )
    return np.fft.ifft2(np.fft.fft2(data) * kernel)


def _oracle_separable_transport(
    modes: "np.ndarray",
    terms: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Reference implementation."""
    fields = np.asarray(modes)
    pieces = np.asarray(terms)
    if fields.ndim != 3 or pieces.ndim != 3 or fields.shape[1:] != pieces.shape[1:]:
        raise ValueError("modes and terms must share the same spatial shape.")
    if min(fields.shape[1:]) < 2:
        raise ValueError("both spatial sides must be at least 2.")
    parameter = float(fresnel_parameter)
    if not np.isfinite(parameter):
        raise ValueError("fresnel_parameter must be finite.")
    factors = []
    for piece in pieces:
        left, values, right = np.linalg.svd(piece, full_matrices=False)
        factors.append((values[0] * left[:, 0], right[0]))
    transported = np.zeros(fields.shape, dtype=np.complex128)
    for index, field in enumerate(fields):
        left, values, right = np.linalg.svd(field, full_matrices=False)
        vertical = values[0] * left[:, 0]
        horizontal = right[0]
        for term_vertical, term_horizontal in factors:
            moved_vertical = _propagate_line(vertical * term_vertical, parameter)
            moved_horizontal = _propagate_line(horizontal * term_horizontal, parameter)
            transported[index] += np.outer(moved_vertical, moved_horizontal)
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
               'model_terms = _review_independent(element_terms, _review_independent(element_transmission, '
               '*element_inputs), 0.99)\n'
               'reference_terms = _review_independent(_oracle_element_terms, '
               '_review_independent(_oracle_element_transmission, *element_inputs), 0.99)',
      'call': '_review_independent(separable_transport, model_modes, model_terms, 3.5)',
      'gold_call': '_review_independent(_oracle_separable_transport, reference_modes, reference_terms, 3.5)',
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
               'element_inputs = ((5, 5), (2.0, 1.2), -0.5, -0.06)\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'model_terms = _review_independent(element_terms, _review_independent(element_transmission, '
               '*element_inputs), 0.9)\n'
               'reference_terms = _review_independent(_oracle_element_terms, '
               '_review_independent(_oracle_element_transmission, *element_inputs), 0.9)',
      'call': '_review_independent(separable_transport, model_modes, model_terms, 0.0)',
      'gold_call': '_review_independent(_oracle_separable_transport, reference_modes, reference_terms, 0.0)',
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
               'element_inputs = ((6, 5), (2.0, 1.2), -0.5, -0.06)\n'
               'model_modes = _review_independent(orthonormal_source_modes, *mode_inputs)\n'
               'reference_modes = _review_independent(_oracle_orthonormal_source_modes, *mode_inputs)\n'
               'model_terms = _review_independent(element_terms, _review_independent(element_transmission, '
               '*element_inputs), 0.9)\n'
               'reference_terms = _review_independent(_oracle_element_terms, '
               '_review_independent(_oracle_element_transmission, *element_inputs), 0.9)',
      'call': '_review_independent(invalid, separable_transport, (model_modes, model_terms, 2.0))',
      'gold_call': '_review_independent(invalid, _oracle_separable_transport, (reference_modes, reference_terms, '
                   '2.0))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'a = ((5, 7), np.array([1.2]), np.array([[0.2, -0.1]]), np.array([0.0]))\n'
               'model = _review_independent(orthonormal_source_modes, *a)\n'
               'reference = _review_independent(_oracle_orthonormal_source_modes, *a)\n'
               'terms = np.ones((1, 5, 7), dtype=complex)',
      'call': '_review_independent(separable_transport, model, terms, 3.5)',
      'gold_call': '_review_independent(_oracle_separable_transport, reference, terms, 3.5)',
      'tol': 1e-10}]
