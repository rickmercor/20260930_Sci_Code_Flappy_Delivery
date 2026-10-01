"""
Build the orthonormal coherent modes of the benchmark source on a pixel grid.

The source is described by mutually incoherent coherent modes sampled on a rectangular grid of `$ny$` rows and `$nx$` columns. Coordinates are the unit-spaced pixel offsets from the grid center, `$x = j - (nx - 1) / 2$` for column `$j$` and `$y = i - (ny - 1) / 2$` for row `$i$`, in the `$xy$` convention, so a column index advances the horizontal coordinate. Before orthonormalization, mode `$n$` is a Gaussian envelope of width ``sigmas[n]`$multiplied by the unit-modulus phase$$exp(i (a x + b y + c x y))$` with carrier tilt ``(a, b) = tilts[n]`` and bilinear coupling ``c = coupling[n]``.



The modes are orthonormalized in the given order by modified Gram-Schmidt on their C-order flattened samples, with the inner product that conjugates its first argument. Each finished mode is phased so that its largest-magnitude sample is real and nonnegative, ties broken by the smallest flat index. Later steps rely on these modes having unit Euclidean norm.

Returns
-------
np.ndarray, modes: Complex128 array of shape ``(n_modes, ny, nx)``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orthonormal_source_modes(
    shape: tuple,
    sigmas: "np.ndarray",
    tilts: "np.ndarray",
    coupling: "np.ndarray",
) -> "np.ndarray":
    """Return the orthonormal coherent modes on the requested grid.

    Column ``n`` is the Gaussian envelope of width ``sigmas[n]``, carrier
    tilt ``tilts[n]``, and bilinear coupling ``coupling[n]``. Coordinates
    are the unit-spaced pixel offsets from the grid center,
    ``x = j - (nx - 1) / 2`` for column ``j`` and ``y = i - (ny - 1) / 2``
    for row ``i``, with ``indexing="xy"``. The vectorized columns, in C
    order, are orthonormalized by modified Gram-Schmidt. Each finished
    column is phased so that its largest-magnitude sample is real and
    nonnegative, with ties broken by the smallest flat index. The inner
    product conjugates its first argument.

    Parameters
    ----------
    shape : tuple
        ``(ny, nx)``, both integers at least 2.
    sigmas : np.ndarray
        Positive finite envelope widths, shape ``(n_modes,)``.
    tilts : np.ndarray
        Finite carrier coefficients, shape ``(n_modes, 2)``.
    coupling : np.ndarray
        Finite bilinear coefficients, shape ``(n_modes,)``.

    Returns
    -------
    modes : np.ndarray
        Complex128 array of shape ``(n_modes, ny, nx)``.

    Raises
    ------
    ValueError
        If a shape is wrong, a width is not positive, or a coefficient
        is not finite.
    """
    return modes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orthonormal_source_modes(
    shape: tuple,
    sigmas: "np.ndarray",
    tilts: "np.ndarray",
    coupling: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two integers of at least 2.")
    widths = np.asarray(sigmas, dtype=np.float64)
    carriers = np.asarray(tilts, dtype=np.float64)
    mix = np.asarray(coupling, dtype=np.float64)
    if widths.ndim != 1 or mix.ndim != 1 or carriers.ndim != 2:
        raise ValueError("sigmas, tilts, and coupling have inconsistent ranks.")
    n_modes = widths.shape[0]
    if n_modes < 1 or carriers.shape != (n_modes, 2) or mix.shape != (n_modes,):
        raise ValueError("sigmas, tilts, and coupling must describe the same modes.")
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("sigmas must be positive and finite.")
    if not np.all(np.isfinite(carriers)) or not np.all(np.isfinite(mix)):
        raise ValueError("tilts and coupling must be finite.")
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    columns = []
    for width, (tilt_x, tilt_y), factor in zip(widths, carriers, mix):
        envelope = np.exp(-(grid_x * grid_x + grid_y * grid_y) / (2.0 * width * width))
        phase = tilt_x * grid_x + tilt_y * grid_y + factor * grid_x * grid_y
        columns.append((envelope * np.exp(1j * phase)).ravel())
    design = np.column_stack(columns).astype(np.complex128)
    basis = np.zeros_like(design)
    for index in range(n_modes):
        column = design[:, index].copy()
        for earlier in range(index):
            column = column - np.vdot(basis[:, earlier], column) * basis[:, earlier]
        norm = np.linalg.norm(column)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("mode construction produced a vanishing column.")
        column = column / norm
        pivot = int(np.argmax(np.abs(column)))
        column = column * np.exp(-1j * np.angle(column[pivot]))
        basis[:, index] = column
    return basis.T.reshape(n_modes, ny, nx)

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
               'shape = (6, 8)\n'
               'sigmas = np.array([1.4, 2.2, 1.1])\n'
               'tilts = np.array([[0.2, -0.1], [0.0, 0.15], [-0.2, 0.05]])\n'
               'coupling = np.array([0.03, -0.05, 0.08])',
      'call': '_review_independent(orthonormal_source_modes, shape, sigmas, tilts, coupling)',
      'gold_call': '_review_independent(_oracle_orthonormal_source_modes, shape, sigmas, tilts, coupling)',
      'tol': 1e-10},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'shape = (5, 5)\n'
               'sigmas = np.array([1.0, 1.7])\n'
               'tilts = np.array([[0.1, 0.1], [-0.3, 0.2]])\n'
               'coupling = np.array([0.0, 0.12])',
      'call': '_review_independent(orthonormal_source_modes, shape, sigmas, tilts, coupling)',
      'gold_call': '_review_independent(_oracle_orthonormal_source_modes, shape, sigmas, tilts, coupling)',
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
               'shape = (6, 6)\n'
               'sigmas = np.array([1.0, 0.0])\n'
               'tilts = np.array([[0.1, 0.0], [0.0, 0.1]])\n'
               'coupling = np.array([0.02, 0.02])',
      'call': '_review_independent(invalid, orthonormal_source_modes, (shape, sigmas, tilts, coupling))',
      'gold_call': '_review_independent(invalid, _oracle_orthonormal_source_modes, (shape, sigmas, tilts, '
                   'coupling))',
      'tol': 1e-12},
     {'setup': '\n'
               'def _review_independent(function, *args, **kwargs):\n'
               '    from copy import deepcopy\n'
               '    independent_args, independent_kwargs = deepcopy((args, kwargs))\n'
               '    return function(*independent_args, **independent_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'a = ((2, 3), np.array([1.2]), np.array([[0.2, -0.1]]), np.array([0.08]))',
      'call': '_review_independent(orthonormal_source_modes, *a)',
      'gold_call': '_review_independent(_oracle_orthonormal_source_modes, *a)',
      'tol': 1e-10}]
