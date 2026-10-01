"""
Implement compute_pca_basis, which computes the PCA mean and truncated projection basis from a set of reference system-averaged descriptors. The reference set of previously collected descriptors is used to find the dominant directions of variation via PCA. Centering removes the mean structural offset; the SVD of the centered matrix gives principal directions ordered by explained variance, of which the leading k are kept to define the reduced collective-variable (CV) space.

ERBS reduces a high-dimensional correlated descriptor to a small set of collective variables using PCA rather than a hand-picked reaction coordinate or a learned embedding. This avoids the curse of dimesionality that would come from biasing along every raw descriptor dimension directly, and the leading singular vectors of the centered reference set capture the system's dominant collective modes of structural variation.

Returns
-------
tuple[np.ndarray, np.ndarray]: mu of shape (D, ) and V_k of shape (D, k), both NumPy float arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_pca_basis(S_ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:

    '''Compute the PCA mean and truncated projection basis from reference descriptors.


    Parameters

    ----------

    S_ref : np.ndarray

        Array of shape (N_ref, D), reference system-averaged descriptors.

    k : int

        Number of principal components to keep (1 <= k <= D).


    Returns

    -------

    mu : np.ndarray

        Array of shape (D,), the empirical mean of S_ref.

    V_k : np.ndarray

        Array of shape (D, k), the first k right-singular vectors of the

        centered reference matrix, ordered by descending singular value.

        Signs are arbitrary. Within a tied singular-value eigenspace, any

        orthonormal basis is accepted. Include null-space directions when

        necessary so that the returned basis has exactly k columns.


    Raises

    ------

    ValueError

        If S_ref is not a 2D array, if S_ref has fewer than 2 rows, if k is

        not an integer with 1 <= k <= D (D = S_ref.shape[1]), or if S_ref

        contains any non-finite values.

    '''

    return mu, V_k  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_pca_basis(S_ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:

    """Reference implementation."""

    S_ref = np.asarray(S_ref, dtype=float)

    if S_ref.ndim != 2:

        raise ValueError("S_ref must be a 2D array of shape (N_ref, D)")

    if S_ref.shape[0] < 2:

        raise ValueError("S_ref must contain at least 2 reference rows")

    if not np.all(np.isfinite(S_ref)):

        raise ValueError("S_ref must contain only finite values")

    D = S_ref.shape[1]

    if not isinstance(k, (int, np.integer)) or not (1 <= k <= D):

        raise ValueError(f"k must be an integer with 1 <= k <= {D}")


    mu = S_ref.mean(axis=0)

    S_hat = S_ref - mu

    _, _, Vt = np.linalg.svd(S_hat, full_matrices=True)

    V = Vt.T

    V_k = V[:, :k]

    return mu.astype(float), V_k.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    """Normal, full-basis, tied-cutoff, thin-input, and invalid-k cases."""

    return [{'setup': 'import numpy as np\n'

               '\n'

               'def _pca_summary(result, X, count):\n'

               '    """Preserve the mean and output shapes; compare PCA scientific '

               'invariants."""\n'

               '    if not isinstance(result, tuple) or len(result) != 2:\n'

               '        raise AssertionError("PCA must return a (mean, basis) tuple")\n'

               '    mean, basis = (np.asarray(v) for v in result)\n'

               '    d = X.shape[1]\n'

               '    if mean.shape != (d,) or basis.shape != (d, count):\n'

               '        raise AssertionError("PCA output shape does not match its contract")\n'

               '    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(basis))):\n'

               '        raise AssertionError("PCA returned non-finite values")\n'

               '    centered = X - X.mean(axis=0)\n'

               '    scatter = centered.T @ centered\n'

               '    leading = np.linalg.eigvalsh(scatter)[::-1][:count]\n'

               '    scale = max(1.0, float(np.linalg.norm(scatter, ord=2)))\n'

               '    residual = (scatter @ basis - basis * leading[None, :]) / scale\n'

               '    # Both columns and ordering are tested. Sign changes and tied eigenspace\n'

               '    # rotations remain valid. Eigenvalues below the cutoff cannot pass.\n'

               '    return np.concatenate((\n'

               '        np.array([mean.ndim, *mean.shape, basis.ndim, *basis.shape], '

               'dtype=float),\n'

               '        mean.ravel(),\n'

               '        (basis.T @ basis).ravel(),\n'

               '        residual.ravel(),\n'

               '    ))\n'

               '\n'

               'S_ref = np.random.default_rng(7).standard_normal((20, 12))\n'

               'k = 3',

      'call': '_pca_summary(compute_pca_basis(S_ref, k), S_ref, k)',

      'gold_call': '_pca_summary(_oracle_compute_pca_basis(S_ref, k), S_ref, k)',

      'tol': 1e-08},

     {'setup': 'import numpy as np\n'

               '\n'

               'def _pca_summary(result, X, count):\n'

               '    """Preserve the mean and output shapes; compare PCA scientific '

               'invariants."""\n'

               '    if not isinstance(result, tuple) or len(result) != 2:\n'

               '        raise AssertionError("PCA must return a (mean, basis) tuple")\n'

               '    mean, basis = (np.asarray(v) for v in result)\n'

               '    d = X.shape[1]\n'

               '    if mean.shape != (d,) or basis.shape != (d, count):\n'

               '        raise AssertionError("PCA output shape does not match its contract")\n'

               '    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(basis))):\n'

               '        raise AssertionError("PCA returned non-finite values")\n'

               '    centered = X - X.mean(axis=0)\n'

               '    scatter = centered.T @ centered\n'

               '    leading = np.linalg.eigvalsh(scatter)[::-1][:count]\n'

               '    scale = max(1.0, float(np.linalg.norm(scatter, ord=2)))\n'

               '    residual = (scatter @ basis - basis * leading[None, :]) / scale\n'

               '    # Both columns and ordering are tested. Sign changes and tied eigenspace\n'

               '    # rotations remain valid. Eigenvalues below the cutoff cannot pass.\n'

               '    return np.concatenate((\n'

               '        np.array([mean.ndim, *mean.shape, basis.ndim, *basis.shape], '

               'dtype=float),\n'

               '        mean.ravel(),\n'

               '        (basis.T @ basis).ravel(),\n'

               '        residual.ravel(),\n'

               '    ))\n'

               '\n'

               'S_ref = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])\n'

               'k = 2',

      'call': '_pca_summary(compute_pca_basis(S_ref, k), S_ref, k)',

      'gold_call': '_pca_summary(_oracle_compute_pca_basis(S_ref, k), S_ref, k)',

      'tol': 1e-08},

     {'setup': 'import numpy as np\n'

               '\n'

               'def _pca_summary(result, X, count):\n'

               '    """Preserve the mean and output shapes; compare PCA scientific '

               'invariants."""\n'

               '    if not isinstance(result, tuple) or len(result) != 2:\n'

               '        raise AssertionError("PCA must return a (mean, basis) tuple")\n'

               '    mean, basis = (np.asarray(v) for v in result)\n'

               '    d = X.shape[1]\n'

               '    if mean.shape != (d,) or basis.shape != (d, count):\n'

               '        raise AssertionError("PCA output shape does not match its contract")\n'

               '    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(basis))):\n'

               '        raise AssertionError("PCA returned non-finite values")\n'

               '    centered = X - X.mean(axis=0)\n'

               '    scatter = centered.T @ centered\n'

               '    leading = np.linalg.eigvalsh(scatter)[::-1][:count]\n'

               '    scale = max(1.0, float(np.linalg.norm(scatter, ord=2)))\n'

               '    residual = (scatter @ basis - basis * leading[None, :]) / scale\n'

               '    # Both columns and ordering are tested. Sign changes and tied eigenspace\n'

               '    # rotations remain valid. Eigenvalues below the cutoff cannot pass.\n'

               '    return np.concatenate((\n'

               '        np.array([mean.ndim, *mean.shape, basis.ndim, *basis.shape], '

               'dtype=float),\n'

               '        mean.ravel(),\n'

               '        (basis.T @ basis).ravel(),\n'

               '        residual.ravel(),\n'

               '    ))\n'

               '\n'

               'S_ref = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])\n'

               'k = 1',

      'call': '_pca_summary(compute_pca_basis(S_ref, k), S_ref, k)',

      'gold_call': '_pca_summary(_oracle_compute_pca_basis(S_ref, k), S_ref, k)',

      'tol': 1e-08},

     {'setup': 'import numpy as np\n'

               '\n'

               'def _pca_summary(result, X, count):\n'

               '    """Preserve the mean and output shapes; compare PCA scientific '

               'invariants."""\n'

               '    if not isinstance(result, tuple) or len(result) != 2:\n'

               '        raise AssertionError("PCA must return a (mean, basis) tuple")\n'

               '    mean, basis = (np.asarray(v) for v in result)\n'

               '    d = X.shape[1]\n'

               '    if mean.shape != (d,) or basis.shape != (d, count):\n'

               '        raise AssertionError("PCA output shape does not match its contract")\n'

               '    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(basis))):\n'

               '        raise AssertionError("PCA returned non-finite values")\n'

               '    centered = X - X.mean(axis=0)\n'

               '    scatter = centered.T @ centered\n'

               '    leading = np.linalg.eigvalsh(scatter)[::-1][:count]\n'

               '    scale = max(1.0, float(np.linalg.norm(scatter, ord=2)))\n'

               '    residual = (scatter @ basis - basis * leading[None, :]) / scale\n'

               '    # Both columns and ordering are tested. Sign changes and tied eigenspace\n'

               '    # rotations remain valid. Eigenvalues below the cutoff cannot pass.\n'

               '    return np.concatenate((\n'

               '        np.array([mean.ndim, *mean.shape, basis.ndim, *basis.shape], '

               'dtype=float),\n'

               '        mean.ravel(),\n'

               '        (basis.T @ basis).ravel(),\n'

               '        residual.ravel(),\n'

               '    ))\n'

               '\n'

               'S_ref = np.array([[1., 2., 3.], [2., 1., 0.]])\n'

               'k = 3',

      'call': '_pca_summary(compute_pca_basis(S_ref, k), S_ref, k)',

      'gold_call': '_pca_summary(_oracle_compute_pca_basis(S_ref, k), S_ref, k)',

      'tol': 1e-08},

     {'setup': 'import numpy as np\n'

               'S_ref = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])\n'

               'k = 5\n'

               'def run_model():\n'

               '    try:\n'

               '        compute_pca_basis(S_ref, k)\n'

               '        return 0\n'

               '    except ValueError:\n'

               '        return 1\n'

               '    except Exception:\n'

               '        return 2\n'

               'def run_gold():\n'

               '    try:\n'

               '        _oracle_compute_pca_basis(S_ref, k)\n'

               '        return 0\n'

               '    except ValueError:\n'

               '        return 1\n'

               '    except Exception:\n'

               '        return 2',

      'call': 'run_model()',

      'gold_call': 'run_gold()'}]
