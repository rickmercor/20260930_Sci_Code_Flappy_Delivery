"""
Step name: 04_project_dynamical_components
Step description: Reduce the descriptor matrix to its slowest linear components by solving the time-lagged generalized eigenvalue problem.

Step scientific background: A Markov state model needs a low-dimensional space in which configurations are separated by dynamical rather than geometric distance. Diagonalising the lagged correlation matrix against the instantaneous covariance yields the linear combinations of descriptors that decorrelate most slowly, which are the coordinates in which slow processes are resolved.

Returns
-------
np.ndarray: float projection of shape (T, n_components) onto the slowest components.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def project_dynamical_components(descriptors: np.ndarray, lag_frames: int = 20,
                                 n_components: int = 5,
                                 rank_cutoff: float = 1e-6) -> np.ndarray:
    """Return the slowest linear components of a descriptor trajectory.

    Conventions fixed by this step. The descriptors are first made mean free
    over the whole trajectory. The instantaneous covariance is
    ``C0 = Y.T @ Y / T`` and the lagged correlation is the symmetrised
    ``Ct = (Yt.T @ Yl + Yl.T @ Yt) / (2 * (T - lag_frames))`` where ``Yt`` and
    ``Yl`` are the first and last ``T - lag_frames`` rows of ``Y``.
    Whitening is done on the eigendecomposition of ``C0``, keeping only the
    eigenvalues larger than ``rank_cutoff`` times the largest one and scaling
    each retained eigenvector by the inverse square root of its eigenvalue.
    The requested number of components is taken in order of decreasing
    eigenvalue of the whitened lagged matrix. Each returned component is
    signed so that the entry of largest absolute value in its descriptor-space
    loading vector is positive; on a tie the earliest such entry decides.

    Parameters
    ----------
    descriptors : np.ndarray
        Float array of shape ``(T, d)``.
    lag_frames : int
        Correlation lag in frames, at least 1 and smaller than ``T``.
    n_components : int
        Number of components returned, at least 1.
    rank_cutoff : float
        Relative eigenvalue floor for the whitening, strictly between 0 and 1.

    Returns
    -------
    np.ndarray
        Float array of shape ``(T, n_components)`` holding the projected
        trajectory, each column with unit variance up to the whitening.

    Raises
    ------
    ValueError
        If ``descriptors`` is not a finite two-dimensional array, if
        ``lag_frames`` is not an integer in ``1 .. T - 1``, if
        ``n_components`` is not an integer of at least 1 or exceeds the
        retained rank, or if ``rank_cutoff`` is not a float strictly between
        0 and 1.
    """
    return components

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _fix_loading_signs(loadings):
    """Return loadings with the largest-magnitude entry of each column positive."""
    import numpy as np
    signed = loadings.copy()
    for column in range(signed.shape[1]):
        pivot = int(np.argmax(np.abs(signed[:, column])))
        if signed[pivot, column] < 0.0:
            signed[:, column] = -signed[:, column]
    return signed


import numpy as np
def _oracle_project_dynamical_components(descriptors: np.ndarray, lag_frames: int = 20,
                                         n_components: int = 5,
                                         rank_cutoff: float = 1e-6) -> np.ndarray:
    """Reference implementation (whitened symmetric lagged eigenproblem)."""
    import numpy as np

    features = np.asarray(descriptors, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("descriptors must be a two-dimensional array with at least two rows")
    if not np.all(np.isfinite(features)):
        raise ValueError("descriptors must be finite")
    n_rows = features.shape[0]
    if not (_is_integer(lag_frames) and 1 <= int(lag_frames) < n_rows):
        raise ValueError("lag_frames must be an integer in 1 .. T - 1")
    if not (_is_integer(n_components) and int(n_components) >= 1):
        raise ValueError("n_components must be an integer of at least 1")
    if isinstance(rank_cutoff, bool) or not isinstance(rank_cutoff, (int, float, np.floating, np.integer)):
        raise ValueError("rank_cutoff must be a float strictly between 0 and 1")
    if not 0.0 < float(rank_cutoff) < 1.0:
        raise ValueError("rank_cutoff must be a float strictly between 0 and 1")

    lag = int(lag_frames)
    centred = features - features.mean(axis=0)
    instant = centred.T @ centred / n_rows
    early, late = centred[:n_rows - lag], centred[lag:]
    lagged = (early.T @ late + late.T @ early) / (2.0 * (n_rows - lag))

    values, vectors = np.linalg.eigh(instant)
    if values.max() <= 0.0:
        raise ValueError("descriptors have no variance")
    keep = values > float(rank_cutoff) * values.max()
    whitener = vectors[:, keep] / np.sqrt(values[keep])
    if int(n_components) > int(keep.sum()):
        raise ValueError("n_components exceeds the retained rank of the covariance")

    reduced = whitener.T @ lagged @ whitener
    reduced = 0.5 * (reduced + reduced.T)
    slow_values, slow_vectors = np.linalg.eigh(reduced)
    order = np.argsort(slow_values)[::-1][:int(n_components)]
    loadings = _fix_loading_signs(whitener @ slow_vectors[:, order])
    return centred @ loadings

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(a):\n"
        "    f = np.asarray(a, dtype=float).ravel()\n"
        "    if f.size == 0:\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(f.size, dtype=float) + 1.0)\n"
        "    return float(np.dot(f, w) + np.abs(f).mean() + 1000.0 * f[0])\n"
        "phi = build_local_descriptors(simulate_surface_trajectory(4000, 5))\n"
        "phi_gold = _oracle_build_local_descriptors(_oracle_simulate_surface_trajectory(4000, 5))\n"
        "rng = np.random.default_rng(1)\n"
        "z = np.cumsum(rng.standard_normal((800, 2)) * 0.1, axis=0)\n"
        "toy = np.column_stack([z[:, 0], z[:, 1], z[:, 0] + 0.5 * z[:, 1], np.ones(800)])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "phi = build_local_descriptors(simulate_surface_trajectory(300, 5))\n"
        "phi_gold = _oracle_build_local_descriptors(_oracle_simulate_surface_trajectory(300, 5))\n"
    )
    return [
        {
            "setup": digest,
            "call": "_sig(project_dynamical_components(phi))",
            "gold_call": "_sig(_oracle_project_dynamical_components(phi_gold))",
        },
        {
            "setup": digest,
            "call": "_sig(project_dynamical_components(phi, lag_frames=5, n_components=2))",
            "gold_call": "_sig(_oracle_project_dynamical_components(phi_gold, lag_frames=5, n_components=2))",
        },
        {
            "setup": digest,
            "call": "_sig(project_dynamical_components(phi, n_components=3))",
            "gold_call": "_sig(_oracle_project_dynamical_components(phi_gold, n_components=3))",
        },
        {
            "setup": digest,
            "call": "_sig(project_dynamical_components(toy, lag_frames=10, n_components=2, rank_cutoff=1e-8))",
            "gold_call": "_sig(_oracle_project_dynamical_components(toy, lag_frames=10, n_components=2, rank_cutoff=1e-8))",
        },
        {
            "setup": digest,
            "call": "float(project_dynamical_components(phi, lag_frames=1, n_components=1)[0, 0] * 1000.0)",
            "gold_call": "float(_oracle_project_dynamical_components(phi_gold, lag_frames=1, n_components=1)[0, 0] * 1000.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: project_dynamical_components(phi, lag_frames=0))",
            "gold_call": "_status(lambda: _oracle_project_dynamical_components(phi_gold, lag_frames=0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: project_dynamical_components(phi, n_components=100))",
            "gold_call": "_status(lambda: _oracle_project_dynamical_components(phi_gold, n_components=100))",
        },
        {
            "setup": status,
            "call": "_status(lambda: project_dynamical_components(phi, rank_cutoff=1.0))",
            "gold_call": "_status(lambda: _oracle_project_dynamical_components(phi_gold, rank_cutoff=1.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: project_dynamical_components(np.zeros(7)))",
            "gold_call": "_status(lambda: _oracle_project_dynamical_components(np.zeros(7)))",
        },
    ]
