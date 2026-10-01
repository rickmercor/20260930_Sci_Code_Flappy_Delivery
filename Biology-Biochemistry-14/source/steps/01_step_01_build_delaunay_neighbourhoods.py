"""
Build the neighbourhood of every cell from the Delaunay triangulation of the cell centroids.

Cells joined by a Delaunay edge are treated as directly adjacent, so the triangulation gives each cell a small neighbourhood over which cell-level signalling values are summarised.

Returns
-------
np.ndarray: (n, n) 0/1 neighbourhood indicator with ones on the diagonal.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_delaunay_neighbourhoods(positions: "np.ndarray") -> "np.ndarray":
    """Return the Delaunay neighbourhood indicator matrix of a set of cell centroids.

    Entry ``[i, k]`` is 1.0 when cell ``k`` belongs to the neighbourhood of
    cell ``i`` and 0.0 otherwise. The neighbourhood of a cell is the cell
    itself together with every cell joined to it by an edge of the Delaunay
    triangulation of all centroids. Centroids are assumed to be in general
    position (no four on the boundary of a common empty circle).

    Parameters
    ----------
    positions : np.ndarray
        Float array of shape ``(n, 2)`` with ``n >= 3`` centroid coordinates.

    Returns
    -------
    np.ndarray
        Symmetric float array of shape ``(n, n)`` with ones on the diagonal.

    Raises
    ------
    ValueError
        If ``positions`` is not a finite numeric array of shape ``(n, 2)``
        with ``n >= 3``, if two centroids coincide, or if all centroids lie
        on one straight line.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_delaunay_neighbourhoods(positions: "np.ndarray") -> "np.ndarray":
    """Reference implementation (Delaunay triangulation of the centroids)."""
    import numpy as np
    from scipy.spatial import Delaunay

    try:
        points = np.array(positions, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("positions must be a numeric array") from None
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("positions must have shape (n, 2) with n >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("centroids must be distinct")
    centred = points - points.mean(axis=0)
    scale = max(float(np.max(np.abs(centred))), 1.0)
    if np.linalg.svd(centred, compute_uv=False)[-1] <= 1e-12 * scale * len(points):
        raise ValueError("centroids must not all lie on one line")
    try:
        simplices = Delaunay(points).simplices
    except RuntimeError:  # QhullError on a degenerate configuration
        raise ValueError("the centroids admit no Delaunay triangulation") from None
    neighbourhoods = np.eye(len(points))
    for simplex in simplices:
        for vertex in simplex:
            # Every pair of vertices of a Delaunay triangle shares an edge.
            neighbourhoods[vertex, simplex] = 1.0
    return neighbourhoods

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    head = (
        "import numpy as np\n"
        "from scipy.spatial import Delaunay\n"
        "def _sig(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.ndim != 2 or a.shape[0] != a.shape[1]:\n"
        "        return -1.0\n"
        "    k = np.arange(a.size, dtype=float)\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a.ravel() * np.cos(0.7 * k)))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    scatter = (
        "P = np.array([[0.0, 0.0], [10.2, 1.3], [4.9, 8.8], [15.7, 9.6], [-3.1, 11.4],\n"
        "              [8.3, 17.9], [19.4, -2.2], [2.6, 4.1]])\n"
    )
    ring = (
        "t = np.linspace(0.0, 2.0 * np.pi, 11)[:-1] + 0.05 * np.sin(np.arange(10))\n"
        "rad = 20.0 + 1.5 * np.cos(3.0 * np.arange(10))\n"
        "P = np.vstack([np.column_stack([rad * np.cos(t), rad * np.sin(t)]),\n"
        "               [[1.3, -0.7], [-6.2, 4.4]]])\n"
        "w = np.arange(1.0, 13.0)\n"
    )
    return [
        {
            "setup": head + scatter,
            "call": "_sig(build_delaunay_neighbourhoods(P))",
            "gold_call": "_sig(_oracle_build_delaunay_neighbourhoods(P))",
        },
        {
            "setup": head,
            "call": "_sig(build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [5.0, 0.5], [1.0, 4.0]])))",
            "gold_call": "_sig(_oracle_build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [5.0, 0.5], [1.0, 4.0]])))",
        },
        {
            "setup": head,
            "call": "_sig(build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])))",
            "gold_call": "_sig(_oracle_build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 5.0]])))",
        },
        {
            "setup": head + ring,
            "call": "float(np.sum(build_delaunay_neighbourhoods(P).sum(axis=1) * w))",
            "gold_call": "float(np.sum(_oracle_build_delaunay_neighbourhoods(P).sum(axis=1) * w))",
        },
        {
            "setup": head + status,
            "call": "_status(lambda: build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])))",
            "gold_call": "_status(lambda: _oracle_build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])))",
        },
        {
            "setup": head + status,
            "call": "_status(lambda: build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [4.0, 1.0], [0.0, 0.0], [1.0, 5.0]])))",
            "gold_call": "_status(lambda: _oracle_build_delaunay_neighbourhoods(np.array([[0.0, 0.0], [4.0, 1.0], [0.0, 0.0], [1.0, 5.0]])))",
        },
        {
            "setup": head + status,
            "call": "_status(lambda: build_delaunay_neighbourhoods(np.ones((5, 3))))",
            "gold_call": "_status(lambda: _oracle_build_delaunay_neighbourhoods(np.ones((5, 3))))",
        },
    ]
