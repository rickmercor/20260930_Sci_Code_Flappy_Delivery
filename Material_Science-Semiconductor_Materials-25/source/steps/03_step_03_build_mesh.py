"""
Build the distorted, non-Delaunay triangular mesh of the unit square (lengths in micrometres) on which the device is solved.

The classical Scharfetter-Gummel finite volume method uses Voronoi cells as control volumes, and those exist with the right properties only when the triangulation is Delaunay, meaning the two angles opposite every interior edge sum to at most 180 degrees. Real device meshes around curved junctions and thin layers often break this condition, which is the situation the DDFV-HA scheme is designed for. The test mesh is built to break it in a controlled, reproducible way. Start from an (M+1) x (M+1) grid of vertices with spacing h = 1/M, shift every interior vertex of every odd row by 0.4 h in x while leaving the side vertices on x = 0 and x = 1 in place, and split each cell along the diagonal from its lower-left to its upper-right vertex. In the rows below odd rows the long diagonals then have opposite angles of about 111.8 degrees each, so roughly one interior edge in six violates the Delaunay condition. Every vertex row stays exactly horizontal, which later allows row averages, and with M even a vertex row lies exactly on the junction at y = 0.5.

Returns
-------
tuple (points, triangles): points float array ((M+1)^2, 2) in micrometres, triangles int array (2M^2, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_mesh(M: int) -> tuple:
    """Shifted-row triangulation of the unit square [0, 1] x [0, 1] (micrometres).

    Parameters
    ----------
    M : int
        Number of cells per side; must be even and >= 2.

    Returns
    -------
    points : "np.ndarray"
        Shape ((M+1)**2, 2), float. Vertex (i, j), i, j = 0..M, has index
        j*(M+1) + i and coordinates (i*h, j*h) with h = 1/M, except that vertices
        with j odd and 0 < i < M are shifted by +0.4*h in x.
    triangles : "np.ndarray"
        Shape (2*M*M, 3), int. For j = 0..M-1 (outer loop) and i = 0..M-1 (inner
        loop), with a, b, c, d the indices of (i,j), (i+1,j), (i+1,j+1), (i,j+1),
        append (a, b, c) then (a, c, d).

    Notes
    -----
    Raise ValueError if M is odd or below 2.
    Include every import your implementation needs inside the function body.
    """
    return points, triangles  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_mesh(M: int) -> tuple:
    M = int(M)
    if M < 2 or M % 2:
        raise ValueError("M must be an even integer >= 2")
    h = 1.0 / M
    j, i = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing="ij")
    shift = (j % 2 == 1) & (i > 0) & (i < M)
    x = i * h + 0.4 * h * shift
    y = j * h
    points = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    tris = []
    for jj in range(M):
        for ii in range(M):
            a = jj * (M + 1) + ii
            b, c, d = a + 1, a + M + 2, a + M + 1
            tris.append((a, b, c))
            tris.append((a, c, d))
    return points, np.array(tris, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: vertex coordinates of the small test mesh.
        {"setup": "import numpy as np\n",
         "call": "build_mesh(8)[0]",
         "gold_call": "_oracle_build_mesh(8)[0]"},
        # Normal: triangle connectivity and orientation of the same mesh.
        {"setup": "import numpy as np\n",
         "call": "build_mesh(8)[1]",
         "gold_call": "_oracle_build_mesh(8)[1]"},
        # Boundary: smallest allowed mesh, one odd row with a single shifted vertex.
        {"setup": "import numpy as np\n",
         "call": "np.concatenate([build_mesh(2)[0].ravel(), build_mesh(2)[1].ravel().astype(float)])",
         "gold_call": "np.concatenate([_oracle_build_mesh(2)[0].ravel(), _oracle_build_mesh(2)[1].ravel().astype(float)])"},
        # Normal: production mesh used for the reported results.
        {"setup": "import numpy as np\n",
         "call": "build_mesh(64)[0]",
         "gold_call": "_oracle_build_mesh(64)[0]"},
    ]
