"""
Compute a frame-averaged radial distribution function from periodically imaged pair distances.

The coordinates are in a synthetic periodic cube. Use minimum-image distances, left-closed radial bins with the final right edge included, and an arithmetic mean over frames. The source RDF definition determines its normalization.

Returns
-------
np.ndarray, the frame-averaged RDF values for the supplied radial bins
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_distribution(positions: "np.ndarray", box_length: float, edges: "np.ndarray") -> "np.ndarray":
    """Return the source Eq. 2 frame-mean RDF on the supplied radial bins.

    Positions have shape (frames>=1, atoms>=2, 3) in a cubic periodic box.
    Pair distances use the minimum-image convention. Bin edges are strictly
    increasing, finite and nonnegative, ending no farther than half the box
    length. Use left-closed bins and include the final right edge. Return one
    radial value per bin. Raise ValueError on invalid geometry or controls.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_radial_distribution(
    positions: "np.ndarray", box_length: float, edges: "np.ndarray"
) -> "np.ndarray":
    """Compute paper Eq. 2 using a stated cubic periodic-box discretization.

    Positions have shape (frames, atoms, 3). Pair distances use minimum-image
    convention. Each shell's mean pair count is multiplied by
    2*volume/(shell_volume*N*(N-1)), as in the source equation.
    """
    x, r = np.asarray(positions, float), np.asarray(edges, float)
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 2 or x.shape[2] != 3:
        raise ValueError("positions must have shape (frames, atoms>=2, 3)")
    if r.ndim != 1 or r.size < 3 or r[0] < 0 or not np.all(np.diff(r) > 0):
        raise ValueError("radial edges must be increasing and nonnegative")
    if not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError("nonfinite geometry")
    if not np.isfinite(box_length) or box_length <= 0 or r[-1] > box_length / 2:
        raise ValueError("radial range exceeds the minimum-image radius")
    n = x.shape[1]
    upper = np.triu_indices(n, 1)
    hist = np.zeros(r.size - 1, dtype=np.float64)
    for frame in x:
        delta = frame[upper[0]] - frame[upper[1]]
        delta -= box_length * np.round(delta / box_length)
        hist += np.histogram(np.linalg.norm(delta, axis=1), bins=r)[0]
    hist /= x.shape[0]
    shell_volume = 4 * np.pi / 3 * (r[1:] ** 3 - r[:-1] ** 3)
    return hist * (2 * box_length**3 / (shell_volume * n * (n - 1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\na=np.array([[[0.,0,0],[1.,0,0]]]); e=np.array([0.,1.5,5.]); ag=a.copy(); eg=e.copy()", "call": "radial_distribution(a,10.,e)", "gold_call": "_oracle_radial_distribution(ag,10.,eg)", "tol": 1e-9},
        {"setup": "import numpy as np\na=np.array([[[.1,0,0],[9.9,0,0]],[[.2,0,0],[9.8,0,0]]]); e=np.array([0.,.3,.6,5.]); ag=a.copy(); eg=e.copy()", "call": "radial_distribution(a,10.,e)", "gold_call": "_oracle_radial_distribution(ag,10.,eg)", "tol": 1e-9},
        {"setup": "import numpy as np\nrng=np.random.default_rng(9); a=rng.uniform(0,10,size=(4,12,3)); e=np.linspace(0,5,11); ag=a.copy(); eg=e.copy()", "call": "radial_distribution(a,10.,e)", "gold_call": "_oracle_radial_distribution(ag,10.,eg)", "tol": 1e-9},
        {"setup": 'import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\na=np.zeros((1,2,3)); e=np.array([0.,2.,6.]); ag=a.copy(); eg=e.copy()', "call": '_raises(lambda: radial_distribution(a,10.,e))', "gold_call": '_raises(lambda: _oracle_radial_distribution(ag,10.,eg))', "tol": 0},
    ]
