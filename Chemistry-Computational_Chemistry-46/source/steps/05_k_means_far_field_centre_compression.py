"""
Partition the pooled far-field periodic image sites, by their in-plane (x, y) coordinates only, into K expansion centres by K-means with k-means++ initialisation and Lloyd iteration, implemented with numpy alone so the step is self-contained.  The reference runs a fixed floor of 64 random restarts and keeps the lowest-inertia clustering.  The prompt states the clustering parameters as K = 3, n_init = 10, random_state = 0, tol = 1e-6 (the scikit-learn ``KMeans`` convention); this reference uses `$random_state$` to seed the restarts and ``tol`$to bound the per-step centre shift as stated, but treats$$n_init$` only as a lower bound on the restart count -- not the exact restart budget -- because plain k-means++ needs more restarts than scikit-learn's greedy variant to reach the same optimum.  At the benchmark K = 3 the 64-restart clustering equals the global optimum ``sklearn.cluster.KMeans(n_init=10)`` finds; at other K the two implementations may settle on different local optima.  Each centre's z coordinate is the mean z of the sites assigned to it.  This step produces only the clustering; the orchestrator then origin-shifts each image's permanent moments onto its assigned centre and sums them.

A 2-D-periodic system requires summing interactions over infinitely many periodic images of every site.  Once the images are far enough away their fine angular detail no longer affects the converged electrostatics: a group of nearby image sites is almost indistinguishable from a single point multipole at their combined centre.  K-means groups the distant images spatially -- k-means++ spreads the initial centres by squared-distance sampling, then Lloyd's algorithm alternates nearest-centre assignment and centre-as-mean update until the centres stop moving -- yielding the small set of expansion centres onto which the distributed moments are then compressed.

Returns
-------
return np.zeros(3 * n_clusters, dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

_MIN_RESTARTS = 64

def kmeans_expansion_centers(xy, z, n_clusters: int = 3, random_state: int = 0,
                             n_init: int = 10, tol: float = 1e-6) -> np.ndarray:
    '''K-means expansion centres for the pooled far-field image sites.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        In-plane (x, y) coordinates of the pooled far-field image sites.
    z : array_like, shape (N,)
        z coordinates of the same sites.
    n_clusters : int
        Number of expansion centres K (default 3).
    random_state : int
        Seed for ``numpy.random.default_rng`` -- fixes the k-means++ seeding.
    n_init : int
        Requested number of random restarts (the reference uses at least a fixed floor).
    tol : float
        Convergence threshold on the summed centre shift per Lloyd step.

    Returns
    -------
    packed : np.ndarray, shape (3*n_clusters + N,)
        The K centre positions [cx, cy, cz] (cz = mean z of assigned sites), ordered by
        ascending cx then cy, followed by the N per-site cluster labels as floats --
        each an index into that ordered centre list.

    Raises
    ------
    ValueError
        If ``xy`` is not shape (N, 2) or ``z`` is not shape (N,); if any input is not
        finite; or if ``n_clusters`` is not an integer in [1, N].
    '''
    return np.zeros(3 * n_clusters, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_MIN_RESTARTS = 64

def _kmeans_pp_lloyd(X, k, rng, tol, max_iter=300):
    """One k-means++ seeding + Lloyd run. Returns (centres (k,d), labels (N,), inertia)."""
    n, d = X.shape

    # --- k-means++ initialisation (squared-distance / D^2 sampling) ---
    centres = np.empty((k, d), dtype=float)
    centres[0] = X[int(rng.integers(n))]
    d2 = np.sum((X - centres[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:
            centres[c] = X[int(rng.integers(n))]
        else:
            r = float(rng.random()) * total
            cand = int(np.searchsorted(np.cumsum(d2), r, side="right"))
            cand = min(cand, n - 1)
            if d2[cand] == 0.0:
                cand = int(np.argmax(d2))
            centres[c] = X[cand]
        d2 = np.minimum(d2, np.sum((X - centres[c]) ** 2, axis=1))

    # --- Lloyd iteration ---
    labels = np.full(n, -1, dtype=int)
    for _ in range(int(max_iter)):
        dist2 = np.sum((X[:, None, :] - centres[None, :, :]) ** 2, axis=2)   # (N, k)
        new_labels = np.argmin(dist2, axis=1)
        new_centres = centres.copy()
        for c in range(k):
            m = new_labels == c
            if m.any():
                new_centres[c] = X[m].mean(axis=0)
            else:  # relocate an empty cluster onto its worst-fit point
                far = int(np.argmax(dist2[np.arange(n), new_labels]))
                new_centres[c] = X[far]
                new_labels[far] = c
        shift = float(np.sqrt(np.sum((new_centres - centres) ** 2, axis=1)).sum())
        stable = np.array_equal(new_labels, labels)
        centres, labels = new_centres, new_labels
        if stable and shift <= float(tol):
            break

    inertia = float(np.sum((X - centres[labels]) ** 2))
    return centres, labels, inertia


def _run_kmeans(X, k, random_state, n_init, tol):
    """k-means with many restarts; keeps the lowest-inertia clustering. Deterministic."""
    rng = np.random.default_rng(int(random_state))
    n_restarts = max(int(n_init), _MIN_RESTARTS)
    best = None
    for _ in range(n_restarts):
        centres, labels, inertia = _kmeans_pp_lloyd(X, int(k), rng, float(tol))
        if best is None or inertia < best[2] - 1e-12:
            best = (centres, labels, inertia)
    return best[0], best[1]

def _oracle_kmeans_expansion_centers(xy, z, n_clusters: int = 3, random_state: int = 0,
                                     n_init: int = 10, tol: float = 1e-6) -> np.ndarray:
    """Reference K-means clustering of the pooled far-field image sites (numpy only)."""
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    n = xy.shape[0]
    if z.shape[0] != n:
        raise ValueError("z must have shape (N,)")
    if not (np.all(np.isfinite(xy)) and np.all(np.isfinite(z))):
        raise ValueError("inputs must be finite")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1 or n_clusters > n:
        raise ValueError("n_clusters must be an integer in [1, N]")

    cxy, raw = _run_kmeans(xy, int(n_clusters), random_state, n_init, tol)

    centres = []
    for k in range(int(n_clusters)):
        m = raw == k
        cz = float(np.mean(z[m])) if np.any(m) else 0.0
        centres.append((float(cxy[k, 0]), float(cxy[k, 1]), cz, k))
    centres.sort(key=lambda c: (round(c[0], 10), round(c[1], 10)))
    old_to_new = {c[3]: i for i, c in enumerate(centres)}
    labels = np.array([old_to_new[int(r)] for r in raw], dtype=float)

    out = np.empty(3 * int(n_clusters) + n, dtype=float)
    for i, (cx, cy, cz, _) in enumerate(centres):
        out[3 * i: 3 * i + 3] = (cx, cy, cz)
    out[3 * int(n_clusters):] = labels
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test-case specifications."""
    return [
        {
            "setup": """import numpy as np
rng = np.random.default_rng(0)
a = rng.normal(loc=[-5.0, 0.0], scale=0.3, size=(12, 2))
b = rng.normal(loc=[ 5.0, 3.0], scale=0.3, size=(12, 2))
xy = np.vstack([a, b])
z = rng.normal(scale=0.4, size=24)
""",
            "call": "kmeans_expansion_centers(xy, z, n_clusters=2)",
            "gold_call": "_oracle_kmeans_expansion_centers(xy, z, n_clusters=2)",
        },
        {
            "setup": """import numpy as np
xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
z = np.array([0.1, -0.2, 0.3])
""",
            "call": "kmeans_expansion_centers(xy, z, n_clusters=3)",
            "gold_call": "_oracle_kmeans_expansion_centers(xy, z, n_clusters=3)",
        },
        {
            # Three well-separated blobs -> the K=3 clustering has a single unambiguous
            # global optimum (any restart count finds it). Compare the K returned centre
            # positions, sorted lexicographically so the check is insensitive to which
            # cluster index gets which label but still catches a wrong clustering.
            "setup": """import numpy as np
rng = np.random.default_rng(3)
xy = np.vstack([np.array([0.0, 0.0]) + 0.15 * rng.standard_normal((8, 2)),
                np.array([12.0, 1.0]) + 0.15 * rng.standard_normal((8, 2)),
                np.array([5.0, 10.0]) + 0.15 * rng.standard_normal((8, 2))])
z = 0.05 * rng.standard_normal(24)
def sorted_centres(fn):
    p = fn(xy, z, n_clusters=3)
    c = p[:9].reshape(3, 3)
    return c[np.lexsort(c.T[::-1])]
""",
            "call": "sorted_centres(kmeans_expansion_centers)",
            "gold_call": "sorted_centres(_oracle_kmeans_expansion_centers)",
        },
        {
            "setup": """import numpy as np
xy = np.zeros((5, 3)); z = np.zeros(5)
def run_model():
    try:
        kmeans_expansion_centers(xy, z); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_kmeans_expansion_centers(xy, z); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
xy = np.zeros((3, 2)); z = np.zeros(3)
def run_model():
    try:
        kmeans_expansion_centers(xy, z, n_clusters=9); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_kmeans_expansion_centers(xy, z, n_clusters=9); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
