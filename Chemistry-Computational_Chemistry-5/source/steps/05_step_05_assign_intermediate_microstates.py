"""
Step name: 05_assign_intermediate_microstates
Step description: Cluster the projected coordinates of the frames outside both endpoint states into discrete microstates that will serve as the Galerkin basis.

Step scientific background: The committor is expanded in indicator functions of discrete microstates, so only the configurations that belong to neither endpoint state need a basis. Clustering those configurations in the slow-coordinate space partitions the transition region into cells within which the commitment probability is close to constant.

Returns
-------
np.ndarray: integer microstate index of shape (T,), -1 on frames outside the intermediate region.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def assign_intermediate_microstates(components: np.ndarray, labels: np.ndarray,
                                    n_microstates: int = 64, seed: int = 0,
                                    max_iter: int = 100) -> np.ndarray:
    """Return the microstate index of every intermediate frame.

    Only the rows of ``components`` whose label is ``2`` are clustered; every
    other frame receives ``-1``.

    Conventions fixed by this step, applied to the intermediate rows in
    trajectory order. A generator ``np.random.default_rng(seed)`` is created
    once. The first centre is the row at index ``rng.integers(n)``. Each later
    centre is the row at index ``rng.choice(n, p=w)`` where ``w`` is the
    running squared distance to the nearest chosen centre divided by its own
    sum. Lloyd iterations then alternate assignment and recentring: a row is
    assigned to the centre of smallest squared Euclidean distance with the
    lowest index winning a tie, a centre with no rows keeps its previous
    position, and the loop stops as soon as an assignment repeats the previous
    one or after ``max_iter`` iterations.

    Parameters
    ----------
    components : np.ndarray
        Float array of shape ``(T, d)`` of projected coordinates.
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    n_microstates : int
        Number of microstates, at least 1 and at most the number of
        intermediate frames.
    seed : int
        Seed of the seeding generator.
    max_iter : int
        Maximum number of Lloyd iterations, at least 1.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(T,)`` holding a microstate index in
        ``0 .. n_microstates - 1`` for intermediate frames and ``-1``
        elsewhere.

    Raises
    ------
    ValueError
        If ``components`` is not a finite two-dimensional array, if ``labels``
        is not an integer array of matching length with values in
        ``{0, 1, 2}``, if ``n_microstates`` is not an integer of at least 1 or
        exceeds the number of intermediate frames, if ``seed`` is not an
        integer, or if ``max_iter`` is not an integer of at least 1.
    """
    return microstates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _seed_centres(points, n_centres: int, rng):
    """Return the initial centres chosen by the squared-distance seeding rule."""
    import numpy as np
    n_points = points.shape[0]
    centres = np.empty((n_centres, points.shape[1]), dtype=float)
    centres[0] = points[int(rng.integers(n_points))]
    nearest = np.sum((points - centres[0]) ** 2, axis=1)
    for index in range(1, n_centres):
        total = nearest.sum()
        weights = nearest / total if total > 0.0 else np.full(n_points, 1.0 / n_points)
        centres[index] = points[int(rng.choice(n_points, p=weights))]
        nearest = np.minimum(nearest, np.sum((points - centres[index]) ** 2, axis=1))
    return centres


def _lloyd_refine(points, centres, max_iter: int):
    """Return the assignment produced by Lloyd iterations from given centres."""
    import numpy as np
    assignment = np.full(points.shape[0], -1, dtype=np.int64)
    for _ in range(int(max_iter)):
        distances = np.sum((points[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        proposal = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(proposal, assignment):
            break
        assignment = proposal
        for index in range(centres.shape[0]):
            member = assignment == index
            if member.any():
                centres[index] = points[member].mean(axis=0)
    return assignment


import numpy as np
def _oracle_assign_intermediate_microstates(components: np.ndarray, labels: np.ndarray,
                                            n_microstates: int = 64, seed: int = 0,
                                            max_iter: int = 100) -> np.ndarray:
    """Reference implementation (squared-distance seeding plus Lloyd refinement)."""
    import numpy as np

    coords = np.asarray(components, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 1 or coords.shape[1] < 1:
        raise ValueError("components must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(coords)):
        raise ValueError("components must be finite")
    tags = np.asarray(labels)
    if tags.ndim != 1 or tags.shape[0] != coords.shape[0] or not np.issubdtype(tags.dtype, np.integer):
        raise ValueError("labels must be an integer array matching the number of rows")
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    if not _is_integer(seed):
        raise ValueError("seed must be an integer")
    if not (_is_integer(max_iter) and int(max_iter) >= 1):
        raise ValueError("max_iter must be an integer of at least 1")
    interior = np.flatnonzero(tags == 2)
    if not (_is_integer(n_microstates) and 1 <= int(n_microstates) <= interior.size):
        raise ValueError("n_microstates must be an integer between 1 and the number of intermediate frames")

    points = coords[interior]
    rng = np.random.default_rng(int(seed))
    centres = _seed_centres(points, int(n_microstates), rng)
    assignment = _lloyd_refine(points, centres, int(max_iter))

    microstates = np.full(coords.shape[0], -1, dtype=np.int64)
    microstates[interior] = assignment
    return microstates

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
        "traj = simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab = label_reaction_endpoints(traj)\n"
        "comp = project_dynamical_components(build_local_descriptors(traj))\n"
        "traj_gold = _oracle_simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab_gold = _oracle_label_reaction_endpoints(traj_gold)\n"
        "comp_gold = _oracle_project_dynamical_components(_oracle_build_local_descriptors(traj_gold))\n"
        "rng = np.random.default_rng(2)\n"
        "blobs = np.concatenate([rng.standard_normal((90, 2)) + np.array([5.0, 0.0]),\n"
        "                        rng.standard_normal((90, 2)) - np.array([5.0, 0.0])])\n"
        "btag = np.full(180, 2, dtype=np.int64)\n"
        "btag[:10] = 0\n"
        "btag[-10:] = 1\n"
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
        "comp = np.linspace(0.0, 1.0, 30).reshape(15, 2)\n"
        "tag = np.array([0, 1] + [2] * 13, dtype=np.int64)\n"
    )
    return [
        {
            "setup": digest,
            "call": "_sig(assign_intermediate_microstates(comp, lab, n_microstates=32))",
            "gold_call": "_sig(_oracle_assign_intermediate_microstates(comp_gold, lab_gold, n_microstates=32))",
        },
        {
            "setup": digest,
            "call": "float(np.bincount(assign_intermediate_microstates(comp, lab, n_microstates=16)[lab == 2], minlength=16).max())",
            "gold_call": "float(np.bincount(_oracle_assign_intermediate_microstates(comp_gold, lab_gold, n_microstates=16)[lab_gold == 2], minlength=16).max())",
        },
        {
            "setup": digest,
            "call": "_sig(assign_intermediate_microstates(blobs, btag, n_microstates=2, seed=4))",
            "gold_call": "_sig(_oracle_assign_intermediate_microstates(blobs, btag, n_microstates=2, seed=4))",
        },
        {
            "setup": digest,
            "call": "_sig(assign_intermediate_microstates(blobs, btag, n_microstates=8, seed=1, max_iter=1))",
            "gold_call": "_sig(_oracle_assign_intermediate_microstates(blobs, btag, n_microstates=8, seed=1, max_iter=1))",
        },
        {
            "setup": digest,
            "call": "float(np.sum(assign_intermediate_microstates(blobs, btag, n_microstates=1) == -1))",
            "gold_call": "float(np.sum(_oracle_assign_intermediate_microstates(blobs, btag, n_microstates=1) == -1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: assign_intermediate_microstates(comp, tag, n_microstates=99))",
            "gold_call": "_status(lambda: _oracle_assign_intermediate_microstates(comp, tag, n_microstates=99))",
        },
        {
            "setup": status,
            "call": "_status(lambda: assign_intermediate_microstates(comp, tag[:5], n_microstates=2))",
            "gold_call": "_status(lambda: _oracle_assign_intermediate_microstates(comp, tag[:5], n_microstates=2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: assign_intermediate_microstates(comp, np.full(15, 3, dtype=np.int64), n_microstates=2))",
            "gold_call": "_status(lambda: _oracle_assign_intermediate_microstates(comp, np.full(15, 3, dtype=np.int64), n_microstates=2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: assign_intermediate_microstates(comp, tag, n_microstates=2, max_iter=0))",
            "gold_call": "_status(lambda: _oracle_assign_intermediate_microstates(comp, tag, n_microstates=2, max_iter=0))",
        },
    ]
