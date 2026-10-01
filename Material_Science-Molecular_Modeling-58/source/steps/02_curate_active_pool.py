"""
Applies the paper's energy-force curation method and requested split.

The source combines density-based filtering in joint energy-force space with explicit
force and relative-energy gates before fine-tuning each molecular-crystal model.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def curate_active_pool(panel: object, radius: float = 3.4, train_fraction: float = 0.85) -> object:
    """Return per-system curation and split diagnostics.

    Parameters
    ----------
    panel : object
        Finite array of shape (n, 9, 4), with n at least 24.
    radius : float
        Positive scale for the specified DBSCAN epsilon, `radius/4`.
    train_fraction : float
        Fraction strictly between zero and one used for the training count.

    Returns
    -------
    object
        A float64 array of shape (9, 7), with columns kept, train,
        validation, mean energy, mean force, population covariance, max robust radius.

    Raises
    ------
    ValueError
        If inputs are nonfinite, shapes or parameters are invalid, or fewer than
        eight configurations survive for a system.

    Notes
    -----
    Within each system, apply the two physical gates before robust scaling and
    density-based filtering. The DBSCAN epsilon is radius/4 and min_samples=5,
    counting each point itself. Use median and 1.4826 times MAD of the physically
    eligible points in that system; if a MAD is zero, use scale 1.0. Retain the
    largest density-connected cluster (tie by smallest original index) and
    stable-sort its indices by leverage before the requested train/validation allocation.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _main_density_cluster(points: np.ndarray, eps: float) -> np.ndarray:
    distance = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    neighbor = distance <= eps
    core = neighbor.sum(axis=1) >= 5
    labels = np.full(len(points), -1, dtype=int)
    cluster_id = 0
    for start in range(len(points)):
        if not core[start] or labels[start] >= 0:
            continue
        labels[start] = cluster_id
        queue = [start]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for nxt in np.flatnonzero(neighbor[current] & core):
                if labels[nxt] < 0:
                    labels[nxt] = cluster_id
                    queue.append(int(nxt))
        cluster_id += 1
    # Assign border points to the first core cluster reaching them.
    for i in range(len(points)):
        if labels[i] < 0:
            adjacent = labels[neighbor[i] & core]
            if adjacent.size:
                labels[i] = int(adjacent.min())
    if cluster_id == 0:
        return np.zeros(len(points), dtype=bool)
    sizes = np.bincount(labels[labels >= 0], minlength=cluster_id)
    best = int(np.argmax(sizes))
    return labels == best

def _oracle_curate_active_pool(panel: object, radius: float = 3.4, train_fraction: float = 0.85) -> object:
    x = np.asarray(panel, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (9, 4) or x.shape[0] < 24 or not np.isfinite(x).all():
        raise ValueError("panel must be finite with shape (n, 9, 4), n >= 24")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(train_fraction) or not (0 < train_fraction < 1):
        raise ValueError("radius and train_fraction are invalid")
    out = np.empty((9, 7), dtype=np.float64)
    for s in range(9):
        eligible = np.flatnonzero((x[:, s, 0] <= 0.2) & (x[:, s, 1] <= 10.0))
        if eligible.size < 8:
            raise ValueError("curation retained too few configurations")
        joint = x[eligible, s, :2]
        med = np.median(joint, axis=0)
        mad = np.median(np.abs(joint - med), axis=0)
        scale = np.where(mad > 0, 1.4826 * mad, 1.0)
        standardized = (joint - med) / scale
        density_keep = _main_density_cluster(standardized, float(radius) / 4.0)
        idx = eligible[density_keep]
        if idx.size < 8:
            raise ValueError("curation retained too few configurations")
        order = idx[np.argsort(x[idx, s, 3], kind="stable")]
        n_train = int(np.floor(float(train_fraction) * order.size))
        n_train = min(max(n_train, 1), order.size - 1)
        vals = x[order, s, :2]
        cov = np.cov(vals.T, ddof=0)
        out[s] = [order.size, n_train, order.size - n_train, vals[:, 0].mean(), vals[:, 1].mean(), cov[0, 1], np.linalg.norm((vals - med) / scale, axis=1).max()]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"pc=build_error_panel(58031,96);pg=_oracle_build_error_panel(58031,96)", "call":"curate_active_pool(pc,3.4,0.85)", "gold_call":"_oracle_curate_active_pool(pg,3.4,0.85)", "tol":1e-9},
        {"setup":"pc=build_error_panel(58079,84);pg=_oracle_build_error_panel(58079,84)", "call":"curate_active_pool(pc,3.0,0.8)", "gold_call":"_oracle_curate_active_pool(pg,3.0,0.8)", "tol":1e-9},
        {"setup":"pc=build_error_panel(58121,72);pg=_oracle_build_error_panel(58121,72)", "call":"curate_active_pool(pc,3.8,0.9)", "gold_call":"_oracle_curate_active_pool(pg,3.8,0.9)", "tol":1e-9},
        {"setup":"pc=build_error_panel(58163,108);pg=_oracle_build_error_panel(58163,108)", "call":"curate_active_pool(pc,4.2,0.75)", "gold_call":"_oracle_curate_active_pool(pg,4.2,0.75)", "tol":1e-9},
        {"setup":"pc=np.zeros((12,9,4));pg=pc.copy()\ndef _candidate_probe():\n    try: curate_active_pool(pc); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_curate_active_pool(pg); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
