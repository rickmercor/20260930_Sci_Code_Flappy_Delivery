"""
Computes geometric packing fidelity for synthetic periodic molecular crystals.

This stage replaces a Gaussian proxy for RMSD15 with a coordinate-based benchmark.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_packing_fidelity(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    """Return mean RMSD15 and its fraction below 0.3 angstrom per system.

    Parameters
    ----------
    summary : object
        Finite curation summary with shape (9, 7).
    seed : int
        Seed for the shared benchmark stream.
    n_frames : int
        Number of frames, at least 60.

    Returns
    -------
    object
        Float64 array of shape (9, 2): mean RMSD15 and fraction below 0.3.

    Raises
    ------
    ValueError
        If the summary, seed, or frame count is invalid.

    Notes
    -----
    The supplied instance-generation convention is defined in the task prompt.
    Use the source's RMSD15 geometry method for each generated frame. This stage
    uses its own RNG stream; Step 6 retains its independently defined stream.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _whole_molecules(frac):
    anchor = frac[:, :1, :]
    delta = frac - anchor
    whole = anchor + delta - np.rint(delta)
    return whole, whole.mean(axis=1)

def _frame_rmsd15(ref, cand, cref, ccand):
    ref_whole, ref_centers = _whole_molecules(ref)
    cand_whole, cand_centers = _whole_molecules(cand)
    origin = ref_centers[13]
    offset = ref_centers - origin
    offset -= np.rint(offset)
    distance = np.linalg.norm(offset @ cref, axis=1)
    selected = np.lexsort((np.arange(27), distance))[:15]
    difference = ref_centers[selected, None, :] - cand_centers[None, :, :]
    difference -= np.rint(difference)
    matched = np.argmin(np.linalg.norm(difference @ ccand, axis=2), axis=1)
    if np.unique(matched).size != 15:
        raise ValueError("molecule correspondence is not unique")
    x = (ref_whole[selected] - np.rint(ref_centers[selected] - origin)[:, None, :]) @ cref
    cand_origin = cand_centers[matched[np.where(selected == 13)[0][0]]]
    y = (cand_whole[matched] - np.rint(cand_centers[matched] - cand_origin)[:, None, :]) @ ccand
    x = x.reshape(-1, 3)
    y = y.reshape(-1, 3)
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    u, _, vt = np.linalg.svd(y.T @ x)
    rotation = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(np.mean(np.sum((x - y @ rotation) ** 2, axis=1))))

def _oracle_compute_packing_fidelity(summary: object, seed: int = 9917, n_frames: int = 240) -> object:
    s = np.asarray(summary, dtype=np.float64)
    if s.shape != (9, 7) or not np.isfinite(s).all() or not isinstance(seed, (int, np.integer)) or not isinstance(n_frames, (int, np.integer)) or n_frames < 60:
        raise ValueError("summary, seed, or n_frames is invalid")
    rng = np.random.default_rng(int(seed))
    offsets = np.stack(np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2), indexing="ij"), axis=-1).reshape(-1, 3)
    motif = np.array([[0.0, 0.0, 0.0], [0.71, 0.21, 0.05], [-0.26, 0.63, -0.08]])
    out = np.empty((9, 2), dtype=np.float64)
    for k in range(9):
        centers = np.mod(np.array([0.93, 0.07, 0.89]) + 0.16 * offsets + np.array([0.003 * k, 0.002 * k, -0.001 * k]), 1.0)
        cell = np.diag([9.1 + 0.07 * k, 8.7 + 0.04 * k, 9.4 + 0.03 * k])
        ref_cart = centers[:, None, :] @ cell + motif[None, :, :]
        ref = np.mod(ref_cart @ np.linalg.inv(cell), 1.0)
        rmsd = np.empty(int(n_frames), dtype=np.float64)
        for t in range(int(n_frames)):
            strain = rng.normal(0.0, 0.012, size=(3, 3))
            other_cell = cell @ (np.eye(3) + strain)
            theta = 0.16 * np.sin((t + 1) * (k + 2) / 37.0)
            rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
            sigma = 0.060 + 0.014 * k + 0.003 * s[k, 3]
            other_cart = centers[:, None, :] @ other_cell + motif[None, :, :] @ rotation
            other_cart += rng.normal(0.0, sigma, size=other_cart.shape)
            other = np.mod(other_cart @ np.linalg.inv(other_cell), 1.0)
            other = other[rng.permutation(27)]
            rmsd[t] = _frame_rmsd15(ref, other, cell, other_cell)
        out[k] = [rmsd.mean(), np.mean(rmsd < 0.3)]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"sc=curate_active_pool(build_error_panel(58031,96));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58031,96))", "call":"compute_packing_fidelity(sc,9917,240)", "gold_call":"_oracle_compute_packing_fidelity(sg,9917,240)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58079,84));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58079,84))", "call":"compute_packing_fidelity(sc,9967,180)", "gold_call":"_oracle_compute_packing_fidelity(sg,9967,180)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58121,72));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58121,72))", "call":"compute_packing_fidelity(sc,10009,60)", "gold_call":"_oracle_compute_packing_fidelity(sg,10009,60)", "tol":1e-9},
        {"setup":"sc=curate_active_pool(build_error_panel(58163,108));sg=_oracle_curate_active_pool(_oracle_build_error_panel(58163,108))", "call":"compute_packing_fidelity(sc,10061,300)", "gold_call":"_oracle_compute_packing_fidelity(sg,10061,300)", "tol":1e-9},
        {"setup":"sc=np.zeros((9,7));sg=sc.copy()\ndef _candidate_probe():\n    try: compute_packing_fidelity(sc,1,59); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef expected_probe():\n    try: _oracle_compute_packing_fidelity(sg,1,59); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call":"_candidate_probe()", "gold_call":"expected_probe()", "tol":1e-9},
    ]
