#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_force_disagreement_scores(force_ensembles: np.ndarray) -> np.ndarray:
    import numpy as np
    arr = np.asarray(force_ensembles, dtype=float)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError("force_ensembles must have shape (n_configs, n_models, n_atoms, 3)")
    if arr.shape[1] < 2:
        raise ValueError("at least two ensemble members are required")
    if arr.shape[0] < 1 or arr.shape[2] < 1:
        raise ValueError("n_configs and n_atoms must be positive")
    std = np.std(arr, axis=1, ddof=0)
    return np.max(np.linalg.norm(std, axis=-1), axis=-1).astype(float)

import numpy as np
def compute_force_labeling_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    import numpy as np
    u = np.asarray(scores, dtype=float).reshape(-1)
    if u.size < 1:
        raise ValueError("scores must be non-empty")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be a finite non-negative float")
    if np.any(~np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("scores must be finite and non-negative")
    return (u > float(threshold)).astype(float)

def compute_hamiltonian_disagreement_scores(h_ensembles: np.ndarray) -> np.ndarray:
    import numpy as np
    arr = np.asarray(h_ensembles, dtype=float)
    if arr.ndim != 4:
        raise ValueError("h_ensembles must have shape (n_configs, n_models, n_orb, n_orb)")
    if arr.shape[1] < 2:
        raise ValueError("at least two ensemble members are required")
    if arr.shape[0] < 1 or arr.shape[2] != arr.shape[3] or arr.shape[2] < 1:
        raise ValueError("invalid Hamiltonian ensemble shape")
    std = np.std(arr, axis=1, ddof=0)
    return np.sqrt((std**2).mean(axis=(-2, -1))).astype(float)

import numpy as np
def compute_balanced_acquisition_indices(
    
    scores: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    n_per_group: int,
) -> np.ndarray:
    u = np.asarray(scores, dtype=float).reshape(-1)
    d = np.asarray(defect_ids).reshape(-1)
    t = np.asarray(temperatures, dtype=float).reshape(-1)
    if not (u.size == d.size == t.size) or u.size < 1:
        raise ValueError("scores, defect_ids, temperatures length mismatch")
    if int(n_per_group) < 1:
        raise ValueError("n_per_group must be >= 1")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(t)):
        raise ValueError("scores and temperatures must be finite")
    selected: list[int] = []
    for defect in sorted(set(d.tolist())):
        for temperature in sorted(set(t.tolist())):
            idxs = np.where((d == defect) & np.isclose(t, float(temperature)))[0]
            if idxs.size < int(n_per_group):
                raise ValueError("insufficient candidates in a (defect, temperature) group")
            order = idxs[np.argsort(-u[idxs], kind="mergesort")]
            selected.extend(int(i) for i in order[: int(n_per_group)])
    return np.asarray(selected, dtype=float)

import numpy as np


def compute_gap_spectrum_table(hamiltonians: np.ndarray) -> np.ndarray:
    arr = np.asarray(hamiltonians, dtype=float)
    single = arr.ndim == 2
    if single:
        if arr.shape != (6, 6):
            raise ValueError("single Hamiltonian must have shape (6, 6)")
        arr2 = arr.reshape(1, 6, 6)
    elif arr.ndim == 3:
        arr2 = arr
    else:
        raise ValueError("hamiltonians must have shape (6, 6) or (n, 6, 6)")
    if arr2.shape[1:] != (6, 6):
        raise ValueError("each Hamiltonian must be 6x6")
    if np.any(~np.isfinite(arr2)):
        raise ValueError("hamiltonians must be finite")
    if np.any(np.abs(arr2 - np.swapaxes(arr2, -1, -2)) > 1e-10):
        raise ValueError("hamiltonians must be symmetric")
    e = np.linalg.eigvalsh(arr2)
    e = np.sort(e, axis=1)
    vbm, ed, cbm = e[:, 2], e[:, 3], e[:, 4]
    if np.any(~((vbm < ed) & (ed < cbm))):
        raise ValueError("mid-gap level is not strictly inside the host gap")
    if np.any(cbm - vbm <= 0.0):
        raise ValueError("non-positive host gap")
    out = np.stack([vbm, ed, cbm, cbm - ed], axis=1)
    return out[0] if single else out

import numpy as np


def compute_asga_ordered_depth_series(
    selected_indices: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    asga_defect_id: int = 4,
) -> np.ndarray:
    sel = np.asarray(selected_indices, dtype=int).reshape(-1)
    d = np.asarray(defect_ids).reshape(-1)
    t = np.asarray(temperatures, dtype=float).reshape(-1)
    H = np.asarray(trained_hamiltonians, dtype=float)
    if H.ndim != 3 or H.shape[1:] != (6, 6):
        raise ValueError("trained_hamiltonians must have shape (n_pool, 6, 6)")
    if not (d.size == t.size == H.shape[0]):
        raise ValueError("defect_ids, temperatures, trained_hamiltonians length mismatch")
    if sel.size < 1:
        raise ValueError("selected_indices must be non-empty")
    if np.any(sel < 0) or np.any(sel >= H.shape[0]):
        raise ValueError("selected index out of range")

    keep = sel[d[sel] == int(asga_defect_id)]
    if keep.size < 1:
        raise ValueError("no As_Ga structures in the selection")

    order = np.lexsort((-keep, t[keep]))
    keep_ordered = keep[order]

    table = compute_gap_spectrum_table(H[keep_ordered])
    if table.ndim == 1:
        table = table.reshape(1, 4)
    return table[:, 3].astype(float)

import numpy as np
def compute_temperature_paired_asga_means(
    ordered_depths: np.ndarray, n_per_temperature: int
) -> np.ndarray:
    
    d = np.asarray(ordered_depths, dtype=float).reshape(-1)
    n = int(n_per_temperature)
    if n < 1:
        raise ValueError("n_per_temperature must be >= 1")
    if d.size < 1:
        raise ValueError("ordered_depths must be non-empty")
    if d.size % n != 0:
        raise ValueError("ordered_depths length must be divisible by n_per_temperature")
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("depths must be finite and non-negative")
    blocks = d.reshape(-1, n)
    return blocks.mean(axis=1).astype(float)

import numpy as np


def orchestrate_defect_al_cbm_depth(
    force_ensembles: np.ndarray,
    force_threshold: float,
    h_ensembles: np.ndarray,
    defect_ids: np.ndarray,
    temperatures: np.ndarray,
    trained_hamiltonians: np.ndarray,
    n_per_group: int | None = None,
    asga_defect_id: int = 4,
) -> float:
    force_scores = compute_force_disagreement_scores(force_ensembles)
    force_mask = compute_force_labeling_mask(force_scores, force_threshold)
    n_labeled = int(np.sum(force_mask))
    if n_labeled < 1:
        raise ValueError("force active learning selected no configurations")
    if n_per_group is None:
        n_per_group = n_labeled // 3
    n_per_group = int(n_per_group)
    if n_per_group < 1:
        raise ValueError("derived or provided n_per_group must be >= 1")

    h_scores = compute_hamiltonian_disagreement_scores(h_ensembles)
    selected = compute_balanced_acquisition_indices(
        h_scores, defect_ids, temperatures, n_per_group
    )
    ordered = compute_asga_ordered_depth_series(
        selected, defect_ids, temperatures, trained_hamiltonians, asga_defect_id
    )
    paired = compute_temperature_paired_asga_means(ordered, n_per_group)
    p = np.asarray(paired, dtype=float).reshape(-1)
    if p.size < 1:
        raise ValueError("empty temperature-paired means")
    return float(p.sum() / float(p.size) + 0.0 * float(n_labeled))
SCICODE_GOLD_EOF
