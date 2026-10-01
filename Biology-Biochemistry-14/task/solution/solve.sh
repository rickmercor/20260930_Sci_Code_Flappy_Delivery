#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def build_delaunay_neighbourhoods(positions: "np.ndarray") -> "np.ndarray":
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

import numpy as np
def derive_diffusion_bandwidth(radius: float, tail_mass: float) -> float:
    """Reference implementation (closed-form tail mass of a planar Gaussian)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(radius) and radius > 0.0):
        raise ValueError("radius must be a finite positive number")
    if not (_is_number(tail_mass) and 0.0 < tail_mass < 1.0):
        raise ValueError("tail_mass must be a finite number in (0, 1)")
    # In polar coordinates the mass of a planar Gaussian beyond distance t is
    # exp(-t^2 / (2 sigma^2)), which increases with sigma; setting it equal
    # to the admissible tail gives the widest kernel.
    sigma = float(radius) / np.sqrt(-2.0 * np.log(float(tail_mass)))
    return float(sigma)

import numpy as np
def compute_ligand_sharing_weights(
    positions: "np.ndarray",
    radius: float,
    tail_mass: float,
) -> "np.ndarray":
    """Reference implementation (Gaussian kernel normalised over each sender)."""
    import numpy as np

    try:
        points = np.array(positions, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("positions must be a numeric array") from None
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    sigma = derive_diffusion_bandwidth(radius, tail_mass)
    if float(radius) < 1e-9:
        raise ValueError("radius must be at least the self-distance 1e-9")
    offsets = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(offsets ** 2, axis=2))
    np.fill_diagonal(distance, 1e-9)
    density = np.exp(-distance ** 2 / (2.0 * sigma ** 2)) / (2.0 * np.pi * sigma ** 2)
    density[distance > float(radius)] = 0.0
    # Row k holds the kernel from sender k to every receiver; dividing by the
    # row total shares the sender's release over its own neighbourhood.
    return density / density.sum(axis=1, keepdims=True)

import numpy as np
def compute_total_expression_rates(
    unspliced: "np.ndarray",
    spliced: "np.ndarray",
    transcription_rate: float,
    splicing_rate: float,
    degradation_rate: float,
) -> "np.ndarray":
    """Reference implementation (sum of the unspliced and spliced balances)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _abundance(values, name):
        try:
            array = np.array(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must have shape (n,) with n >= 1")
        if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
            raise ValueError(f"{name} must be finite and nonnegative")
        return array

    u = _abundance(unspliced, "unspliced")
    s = _abundance(spliced, "spliced")
    if u.shape != s.shape:
        raise ValueError("unspliced and spliced must have the same length")
    if not (_is_number(transcription_rate) and transcription_rate >= 0.0):
        raise ValueError("transcription_rate must be a finite nonnegative number")
    if not (_is_number(splicing_rate) and splicing_rate > 0.0):
        raise ValueError("splicing_rate must be a finite positive number")
    if not (_is_number(degradation_rate) and degradation_rate > 0.0):
        raise ValueError("degradation_rate must be a finite positive number")
    unspliced_rate = float(transcription_rate) - float(splicing_rate) * u
    spliced_rate = float(splicing_rate) * u - float(degradation_rate) * s
    return np.column_stack([u + s, unspliced_rate + spliced_rate])

import numpy as np
def compute_cell_signalling_velocity_terms(
    sharing_weights: "np.ndarray",
    ligand_rates: "np.ndarray",
    receptor_subunit_rates: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation (shared ligand, summed receptor subunits, product rule)."""
    import numpy as np

    def _array(values, name):
        try:
            array = np.array(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array

    weights = _array(sharing_weights, "sharing_weights")
    ligand = _array(ligand_rates, "ligand_rates")
    receptor = _array(receptor_subunit_rates, "receptor_subunit_rates")
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] < 1:
        raise ValueError("sharing_weights must have shape (n, n) with n >= 1")
    n = weights.shape[0]
    if np.any(weights < 0.0):
        raise ValueError("sharing_weights must be nonnegative")
    if ligand.shape != (n, 2) or np.any(ligand[:, 0] < 0.0):
        raise ValueError("ligand_rates must have shape (n, 2) with nonnegative abundances")
    if receptor.ndim != 3 or receptor.shape[0] < 1 or receptor.shape[1:] != (n, 2):
        raise ValueError("receptor_subunit_rates must have shape (m, n, 2) with m >= 1")
    if np.any(receptor[:, :, 0] < 0.0):
        raise ValueError("receptor subunit abundances must be nonnegative")
    # Receiver i collects column i of the sender-by-receiver weights.
    available = weights.T @ ligand[:, 0]
    available_rate = weights.T @ ligand[:, 1]
    receptor_total = receptor[:, :, 0].sum(axis=0)
    receptor_rate = receptor[:, :, 1].sum(axis=0)
    return np.column_stack([receptor_total * available_rate, available * receptor_rate])

import numpy as np
def average_over_neighbourhoods(values: "np.ndarray", neighbourhoods: "np.ndarray") -> "np.ndarray":
    """Reference implementation (row-normalised membership matrix)."""
    import numpy as np

    try:
        members = np.array(neighbourhoods, dtype=float)
        data = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("values and neighbourhoods must be numeric arrays") from None
    if members.ndim != 2 or members.shape[0] != members.shape[1] or members.shape[0] < 1:
        raise ValueError("neighbourhoods must have shape (n, n) with n >= 1")
    if not np.all((members == 0.0) | (members == 1.0)):
        raise ValueError("neighbourhoods must hold only zeros and ones")
    if not (np.all(np.diag(members) == 1.0) and np.array_equal(members, members.T)):
        raise ValueError("neighbourhoods must be symmetric with ones on the diagonal")
    n = members.shape[0]
    if data.ndim not in (1, 2) or data.shape[0] != n:
        raise ValueError("values must have shape (n,) or (n, d)")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must be finite")
    means = members / members.sum(axis=1, keepdims=True)
    return means @ data

import numpy as np
def estimate_neighbourhood_signalling_velocity(
    positions: "np.ndarray",
    ligand_counts: "np.ndarray",
    ligand_kinetics: tuple,
    receptor_subunit_counts: "np.ndarray",
    receptor_subunit_kinetics: "np.ndarray",
    cell_index: int,
    radius: float = 200.0,
    tail_mass: float = 1e-9,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _has_boolean(values):
        return any(isinstance(v, (bool, np.bool_)) for v in np.asarray(values, dtype=object).ravel())

    try:
        if _has_boolean(ligand_kinetics) or _has_boolean(receptor_subunit_kinetics):
            raise ValueError("kinetic rates must be numbers, not booleans")
        points = np.array(positions, dtype=float)
        ligand = np.array(ligand_counts, dtype=float)
        kinetics = np.array(ligand_kinetics, dtype=float)
        receptor = np.array(receptor_subunit_counts, dtype=float)
        subunit_kinetics = np.array(receptor_subunit_kinetics, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("inputs must be numeric arrays") from None
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n, 2)")
    n = points.shape[0]
    if ligand.shape != (n, 2) or kinetics.shape != (3,):
        raise ValueError("ligand_counts must be (n, 2) and ligand_kinetics must hold three rates")
    if receptor.ndim != 3 or receptor.shape[0] < 1 or receptor.shape[1:] != (n, 2):
        raise ValueError("receptor_subunit_counts must have shape (m, n, 2) with m >= 1")
    if subunit_kinetics.shape != (receptor.shape[0], 3):
        raise ValueError("receptor_subunit_kinetics must have shape (m, 3)")
    if not (_is_integer(cell_index) and 0 <= cell_index < n):
        raise ValueError("cell_index must be an integer in [0, n)")
    neighbourhoods = build_delaunay_neighbourhoods(points)
    weights = compute_ligand_sharing_weights(points, radius, tail_mass)
    ligand_rates = compute_total_expression_rates(
        ligand[:, 0], ligand[:, 1], float(kinetics[0]), float(kinetics[1]), float(kinetics[2])
    )
    subunit_rates = np.stack([
        compute_total_expression_rates(
            receptor[j, :, 0], receptor[j, :, 1],
            float(subunit_kinetics[j, 0]), float(subunit_kinetics[j, 1]), float(subunit_kinetics[j, 2]),
        )
        for j in range(receptor.shape[0])
    ])
    terms = compute_cell_signalling_velocity_terms(weights, ligand_rates, subunit_rates)
    neighbourhood_terms = average_over_neighbourhoods(terms, neighbourhoods)
    velocity = float(np.sum(neighbourhood_terms[int(cell_index)]))
    if not np.isfinite(velocity):
        raise ValueError("the neighbourhood signalling velocity is not finite")
    return velocity
SCICODE_GOLD_EOF
