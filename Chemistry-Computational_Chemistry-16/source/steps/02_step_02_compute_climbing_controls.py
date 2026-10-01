"""
Call public Step 1. For each mechanism choose the highest-energy internal image. Normalize the centered secant path[k+1]-path[k-1] as tau. Reflect the true force through tau using Fclimb=F-2*(F dot tau)*tau and record FCI=norm(Fclimb). Set the deterministic latch history to 3+((7*m+seed) mod 6). Define control_checksum=dot(indices,1+m/23)+dot(tangents.ravel(),0.4+(k mod 13)/17)+dot(FCI,2+m/29). Return indices, tangents, reflected forces, FCI, latch counts, F0, unchanged panel_checksum, and control_checksum.

The climbing-image and minimum-mode forces share a reflection structure. The source is needed later to decide when the second reflection is allowed to take control.

Returns
-------
tuple : indices, tangents, climbing forces, norms, latch counts, baselines, and checksums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_climbing_controls(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    """Return climbing-image controls and checksums for the complete panel.

    Raises:
        ValueError: If an upstream panel argument is inadmissible.
    """
    return None, None, None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_climbing_controls(seed: int = 260529, n_mechanisms: int = 18, n_images: int = 9) -> tuple:
    paths, energies, forces, baselines, panel_checksum = _oracle_generate_oci_neb_panel(seed, n_mechanisms, n_images)
    indices = np.argmax(energies[:, 1:-1], axis=1) + 1
    tangents = np.empty((n_mechanisms, 3), dtype=float)
    climb_forces = np.empty_like(tangents)
    norms = np.empty(n_mechanisms, dtype=float)
    stable = np.empty(n_mechanisms, dtype=int)
    for m, k in enumerate(indices):
        tangent = paths[m, k + 1] - paths[m, k - 1]
        tangent /= np.linalg.norm(tangent)
        tangents[m] = tangent
        force = forces[m, k]
        climb = force - 2.0 * np.dot(force, tangent) * tangent
        climb_forces[m] = climb
        norms[m] = np.linalg.norm(climb)
        stable[m] = 3 + ((m * 7 + seed) % 6)
    checksum = float(np.dot(indices, 1.0 + np.arange(n_mechanisms) / 23.0)
                     + np.dot(tangents.ravel(), 0.4 + (np.arange(tangents.size) % 13) / 17.0)
                     + np.dot(norms, 2.0 + np.arange(n_mechanisms) / 29.0))
    return indices, tangents, climb_forces, norms, stable, baselines, panel_checksum, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    flat_setup = 'np=__import__("numpy")\ndef _tree(value):\n    if isinstance(value, np.ndarray):\n        return [float(value.ndim), *[float(x) for x in value.shape], *value.astype(float).ravel().tolist()]\n    if isinstance(value, (tuple, list)):\n        out = [float(len(value))]\n        for item in value:\n            out.extend(_tree(item))\n        return out\n    return [float(value)]'
    error_setup = 'def _value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0'
    return [
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(compute_climbing_controls())','gold_call':'_tree(_oracle_compute_climbing_controls())'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(compute_climbing_controls(260601,8,7))','gold_call':'_tree(_oracle_compute_climbing_controls(260601,8,7))'},
      {'tol':1e-9,'setup':flat_setup,'call':'_tree(compute_climbing_controls(260777,24,11))','gold_call':'_tree(_oracle_compute_climbing_controls(260777,24,11))'},
      {'tol':1e-9,'setup':error_setup+'\na=(260529,7,9)','call':'_value_error(compute_climbing_controls,*a)','gold_call':'_value_error(_oracle_compute_climbing_controls,*a)'},
    ]
