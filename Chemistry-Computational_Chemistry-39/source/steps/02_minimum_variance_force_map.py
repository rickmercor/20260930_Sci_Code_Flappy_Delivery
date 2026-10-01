"""
Determine the compatible linear force map that minimizes the empirical second moment of the mapped atomistic forces with quadratic ridge regularization.

For a coordinate map M, a compatible linear force map T must satisfy T M^T = I. Among the compatible maps, the statistically optimal aggregation minimizes the empirical mapped-force second moment. The ridge term stabilizes the force covariance without replacing the compatibility constraint.

Returns
-------
Return the compatible force-map matrix with shape (beads, atoms).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def minimum_variance_force_map(
    mapping: np.ndarray,
    atomistic_forces: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Find the compatible force map with minimum empirical second moment.

    The returned matrix T minimizes the mean squared mapped force plus
    ridge times its squared Frobenius norm, subject to T @ mapping.T = I.

    Parameters
    ----------
    mapping
        Coordinate map with shape (beads, atoms).
    atomistic_forces
        Force samples with shape (frames, atoms, components).
    ridge
        Nonnegative quadratic regularization.

    Returns
    -------
    np.ndarray
        Force map with shape (beads, atoms).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_minimum_variance_force_map(mapping, atomistic_forces, ridge):
    import numpy as np

    coordinate_map = np.asarray(mapping, dtype=float)
    forces = np.asarray(atomistic_forces, dtype=float)
    if forces.ndim != 3 or min(forces.shape) == 0:
        raise ValueError("atomistic_forces must be a nonempty 3D array")
    if (
        coordinate_map.ndim != 2
        or coordinate_map.shape[0] == 0
        or coordinate_map.shape[1] != forces.shape[1]
    ):
        raise ValueError("mapping has incompatible shape")
    if np.any(~np.isfinite(forces)) or np.any(~np.isfinite(coordinate_map)):
        raise ValueError("forces and mapping must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if np.linalg.matrix_rank(coordinate_map) != coordinate_map.shape[0]:
        raise ValueError("mapping must have full row rank")

    observations = np.transpose(forces, (0, 2, 1)).reshape(-1, forces.shape[1])
    second_moment = observations.T @ observations / len(observations)
    regularized = second_moment + ridge * np.eye(forces.shape[1])
    try:
        map_times_inverse = np.linalg.solve(
            regularized, coordinate_map.T
        ).T
        force_map = np.linalg.solve(
            coordinate_map @ map_times_inverse.T,
            map_times_inverse,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("force-map system is singular") from exc
    return force_map

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(3); f=rng.normal(size=(7,4,3)); "
            "M=np.array([[.6,.4,0,0],[0,0,.3,.7]])",
            "call": "minimum_variance_force_map(M,f,.01)",
            "gold_call": "_oracle_minimum_variance_force_map(M,f,.01)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(9); f=rng.normal(size=(5,3,2)); "
            "M=np.eye(3)",
            "call": "minimum_variance_force_map(M,f,.2)",
            "gold_call": "_oracle_minimum_variance_force_map(M,f,.2)",
        },
        {
            "setup": "import numpy as np\nf=np.array([[[1.,2.],[-1.,.5]],[[.2,-.4],[.7,1.1]]]); "
            "M=np.array([[.25,.75]])",
            "call": "minimum_variance_force_map(M,f,.05)",
            "gold_call": "_oracle_minimum_variance_force_map(M,f,.05)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(27); "
            "f=rng.normal(size=(8,5,2)); "
            "M=np.array([[.55,.25,.20,0.,0.],[0.,.10,.25,.30,.35]])",
            "call": "minimum_variance_force_map(M,f,0.0)",
            "gold_call": "_oracle_minimum_variance_force_map(M,f,0.0)",
        },
    ]
