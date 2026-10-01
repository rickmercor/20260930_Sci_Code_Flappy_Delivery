"""
Construct the atom-first joint-force tensor for the extended atomistic and noised coarse-grained system.

For Gaussian conditional mapping, the displacement R-Mr contributes opposite score terms to the atomistic and noise variables. The atomistic block receives the mapping-transposed displacement divided by sigma squared, while the noise block receives its negative directly. The two blocks are concatenated along the site axis with the atomistic sites first.

Returns
-------
Return the joint-force tensor with shape (replicates, frames, atoms + beads, components).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_extended_forces(
    atomistic_forces: np.ndarray,
    mapping: np.ndarray,
    mapped_positions: np.ndarray,
    noised_positions: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Build atomistic and noise forces on the extended system.

    The atomistic block is followed by the noise-bead block along axis 2.

    Parameters
    ----------
    atomistic_forces
        Array with shape (frames, atoms, components).
    mapping
        Coordinate map with shape (beads, atoms).
    mapped_positions
        Deterministic positions with shape (frames, beads, components).
    noised_positions
        Noised positions with shape (replicates, frames, beads, components).
    sigma
        Positive Gaussian standard deviation.

    Returns
    -------
    np.ndarray
        Joint forces with shape
        (replicates, frames, atoms + beads, components).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_extended_forces(
    atomistic_forces,
    mapping,
    mapped_positions,
    noised_positions,
    sigma,
):
    import numpy as np

    forces = np.asarray(atomistic_forces, dtype=float)
    coordinate_map = np.asarray(mapping, dtype=float)
    mapped = np.asarray(mapped_positions, dtype=float)
    noised = np.asarray(noised_positions, dtype=float)
    if forces.ndim != 3 or coordinate_map.ndim != 2:
        raise ValueError("forces and mapping have invalid rank")
    if mapped.ndim != 3 or noised.ndim != 4:
        raise ValueError("mapped and noised positions have invalid rank")
    if coordinate_map.shape != (mapped.shape[1], forces.shape[1]):
        raise ValueError("mapping has incompatible shape")
    if (
        forces.shape[0] != mapped.shape[0]
        or forces.shape[2] != mapped.shape[2]
        or noised.shape[1:] != mapped.shape
    ):
        raise ValueError("trajectory arrays have incompatible shapes")
    if any(
        np.any(~np.isfinite(value))
        for value in (forces, coordinate_map, mapped, noised)
    ):
        raise ValueError("all arrays must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    displacement = noised - mapped[None, ...]
    inverse_variance = 1.0 / (sigma * sigma)
    atom_block = forces[None, ...] + np.einsum(
        "qa,sfqc->sfac", coordinate_map, displacement
    ) * inverse_variance
    noise_block = -displacement * inverse_variance
    return np.concatenate([atom_block, noise_block], axis=2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nf=np.arange(24,dtype=float).reshape(2,4,3)/7; "
            "M=np.array([[.6,.4,0,0],[0,0,.3,.7]]); "
            "R0=np.einsum('qa,fac->fqc',M,np.arange(24,dtype=float).reshape(2,4,3)/9); "
            "z=np.linspace(-1,1,36).reshape(3,2,2,3); R=R0[None]+.2*z",
            "call": "build_extended_forces(f,M,R0,R,.2)",
            "gold_call": "_oracle_build_extended_forces(f,M,R0,R,.2)",
        },
        {
            "setup": "import numpy as np\nf=np.array([[[1.,-2.],[.5,.3]]]); "
            "M=np.array([[.4,.6]]); R0=np.zeros((1,1,2)); "
            "R=np.zeros((2,1,1,2))",
            "call": "build_extended_forces(f,M,R0,R,.1)",
            "gold_call": "_oracle_build_extended_forces(f,M,R0,R,.1)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(6); f=rng.normal(size=(4,3,1)); "
            "M=np.eye(3); R0=rng.normal(size=(4,3,1)); "
            "R=R0[None]+.05*rng.normal(size=(5,4,3,1))",
            "call": "build_extended_forces(f,M,R0,R,.05)",
            "gold_call": "_oracle_build_extended_forces(f,M,R0,R,.05)",
        },
        {
            "setup": "import numpy as np\nf=np.array([[[1.],[-2.],[.5]]]); "
            "M=np.array([[.6,.4,0.],[0.,.25,.75]]); "
            "R0=np.array([[[.2],[-.3]]]); "
            "R=np.array([[[[.3],[-.5]]],[[[.05],[.1]]]])",
            "call": "build_extended_forces(f,M,R0,R,.25)",
            "gold_call": "_oracle_build_extended_forces(f,M,R0,R,.25)",
        },
    ]
