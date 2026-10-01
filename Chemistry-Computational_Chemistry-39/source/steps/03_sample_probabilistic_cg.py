"""
Apply the Gaussian conditional coarse-graining map to deterministic mapped coordinates using the supplied standard-normal draws.

The probabilistic coarse-graining map replaces the deterministic point assignment Mr with samples R = Mr + sigma xi. The standard-normal tensor retains separate replicate, frame, bead, and Cartesian-component axes. Sigma is the positive Gaussian standard deviation.

Returns
-------
Return the noised coordinates with shape (replicates, frames, beads, components).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sample_probabilistic_cg(
    mapped_positions: np.ndarray,
    standard_normals: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Sample the Gaussian probabilistic coarse-graining map.

    Parameters
    ----------
    mapped_positions
        Deterministic positions M r with shape (frames, beads, components).
    standard_normals
        Fixed standard-normal draws with shape
        (replicates, frames, beads, components).
    sigma
        Positive Gaussian standard deviation.

    Returns
    -------
    np.ndarray
        Noised positions M r + sigma * normal.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sample_probabilistic_cg(mapped_positions, standard_normals, sigma):
    import numpy as np

    mapped = np.asarray(mapped_positions, dtype=float)
    normals = np.asarray(standard_normals, dtype=float)
    if mapped.ndim != 3 or min(mapped.shape) == 0:
        raise ValueError("mapped_positions must be a nonempty 3D array")
    if normals.ndim != 4 or normals.shape[1:] != mapped.shape:
        raise ValueError("standard_normals has incompatible shape")
    if np.any(~np.isfinite(mapped)) or np.any(~np.isfinite(normals)):
        raise ValueError("positions and normals must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    return mapped[None, ...] + sigma * normals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nR=np.arange(18,dtype=float).reshape(2,3,3)/5; "
            "z=np.linspace(-1,1,72).reshape(4,2,3,3)",
            "call": "sample_probabilistic_cg(R,z,.2)",
            "gold_call": "_oracle_sample_probabilistic_cg(R,z,.2)",
        },
        {
            "setup": "import numpy as np\nR=np.zeros((1,1,2)); z=np.array([[[[1.,-1.]]]])",
            "call": "sample_probabilistic_cg(R,z,1e-6)",
            "gold_call": "_oracle_sample_probabilistic_cg(R,z,1e-6)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(4); R=rng.normal(size=(5,2,1)); "
            "z=rng.normal(size=(3,5,2,1))",
            "call": "sample_probabilistic_cg(R,z,.35)",
            "gold_call": "_oracle_sample_probabilistic_cg(R,z,.35)",
        },
        {
            "setup": "import numpy as np\nR=np.array([[[1.,-2.],[.5,3.]],"
            "[[-1.,.25],[2.,-4.]]]); z=np.zeros((3,2,2,2))",
            "call": "sample_probabilistic_cg(R,z,.7)",
            "gold_call": "_oracle_sample_probabilistic_cg(R,z,.7)",
        },
    ]
