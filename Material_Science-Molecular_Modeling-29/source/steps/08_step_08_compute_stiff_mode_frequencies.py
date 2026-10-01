"""
Extract the stiff normal-mode angular frequencies from each mass-scaled Hessian block.

The free energy charged for the eliminated degrees of freedom is the harmonic free energy of the stiff normal modes, so those modes must be characterized at whatever configuration the free energy is being evaluated at. Because the second-derivative matrix has already been scaled by the inverse square roots of the atomic masses, its eigenvalues are squared angular frequencies directly, and the angular frequency of a mode is the square root of its eigenvalue. No further mass factor enters.

The selection rule is the same rank-based one used to build the subspace pseudoinverse, and it must be the same: the modes whose free energy is charged have to be the modes that were driven to equilibrium, otherwise the relaxation and the harmonic correction refer to different subspaces and the resulting model is not thermodynamically consistent. The eigenvalues are ranked and the prescribed number of largest ones is retained; the remaining directions, which approximate the rigid-body motions of the monomer, are left to the coarse-grained description.

This step is performed twice in the calculation, and the difference between the two invocations is the whole point. Evaluated at the starting configuration it gives the frequencies of the conventional frozen treatment; evaluated after the relaxation it gives the frequencies of the relaxed treatment, and the block Hessian genuinely has to be rebuilt at the relaxed point rather than reused, since it is the shift in curvature caused by the relaxation that carries the entropic part of the free-energy difference. In a hydrogen-bonded cluster that shift is not small and it is not uniform: the O-H stretch that donates a hydrogen bond softens substantially, while the bend stiffens slightly, so the frequencies move in opposite directions and no scalar rescaling could stand in for the recomputation.

Only the frequencies are returned here. The mode directions, which the all-atom reconstruction also needs, are obtained separately from the same eigendecomposition; keeping the two outputs in separate steps means each can be checked against a reference on its own terms rather than as a bundle.

Returns
-------
np.ndarray of shape (n_monomers, n_stiff), float: the stiff angular frequencies of each block, ordered from stiffest to softest.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_stiff_mode_frequencies(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    """Return the stiff angular frequencies of each mass-scaled Hessian block.

    Parameters
    ----------
    blocks : np.ndarray
        Array of shape (n_monomers, n_dim, n_dim) holding one symmetric
        mass-scaled second-derivative block per monomer.
    n_stiff : int
        Number of stiff degrees of freedom retained per block, at least 1 and
        at most n_dim.

    Returns
    -------
    frequencies : np.ndarray
        Array of shape (n_monomers, n_stiff) holding the angular frequency of
        each retained mode in (kcal/(mol angstrom**2 u))**(1/2), ordered from
        stiffest to softest within each monomer.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, non-finite, non-square, or non-symmetric; if
        ``n_stiff`` is not an integer between one and the block dimension; or
        if any retained stiff eigenvalue is non-positive.
    """
    return frequencies  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_stiff_mode_frequencies(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2] or blocks.shape[1] < 1:
        raise ValueError("blocks must have shape (n_monomers, n_dim, n_dim) with n_dim >= 1")
    if blocks.shape[0] < 1:
        raise ValueError("blocks must contain at least one block")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("blocks must contain only finite entries")
    if not (isinstance(n_stiff, (int, np.integer)) and not isinstance(n_stiff, bool)):
        raise ValueError("n_stiff must be an integer")
    n_stiff = int(n_stiff)
    if not 1 <= n_stiff <= blocks.shape[1]:
        raise ValueError("n_stiff must lie between 1 and the block dimension")

    n_monomers = blocks.shape[0]
    frequencies = np.zeros((n_monomers, n_stiff), dtype=float)

    for index in range(n_monomers):
        block = blocks[index]
        if not np.allclose(block, block.T, rtol=0.0, atol=1.0e-8 * max(1.0, np.abs(block).max())):
            raise ValueError("every block must be symmetric")
        eigenvalues = np.linalg.eigvalsh(block)
        order = np.argsort(eigenvalues)[::-1][:n_stiff]
        if np.any(eigenvalues[order] <= 0.0):
            raise ValueError("a retained stiff mode has a non-positive eigenvalue")
        frequencies[index] = np.sqrt(eigenvalues[order])

    return frequencies

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: blocks with a clean stiff/soft separation (normal case) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(23)
blocks = np.zeros((3, 9, 9))
for i in range(3):
    basis = np.linalg.qr(rng.normal(size=(9, 9)))[0]
    spectrum = np.array([1296.0, 1239.0, 214.0, 23.5, 3.9, 2.8, 1.9, 0.85, -0.68]) + 2.0 * i
    blocks[i] = 0.5 * ((basis @ np.diag(spectrum) @ basis.T)
                       + (basis @ np.diag(spectrum) @ basis.T).T)
""",
            "call": "compute_stiff_mode_frequencies(blocks, 3)",
            "gold_call": "_oracle_compute_stiff_mode_frequencies(blocks, 3)",
        },
        # --- Valid: a single block with only one stiff direction requested ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(31)
basis = np.linalg.qr(rng.normal(size=(6, 6)))[0]
spectrum = np.array([840.0, 120.0, 9.0, 1.2, 0.3, -0.15])
blocks = (basis @ np.diag(spectrum) @ basis.T)[None, :, :]
blocks = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
""",
            "call": "compute_stiff_mode_frequencies(blocks, 1)",
            "gold_call": "_oracle_compute_stiff_mode_frequencies(blocks, 1)",
        },
        # --- Boundary: a diagonal block, where the modes are the coordinate
        #     axes and only the ordering can go wrong ---
        {
            "setup": """import numpy as np
blocks = np.diag(np.array([4.0, 900.0, 100.0, 0.25, 16.0]))[None, :, :]
""",
            "call": "compute_stiff_mode_frequencies(blocks, 3)",
            "gold_call": "_oracle_compute_stiff_mode_frequencies(blocks, 3)",
        },
        # --- Edge: a block whose discarded directions are strongly negative,
        #     which a magnitude-based ranking would wrongly select ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(41)
basis = np.linalg.qr(rng.normal(size=(7, 7)))[0]
spectrum = np.array([300.0, 280.0, 60.0, 2.0, -55.0, -190.0, -410.0])
blocks = (basis @ np.diag(spectrum) @ basis.T)[None, :, :]
blocks = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
""",
            "call": "compute_stiff_mode_frequencies(blocks, 3)",
            "gold_call": "_oracle_compute_stiff_mode_frequencies(blocks, 3)",
        },
        # --- Invalid: a retained mode with a non-positive eigenvalue ---
        {
            "setup": """import numpy as np
blocks = np.diag(np.array([5.0, 1.0, -2.0, -9.0]))[None, :, :]
def run_model():
    try:
        compute_stiff_mode_frequencies(blocks, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_stiff_mode_frequencies(blocks, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-square block array ---
        {
            "setup": """import numpy as np
blocks = np.zeros((2, 9, 8))
def run_model():
    try:
        compute_stiff_mode_frequencies(blocks, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_stiff_mode_frequencies(blocks, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
