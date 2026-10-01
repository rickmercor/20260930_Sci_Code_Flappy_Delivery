"""
Extract the mass-scaled eigenvectors of the stiff normal modes from each Hessian block, under a fixed sign convention.

A frequency alone says how fast a mode oscillates but not what moves; the direction is a separate object and it is needed wherever the harmonic model has to be turned back into atomic positions. The eigendecomposition of a mass-scaled block supplies both at once, and this step is the half that returns the directions, retaining the columns belonging to the same rank-selected stiff eigenvalues so that the vectors and the frequencies index the same modes in the same order.

The directions are unit vectors in mass-scaled coordinates, not in Cartesian ones, which is the distinction most easily lost. A displacement generated along such a vector has to be divided by the square root of each atomic mass before it is added to a position, so a routine that treats the columns as Cartesian directions will move the hydrogens by a factor of four too little relative to the oxygen. Within a block the columns are orthonormal by construction, but they carry no relation to the columns of any other block, since the blocks were diagonalized independently and each lives in its own monomer's frame.

The mode directions matter because the reconstruction of a canonical all-atom ensemble displaces along them, so an error in the vectors misdirects every reconstructed fluctuation even when the frequencies are right. Their overall sign, on the other hand, is pure gauge: an eigenvector and its negative describe the same mode, and a decomposition routine may return either. Because a reproducible reconstruction requires a reproducible basis, the sign is fixed here by an explicit convention, making the entry of largest magnitude positive in every column.

The convention is a choice, not a derivation, so it has to be stated and obeyed exactly rather than inferred: the largest-magnitude entry of a column is made positive, and an exact tie between two entries of equal magnitude and opposite sign is broken in favour of the lower index. Any consistent rule would serve the physics equally well, but only the stated one reproduces the reference reconstruction entry by entry.

Returns
-------
np.ndarray of shape (n_monomers, n_dim, n_stiff), float: the sign-fixed orthonormal mass-scaled eigenvectors of the stiff modes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_stiff_mode_vectors(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    """Return the stiff mode vectors of each mass-scaled Hessian block.

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
    modes : np.ndarray
        Array of shape (n_monomers, n_dim, n_stiff) whose columns are the
        orthonormal mass-scaled eigenvectors of the retained stiff modes,
        ordered from stiffest to softest within each monomer. Each column is
        sign-fixed so that its entry of largest magnitude is positive, ties
        being resolved in favour of the lowest index.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, non-finite, non-square, or non-symmetric; if
        ``n_stiff`` is not an integer between one and the block dimension; or
        if any retained stiff eigenvalue is non-positive.
    """
    return modes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_stiff_mode_vectors(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
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

    n_monomers, n_dim = blocks.shape[0], blocks.shape[1]
    modes = np.zeros((n_monomers, n_dim, n_stiff), dtype=float)

    for index in range(n_monomers):
        block = blocks[index]
        if not np.allclose(block, block.T, rtol=0.0, atol=1.0e-8 * max(1.0, np.abs(block).max())):
            raise ValueError("every block must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        order = np.argsort(eigenvalues)[::-1][:n_stiff]
        if np.any(eigenvalues[order] <= 0.0):
            raise ValueError("a retained stiff mode has a non-positive eigenvalue")
        selected = eigenvectors[:, order]
        # Fix the arbitrary eigenvector sign so the modes are reproducible.
        dominant = np.argmax(np.abs(selected), axis=0)
        signs = np.sign(selected[dominant, np.arange(n_stiff)])
        signs[signs == 0.0] = 1.0
        modes[index] = selected * signs

    return modes

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
            "call": "compute_stiff_mode_vectors(blocks, 3)",
            "gold_call": "_oracle_compute_stiff_mode_vectors(blocks, 3)",
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
            "call": "compute_stiff_mode_vectors(blocks, 1)",
            "gold_call": "_oracle_compute_stiff_mode_vectors(blocks, 1)",
        },
        # --- Boundary: a diagonal block, where the modes are the coordinate
        #     axes and only the ordering can go wrong ---
        {
            "setup": """import numpy as np
blocks = np.diag(np.array([4.0, 900.0, 100.0, 0.25, 16.0]))[None, :, :]
""",
            "call": "compute_stiff_mode_vectors(blocks, 3)",
            "gold_call": "_oracle_compute_stiff_mode_vectors(blocks, 3)",
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
            "call": "compute_stiff_mode_vectors(blocks, 3)",
            "gold_call": "_oracle_compute_stiff_mode_vectors(blocks, 3)",
        },
        # --- Invalid: a retained mode with a non-positive eigenvalue ---
        {
            "setup": """import numpy as np
blocks = np.diag(np.array([5.0, 1.0, -2.0, -9.0]))[None, :, :]
def run_model():
    try:
        compute_stiff_mode_vectors(blocks, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_stiff_mode_vectors(blocks, 3)
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
        compute_stiff_mode_vectors(blocks, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_stiff_mode_vectors(blocks, 3)
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
