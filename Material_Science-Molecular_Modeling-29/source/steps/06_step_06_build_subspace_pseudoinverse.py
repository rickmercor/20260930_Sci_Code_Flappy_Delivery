"""
Invert each mass-scaled Hessian block on the subspace spanned by its stiffest eigenvectors only, producing the operator that projects a gradient onto the fast degrees of freedom.

A group of atoms treated as a rigid unit has more Cartesian coordinates than it has stiff internal degrees of freedom: a three-atom monomer has nine coordinates but only three internal vibrations, the remaining six being the rigid-body translations and rotations that the coarse-grained description owns. The relaxation must drive the monomer to equilibrium along the three stiff directions and leave the other six untouched, since moving along them would change the coarse-grained configuration the calculation is supposed to hold fixed. The operator that enforces that split is a pseudoinverse of the mass-scaled block, restricted to the stiff subspace.

The restriction is defined by rank rather than by threshold: the block is diagonalized, its eigenvalues are ranked, and the pseudoinverse is assembled as the sum over the prescribed number of *largest* eigenvalues of the reciprocal eigenvalue times the outer product of its eigenvector with itself. Every other eigendirection is assigned exactly zero, so a gradient component along a rigid-body direction produces no displacement.

Selecting the largest eigenvalues, rather than discarding those below some tolerance, is not a cosmetic difference here. Because the block is a diagonal block of the full cluster Hessian, it carries the curvature of the intermolecular interactions as well as the intramolecular ones, and it is therefore not positive semi-definite: the would-be rigid-body eigenvalues are neither zero nor necessarily positive, and at a configuration that is not an intermolecular minimum some of them are small and negative. A conventional Moore-Penrose construction that keeps everything above a numerical tolerance would then invert those small eigenvalues, producing enormous reciprocals along soft or unstable directions and throwing the subsequent Newton step far outside the physical range. Ranking by eigenvalue and keeping a fixed number of the stiffest is what makes the operator well defined regardless of where in configuration space it is evaluated.

The number kept is a property of the partitioning, not of the numerics. For a free monomer it is three times the atom count minus six; a group covalently connected to another group carries additional constraints and keeps more.

Returns
-------
np.ndarray of shape (n_monomers, n_dim, n_dim), float: the stiff-subspace pseudoinverse of each mass-scaled Hessian block.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_subspace_pseudoinverse(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
    """Invert each Hessian block on the span of its stiffest eigenvectors.

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
    pseudoinverse : np.ndarray
        Array with the same shape as ``blocks`` holding the subspace
        pseudoinverse of each block, in mol angstrom**2 u/kcal.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, non-finite, non-square, or non-symmetric; if
        ``n_stiff`` is not an integer between one and the block dimension; or
        if any retained stiff eigenvalue is non-positive.
    """
    return pseudoinverse  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_subspace_pseudoinverse(blocks: np.ndarray, n_stiff: int = 3) -> np.ndarray:
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

    pseudoinverse = np.zeros_like(blocks)
    for index in range(blocks.shape[0]):
        block = blocks[index]
        if not np.allclose(block, block.T, rtol=0.0, atol=1.0e-8 * max(1.0, np.abs(block).max())):
            raise ValueError("every block must be symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        # Rank by eigenvalue and keep the stiffest directions; a tolerance-based
        # selection is not equivalent, because the block is not positive
        # semi-definite.
        order = np.argsort(eigenvalues)[::-1][:n_stiff]
        if np.any(eigenvalues[order] <= 0.0):
            raise ValueError("the retained stiff subspace contains a non-positive eigenvalue")
        selected_values = eigenvalues[order]
        selected_vectors = eigenvectors[:, order]
        pseudoinverse[index] = (selected_vectors / selected_values) @ selected_vectors.T

    return pseudoinverse

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: two blocks with a wide stiff/soft separation (normal case) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
blocks = np.zeros((2, 9, 9))
for i in range(2):
    basis = np.linalg.qr(rng.normal(size=(9, 9)))[0]
    spectrum = np.array([1300.0, 1240.0, 214.0, 23.0, 4.0, 2.8, 1.8, 0.9, -0.7]) + 3.0 * i
    blocks[i] = basis @ np.diag(spectrum) @ basis.T
    blocks[i] = 0.5 * (blocks[i] + blocks[i].T)
""",
            "call": "build_subspace_pseudoinverse(blocks, 3)",
            "gold_call": "_oracle_build_subspace_pseudoinverse(blocks, 3)",
        },
        # --- Valid: a spectrum containing small negative eigenvalues, which a
        #     tolerance-based pseudoinverse would invert and this one must not ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
basis = np.linalg.qr(rng.normal(size=(9, 9)))[0]
spectrum = np.array([980.0, 940.0, 190.0, 12.0, -1.0e-9, -0.4, -2.0, -5.0, -30.0])
blocks = (basis @ np.diag(spectrum) @ basis.T)[None, :, :]
blocks = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
""",
            "call": "build_subspace_pseudoinverse(blocks, 3)",
            "gold_call": "_oracle_build_subspace_pseudoinverse(blocks, 3)",
        },
        # --- Boundary: keeping every direction, so the result is a true inverse ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3)
basis = np.linalg.qr(rng.normal(size=(4, 4)))[0]
blocks = (basis @ np.diag(np.array([9.0, 5.0, 2.0, 0.5])) @ basis.T)[None, :, :]
blocks = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
""",
            "call": "build_subspace_pseudoinverse(blocks, 4)",
            "gold_call": "_oracle_build_subspace_pseudoinverse(blocks, 4)",
        },
        # --- Edge: nearly degenerate stiff eigenvalues straddling the cut ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(19)
basis = np.linalg.qr(rng.normal(size=(6, 6)))[0]
spectrum = np.array([500.0, 500.0 - 1.0e-7, 499.9, 1.0, 0.2, -0.05])
blocks = (basis @ np.diag(spectrum) @ basis.T)[None, :, :]
blocks = 0.5 * (blocks + np.transpose(blocks, (0, 2, 1)))
""",
            "call": "build_subspace_pseudoinverse(blocks, 3)",
            "gold_call": "_oracle_build_subspace_pseudoinverse(blocks, 3)",
        },
        # --- Invalid: more stiff directions requested than the block has ---
        {
            "setup": """import numpy as np
blocks = np.eye(4)[None, :, :] * 3.0
def run_model():
    try:
        build_subspace_pseudoinverse(blocks, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_subspace_pseudoinverse(blocks, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the retained subspace is not positive definite ---
        {
            "setup": """import numpy as np
blocks = (np.diag(np.array([-1.0, -2.0, -3.0, -4.0])))[None, :, :]
def run_model():
    try:
        build_subspace_pseudoinverse(blocks, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_subspace_pseudoinverse(blocks, 2)
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
