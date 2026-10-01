"""
Build a deterministic orthonormal block-Krylov basis with two-pass rank deflation.

Finite projection depth makes the selected subspace part of the numerical model. Dependent probe directions must be deflated without replacing raw powers by recursively normalized blocks.

Returns
-------
np.ndarray of shape (n,r), with canonical orthonormal columns.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def canonical_block_krylov_basis(
    operator: np.ndarray,
    block: np.ndarray,
    depth: int,
    rank_tolerance: float = 1e-12,
) -> np.ndarray:
    """Construct a canonical two-pass block-Krylov basis.

    Parameters
    ----------
    operator : np.ndarray
        Finite real symmetric matrix of shape (n, n).
    block : np.ndarray
        Finite starting block of shape (n, m), with at least one nonzero direction.
    depth : int
        Positive number of blocks in [B, A B, ..., A**(depth-1) B].
    rank_tolerance : float
        Positive relative threshold multiplying max(1, ||B||_F).

    Returns
    -------
    basis : np.ndarray
        Canonical orthonormal basis of shape (n, r). Symmetry tolerance is 1e-12.
        Process the raw blocks [B,AB,...] in block order then column order.
        For each candidate perform two sequential modified Gram-Schmidt passes
        against previously retained columns, discarding it when its residual
        norm <= rank_tolerance*max(1,||B||_F). Normalize accepted columns and
        flip their sign if the first largest-magnitude component is negative.

    Raises
    ------
    ValueError
        If shapes, symmetry, finiteness, depth, tolerance, or numerical rank are invalid.
    """
    return np.empty((operator.shape[0], 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_canonical_block_krylov_basis(operator: np.ndarray, block: np.ndarray, depth: int, rank_tolerance: float=1e-12) -> np.ndarray:
    import numpy as np

    a = np.asarray(operator, dtype=float)
    b = np.asarray(block, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError("operator must be a nonempty square matrix")
    if b.ndim != 2 or b.shape[0] != a.shape[0] or b.shape[1] < 1:
        raise ValueError("block must have shape (n,m) with m >= 1")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("operator and block must be finite")
    if not np.allclose(a, a.T, rtol=0.0, atol=1e-12):
        raise ValueError("operator must be symmetric")
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or int(depth) < 1:
        raise ValueError("depth must be a positive integer")
    if not np.isfinite(rank_tolerance) or float(rank_tolerance) <= 0.0:
        raise ValueError("rank_tolerance must be positive and finite")
    depth = int(depth)
    threshold = float(rank_tolerance) * max(1.0, float(np.linalg.norm(b, ord="fro")))
    accepted = []
    current = b.copy()
    for level in range(depth):
        for column in range(current.shape[1]):
            vector = current[:, column].copy()
            for _pass in range(2):
                for q in accepted:
                    vector -= q * float(np.dot(q, vector))
            norm = float(np.linalg.norm(vector))
            if norm > threshold:
                vector /= norm
                pivot = int(np.argmax(np.abs(vector)))
                if vector[pivot] < 0.0:
                    vector = -vector
                accepted.append(vector)
        if level + 1 < depth:
            current = a @ current
    if not accepted:
        raise ValueError("the block Krylov matrix has zero numerical rank")
    return np.column_stack(accepted)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge, and invalid cases."""
    return [{'setup': 'import numpy as np\n'
               'A=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,2.]])\n'
               'B=np.array([[1.,0.],[0.,1.],[1.,0.]])',
      'call': 'canonical_block_krylov_basis(A,B,3)',
      'gold_call': '_oracle_canonical_block_krylov_basis(A,B,3)'},
     {'setup': 'import numpy as np\nA=np.diag([1.,2.,3.]); B=np.array([[1.,2.],[0.,0.],[0.,0.]])',
      'call': 'canonical_block_krylov_basis(A,B,4)',
      'gold_call': '_oracle_canonical_block_krylov_basis(A,B,4)'},
     {'setup': 'import numpy as np\nA=np.array([[2.,0.],[0.,3.]]); B=np.array([[-2.],[0.]])',
      'call': 'canonical_block_krylov_basis(A,B,1)',
      'gold_call': '_oracle_canonical_block_krylov_basis(A,B,1)'},
     {'setup': 'import numpy as np\n'
               'i=np.arange(6); A=1.0/(i[:,None]+i[None,:]+1.0); '
               'B=np.column_stack([np.ones(6),np.ones(6)+1e-4*((-1.0)**i)])',
      'call': 'canonical_block_krylov_basis(A,B,2)',
      'gold_call': '_oracle_canonical_block_krylov_basis(A,B,2)'},
     {'setup': 'import numpy as np\nA=np.eye(2);B=np.diag([1.,1e-12])',
      'call': 'canonical_block_krylov_basis(A,B,1)',
      'gold_call': '_oracle_canonical_block_krylov_basis(A,B,1)'},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1.,2.],[0.,1.]]); B=np.ones((2,1))\n'
               'def run_model():\n'
               '    try:\n'
               '        canonical_block_krylov_basis(A,B,2)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_canonical_block_krylov_basis(A,B,2)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'A=np.eye(2); B=np.zeros((2,1))\n'
               'def run_model():\n'
               '    try:\n'
               '        canonical_block_krylov_basis(A,B,1)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_canonical_block_krylov_basis(A,B,1)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
