"""
Return the gauge-independent projector that maximizes the retained stationary agent-memory weight at a fixed rank.

Agent-local compression must leave the reference generator untouched. The retained support therefore acts only on the agent memory, even though the routed stationary state lives on the larger joint bond. Use the variational retained-weight principle to identify the appropriate density operator and its dominant spectral subspace. Return the orthogonal projector, not basis vectors whose phases or internal rotations would be arbitrary. A degenerate eigenvalue cut does not specify a unique projector and is excluded.

Returns
-------
np.ndarray, complex rank-rank orthogonal projector of shape (d, d).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np

def select_memory_subspace(blocks: np.ndarray, rank: int) -> np.ndarray:
    """Select a unique maximal-weight memory subspace.

    Parameters
    ----------
    blocks : np.ndarray
        Finite complex array (C, d, d), C,d >= 1, of Hermitian positive
        semidefinite source-conditioned blocks with total trace one.
        Hermiticity, positivity and normalization use absolute tolerance 1e-10.
    rank : int
        Retained memory dimension in [1,d], excluding booleans. At an interior
        spectral cut the retained/discarded eigenvalue gap must exceed 1e-10.
        Full retention is defined even for degenerate spectra.

    Returns
    -------
    projector : np.ndarray
        Complex (d,d) orthogonal projector. Inputs are not mutated.

    Raises
    ------
    ValueError
        If a density-block condition fails, rank is invalid, or an interior
        eigenvalue cut has gap at most 1e-10.
    """
    return np.zeros((blocks.shape[-1], blocks.shape[-1]), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_memory_subspace(blocks: "np.ndarray", rank: int) -> "np.ndarray":
    import numpy as np
    from numbers import Integral

    try:
        blocks = np.asarray(blocks, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric blocks are required") from exc
    if blocks.ndim != 3 or min(blocks.shape) < 1 or blocks.shape[1] != blocks.shape[2] or not np.isfinite(blocks).all():
        raise ValueError("invalid block array")
    d = blocks.shape[-1]
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral) or not 1 <= rank <= d:
        raise ValueError("invalid retained dimension")
    if not np.allclose(blocks, blocks.conj().transpose(0,2,1), atol=1e-10, rtol=0):
        raise ValueError("blocks must be Hermitian")
    if np.min(np.linalg.eigvalsh(blocks)) < -1e-10 or abs(np.trace(blocks, axis1=1, axis2=2).sum()-1) > 1e-10:
        raise ValueError("blocks must be positive and jointly normalized")
    rho = blocks.sum(axis=0)
    if rank == d:
        return np.eye(d, dtype=complex)
    values, vectors = np.linalg.eigh((rho + rho.conj().T)/2)
    if values[d-rank] - values[d-rank-1] <= 1e-10:
        raise ValueError("spectral cut is not unique")
    retained = vectors[:, -int(rank):]
    return retained @ retained.conj().T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic case specifications."""
    return [
        {
            "setup": """import numpy as np
n=4
S = np.tril(np.ones((n,n))) / np.sqrt(np.arange(n,0,-1))[None,:]
blocks = np.stack([(S*w)@S.T for w in (np.array([.3,.15,.10,.05]),np.array([.1,.1,.1,.1]))])
""",
            "call": 'select_memory_subspace(blocks, 2)',
            "gold_call": '_oracle_select_memory_subspace(blocks, 2)',
        },
        {
            "setup": """import numpy as np
blocks=np.eye(3, dtype=complex)[None]/3
""",
            "call": 'select_memory_subspace(blocks, 3)',
            "gold_call": '_oracle_select_memory_subspace(blocks, 3)',
        },
        {
            "setup": """import numpy as np
blocks=np.diag([.5+1e-8,.5-1e-8])[None].astype(complex)
""",
            "call": 'select_memory_subspace(blocks, 1)',
            "gold_call": '_oracle_select_memory_subspace(blocks, 1)',
        },
        {
            "setup": """import numpy as np
v=np.array([1,1j,2],complex)/np.sqrt(6)
blocks=(.7*np.outer(v,v.conj())+.1*np.eye(3))[None]
""",
            "call": 'select_memory_subspace(blocks, 1)',
            "gold_call": '_oracle_select_memory_subspace(blocks, 1)',
        },
        {
            "setup": """import numpy as np
blocks=np.eye(3)[None]/3

def _run_model():
    try:
        select_memory_subspace(blocks, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_select_memory_subspace(blocks, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """import numpy as np
n=4
S = np.tril(np.ones((n,n))) / np.sqrt(np.arange(n,0,-1))[None,:]
blocks = np.stack([(S*w)@S.T for w in (np.array([.3,.15,.10,.05]),np.array([.1,.1,.1,.1]))])

def _run_model():
    try:
        select_memory_subspace(blocks, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_select_memory_subspace(blocks, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
        {
            "setup": """import numpy as np
n=4
S = np.tril(np.ones((n,n))) / np.sqrt(np.arange(n,0,-1))[None,:]
blocks = np.stack([(S*w)@S.T for w in (np.array([.3,.15,.10,.05]),np.array([.1,.1,.1,.1]))])
blocks=blocks.astype(complex); blocks[0,0,1] += .1j

def _run_model():
    try:
        select_memory_subspace(blocks, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _run_gold():
    try:
        _oracle_select_memory_subspace(blocks, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_run_model()',
            "gold_call": '_run_gold()',
        },
    ]
