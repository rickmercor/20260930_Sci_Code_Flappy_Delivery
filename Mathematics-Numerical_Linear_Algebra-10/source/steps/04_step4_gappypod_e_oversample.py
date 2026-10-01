"""
Extend a Q-DEIM row set using GappyPOD+E oversampling.

Oversampling adds measurement locations beyond the basis rank to improve the robustness of a sampled reduced representation. GappyPOD+E targets this stability objective through information tied to the smallest spectral directions of the already sampled basis.

Returns
-------
np.ndarray, one-based integer row indices of length $s$, preserving the initial order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def gappypod_e_oversample(V: np.ndarray, initial_indices: np.ndarray, s: int, tie_tol: float = 1e-12) -> np.ndarray:
    r"""Extend a Q-DEIM row set using GappyPOD+E oversampling.

    Parameters
    ----------
    V : np.ndarray
        Finite full-column-rank basis matrix of shape $n\times m$. For a
        single-column basis ($m=1$) no eigengap exists; it is taken as
        zero, so every candidate scores zero and ties resolve to the
        smallest one-based index.
    initial_indices : np.ndarray
        One-dimensional integer array of exactly $m$ distinct one-based
        row indices whose sampled basis has full column rank.
    s : int
        Target sketch size satisfying $m\le s\le n$.
    tie_tol : float, optional
        Finite nonnegative tolerance used to resolve numerically tied
        oversampling scores. The default is 1e-12.

    Returns
    -------
    indices : np.ndarray
        One-dimensional integer array of length $s$ containing the
        one-based selected rows, preserving the initial index order.

    Raises
    ------
    ValueError
        If V is invalid or rank deficient, if initial_indices is not a
        valid full-rank $m$-row sample, if $s$ is outside $[m,n]$, if
        tie_tol is invalid, or if no admissible oversampling row exists.
    """
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_gappypod_e_oversample(V, initial_indices, s, tie_tol=1e-12):
    V = np.asarray(V, dtype=np.float64)
    idx = np.asarray(initial_indices)
    if V.ndim != 2 or min(V.shape) < 1 or not np.all(np.isfinite(V)):
        raise ValueError("V must be finite nonempty 2D")
    n, m = V.shape
    if n < m or np.linalg.matrix_rank(V) != m:
        raise ValueError("V must be full column rank")
    if idx.ndim != 1 or idx.size != m or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("initial_indices must contain m integers")
    idx = idx.astype(np.int64)
    if np.unique(idx).size != m or np.any(idx < 1) or np.any(idx > n):
        raise ValueError("invalid initial indices")
    if np.linalg.matrix_rank(V[idx - 1, :]) != m:
        raise ValueError("initial sampled basis must be full rank")
    if not isinstance(s, (int, np.integer)) or not (m <= int(s) <= n):
        raise ValueError("invalid s")
    if not isinstance(tie_tol, (int, float, np.integer, np.floating)) or not np.isfinite(tie_tol) or float(tie_tol) < 0:
        raise ValueError("invalid tie_tol")
    chosen = list((idx - 1).tolist())
    tol = float(tie_tol)
    while len(chosen) < int(s):
        SV = V[chosen, :]
        _, sing, Vh = np.linalg.svd(SV, full_matrices=False)
        g = float(sing[-2] ** 2 - sing[-1] ** 2) if m > 1 else 0.0
        W = Vh @ V.T
        y = np.sum(W * W, axis=0)
        disc = np.maximum((g + y) ** 2 - 4.0 * g * (W[-1, :] ** 2), 0.0)
        score = g + y - np.sqrt(disc)
        score[chosen] = -np.inf
        mx = float(np.max(score))
        cand = np.flatnonzero(np.abs(score - mx) <= tol)
        if cand.size == 0:
            raise ValueError("no admissible oversampling row")
        chosen.append(int(cand[0]))
    return np.asarray(chosen, dtype=np.int64) + 1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    def _base_test_cases():
        return [{'setup': 'import numpy as np\nV=np.array([[1.,0.],[0.,1.],[.8,.2],[.2,.9],[.6,.7]])\ninitial_indices=np.array([1,2]); s=4', 'call': 'gappypod_e_oversample(V,initial_indices,s)', 'gold_call': '_oracle_gappypod_e_oversample(V,initial_indices,s)'}, {'setup': 'import numpy as np\nV=np.array([[1.,0.],[0.,1.],[1.,1.]])\ninitial_indices=np.array([1,2]); s=2', 'call': 'gappypod_e_oversample(V,initial_indices,s)', 'gold_call': '_oracle_gappypod_e_oversample(V,initial_indices,s)'}, {'setup': 'import numpy as np\nV=np.array([[1.],[2.],[-1.],[.5]])\ninitial_indices=np.array([2]); s=3', 'call': 'gappypod_e_oversample(V,initial_indices,s)', 'gold_call': '_oracle_gappypod_e_oversample(V,initial_indices,s)'}, {'setup': 'import numpy as np\nV=np.eye(3); initial_indices=np.array([1,2]); s=3\ndef run_model():\n    try: gappypod_e_oversample(V,initial_indices,s); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_gappypod_e_oversample(V,initial_indices,s); return 0\n    except ValueError: return 1\n    except Exception: return 2', 'call': 'run_model()', 'gold_call': 'run_gold()'}, {'setup': 'import numpy as np\nV=np.array([[1.,0.],[0.,1.],[1.,1.]])\ninitial_indices=np.array([1,1]); s=3\ndef run_model():\n    try: gappypod_e_oversample(V,initial_indices,s); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_gappypod_e_oversample(V,initial_indices,s); return 0\n    except ValueError: return 1\n    except Exception: return 2', 'call': 'run_model()', 'gold_call': 'run_gold()'}, {'setup': 'import numpy as np\nV=np.array([[1.,0.],[0.,1.],[1.,1.]])\ninitial_indices=np.array([1,2]); s=4\ndef run_model():\n    try: gappypod_e_oversample(V,initial_indices,s); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_gappypod_e_oversample(V,initial_indices,s); return 0\n    except ValueError: return 1\n    except Exception: return 2', 'call': 'run_model()', 'gold_call': 'run_gold()'}]
    cases = _base_test_cases()
    for seed, n, m, s in [(18, 11, 4, 8), (42, 9, 3, 6), (71, 13, 5, 9)]:
        setup = f'import numpy as np\nrng=np.random.default_rng({seed})\nV=rng.normal(size=({n},{m}))@np.diag(np.geomspace(.1,3.,{m}))\ninitial_indices=np.arange(1,{m + 1}); s={s}'
        cases.append({'setup': setup, 'call': 'gappypod_e_oversample(V,initial_indices,s)', 'gold_call': '_oracle_gappypod_e_oversample(V,initial_indices,s)'})
    return cases
