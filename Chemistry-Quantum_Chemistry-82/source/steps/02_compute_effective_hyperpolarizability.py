"""
Apply the source's unit-sphere representation. For each incident unit direction u, contract the final two tensor axes with u twice so the kth output component is the sum of beta[k,i,j] u[i] u[j]. Preserve the input Cartesian frame.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
np.ndarray: Effective response vectors.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_effective_hyperpolarizability(beta: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Map a rank-three hyperpolarizability tensor onto unit directions.

    Parameters
    ----------
    beta : np.ndarray
        Finite tensor with shape (3,3,3).
    directions : np.ndarray
        At least four Cartesian unit directions with shape (n,3).
    Returns
    -------
    np.ndarray
        Effective response vectors with shape (n,3).
    Raises
    ------
    ValueError
        If a shape, finiteness, count, or unit-norm contract fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _oracle_compute_effective_hyperpolarizability(
    beta: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if beta.shape != (3, 3, 3) or not np.all(np.isfinite(beta)):
        raise ValueError("beta must be a finite (3,3,3) tensor")
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] < 4:
        raise ValueError("directions must have shape (n,3) with n at least 4")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions must be finite")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("every direction must be a unit vector")
    return np.einsum("kij,ni,nj->nk", beta, directions, directions)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common="""import numpy as np
b=np.arange(27,dtype=float).reshape(3,3,3)/17
u=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[-1,1,1]],dtype=float);u/=np.linalg.norm(u,axis=1,keepdims=True)"""
    return [
        {"setup":common,"call":"compute_effective_hyperpolarizability(b,u)","gold_call":"_oracle_compute_effective_hyperpolarizability(b,u)","tol":1e-12},
        {"setup":common+chr(10)+"b=np.zeros((3,3,3));b[2,1,1]=1e-12","call":"compute_effective_hyperpolarizability(b,u)","gold_call":"_oracle_compute_effective_hyperpolarizability(b,u)","tol":1e-12},
        {"setup":common+chr(10)+"u=np.vstack((u,-u))","call":"compute_effective_hyperpolarizability(b,u)","gold_call":"_oracle_compute_effective_hyperpolarizability(b,u)","tol":1e-12},
        {"setup":"""import numpy as np
b=np.zeros((3,3,3));u=np.ones((4,3))
def f(g):
 try:g(b,u)
 except ValueError:return 1.0
 return 0.0""","call":"f(compute_effective_hyperpolarizability)","gold_call":"f(_oracle_compute_effective_hyperpolarizability)","tol":1e-12}
    ]
