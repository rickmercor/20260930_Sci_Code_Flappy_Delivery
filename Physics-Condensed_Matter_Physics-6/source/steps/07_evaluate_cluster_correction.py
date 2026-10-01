"""
Evaluate the numerical connected-cluster contributions and their total correction.

Connected-cluster amplitudes shift the logarithm of the partition function
away from its BP baseline.  Their signed contributions reflect the effects of
repeated and overlapping loop excitations at the chosen cutoff.

Loop weights are indexed by the loop columns of the supplied multiplicity
matrix.  Cluster rows correspond one-to-one with the supplied coefficient
array.  The returned contribution array preserves cluster-row order and its sum
is returned separately for downstream use.

Parameters
----------
loop_weights : np.ndarray
    Finite float array of shape ``(L,)``.
multiplicities : np.ndarray
    Nonnegative integer array of shape ``(C,L)``.
ursell_coefficients : np.ndarray
    Finite float array of shape ``(C,)`` in the same cluster-row order.

Returns
-------
cluster_contributions : np.ndarray
    Float array of shape ``(C,)`` preserving cluster-row order.
delta : float
    Sum of all returned connected-cluster contributions.

Raises
------
ValueError
    If dimensions are inconsistent, a required value is nonfinite, or a
    multiplicity is negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_cluster_correction(
    loop_weights: "np.ndarray", multiplicities: "np.ndarray",
    ursell_coefficients: "np.ndarray",
) -> "tuple[np.ndarray, float]":
    '''Return per-cluster numerical terms and their total correction.

    Parameters
    ----------
    loop_weights : np.ndarray
        Finite float array of shape ``(L,)``.
    multiplicities : np.ndarray
        Nonnegative integer array of shape ``(C,L)``.
    ursell_coefficients : np.ndarray
        Finite float array of shape ``(C,)`` in the same cluster-row order.

    Returns
    -------
    cluster_contributions : np.ndarray
        Float array of shape ``(C,)`` preserving cluster-row order.
    delta : float
        Sum of all returned connected-cluster contributions.

    Raises
    ------
    ValueError
        If dimensions are inconsistent, a required value is nonfinite, or a
        multiplicity is negative.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_cluster_correction(
    loop_weights: "np.ndarray", multiplicities: "np.ndarray",
    ursell_coefficients: "np.ndarray",
) -> "tuple[np.ndarray, float]":
    loop_weights = np.asarray(loop_weights, dtype=float)
    multiplicities = np.asarray(multiplicities)
    ursell_coefficients = np.asarray(ursell_coefficients, dtype=float)
    if loop_weights.ndim != 1 or multiplicities.ndim != 2 or multiplicities.shape[1] != loop_weights.size:
        raise ValueError("loop weights and multiplicities have inconsistent shapes")
    if ursell_coefficients.shape != (multiplicities.shape[0],):
        raise ValueError("coefficient length must match cluster rows")
    if not np.issubdtype(multiplicities.dtype, np.integer) or np.any(multiplicities < 0):
        raise ValueError("multiplicities must be nonnegative integers")
    if not (np.all(np.isfinite(loop_weights)) and np.all(np.isfinite(ursell_coefficients))):
        raise ValueError("weights and coefficients must be finite")
    if multiplicities.shape[0] == 0:
        return np.empty(0, dtype=float), 0.0
    contributions = ursell_coefficients * np.prod(
        np.power(loop_weights[None, :], multiplicities), axis=1
    )
    return contributions.astype(float), float(np.sum(contributions))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return singleton, distinct-pair, repeated-copy, and empty-cluster cases."""
    helper = """import numpy as np

def run_midchain(build_fn, solve_fn, vacuum_fn, loop_fn, weight_fn, cluster_fn, eval_fn, Kh, Kv, h, cutoff):
    e,n,d,t,b = build_fn(Kh.copy(),Kv.copy(),h.copy())
    m = solve_fn(e.copy(),n.copy(),d.copy(),t.copy(),1e-13)
    sf,_,p0,pp,_ = vacuum_fn(e.copy(),n.copy(),d.copy(),t.copy(),m.copy())
    ls = loop_fn(e.copy(),cutoff)
    lw = weight_fn(e.copy(),h.copy(),b.copy(),p0.copy(),pp.copy(),sf.copy(),ls.copy())
    mult,coef,_ = cluster_fn(e.copy(),ls.copy(),cutoff)
    vals,delta = eval_fn(lw.copy(),mult.copy(),coef.copy())
    return (vals,float(delta))
"""
    return [
        {
            "setup": helper + "\nKh=np.array([[0.31,0.47],[0.38,0.29]])\nKv=np.array([[0.42,0.35,0.51]])\nh=np.array([[0.03,-0.06,0.08],[-0.04,0.05,-0.02]])",
            "call": "run_midchain(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,enumerate_connected_generalized_loops,compute_normalized_loop_weights,enumerate_connected_loop_clusters,evaluate_cluster_correction,Kh,Kv,h,6)",
            "gold_call": "run_midchain(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,_oracle_enumerate_connected_generalized_loops,_oracle_compute_normalized_loop_weights,_oracle_enumerate_connected_loop_clusters,_oracle_evaluate_cluster_correction,Kh,Kv,h,6)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.31,0.47],[0.38,0.29]])\nKv=np.array([[0.42,0.35,0.51]])\nh=np.array([[0.03,-0.06,0.08],[-0.04,0.05,-0.02]])",
            "call": "run_midchain(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,enumerate_connected_generalized_loops,compute_normalized_loop_weights,enumerate_connected_loop_clusters,evaluate_cluster_correction,Kh,Kv,h,8)",
            "gold_call": "run_midchain(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,_oracle_enumerate_connected_generalized_loops,_oracle_compute_normalized_loop_weights,_oracle_enumerate_connected_loop_clusters,_oracle_evaluate_cluster_correction,Kh,Kv,h,8)",
        },
        {
            "setup": "import numpy as np\nlw=np.array([0.2])\nmult=np.array([[1],[2],[3]],dtype=int)\ncoef=np.array([1.0,-0.5,1.0/3.0])",
            "call": "evaluate_cluster_correction(lw.copy(),mult.copy(),coef.copy())",
            "gold_call": "_oracle_evaluate_cluster_correction(lw.copy(),mult.copy(),coef.copy())",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\nlw=np.array([0.14,-0.06])\nmult=np.empty((0,2),dtype=int)\ncoef=np.empty(0)",
            "call": "evaluate_cluster_correction(lw.copy(),mult.copy(),coef.copy())",
            "gold_call": "_oracle_evaluate_cluster_correction(lw.copy(),mult.copy(),coef.copy())",
            "tol": 1e-13,
        },
    ]
