"""
Determine the global discrepancy between two solution surfaces defined on the same computational mesh.



The comparison must be performed node by node over the complete domain, without discarding boundary nodes or restricting the calculation to a selected region. The function should return the single scalar quantity specified by the benchmark's error definition.

A numerical approximation can be assessed by comparing it directly with an analytical reference at corresponding mesh nodes. For the benchmark considered here, the relevant global measure is based on the largest pointwise discrepancy over the complete discrete domain.



This measure provides a deterministic summary of the numerical error and is independent of any particular visual representation of the solution surface.

Returns
-------
Returns the maximum absolute nodal discrepancy as a native Python floating-point scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_max_nodal_error(numerical: "np.ndarray", reference: "np.ndarray") -> float:
    """Return the maximum absolute nodal discrepancy.

    Parameters
    ----------
    numerical : numpy.ndarray
        Numerical surface.
    reference : numpy.ndarray
        Analytical reference surface.

    Returns
    -------
    float
        Maximum absolute difference between corresponding finite mesh values.

    Raises
    ------
    ValueError
        If the inputs are not finite non-empty two-dimensional arrays with equal shapes.
    
    Notes
    -----
    The inputs are treated as corresponding values on the same mesh.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_max_nodal_error(numerical: "np.ndarray", reference: "np.ndarray") -> float:
    import numpy as np
    numerical=np.asarray(numerical,dtype=np.float64); reference=np.asarray(reference,dtype=np.float64)
    if numerical.ndim!=2 or reference.ndim!=2: raise ValueError("both surfaces must be 2D")
    if numerical.shape!=reference.shape or numerical.size==0: raise ValueError("surfaces must have the same non-empty shape")
    if not (np.all(np.isfinite(numerical)) and np.all(np.isfinite(reference))): raise ValueError("surfaces must be finite")
    return float(np.max(np.abs(numerical-reference)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\na=np.array([[0.0,1.0],[2.0,3.0]]); b=np.array([[0.0,0.5],[2.5,2.0]])","call":"compute_max_nodal_error(a,b)","gold_call":"_oracle_compute_max_nodal_error(a,b)"},
        {"setup":"import numpy as np\na=np.zeros((1,1)); b=np.zeros((1,1))","call":"compute_max_nodal_error(a,b)","gold_call":"_oracle_compute_max_nodal_error(a,b)"},
        {"setup":"import numpy as np\na=np.array([[1.0,-2.0],[3.0,-4.0]]); b=np.zeros((2,2))","call":"compute_max_nodal_error(a,b)","gold_call":"_oracle_compute_max_nodal_error(a,b)"},
        {"setup":"import numpy as np\na=np.zeros((2,2)); b=np.zeros((2,3))\ndef run_model():\n    try: compute_max_nodal_error(a,b); return 0\n    except ValueError: return 1\ndef run_gold():\n    try: _oracle_compute_max_nodal_error(a,b); return 0\n    except ValueError: return 1","call":"run_model()","gold_call":"run_gold()"}
    ]
