"""
Compute the population of the +1 eigenstate of axis·sigma from the final basis-correlation matrix and the initial Bloch vector.

For measurement direction a and initial Bloch vector r, the population is (1,a)^T C (1,r)/2. The vectors use basis order (I,x,y,z). Include the identity column and the initial coherences. Return the computed value without clipping or renormalization.

Returns
-------
A native Python float giving the population of the +1 eigenstate of axis·σ.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def measure_population(correlation, initial, axis):
    """Recover a qubit measurement probability.

    Parameters
    ----------
    correlation : array_like, real, shape (4, 4)
        Bath-weighted Heisenberg basis-correlation matrix in the
        ordered normalized basis (I, X, Y, Z)/sqrt(2).
    initial : array_like, real, shape (3,)
        Initial Bloch vector, with norm <= 1.
    axis : array_like, real, shape (3,)
        Unit Bloch vector defining the measured +1 eigenstate.

    Returns
    -------
    float
        Reconstructed population. Do not clip or renormalize it.

    Raises
    ------
    ValueError
        If shapes are incorrect, an entry is nonfinite, the initial
        norm exceeds 1+1e-12, or the axis norm differs from one
        by more than 1e-12.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_measure_population(correlation, initial, axis):
    import numpy as np

    c = np.asarray(correlation, dtype=float)
    r = np.asarray(initial, dtype=float)
    a = np.asarray(axis, dtype=float)

    if (
        c.shape != (4, 4)
        or r.shape != (3,)
        or a.shape != (3,)
        or not all(
            np.isfinite(value).all()
            for value in (c, r, a)
        )
        or np.linalg.norm(r) > 1 + 1e-12
        or abs(np.linalg.norm(a) - 1) > 1e-12
    ):
        raise ValueError(
            "Invalid correlation matrix or Bloch vectors."
        )

    return float(
        np.r_[1.0, a] @ c @ np.r_[1.0, r] / 2
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
c = np.eye(4)
c[3] = [.1, .2, -.3, .8]
""",
            "call": "measure_population(c, [.3,-.4,.5], [0.,0.,1.])",
            "gold_call": "_oracle_measure_population(c, [.3,-.4,.5], [0.,0.,1.])",
        },
        {
            "setup": """import numpy as np
c = np.eye(4)
""",
            "call": "measure_population(c, [0.,0.,1.], [0.,0.,1.])",
            "gold_call": "_oracle_measure_population(c, [0.,0.,1.], [0.,0.,1.])",
        },
        {
            "setup": """import numpy as np
c = np.eye(4)
c[1, 0] = .2
""",
            "call": "measure_population(c, [0.,0.,0.], [1.,0.,0.])",
            "gold_call": "_oracle_measure_population(c, [0.,0.,0.], [1.,0.,0.])",
        },
    ]
