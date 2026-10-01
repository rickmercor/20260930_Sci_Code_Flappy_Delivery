"""
Convert a complex walker overlap change into a phaseless multiplier.

Importance sampling changes a walker's weight according to its trial-overlap ratio and the Gaussian shift introduced by the force bias. The phaseless constraint converts this complex importance factor into a nonnegative multiplier and suppresses walkers whose overlap rotation lies outside the accepted cosine sector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phaseless_importance(
    trial: 'np.ndarray',
    old_walker: 'np.ndarray',
    new_walker: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
) -> float:
    """Evaluate the phaseless importance multiplier for one walker step.

    Parameters
    ----------
    trial
        Trial Slater determinant.
    old_walker, new_walker
        Physical walker before and after the step.
    auxiliary_field, force_bias
        Matching sampled-field and importance-shift vectors.

    Notes
    -----
    Let
    ``S = det(trial.conj().T @ new_walker) /
    det(trial.conj().T @ old_walker)``
    and
    ``F = dot(x, xbar) - 0.5*dot(xbar, xbar)``,
    where both dots are unconjugated. Return
    ``abs(S*exp(F)) * max(0, cos(angle(S)))``.

    Returns
    -------
    float
        Nonnegative phaseless importance multiplier.

    Raises
    ------
    ValueError
        If shapes or finiteness are invalid, the old overlap is zero, or
        the importance magnitude exceeds the floating-point range.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_phaseless_importance(
    trial: 'np.ndarray',
    old_walker: 'np.ndarray',
    new_walker: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
) -> float:
    import numpy as np

    A = np.asarray(trial, dtype=complex)
    B0 = np.asarray(old_walker, dtype=complex)
    B1 = np.asarray(new_walker, dtype=complex)
    x = np.asarray(auxiliary_field, dtype=float)
    xb = np.asarray(force_bias, dtype=complex)
    if (
        A.ndim != 2
        or min(A.shape) == 0
        or B0.shape != A.shape
        or B1.shape != A.shape
    ):
        raise ValueError("trial and walkers must be matching nonempty matrices")
    if x.ndim != 1 or xb.shape != x.shape or x.size == 0:
        raise ValueError("field and force bias must be matching vectors")
    if any(np.any(~np.isfinite(v)) for v in (A, B0, B1, x, xb)):
        raise ValueError("inputs must be finite")

    old_phase, old_logabs = np.linalg.slogdet(A.conj().T @ B0)
    if old_phase == 0:
        raise ValueError("old trial-walker overlap is zero")

    new_phase, new_logabs = np.linalg.slogdet(A.conj().T @ B1)
    with np.errstate(over="ignore", invalid="ignore"):
        force_factor = np.dot(x, xb) - 0.5 * np.dot(xb, xb)
    if not np.isfinite(force_factor):
        raise ValueError("importance magnitude exceeds the floating-point range")
    log_magnitude = new_logabs - old_logabs + float(np.real(force_factor))
    if new_phase == 0:
        magnitude = 0.0
        ratio_phase = 0.0j
    else:
        if not np.isfinite(log_magnitude) or log_magnitude > np.log(np.finfo(float).max):
            raise ValueError("importance magnitude exceeds the floating-point range")
        if log_magnitude < np.log(np.nextafter(0.0, 1.0)):
            magnitude = 0.0
        else:
            magnitude = float(np.exp(log_magnitude))
        ratio_phase = new_phase / old_phase
    cosine_gate = max(0.0, float(np.cos(np.angle(ratio_phase))))
    return float(magnitude * cosine_gate)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nA=np.eye(2); B0=np.array([[1.,.1j],[.2,1.]],complex); B1=np.array([[1.03+.04j,.08j],[.18,1.01-.02j]],complex); x=np.array([.4,-.7]); xb=np.array([-.03+.02j,.01-.01j])",
            "call": "phaseless_importance(A,B0,B1,x,xb)",
            "gold_call": "_oracle_phaseless_importance(A,B0,B1,x,xb)",
        },
        {
            "setup": "import numpy as np\nA=np.array([[1.],[0.]]); B0=np.array([[1.],[.2]]); B1=np.array([[-1.],[.3]]); x=np.array([0.]); xb=np.array([0j])",
            "call": "phaseless_importance(A,B0,B1,x,xb)",
            "gold_call": "_oracle_phaseless_importance(A,B0,B1,x,xb)",
        },
        {
            "setup": "import numpy as np\nA=np.array([[1.],[0.]]); B0=np.array([[2.],[1.]]); B1=np.array([[3.],[4.]]); x=np.array([.5]); xb=np.array([.2])",
            "call": "phaseless_importance(A,B0,B1,x,xb)",
            "gold_call": "_oracle_phaseless_importance(A,B0,B1,x,xb)",
        },
        {
            "setup": "import numpy as np\nA=np.eye(2); B0=np.eye(2,dtype=complex); B1=np.diag([1j,1.]); x=np.array([1.,-1.]); xb=np.array([.1j,-.2j])",
            "call": "phaseless_importance(A,B0,B1,x,xb)",
            "gold_call": "_oracle_phaseless_importance(A,B0,B1,x,xb)",
        },
        {
            "setup": "import numpy as np\nA=np.array([[1.]]); B0=np.array([[1e-15]]); B1=np.array([[2e-15]]); x=np.array([0.]); xb=np.array([0j])",
            "call": "phaseless_importance(A,B0,B1,x,xb)",
            "gold_call": "_oracle_phaseless_importance(A,B0,B1,x,xb)",
        },
    ]
