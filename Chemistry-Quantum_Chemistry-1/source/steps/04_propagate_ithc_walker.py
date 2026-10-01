"""
Apply the diagonal ITHC field and finish a projected walker step.

The enlarged-space ITHC interaction permits the sampled two-body field to act diagonally on the rows of the rotated Slater matrix. The propagated state must then be projected back to the physical orbital space before applying the second one-body half-step. The field is shifted by the complex force bias obtained from importance sampling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_ithc_walker(
    half_propagated_walker: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    channels: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    """Complete one ITHC walker step after its first one-body half-step.

    Parameters
    ----------
    half_propagated_walker
        Walker after the first one-body half-step.
    half_one_body
        Physical-basis one-body half-step propagator.
    isometry
        Real row-orthonormal physical-to-auxiliary map.
    channels
        Channel factor with shape ``(n_fields, n_auxiliary)``.
    auxiliary_field, force_bias
        Matching sampled-field and importance-shift vectors.
    time_step
        Finite nonnegative propagation interval.

    Notes
    -----
    Form ``B_t = u.T @ half_propagated_walker`` and
    ``z = sqrt(complex(-time_step)) * ((auxiliary_field-force_bias) @ channels)``
    using the principal complex square root. Return
    ``half_one_body @ (u @ (exp(z)[:,None] * B_t))``.

    Returns
    -------
    np.ndarray
        Propagated physical-basis Slater matrix.

    Raises
    ------
    ValueError
        If shapes, finiteness, row orthonormality, or the time-step domain
        are invalid, or if propagation exceeds the floating-point range.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_ithc_walker(
    half_propagated_walker: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    channels: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    import numpy as np

    B = np.asarray(half_propagated_walker, dtype=complex)
    H = np.asarray(half_one_body, dtype=complex)
    u = np.asarray(isometry, dtype=float)
    w = np.asarray(channels, dtype=float)
    x = np.asarray(auxiliary_field, dtype=float)
    xb = np.asarray(force_bias, dtype=complex)
    dt = float(time_step)
    if B.ndim != 2 or min(B.shape) == 0:
        raise ValueError("walker must be a nonempty matrix")
    if H.shape != (B.shape[0], B.shape[0]):
        raise ValueError("half_one_body has incompatible shape")
    if u.ndim != 2 or u.shape[0] != B.shape[0]:
        raise ValueError("isometry has incompatible shape")
    if w.ndim != 2 or w.shape[1] != u.shape[1]:
        raise ValueError("channels have incompatible shape")
    if x.shape != (w.shape[0],) or xb.shape != x.shape:
        raise ValueError("field and force bias have incompatible shape")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("time_step must be finite and nonnegative")
    if any(np.any(~np.isfinite(v)) for v in (B, H, u, w, x, xb)):
        raise ValueError("inputs must be finite")
    if not np.allclose(u @ u.T, np.eye(u.shape[0]), rtol=1e-11, atol=1e-11):
        raise ValueError("isometry rows must be orthonormal")

    extended = u.T @ B
    exponent = np.sqrt(complex(-dt)) * ((x - xb) @ w)
    with np.errstate(over="ignore", invalid="ignore"):
        extended = np.exp(exponent)[:, None] * extended
        propagated = H @ (u @ extended)
    if np.any(~np.isfinite(propagated)):
        raise ValueError("propagation exceeds the floating-point range")
    return propagated

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nB=np.array([[1.,.1j],[.2,1.],[-.1j,.3]],complex); H=np.array([[1.02,.01,0.],[.01,.98,.02],[0.,.02,1.01]]); u=np.array([[1.,0.,0.,0.],[0.,2**-.5,2**-.5,0.],[0.,0.,0.,1.]]); w=np.array([[.2,-.3,.1,.4],[-.1,.2,.25,-.2]]); x=np.array([.4,-.7]); xb=np.array([-.03+.01j,.02-.02j])",
            "call": "propagate_ithc_walker(B,H,u,w,x,xb,.04)",
            "gold_call": "_oracle_propagate_ithc_walker(B,H,u,w,x,xb,.04)",
        },
        {
            "setup": "import numpy as np\nB=np.array([[1.],[2j]]); H=np.eye(2); u=np.eye(2); w=np.array([[.3,.4]]); x=np.array([2.]); xb=np.array([-.5j])",
            "call": "propagate_ithc_walker(B,H,u,w,x,xb,0.)",
            "gold_call": "_oracle_propagate_ithc_walker(B,H,u,w,x,xb,0.)",
        },
        {
            "setup": "import numpy as np\nB=np.array([[1.+.2j]]); H=np.array([[.9]]); u=np.array([[.5,.5,-.5,.5]]); w=np.array([[.2,0.,-.1,.3]]); x=np.array([0.]); xb=np.array([.1j])",
            "call": "propagate_ithc_walker(B,H,u,w,x,xb,.09)",
            "gold_call": "_oracle_propagate_ithc_walker(B,H,u,w,x,xb,.09)",
        },
        {
            "setup": "import numpy as np\nB=np.array([[1.,0.],[0.,1.]],complex); H=np.array([[1.1,.05],[.05,.95]]); u=np.eye(2); w=np.zeros((2,2)); x=np.array([1.,-2.]); xb=np.zeros(2,complex)",
            "call": "propagate_ithc_walker(B,H,u,w,x,xb,.2)",
            "gold_call": "_oracle_propagate_ithc_walker(B,H,u,w,x,xb,.2)",
        },
        {
            "setup": "import numpy as np\ndef expect_value_error(fn,*args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\nB=np.array([[1.]]); H=np.array([[1.]]); u=np.array([[1.]]); w=np.array([[1.]]); x=np.array([0.]); xb=np.array([1000j])",
            "call": "expect_value_error(propagate_ithc_walker,B,H,u,w,x,xb,1.)",
            "gold_call": "expect_value_error(_oracle_propagate_ithc_walker,B,H,u,w,x,xb,1.)",
        },
    ]
