"""
Evaluate the enlarged-basis importance-sampling force bias.

Importance sampling shifts the auxiliary Gaussian fields according to a trial-dependent mixed estimator. For the ITHC interaction, this estimator is formed from the occupation diagonal of the mixed Green matrix in the extended basis and the supplied auxiliary-channel factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ithc_force_bias(
    extended_green: 'np.ndarray',
    channels: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    """Form the importance-sampling force bias in the extended basis.

    Parameters
    ----------
    extended_green
        Square mixed Green matrix in the auxiliary basis.
    channels
        Channel factor with shape ``(n_fields, n_auxiliary)``.
    time_step
        Finite nonnegative propagation interval.

    Notes
    -----
    The supplied real channels factor the positive interaction kernel. Define
    the imaginary auxiliary-field operators as ``v = 1j * channels`` and
    return ``-sqrt(time_step) * v @ diag(extended_green)``. Only diagonal
    occupations of ``extended_green`` enter the contraction.

    Returns
    -------
    np.ndarray
        Complex force-bias vector with one entry per auxiliary field.

    Raises
    ------
    ValueError
        If shapes, finiteness, or the time-step domain are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_ithc_force_bias(
    extended_green: 'np.ndarray',
    channels: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    import numpy as np

    G = np.asarray(extended_green, dtype=complex)
    w = np.asarray(channels, dtype=float)
    dt = float(time_step)
    if G.ndim != 2 or G.shape[0] == 0 or G.shape[0] != G.shape[1]:
        raise ValueError("extended_green must be a nonempty square matrix")
    if w.ndim != 2 or w.shape[1] != G.shape[0] or w.shape[0] == 0:
        raise ValueError("channels have incompatible shape")
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("time_step must be finite and nonnegative")
    if np.any(~np.isfinite(G)) or np.any(~np.isfinite(w)):
        raise ValueError("inputs must be finite")
    auxiliary_operators = 1.0j * w
    return -np.sqrt(dt) * (auxiliary_operators @ np.diag(G))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nG=np.array([[.7+.1j,.2],[-.1j,.4-.05j]]); w=np.array([[.3,-.4],[.2,.5],[.1,.1]])",
            "call": "ithc_force_bias(G,w,.04)",
            "gold_call": "_oracle_ithc_force_bias(G,w,.04)",
        },
        {
            "setup": "import numpy as np\nG=np.array([[1.,2.],[3.,-1.]],complex); w=np.array([[2.,-3.]])",
            "call": "ithc_force_bias(G,w,0.)",
            "gold_call": "_oracle_ithc_force_bias(G,w,0.)",
        },
        {
            "setup": "import numpy as np\nG=np.array([[.25+.75j]]); w=np.array([[-.6],[.4]])",
            "call": "ithc_force_bias(G,w,.01)",
            "gold_call": "_oracle_ithc_force_bias(G,w,.01)",
        },
        {
            "setup": "import numpy as np\nG=np.array([[.2,.7j,0.],[-.3j,.6,.1],[0.,-.2,.9-.1j]]); w=np.array([[1.,0.,-1.],[0.,.5,.5]])",
            "call": "ithc_force_bias(G,w,.125)",
            "gold_call": "_oracle_ithc_force_bias(G,w,.125)",
        },
    ]
