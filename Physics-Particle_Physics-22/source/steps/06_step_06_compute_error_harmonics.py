"""
Extract the continuous CP-phase harmonics of appearance and survival probability errors.

Write $F=R_{23}\operatorname{diag}(1,1,e^{i\sigma\delta})$ and restore each

profile amplitude by $S_f=F\widetilde SF^\dagger$. For a muon initial state,



$$

(S_f)_{e\mu}=c_{23}\widetilde S_{e\mu}+s_{23}e^{-i\sigma\delta}\widetilde S_{e\tau},

$$



$$

(S_f)_{\mu\mu}=c_{23}^2\widetilde S_{\mu\mu}+s_{23}^2\widetilde S_{\tau\tau}

+s_{23}c_{23}(e^{-i\sigma\delta}\widetilde S_{\mu\tau}+e^{i\sigma\delta}\widetilde S_{\tau\mu}).

$$



Subtract the exact squared moduli from their unrefined counterparts without

clipping or probability completion. Each error is a real trigonometric polynomial



$$

f(\delta)=h_0+h_1\cos\delta+h_2\sin\delta+h_3\cos2\delta+h_4\sin2\delta.

$$



For amplitude $b_-e^{-i\delta}+b_0+b_+e^{i\delta}$, its positive-frequency

probability coefficients are $d_1=b_0b_-^*+b_+b_0^*$ and $d_2=b_+b_-^*$;

the corresponding real coefficients are $(2\Re d_k,-2\Im d_k)$.

This relation is valid for nonunitary approximate amplitudes.

Returns
-------
A real array of shape (2, 5) containing unscaled error harmonics for muon appearance and survival.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_error_harmonics(
    amplitudes: "np.ndarray", s23_sq: float, charge: int = 1
) -> "np.ndarray":
    r"""Extract the continuous CP-phase harmonics of appearance and survival probability errors.

    Parameters
    ----------
    amplitudes : np.ndarray
        Finite complex shape (2, 3, 3), approximate then exact, in the real
        propagation basis. Exact unitarity is not required by this function.
    s23_sq : float
        Finite squared sine in [0, 1].
    charge : int, default 1
        +1 or -1; determines the sign of the physical CP phase.

    Returns
    -------
    coefficients : np.ndarray
        Finite real shape (2, 5). Rows are muon-to-electron and muon-to-muon
        errors; columns are constant, cos(delta), sin(delta), cos(2 delta),
        sin(2 delta). Errors are unscaled probability differences.

    Raises
    ------
    ValueError
        If amplitude shape or finiteness, the squared sine or charge is
        invalid, or squared amplitudes overflow to nonfinite coefficients.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _channel_harmonics(amplitude, s23_sq, charge):
    sine, cosine = np.sqrt([s23_sq, 1.0 - s23_sq])
    channels = np.array(
        [
            [sine * amplitude[0, 2], cosine * amplitude[0, 1], 0.0],
            [
                sine * cosine * amplitude[1, 2],
                (1.0 - s23_sq) * amplitude[1, 1] + s23_sq * amplitude[2, 2],
                sine * cosine * amplitude[2, 1],
            ],
        ],
        dtype=complex,
    )
    negative, constant, positive = channels.T
    first = constant * negative.conj() + positive * constant.conj()
    second = positive * negative.conj()
    return np.column_stack(
        (
            np.sum(np.abs(channels) ** 2, axis=1),
            2.0 * first.real,
            -2.0 * charge * first.imag,
            2.0 * second.real,
            -2.0 * charge * second.imag,
        )
    )


def _oracle_compute_error_harmonics(
    amplitudes: "np.ndarray", s23_sq: float, charge: int = 1
) -> "np.ndarray":
    matrices = np.asarray(amplitudes, dtype=complex)
    if matrices.shape != (2, 3, 3) or not np.all(np.isfinite(matrices)):
        raise ValueError("amplitudes must have finite shape (2, 3, 3)")
    angle = _finite_scalar(s23_sq, "s23_sq")
    if not 0.0 <= angle <= 1.0:
        raise ValueError("s23_sq must be in [0, 1]")
    if (
        isinstance(charge, (bool, np.bool_))
        or not isinstance(charge, (int, np.integer))
        or charge not in (-1, 1)
    ):
        raise ValueError("charge must be +1 or -1")
    result = _channel_harmonics(matrices[0], angle, charge) - _channel_harmonics(
        matrices[1], angle, charge
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("probability harmonics must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nlayers = np.array([[250.,1.3],[1900.,4.2],[700.,2.1]])\nparams = np.array([.3,.03,8e-5,.0024])\na_model = propagate_spectral_pair(layers.copy(), 3., params.copy())\na_gold = _oracle_propagate_spectral_pair(layers.copy(), 3., params.copy())\n",
            "call": "compute_error_harmonics(a_model.copy(), .561)",
            "gold_call": "_oracle_compute_error_harmonics(a_gold.copy(), .561)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\na=np.array([np.eye(3),np.eye(3)],dtype=complex)\n",
            "call": "compute_error_harmonics(a.copy(), 0.0)",
            "gold_call": "_oracle_compute_error_harmonics(a.copy(), 0.0)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\na=np.array([[[.8,.2j,.1],[.3,.9j,.2+.1j],[.2j,.3-.1j,.7]],np.eye(3)],dtype=complex)\n",
            "call": "compute_error_harmonics(a.copy(), .3, -1)",
            "gold_call": "_oracle_compute_error_harmonics(a.copy(), .3, -1)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\na=np.zeros((2,2,2),dtype=complex)\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: compute_error_harmonics(a.copy(), .5))",
            "gold_call": "_raises_value_error(lambda: _oracle_compute_error_harmonics(a.copy(), .5))",
            "tol": 0.0,
        },
    ]
