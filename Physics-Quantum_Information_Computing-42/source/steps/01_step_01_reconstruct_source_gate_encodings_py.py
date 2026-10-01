"""
Reconstruct representative gate matrices from the literal-weight semantics of

Quokka#'s computational-basis encoding.



This step independently validates the paper-specific gate semantics before any

synthesis optimization is performed. It reconstructs H, T, T†, RX(theta),

RZ(theta), and CX from the same amplitudes represented by the source weighted

Boolean encodings.



The RZ reconstruction follows the source computational-basis convention

diag(1, exp(i*theta)), which differs from the conventional physical

RZ(theta)=diag(exp(-i*theta/2), exp(i*theta/2)) only by a global phase.



This is deliberately a local gate-semantics validation step. Its 4x4 matrices

do not restrict the later synthesis benchmark to two qubits; the synthesis

instance itself is three-qubit.

Reconstruct representative gate semantics from the source paper and pinned

implementation using its computational-basis weighted Boolean construction.



The result should reproduce the source behavior of the requested gates.

Equivalent choices of auxiliary-variable polarity are acceptable when they

define the same weighted transition function.

Returns
-------
np.ndarray of shape (6,4,4,2), containing [H0,T0,Tdg0,RX0,RZ0,CX01] with real and imaginary parts on the final axis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconstruct_source_gate_encodings(
    theta_rx: float,
    theta_rz: float,
) -> np.ndarray:
    """
    Reconstruct representative source computational-basis gate matrices.

    Parameters
    ----------
    theta_rx : float
        RX rotation angle in radians.
    theta_rz : float
        Source-convention RZ rotation angle in radians.

    Returns
    -------
    np.ndarray
        Array of shape (6,4,4,2). Gate order is
        [H0, T0, Tdg0, RX0, RZ0, CX01].
        The final axis stores real and imaginary components.

    Raises
    ------
    ValueError
        If either angle is non-finite.
    """
    return np.empty((6, 4, 4, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import cmath
import math
import numpy as np


def _oracle_reconstruct_source_gate_encodings(
    theta_rx: float,
    theta_rz: float,
) -> np.ndarray:
    values = np.asarray(
        [theta_rx, theta_rz],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "rotation angles must be finite"
        )

    I = np.eye(
        2,
        dtype=complex,
    )

    # Source Hadamard semantics:
    # h <-> (q and q')
    # W(h) = -1/sqrt(2)
    # W(~h) = +1/sqrt(2)
    H = (
        np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=complex,
        )
        / math.sqrt(2.0)
    )

    T = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                1j
                * math.pi
                / 4.0
            ),
        ]
    )

    Tdg = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                -1j
                * math.pi
                / 4.0
            ),
        ]
    )

    c = math.cos(
        theta_rx / 2.0
    )

    s = (
        -1j
        * math.sin(
            theta_rx / 2.0
        )
    )

    RX = np.array(
        [
            [c, s],
            [s, c],
        ],
        dtype=complex,
    )

    # Source computational-basis RZ convention.
    RZ_source = np.diag(
        [
            1.0 + 0.0j,
            cmath.exp(
                1j
                * theta_rz
            ),
        ]
    )

    H0 = np.kron(
        H,
        I,
    )

    T0 = np.kron(
        T,
        I,
    )

    Tdg0 = np.kron(
        Tdg,
        I,
    )

    RX0 = np.kron(
        RX,
        I,
    )

    RZ0 = np.kron(
        RZ_source,
        I,
    )

    CX01 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )

    gates = np.stack(
        [
            H0,
            T0,
            Tdg0,
            RX0,
            RZ0,
            CX01,
        ],
        axis=0,
    )

    return np.stack(
        [
            gates.real,
            gates.imag,
        ],
        axis=-1,
    ).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

theta_rx = 0.8453
theta_rz = 0.4217
""",
            "call": "reconstruct_source_gate_encodings(theta_rx, theta_rz)",
            "gold_call": "_oracle_reconstruct_source_gate_encodings(theta_rx, theta_rz)",
        },
        {
            "setup": """import numpy as np

theta_rx = 0.0
theta_rz = 0.0
""",
            "call": "reconstruct_source_gate_encodings(theta_rx, theta_rz)",
            "gold_call": "_oracle_reconstruct_source_gate_encodings(theta_rx, theta_rz)",
        },
        {
            "setup": """import numpy as np

theta_rx = np.nan
theta_rz = 0.0

def run_model():
    try:
        reconstruct_source_gate_encodings(
            theta_rx,
            theta_rz,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_reconstruct_source_gate_encodings(
            theta_rx,
            theta_rz,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
