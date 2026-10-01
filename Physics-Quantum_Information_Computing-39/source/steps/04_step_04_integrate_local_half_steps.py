"""
Integrate the sinusoidal collective drive on both halves of every interval.

The local Hamiltonians commute at distinct times because they are all

proportional to $J_x$. Let $t_j=j\Delta t$ and $\Delta t=2\pi/(M\omega_d)$.

The two half-interval channels must therefore use their separate integrated angles:



$$

u(t_b,t_a)=\exp\left[-iJ_x\int_{t_a}^{t_b}(\Omega+\epsilon\cos(\omega_d t+\phi_0))\,dt\right].

$$



The system Liouville channel in row-major triplet order is $u\otimes u^*$.

The early half ends at $t_j+\Delta t/2$ and the late half starts there.

Returns
-------
Complex $(M,2,9,9)$ local channel array; index one selects the early half at zero and the late half at one, in chronological order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_local_half_steps(
    omega: float, amplitude: float, drive_frequency: float, intervals: int, phase: float
) -> "np.ndarray":
    r"""Integrate the sinusoidal collective drive on both halves of every interval.

    Parameters
    ----------
    omega : float
        Finite positive static scale $\Omega$.
    amplitude : float
        Finite nonnegative drive amplitude $\epsilon$.
    drive_frequency : float
        Finite positive angular frequency $\omega_d$.
    intervals : int
        Number $M\ge4$ of intervals per period.
    phase : float
        Finite initial phase $\phi_0$ in radians.

    Returns
    -------
    half_steps : np.ndarray
        Complex $(M,2,9,9)$ local channel array; index one selects the
        early half at zero and the late half at one, in chronological order.

    Raises
    ------
    ValueError
        If a scalar is nonreal or nonfinite, omega or drive_frequency is
        nonpositive, amplitude is negative, or intervals is not an integer at least four.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_integrate_local_half_steps(
    omega: float, amplitude: float, drive_frequency: float, intervals: int, phase: float
) -> "np.ndarray":
    """Use exact integrals for two generally unequal local half intervals."""
    omega, amplitude, drive_frequency, phase = map(
        _scalar, (omega, amplitude, drive_frequency, phase)
    )
    intervals = _integer(intervals, 4)
    if min(omega, drive_frequency) <= 0 or amplitude < 0:
        raise ValueError("positive frequencies and nonnegative amplitude required")
    dt = 2 * np.pi / (intervals * drive_frequency)
    jx, _ = _spin_matrices()
    result = np.empty((intervals, 2, 9, 9), dtype=complex)
    for j in range(intervals):
        for half in range(2):
            start = phase + (j + half / 2) * 2 * np.pi / intervals
            end = start + np.pi / intervals
            angle = omega * dt / 2 + amplitude / drive_frequency * (
                np.sin(end) - np.sin(start)
            )
            unitary = expm(-1j * angle * jx)
            result[j, half] = np.kron(unitary, unitary.conj())
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test local half-interval channels and invalid drive frequency."""
    return [
        {
            "setup": "import numpy as np\n",
            "call": "integrate_local_half_steps(1.0, 1.2, 0.66, 24, np.pi/7)",
            "gold_call": "_oracle_integrate_local_half_steps(1.0, 1.2, 0.66, 24, np.pi/7)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n",
            "call": "integrate_local_half_steps(0.8, 0.0, 1.1, 4, 0.0)",
            "gold_call": "_oracle_integrate_local_half_steps(0.8, 0.0, 1.1, 4, 0.0)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n",
            "call": "integrate_local_half_steps(1.0, 2.3, 3.2, 7, -np.pi/3)",
            "gold_call": "_oracle_integrate_local_half_steps(1.0, 2.3, 3.2, 7, -np.pi/3)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: integrate_local_half_steps(1.0, 0.5, 0.0, 16, 0.0))",
            "gold_call": "_raises(lambda: _oracle_integrate_local_half_steps(1.0, 0.5, 0.0, 16, 0.0))",
            "tol": 1e-07,
        },
    ]
