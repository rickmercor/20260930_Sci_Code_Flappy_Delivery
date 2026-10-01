"""
Evaluate the period-averaged entanglement estimator for each candidate drive.

Entanglement is nonlinear in the density matrix. At each left interval

boundary, evaluate concurrence first and only then average over a period.

For every amplitude-frequency pair compute two resolutions independently:



$$

\overline{C}_M=\frac{1}{M}\sum_{j=0}^{M-1}C(\rho_j),\qquad

E_M=\frac{4\overline{C}_{2M}-\overline{C}_M}{3}.

$$



The estimator uses both time resolutions at the supplied oscillator cutoff.

Returns
-------
Real array of shape $(n_\epsilon,n_\omega)$ containing $E_M$, preserving the supplied amplitude-row and frequency-column order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_stabilization_grid(
    bath_generator: "np.ndarray",
    omega: float,
    levels: int,
    amplitudes: "np.ndarray",
    frequencies: "np.ndarray",
    intervals: int,
    phase: float,
) -> "np.ndarray":
    r"""Evaluate the period-averaged entanglement estimator for each candidate drive.

    Parameters
    ----------
    bath_generator : np.ndarray
        Finite complex $(D,D)$ uniform auxiliary generator, $D=9N^2$.
    omega : float
        Finite positive static scale $\Omega$.
    levels : int
        Oscillator cutoff $N\ge2$.
    amplitudes : np.ndarray
        Nonempty finite real one-dimensional array of nonnegative amplitudes.
    frequencies : np.ndarray
        Nonempty finite real one-dimensional array of positive angular frequencies.
    intervals : int
        Coarse resolution $M\ge4$; fine resolution is $2M$.
    phase : float
        Finite initial drive phase in radians.

    Returns
    -------
    estimates : np.ndarray
        Real array of shape $(n_\epsilon,n_\omega)$ containing $E_M$,
        preserving the supplied amplitude-row and frequency-column order.

    Raises
    ------
    ValueError
        If grid arrays, dimensions, finite values, positive frequencies, or
        integer bounds fail, or an upstream fixed-point/density/closure check
        fails under its documented tolerances.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _mean_driven_concurrence(
    bath_generator, omega, levels, amplitude, frequency, intervals, phase
):
    local = _oracle_integrate_local_half_steps(
        omega, amplitude, frequency, intervals, phase
    )
    dt = 2 * np.pi / (intervals * frequency)
    cycle = _oracle_contract_auxiliary_cycle(bath_generator, local, dt, levels)
    states = _oracle_recover_periodic_qubit_states(cycle, levels)
    return float(np.mean([_concurrence(state) for state in states]))


def _oracle_evaluate_stabilization_grid(
    bath_generator: "np.ndarray",
    omega: float,
    levels: int,
    amplitudes: "np.ndarray",
    frequencies: "np.ndarray",
    intervals: int,
    phase: float,
) -> "np.ndarray":
    """Average concurrence before combining the two time resolutions."""
    levels, intervals = _integer(levels, 2), _integer(intervals, 4)
    bath_generator = _array(bath_generator, (9 * levels**2, 9 * levels**2))
    amplitudes, frequencies = (
        _array(amplitudes, real=True),
        _array(frequencies, real=True),
    )
    omega, phase = _scalar(omega), _scalar(phase)
    if (
        amplitudes.ndim != 1
        or frequencies.ndim != 1
        or not len(amplitudes)
        or not len(frequencies)
        or np.min(amplitudes) < 0
        or np.min(frequencies) <= 0
        or omega <= 0
    ):
        raise ValueError("nonempty physical amplitude and frequency grids are required")
    result = np.empty((len(amplitudes), len(frequencies)))
    for row, amplitude in enumerate(amplitudes):
        for column, frequency in enumerate(frequencies):
            coarse = _mean_driven_concurrence(
                bath_generator, omega, levels, amplitude, frequency, intervals, phase
            )
            fine = _mean_driven_concurrence(
                bath_generator,
                omega,
                levels,
                amplitude,
                frequency,
                2 * intervals,
                phase,
            )
            result[row, column] = (4 * fine - coarse) / 3
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test drive-grid estimates and a nonpositive frequency."""
    return [
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(3,1.0,1.8,0.7,0.5)[0]\n"
            "a = np.array([0.4,1.2]); f = np.array([0.66,0.9])\n",
            "call": "evaluate_stabilization_grid(l.copy(), 1.0, 3, a.copy(), f.copy(), 12, np.pi/7)",
            "gold_call": "_oracle_evaluate_stabilization_grid(l.copy(), 1.0, 3, a.copy(), f.copy(), "
            "12, np.pi/7)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(2,1.0,1.8,0.7,0.5)[0]\n"
            "a = np.array([0.0]); f = np.array([1.0])\n",
            "call": "evaluate_stabilization_grid(l.copy(), 1.0, 2, a.copy(), f.copy(), 4, 0.0)",
            "gold_call": "_oracle_evaluate_stabilization_grid(l.copy(), 1.0, 2, a.copy(), f.copy(), 4, "
            "0.0)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(2,0.8,1.4,0.5,0.3)[0]\n"
            "a = np.array([1.1,0.6]); f = np.array([1.5,0.8])\n",
            "call": "evaluate_stabilization_grid(l.copy(), 0.8, 2, a.copy(), f.copy(), 7, -0.3)",
            "gold_call": "_oracle_evaluate_stabilization_grid(l.copy(), 0.8, 2, a.copy(), f.copy(), 7, "
            "-0.3)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = np.zeros((36,36)); a = np.array([0.5]); f = np.array([0.0])\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: evaluate_stabilization_grid(l.copy(), 1.0, 2, a.copy(), f.copy(), "
            "4, 0.0))",
            "gold_call": "_raises(lambda: _oracle_evaluate_stabilization_grid(l.copy(), 1.0, 2, "
            "a.copy(), f.copy(), 4, 0.0))",
            "tol": 1e-07,
        },
    ]
