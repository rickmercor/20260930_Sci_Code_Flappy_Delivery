"""
Select the transient resonance and compute the strongest driven entanglement estimator.

The quench starts in $|00,0\rangle$. Its full undriven spectral

resolution determines the selected positive-frequency pair

$-\Gamma_*+i\nu_*$ by admissible one-pair concurrence.

Each supplied dimensionless detuning $d_k$ defines a drive frequency

$\omega_k=\nu_*+d_k\Gamma_*$. The final scalar is



$$

\max_{\epsilon,\omega_k}\frac{4\overline{C}_{2M}(\epsilon,\omega_k)-\overline{C}_M(\epsilon,\omega_k)}{3}.

$$



Both resolutions retain the same auxiliary cutoff and local drive phase.

Rescaling all rates and amplitudes by a common positive factor leaves this

dimensionless estimator unchanged.

Returns
-------
Largest finite, dimensionless two-resolution concurrence estimator over the mode-selected drive grid, with all preceding conventions retained.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_stabilized_entanglement(
    levels: int,
    omega: float,
    bath_frequency: float,
    coupling: float,
    decay: float,
    amplitudes: "np.ndarray",
    detunings: "np.ndarray",
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
    intervals: int,
    phase: float,
) -> float:
    r"""Select the transient resonance and compute the strongest driven entanglement estimator.

    Parameters
    ----------
    levels : int
        Oscillator cutoff $N\ge2$.
    omega : float
        Finite positive static scale $\Omega$.
    bath_frequency : float
        Finite positive oscillator frequency $\omega_b$.
    coupling : float
        Finite nonnegative collective coupling $g$.
    decay : float
        Finite positive damping rate $\kappa$.
    amplitudes : np.ndarray
        Nonempty finite real vector of nonnegative drive amplitudes.
    detunings : np.ndarray
        Nonempty finite real vector $d_k$, whose derived frequencies must be positive.
    phase_count : int
        Number of spectral-scoring phases, at least four.
    decay_limit : float
        Finite positive strict decay-window upper bound.
    frequency_limit : float
        Finite positive strict frequency-window upper bound.
    intervals : int
        Coarse time resolution $M\ge4$; fine resolution is $2M$.
    phase : float
        Finite initial drive phase in radians.

    Returns
    -------
    entanglement : float
        Largest finite, dimensionless two-resolution concurrence estimator over
        the mode-selected drive grid, with all preceding conventions retained.

    Raises
    ------
    ValueError
        If an upstream parameter violates its declared shape, finiteness,
        integer or physical bound; the quench spectrum is not simple and
        attracting; no admissible mode is found; a derived drive frequency
        is nonpositive; or an upstream numerical physical-state check fails.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_stabilized_entanglement(
    levels: int,
    omega: float,
    bath_frequency: float,
    coupling: float,
    decay: float,
    amplitudes: "np.ndarray",
    detunings: "np.ndarray",
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
    intervals: int,
    phase: float,
) -> float:
    """Couple the transient-mode selection to the full driven stationary calculation."""
    generators = _oracle_build_collective_generators(
        levels, omega, bath_frequency, coupling, decay
    )
    initial = np.zeros(9 * levels**2)
    initial[0] = 1
    modes = _oracle_resolve_quench_modes(generators[1], initial, levels)
    selection = _oracle_select_entangling_mode(
        modes, levels, phase_count, decay_limit, frequency_limit
    )
    detunings = _array(detunings, real=True)
    if detunings.ndim != 1 or not len(detunings):
        raise ValueError("detunings must be a nonempty vector")
    frequencies = selection[1] + selection[0] * detunings
    estimates = _oracle_evaluate_stabilization_grid(
        generators[0], omega, levels, amplitudes, frequencies, intervals, phase
    )
    return float(np.max(estimates))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test complete entanglement calculations and invalid detunings."""
    return [
        {
            "setup": "import numpy as np\na = np.array([0.4,0.8,1.2]); d = np.array([-0.5,0.0,0.5])\n",
            "call": "compute_stabilized_entanglement(4, 1.0, 1.8, 0.7, 0.5, a.copy(), d.copy(), 128, "
            "0.8, 4.0, 24, np.pi/7)",
            "gold_call": "_oracle_compute_stabilized_entanglement(4, 1.0, 1.8, 0.7, 0.5, a.copy(), "
            "d.copy(), 128, 0.8, 4.0, 24, np.pi/7)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\na = np.array([0.8]); d = np.array([0.0])\n",
            "call": "compute_stabilized_entanglement(2, 1.0, 1.8, 0.7, 0.5, a.copy(), d.copy(), 16, "
            "2.0, 8.0, 8, 0.0)",
            "gold_call": "_oracle_compute_stabilized_entanglement(2, 1.0, 1.8, 0.7, 0.5, a.copy(), "
            "d.copy(), 16, 2.0, 8.0, 8, 0.0)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\na = np.array([0.7,1.1]); d = np.array([0.4,0.0])\n",
            "call": "compute_stabilized_entanglement(3, 0.8, 1.4, 0.5, 0.3, a.copy(), d.copy(), 32, "
            "0.6, 3.5, 12, -0.3)",
            "gold_call": "_oracle_compute_stabilized_entanglement(3, 0.8, 1.4, 0.5, 0.3, a.copy(), "
            "d.copy(), 32, 0.6, 3.5, 12, -0.3)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "a = np.array([0.8]); d = np.array([-100.0])\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: compute_stabilized_entanglement(2, 1.0, 1.8, 0.7, 0.5, a.copy(), "
            "d.copy(), 16, 2.0, 8.0, 8, 0.0))",
            "gold_call": "_raises(lambda: _oracle_compute_stabilized_entanglement(2, 1.0, 1.8, 0.7, "
            "0.5, a.copy(), d.copy(), 16, 2.0, 8.0, 8, 0.0))",
            "tol": 1e-07,
        },
    ]
