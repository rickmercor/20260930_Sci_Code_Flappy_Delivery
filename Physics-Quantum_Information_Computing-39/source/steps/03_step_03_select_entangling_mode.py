"""
Select a transient conjugate pair by its physically admissible entanglement score.

Trace each weighted joint mode over the oscillator and embed the

triplet sector into the two-qubit Hilbert space. For the stationary reduced

state $\rho_{\mathrm{ss}}$ and a positive-frequency contribution $\rho_n$, form



$$

\rho^{(n)}(\theta)=\rho_{\mathrm{ss}}+e^{i\theta}\rho_n+e^{-i\theta}\rho_n^\dagger.

$$



Admissible reconstructions have unit trace and a positive semidefinite spectrum. Two-qubit concurrence is

$\max(0,s_1-s_2-s_3-s_4)$, where the decreasing $s_i$ are the spin-flip

singular values of $\sqrt{\rho}(Y\otimes Y)\sqrt{\rho}^*$.

Returns
-------
Real $(3,)$ vector $(\Gamma_*,\nu_*,C_*)$ for the winning pair in the strict window $0<\Gamma<\Gamma_{\max}$ and $0<\nu<\nu_{\max}$. Eigenvalues at most $10^{-10}$ below zero are clipped only when evaluating concurrence. Each pair is scored by its maximum over admissible phases. Score ties within $10^{-10}$ use the smallest positive frequency across pairs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_entangling_mode(
    modes: "np.ndarray",
    levels: int,
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
) -> "np.ndarray":
    r"""Select a transient conjugate pair by its physically admissible entanglement score.

    Parameters
    ----------
    modes : np.ndarray
        Weighted mode table $(D,D+1)$ returned by the preceding spectral step;
        row zero is the stationary contribution.
    levels : int
        Oscillator cutoff $N\ge2$, with $D=9N^2$.
    phase_count : int
        Number $K\ge4$ of phases $\theta_k=2\pi k/K$.
    decay_limit : float
        Finite positive upper bound for $\Gamma=-\operatorname{Re}\gamma$.
    frequency_limit : float
        Finite positive upper bound for $\nu=\operatorname{Im}\gamma$.

    Returns
    -------
    selection : np.ndarray
        Real $(3,)$ vector $(\Gamma_*,\nu_*,C_*)$ for the winning
        pair in the strict window $0<\Gamma<\Gamma_{\max}$ and $0<\nu<\nu_{\max}$.
        Eigenvalues at most $10^{-10}$ below zero are clipped only when evaluating
        concurrence. Each pair is scored by its maximum over admissible phases.
        Score ties within $10^{-10}$ use the smallest positive frequency across pairs.

    Raises
    ------
    ValueError
        If dimensions, finite values, integer bounds, or positive limits fail;
        the stationary state fails density checks; or no mode has an admissible
        phase in the specified strict spectral window.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _concurrence(density):
    _physical_density(density, 1e-8)
    density = (density + density.conj().T) / 2
    values, vectors = np.linalg.eigh(density)
    root = (vectors * np.sqrt(np.maximum(values, 0))[None, :]) @ vectors.conj().T
    y = np.array([[0, -1j], [1j, 0]])
    singular = np.linalg.svd(root @ np.kron(y, y) @ root.conj(), compute_uv=False)
    return float(max(0.0, singular[0] - np.sum(singular[1:])))


def _oracle_select_entangling_mode(
    modes: "np.ndarray",
    levels: int,
    phase_count: int,
    decay_limit: float,
    frequency_limit: float,
) -> "np.ndarray":
    """Score weighted conjugate pairs, rejecting nonpositive reconstructions."""
    levels, phase_count = _integer(levels, 2), _integer(phase_count, 4)
    dimension = 9 * levels**2
    modes = _array(modes, (dimension, dimension + 1))
    decay_limit, frequency_limit = map(_scalar, (decay_limit, frequency_limit))
    if min(decay_limit, frequency_limit) <= 0:
        raise ValueError("spectral limits must be positive")
    stationary = _reduce_qubits(modes[0, 1:], levels)
    _physical_density(stationary)
    records = []
    for mode in modes[1:]:
        decay, frequency = -mode[0].real, mode[0].imag
        if not (0 < decay < decay_limit and 1e-9 < frequency < frequency_limit):
            continue
        contribution = _reduce_qubits(mode[1:], levels)
        scores = []
        for k in range(phase_count):
            phase = 2 * np.pi * k / phase_count
            oscillation = np.exp(1j * phase) * contribution
            density = stationary + oscillation + oscillation.conj().T
            density = (density + density.conj().T) / 2
            if np.linalg.eigvalsh(density).min() >= -1e-10:
                scores.append(_concurrence(density))
        if scores:
            records.append([decay, frequency, max(scores)])
    if not records:
        raise ValueError("no admissible transient pair in the spectral window")
    maximum = max(record[2] for record in records)
    tied = [record for record in records if record[2] >= maximum - 1e-10]
    return np.asarray(min(tied, key=lambda record: record[1]), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test three-component mode selection and an empty admissible window."""
    return [
        {
            "setup": """import numpy as np
import scipy.linalg


def case_quench_modes(levels, omega, bath_frequency, coupling, decay):
    # Weighted mode table for the |s=0,n=0> quench, rebuilt with numpy/scipy
    # only from the documented model and spectral convention; no task code.
    size = 3 * levels
    lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
    jump = np.kron(np.eye(3), lowering)
    number = jump.T @ jump
    spin_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    spin_z = np.diag([1, 0, -1])
    bath_h = bath_frequency * np.kron(np.eye(3), lowering.T @ lowering)
    bath_h += coupling * np.kron(spin_z, lowering + lowering.T)
    system_h = omega * np.kron(spin_x, np.eye(levels))

    # Column k of the generator is the image of the k-th grouped matrix unit.
    columns = []
    for k in range(size * size):
        unit = np.zeros(size * size)
        unit[k] = 1
        rho = unit.reshape(3, 3, levels, levels).transpose(0, 2, 1, 3).reshape(size, size)
        image = -1j * (bath_h @ rho - rho @ bath_h) + decay * (
            jump @ rho @ jump.T - 0.5 * (number @ rho + rho @ number))
        image = image + -1j * (system_h @ rho - rho @ system_h)
        columns.append(image.reshape(3, levels, 3, levels).transpose(0, 2, 1, 3).reshape(-1))
    generator = np.column_stack(columns)

    rates, right = scipy.linalg.eig(generator)
    stationary = int(np.argmin(np.abs(rates)))
    initial = np.zeros(size * size, dtype=complex)
    initial[0] = 1
    weights = np.linalg.solve(right, initial)
    weighted = (right * weights[None, :]).T
    transient = sorted(
        (n for n in range(size * size) if n != stationary),
        key=lambda n: (round(float(rates[n].imag), 10), float(rates[n].real)),
    )
    order = [stationary] + transient
    modes = np.column_stack((rates[order], weighted[order]))
    modes[0, 0] = 0
    return modes


m = case_quench_modes(4, 1.0, 1.8, 0.7, 0.5)
""",
            "call": "select_entangling_mode(m.copy(), 4, 128, 0.8, 4.0)",
            "gold_call": "_oracle_select_entangling_mode(m.copy(), 4, 128, 0.8, 4.0)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(2,1.0,1.8,0.7,0.5)[1]\n"
            "x = np.zeros(36); x[0] = 1\n"
            "m = _oracle_resolve_quench_modes(l,x,2)\n",
            "call": "select_entangling_mode(m.copy(), 2, 4, 2.0, 8.0)",
            "gold_call": "_oracle_select_entangling_mode(m.copy(), 2, 4, 2.0, 8.0)",
            "tol": 1e-07,
        },
        {
            "setup": """import numpy as np
import scipy.linalg


def case_quench_modes(levels, omega, bath_frequency, coupling, decay):
    # Weighted mode table for the |s=0,n=0> quench, rebuilt with numpy/scipy
    # only from the documented model and spectral convention; no task code.
    size = 3 * levels
    lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
    jump = np.kron(np.eye(3), lowering)
    number = jump.T @ jump
    spin_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    spin_z = np.diag([1, 0, -1])
    bath_h = bath_frequency * np.kron(np.eye(3), lowering.T @ lowering)
    bath_h += coupling * np.kron(spin_z, lowering + lowering.T)
    system_h = omega * np.kron(spin_x, np.eye(levels))

    # Column k of the generator is the image of the k-th grouped matrix unit.
    columns = []
    for k in range(size * size):
        unit = np.zeros(size * size)
        unit[k] = 1
        rho = unit.reshape(3, 3, levels, levels).transpose(0, 2, 1, 3).reshape(size, size)
        image = -1j * (bath_h @ rho - rho @ bath_h) + decay * (
            jump @ rho @ jump.T - 0.5 * (number @ rho + rho @ number))
        image = image + -1j * (system_h @ rho - rho @ system_h)
        columns.append(image.reshape(3, levels, 3, levels).transpose(0, 2, 1, 3).reshape(-1))
    generator = np.column_stack(columns)

    rates, right = scipy.linalg.eig(generator)
    stationary = int(np.argmin(np.abs(rates)))
    initial = np.zeros(size * size, dtype=complex)
    initial[0] = 1
    weights = np.linalg.solve(right, initial)
    weighted = (right * weights[None, :]).T
    transient = sorted(
        (n for n in range(size * size) if n != stationary),
        key=lambda n: (round(float(rates[n].imag), 10), float(rates[n].real)),
    )
    order = [stationary] + transient
    modes = np.column_stack((rates[order], weighted[order]))
    modes[0, 0] = 0
    return modes


m = case_quench_modes(3, 0.8, 1.4, 0.5, 0.3)
""",
            "call": "select_entangling_mode(m.copy(), 3, 32, 0.6, 3.5)",
            "gold_call": "_oracle_select_entangling_mode(m.copy(), 3, 32, 0.6, 3.5)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "m = np.zeros((36,37)); m[0,1] = 1\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: select_entangling_mode(m.copy(), 2, 16, 0.5, 1.0))",
            "gold_call": "_raises(lambda: _oracle_select_entangling_mode(m.copy(), 2, 16, 0.5, 1.0))",
            "tol": 1e-07,
        },
    ]
