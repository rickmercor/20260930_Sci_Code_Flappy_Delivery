"""
Recover the attracting auxiliary cycle and its reduced qubit micromotion.

The periodic joint state is the normalized stationary eigenvector of the complete auxiliary cycle.

If $\tau^{T}$ is the full trace row and $w=\operatorname{vec}(I_{3N}/(3N))$, then



$$

(I-F+w\tau^{T})x_0=w,\qquad x_{j+1}=Q_jx_j.

$$



Only after propagating the full joint state is the oscillator traced out.

The triplet embedding sends its middle basis state to

$(|01\rangle+|10\rangle)/\sqrt{2}$. Return states at the left boundaries,

before their corresponding interval channels act.

Returns
-------
Complex $(M,4,4)$ normalized positive two-qubit states, in the order $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$, at left boundaries.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_periodic_qubit_states(
    cycle_stack: "np.ndarray", levels: int
) -> "np.ndarray":
    r"""Recover the attracting auxiliary cycle and its reduced qubit micromotion.

    Parameters
    ----------
    cycle_stack : np.ndarray
        Finite complex $(M+1,D,D)$ array with $M\ge4$, $D=9N^2$;
        the first $M$ slices are physical channels and the last is their product.
    levels : int
        Oscillator cutoff $N\ge2$.

    Returns
    -------
    qubit_states : np.ndarray
        Complex $(M,4,4)$ normalized positive two-qubit states, in the
        order $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)$, at left boundaries.

    Raises
    ------
    ValueError
        If shape, finite values or integer bounds fail; the normalized fixed-point
        system has condition number above 1e12; a reduced state violates density
        checks at 1e-8; or the propagated joint cycle fails closure by more than 1e-8.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recover_periodic_qubit_states(
    cycle_stack: "np.ndarray", levels: int
) -> "np.ndarray":
    """Normalize once in auxiliary space, then trace each micromotion state."""
    levels = _integer(levels, 2)
    dimension = 9 * levels**2
    cycle_stack = _array(cycle_stack)
    if (
        cycle_stack.ndim != 3
        or cycle_stack.shape[1:] != (dimension, dimension)
        or len(cycle_stack) < 5
    ):
        raise ValueError(
            "cycle_stack must contain at least four channels and their product"
        )
    trace = _trace_row(levels)
    uniform = trace / (3 * levels)
    system = np.eye(dimension) - cycle_stack[-1] + np.outer(uniform, trace)
    if np.linalg.cond(system) > 1e12:
        raise ValueError("the normalized periodic state is not uniquely resolved")
    initial = np.linalg.solve(system, uniform)
    state = initial.copy()
    result = []
    for channel in cycle_stack[:-1]:
        reduced = _reduce_qubits(state, levels)
        _physical_density(reduced)
        result.append(reduced)
        state = channel @ state
    if np.linalg.norm(state - initial) > 1e-8:
        raise ValueError("joint micromotion fails cycle closure")
    return np.asarray(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test periodic states and rejection of an unresolved fixed point."""
    return [
        {
            "setup": """import numpy as np
import scipy.linalg


def case_cycle_stack():
    # Twelve local channels and their product for N=3, rebuilt with
    # numpy/scipy only from the documented model; no task code.
    levels, intervals = 3, 12
    omega, amplitude, drive_frequency, phase = 1.0, 1.2, 0.66, np.pi / 7
    bath_frequency, coupling, decay = 1.8, 0.7, 0.5
    size = 3 * levels
    lowering = np.diag(np.sqrt(np.arange(1, levels)), 1)
    jump = np.kron(np.eye(3), lowering)
    number = jump.T @ jump
    spin_x = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    spin_z = np.diag([1, 0, -1])
    bath_h = bath_frequency * np.kron(np.eye(3), lowering.T @ lowering)
    bath_h += coupling * np.kron(spin_z, lowering + lowering.T)

    # Column k of the bath generator is the image of the k-th grouped matrix unit.
    columns = []
    for k in range(size * size):
        unit = np.zeros(size * size)
        unit[k] = 1
        rho = unit.reshape(3, 3, levels, levels).transpose(0, 2, 1, 3).reshape(size, size)
        image = -1j * (bath_h @ rho - rho @ bath_h) + decay * (
            jump @ rho @ jump.T - 0.5 * (number @ rho + rho @ number))
        columns.append(image.reshape(3, levels, 3, levels).transpose(0, 2, 1, 3).reshape(-1))
    bath_generator = np.column_stack(columns)

    dt = 2 * np.pi / (intervals * drive_frequency)
    bath = scipy.linalg.expm(dt * bath_generator)
    stack = np.empty((intervals + 1, size * size, size * size), dtype=complex)
    cycle = np.eye(size * size, dtype=complex)
    for j in range(intervals):
        halves = []
        for half in range(2):
            start = phase + (j + half / 2) * 2 * np.pi / intervals
            end = start + np.pi / intervals
            angle = omega * dt / 2 + amplitude / drive_frequency * (
                np.sin(end) - np.sin(start))
            unitary = scipy.linalg.expm(-1j * angle * spin_x)
            local = np.kron(unitary, unitary.conj())
            halves.append(np.kron(local, np.eye(levels**2)))
        stack[j] = halves[1] @ bath @ halves[0]
        cycle = stack[j] @ cycle
    stack[-1] = cycle
    return stack


s = case_cycle_stack()
""",
            "call": "recover_periodic_qubit_states(s.copy(), 3)",
            "gold_call": "_oracle_recover_periodic_qubit_states(s.copy(), 3)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "trace = _trace_row(2); reset = np.outer(trace/6,trace)\n"
            "s = np.repeat(reset[None,:,:],5,axis=0)\n",
            "call": "recover_periodic_qubit_states(s.copy(), 2)",
            "gold_call": "_oracle_recover_periodic_qubit_states(s.copy(), 2)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "trace = _trace_row(2); pure = np.zeros(36); pure[0] = 1\n"
            "s = np.repeat(np.outer(pure,trace)[None,:,:],5,axis=0)\n",
            "call": "recover_periodic_qubit_states(s.copy(), 2)",
            "gold_call": "_oracle_recover_periodic_qubit_states(s.copy(), 2)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "s = np.repeat(np.eye(36)[None,:,:],5,axis=0)\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: recover_periodic_qubit_states(s.copy(), 2))",
            "gold_call": "_raises(lambda: _oracle_recover_periodic_qubit_states(s.copy(), 2))",
            "tol": 1e-07,
        },
    ]
