"""
Build the uniform bath generator and the undriven collective generator.

The qubits occupy the symmetric triplet sector, with

$J_x=(|0\rangle\langle1|+|1\rangle\langle0|+|1\rangle\langle2|+|2\rangle\langle1|)/\sqrt{2}$

and $J_z=\operatorname{diag}(1,0,-1)$. The auxiliary oscillator has exactly

$N$ states, $a_{n-1,n}=\sqrt{n}$, and vacuum damping.



$$

H_B=\omega_b I_3\otimes a^\dagger a+gJ_z\otimes(a+a^\dagger),\qquad

\mathcal{L}_B(R)=-i[H_B,R]+\kappa\left[cRc^\dagger-\frac{1}{2}\{c^\dagger c,R\}\right],

\quad c=I_3\otimes a.

$$



The static system term is $\Omega J_x$. The ordered joint Hilbert basis

is $|s,n\rangle$ with $s=0,1,2$ first and $n=0,\ldots,N-1$ second.

Grouped Liouville coordinates are

$x_{N^2(3s+t)+Ne+f}=R_{Ns+e,Nt+f}$.

Keeping all auxiliary indices gives dimension $D=9N^2$.

Returns
-------
Complex $(2,D,D)$ array in grouped Liouville order; slice zero is $\mathcal{L}_B$ and slice one is $\mathcal{L}_0=\mathcal{L}_B-i[\Omega J_x\otimes I_N,\cdot]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_collective_generators(
    levels: int, omega: float, bath_frequency: float, coupling: float, decay: float
) -> "np.ndarray":
    r"""Build the uniform bath generator and the undriven collective generator.

    Parameters
    ----------
    levels : int
        Oscillator cutoff $N\ge2$.
    omega : float
        Finite positive static drive scale $\Omega$ in inverse time units.
    bath_frequency : float
        Finite positive oscillator frequency $\omega_b$ in inverse time units.
    coupling : float
        Finite nonnegative collective coupling $g$ in inverse time units.
    decay : float
        Finite positive oscillator damping rate $\kappa$ in inverse time units.

    Returns
    -------
    generators : np.ndarray
        Complex $(2,D,D)$ array in grouped Liouville order; slice zero is
        $\mathcal{L}_B$ and slice one is $\mathcal{L}_0=\mathcal{L}_B-i[\Omega J_x\otimes I_N,\cdot]$.

    Raises
    ------
    ValueError
        If levels is not an integer at least two, parameters are nonreal or
        nonfinite, a frequency or decay is nonpositive, or coupling is negative.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _integer(value, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError("invalid integer parameter")
    return int(value)


def _array(value, shape=None, real=False):
    result = np.asarray(value, dtype=complex)
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        raise ValueError("invalid array shape or nonfinite entries")
    if real:
        if np.any(result.imag != 0):
            raise ValueError("real data required")
        return result.real
    return result


def _scalar(value):
    return float(_array(value, (), real=True))


def _spin_matrices():
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2), np.diag([1, 0, -1])


def _joint_vector(matrix, levels):
    return matrix.reshape(3, levels, 3, levels).transpose(0, 2, 1, 3).reshape(-1)


def _joint_matrix(vector, levels):
    return (
        vector.reshape(3, 3, levels, levels)
        .transpose(0, 2, 1, 3)
        .reshape(3 * levels, 3 * levels)
    )


def _trace_row(levels):
    return np.kron(np.eye(3).reshape(-1), np.eye(levels).reshape(-1))


def _reduce_qubits(vector, levels):
    joint = _joint_matrix(vector, levels).reshape(3, levels, 3, levels)
    triplet = np.trace(joint, axis1=1, axis2=3)
    embedding = np.array(
        [[1, 0, 0], [0, 1 / np.sqrt(2), 0], [0, 1 / np.sqrt(2), 0], [0, 0, 1]]
    )
    return embedding @ triplet @ embedding.T


def _oracle_build_collective_generators(
    levels: int, omega: float, bath_frequency: float, coupling: float, decay: float
) -> "np.ndarray":
    """Retain the oscillator while restricting only the qubit symmetry sector."""
    levels = _integer(levels, 2)
    omega, bath_frequency, coupling, decay = map(
        _scalar, (omega, bath_frequency, coupling, decay)
    )
    if min(omega, bath_frequency, decay) <= 0 or coupling < 0:
        raise ValueError(
            "positive frequencies and decay, nonnegative coupling required"
        )
    jx, jz = _spin_matrices()
    annihilation = np.diag(np.sqrt(np.arange(1, levels)), 1)
    dimension = 3 * levels
    identity = np.eye(dimension)
    bath_h = bath_frequency * np.kron(np.eye(3), annihilation.T @ annihilation)
    bath_h += coupling * np.kron(jz, annihilation + annihilation.T)
    jump = np.kron(np.eye(3), annihilation)
    number = jump.T @ jump
    bath_l = -1j * (np.kron(bath_h, identity) - np.kron(identity, bath_h.T))
    bath_l += decay * (
        np.kron(jump, jump)
        - 0.5 * (np.kron(number, identity) + np.kron(identity, number.T))
    )
    system_h = omega * np.kron(jx, np.eye(levels))
    system_l = -1j * (np.kron(system_h, identity) - np.kron(identity, system_h.T))
    permutation = (
        np.arange(dimension**2)
        .reshape(3, levels, 3, levels)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    grouped = np.ix_(permutation, permutation)
    return np.array([bath_l[grouped], (bath_l + system_l)[grouped]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test collective generators and the oscillator cutoff bound."""
    return [
        {
            "setup": "import numpy as np\n",
            "call": "build_collective_generators(4, 1.0, 1.8, 0.7, 0.5)",
            "gold_call": "_oracle_build_collective_generators(4, 1.0, 1.8, 0.7, 0.5)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n",
            "call": "build_collective_generators(2, 1.0, 1.3, 0.0, 0.4)",
            "gold_call": "_oracle_build_collective_generators(2, 1.0, 1.3, 0.0, 0.4)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n",
            "call": "build_collective_generators(3, 0.6, 2.1, 1.2, 0.07)",
            "gold_call": "_oracle_build_collective_generators(3, 0.6, 2.1, 1.2, 0.07)",
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
            "call": "_raises(lambda: build_collective_generators(1, 1.0, 1.8, 0.7, 0.5))",
            "gold_call": "_raises(lambda: _oracle_build_collective_generators(1, 1.0, 1.8, 0.7, 0.5))",
            "tol": 1e-07,
        },
    ]
