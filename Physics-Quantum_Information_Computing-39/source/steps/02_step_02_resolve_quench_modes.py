"""
Resolve a normalized quench into stationary and weighted transient contributions.

For a diagonalizable undriven generator, write $\mathcal{L}_0 V=V\Lambda$

and resolve the initial joint state $x_{\mathrm{in}}$ through its biorthogonal

spectral coefficients. Each density contribution includes its quench weight.



$$

x(t)=\sum_n v_n c_n e^{\gamma_n t},\qquad c=V^{-1}x_{\mathrm{in}}.

$$



This form is invariant to the normalization and complex phase of each

right eigenvector. The unique stationary term has trace one, while every

transient contribution has zero trace. The supported domain has a simple

spectrum, one zero eigenvalue, and strictly decaying other modes.

Returns
-------
Complex $(D,D+1)$ array. Each row contains its rate in column zero and the weighted joint density contribution in the remaining columns. The stationary row comes first with rate exactly zero; transient rows are sorted by imaginary rate rounded to ten decimals, then real rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_quench_modes(
    generator: "np.ndarray", initial_state: "np.ndarray", levels: int
) -> "np.ndarray":
    r"""Resolve a normalized quench into stationary and weighted transient contributions.

    Parameters
    ----------
    generator : np.ndarray
        Finite complex $(D,D)$ undriven generator in grouped Liouville order,
        $D=9N^2$, with a simple attracting spectrum.
    initial_state : np.ndarray
        Finite complex $(D,)$ vector of a normalized positive joint density matrix.
    levels : int
        Oscillator cutoff $N\ge2$.

    Returns
    -------
    modes : np.ndarray
        Complex $(D,D+1)$ array. Each row contains its rate in column zero
        and the weighted joint density contribution in the remaining columns.
        The stationary row comes first with rate exactly zero; transient rows
        are sorted by imaginary rate rounded to ten decimals, then real rate.

    Raises
    ------
    ValueError
        If shapes or finiteness fail, the initial state fails density checks
        at 1e-8, the zero eigenvalue is not unique within 1e-9, another rate
        has real part at least -1e-10, eigenvalue separation is below 1e-8,
        or the right-eigenvector matrix has condition number above 1e10.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eig


def _physical_density(matrix, tolerance=1e-8):
    if (
        np.linalg.norm(matrix - matrix.conj().T) > tolerance
        or abs(np.trace(matrix) - 1) > tolerance
    ):
        raise ValueError("a Hermitian trace-one density matrix is required")
    values = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2)
    if values.min() < -tolerance:
        raise ValueError("density matrix is not positive")


def _oracle_resolve_quench_modes(
    generator: "np.ndarray", initial_state: "np.ndarray", levels: int
) -> "np.ndarray":
    """Weight every mode by its dual spectral projection of the quench."""
    levels = _integer(levels, 2)
    dimension = 9 * levels**2
    generator = _array(generator, (dimension, dimension))
    initial_state = _array(initial_state, (dimension,))
    _physical_density(_joint_matrix(initial_state, levels))
    rates, right = eig(generator)
    stationary = np.flatnonzero(np.abs(rates) < 1e-9)
    separation = np.abs(rates[:, None] - rates[None, :]) + np.eye(dimension)
    if len(stationary) != 1 or separation.min() < 1e-8 or np.linalg.cond(right) > 1e10:
        raise ValueError("a simple, well-resolved stationary spectrum is required")
    stationary = int(stationary[0])
    transient = [n for n in range(dimension) if n != stationary]
    if np.max(rates[transient].real) >= -1e-10:
        raise ValueError("all transient rates must decay")
    coefficients = np.linalg.solve(right, initial_state)
    weighted = (right * coefficients[None, :]).T
    transient.sort(
        key=lambda n: (round(float(rates[n].imag), 10), float(rates[n].real))
    )
    order = [stationary] + transient
    result = np.column_stack((rates[order], weighted[order]))
    result[0, 0] = 0
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test weighted quench modes and rejection of an unresolved stationary spectrum."""
    return [
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(4,1.0,1.8,0.7,0.5)[1]\n"
            "x = np.zeros(144); x[0] = 1\n",
            "call": "resolve_quench_modes(l.copy(), x.copy(), 4)",
            "gold_call": "_oracle_resolve_quench_modes(l.copy(), x.copy(), 4)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(2,1.0,1.8,0.7,0.5)[1]\n"
            "x = _joint_vector(np.eye(6)/6,2)\n",
            "call": "resolve_quench_modes(l.copy(), x.copy(), 2)",
            "gold_call": "_oracle_resolve_quench_modes(l.copy(), x.copy(), 2)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(3,0.8,1.4,0.5,0.3)[1]\n"
            "rho = np.zeros((9,9)); rho[-1,-1] = 1; x = _joint_vector(rho,3)\n",
            "call": "resolve_quench_modes(l.copy(), x.copy(), 3)",
            "gold_call": "_oracle_resolve_quench_modes(l.copy(), x.copy(), 3)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = np.zeros((36,36)); x = np.zeros(36); x[0] = 1\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: resolve_quench_modes(l.copy(), x.copy(), 2))",
            "gold_call": "_raises(lambda: _oracle_resolve_quench_modes(l.copy(), x.copy(), 2))",
            "tol": 1e-07,
        },
    ]
