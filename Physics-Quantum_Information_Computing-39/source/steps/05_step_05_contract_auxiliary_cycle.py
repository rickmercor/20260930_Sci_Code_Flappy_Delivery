"""
Dress the uniform auxiliary map and contract the chronological cycle.

The bath generator excludes the local system Hamiltonian, so its

uniform transfer map is $q=e^{\Delta t\mathcal{L}_B}$.

Local channels act only on the physical triplet Liouville index and

leave the $N^2$ auxiliary indices intact:



$$

Q_j=(L_{j,\mathrm{late}}\otimes I_{N^2})q

(L_{j,\mathrm{early}}\otimes I_{N^2}),\qquad

F=Q_{M-1}\cdots Q_0.

$$



Both local factors advance the state forward in time.

The same undressed auxiliary map is used at every phase of a given period;

its time step changes when the drive frequency or resolution changes.

Returns
-------
Complex $(M+1,D,D)$ array; the first $M$ slices are the chronological $Q_j$ and the final slice is their cycle product $F$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contract_auxiliary_cycle(
    bath_generator: "np.ndarray", half_steps: "np.ndarray", dt: float, levels: int
) -> "np.ndarray":
    r"""Dress the uniform auxiliary map and contract the chronological cycle.

    Parameters
    ----------
    bath_generator : np.ndarray
        Finite complex $(D,D)$ uniform generator $\mathcal{L}_B$, $D=9N^2$.
    half_steps : np.ndarray
        Finite complex $(M,2,9,9)$ early/late local channels, $M\ge4$.
    dt : float
        Finite positive interval length used for the supplied half steps.
    levels : int
        Auxiliary cutoff $N\ge2$.

    Returns
    -------
    cycle_stack : np.ndarray
        Complex $(M+1,D,D)$ array; the first $M$ slices are the chronological
        $Q_j$ and the final slice is their cycle product $F$.

    Raises
    ------
    ValueError
        If integer bounds, shapes or finite-value checks fail, or dt is not positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_contract_auxiliary_cycle(
    bath_generator: "np.ndarray", half_steps: "np.ndarray", dt: float, levels: int
) -> "np.ndarray":
    """Preserve auxiliary memory through both dressing and period contraction."""
    levels = _integer(levels, 2)
    dimension = 9 * levels**2
    bath_generator = _array(bath_generator, (dimension, dimension))
    half_steps = _array(half_steps)
    dt = _scalar(dt)
    if (
        half_steps.ndim != 4
        or half_steps.shape[1:] != (2, 9, 9)
        or len(half_steps) < 4
        or dt <= 0
    ):
        raise ValueError("invalid half-step stack or nonpositive dt")
    count = len(half_steps)
    result = np.empty((count + 1, dimension, dimension), dtype=complex)
    bath = expm(dt * bath_generator)
    cycle = np.eye(dimension, dtype=complex)
    for j, (early, late) in enumerate(half_steps):
        result[j] = (
            np.kron(late, np.eye(levels**2)) @ bath @ np.kron(early, np.eye(levels**2))
        )
        cycle = result[j] @ cycle
    result[-1] = cycle
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cycle contraction and a nonpositive time step."""
    return [
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(3,1.0,1.8,0.7,0.5)[0]\n"
            "h = _oracle_integrate_local_half_steps(1.0,1.2,0.66,12,np.pi/7)\n",
            "call": "contract_auxiliary_cycle(l.copy(), h.copy(), 2*np.pi/(12*0.66), 3)",
            "gold_call": "_oracle_contract_auxiliary_cycle(l.copy(), h.copy(), 2*np.pi/(12*0.66), 3)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\nl = np.zeros((36,36)); h = np.tile(np.eye(9),(4,2,1,1))\n",
            "call": "contract_auxiliary_cycle(l.copy(), h.copy(), 0.1, 2)",
            "gold_call": "_oracle_contract_auxiliary_cycle(l.copy(), h.copy(), 0.1, 2)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = _oracle_build_collective_generators(2,0.8,1.4,0.5,0.3)[0]\n"
            "h = _oracle_integrate_local_half_steps(0.8,0.0,1.3,5,-0.4)\n",
            "call": "contract_auxiliary_cycle(l.copy(), h.copy(), 2*np.pi/(5*1.3), 2)",
            "gold_call": "_oracle_contract_auxiliary_cycle(l.copy(), h.copy(), 2*np.pi/(5*1.3), 2)",
            "tol": 1e-07,
        },
        {
            "setup": "import numpy as np\n"
            "l = np.zeros((36,36)); h = np.tile(np.eye(9),(4,2,1,1))\n"
            "\n"
            "def _raises(function):\n"
            "    try:\n"
            "        function()\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    return 0\n",
            "call": "_raises(lambda: contract_auxiliary_cycle(l.copy(), h.copy(), 0.0, 2))",
            "gold_call": "_raises(lambda: _oracle_contract_auxiliary_cycle(l.copy(), h.copy(), 0.0, "
            "2))",
            "tol": 1e-07,
        },
    ]
