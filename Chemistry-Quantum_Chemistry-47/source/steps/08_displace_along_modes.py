"""
Generate a displaced chain geometry from a normal-mode sample on a quadratic energy shell.

Training geometries are sampled in the normal-mode coordinates of each molecule. In the local quadratic approximation the energy above equilibrium is half the sum over modes of the squared frequency times the squared mode displacement, so a fixed excess energy picks out a shell in the coordinates obtained by scaling each mode displacement with its frequency. A sample is given by the excess energy and a direction in that scaled space; the mode displacements are converted back to bond-length changes with the mode vectors and the atom positions are rebuilt from the bond lengths.

Returns
-------
np.ndarray, shape (K + 1,) float atom positions with the first atom at 0 and consecutive atoms separated by the displaced bond lengths
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def displace_along_modes(eq_bonds: "np.ndarray", modes: "np.ndarray", omegas: "np.ndarray", eps: float, direction: "np.ndarray") -> "np.ndarray":
    """Return atom positions displaced along normal modes to quadratic energy eps.

    Parameters
    ----------
    eq_bonds : np.ndarray
        Shape (K,) equilibrium bond lengths between consecutive atoms of a
        chain of K + 1 atoms.
    modes : np.ndarray
        Shape (K, K) array whose row k is the unit normal-mode vector k in
        bond-length coordinates.
    omegas : np.ndarray
        Shape (K,) positive harmonic frequencies of the modes.
    eps : float
        Non-negative excess energy of the sample in the local quadratic model.
    direction : np.ndarray
        Shape (K,) nonzero direction of the sample in the coordinates obtained
        by scaling each mode displacement with its frequency; only its
        orientation matters.

    Returns
    -------
    positions : np.ndarray
        Shape (K + 1,) float positions with the first atom at 0 and consecutive
        atoms separated by the displaced bond lengths, which are the equilibrium
        bonds plus each mode vector times its mode displacement, where the mode
        displacements are the point on the quadratic energy shell at eps that
        lies along the given direction in the frequency-scaled coordinates.

    Raises
    ------
    ValueError
        If eps is negative, the direction has zero length, or the array shapes
        are inconsistent.
    """
    return positions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_displace_along_modes(eq_bonds: "np.ndarray", modes: "np.ndarray", omegas: "np.ndarray", eps: float, direction: "np.ndarray") -> "np.ndarray":
    b0 = np.asarray(eq_bonds, dtype=float)
    V = np.asarray(modes, dtype=float)
    w = np.asarray(omegas, dtype=float)
    u = np.asarray(direction, dtype=float)
    k = b0.shape[0]
    if V.shape != (k, k) or w.shape != (k,) or u.shape != (k,):
        raise ValueError("modes must be (K, K) and omegas, direction (K,)")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    norm = np.linalg.norm(u)
    if norm == 0.0:
        raise ValueError("direction must be nonzero")
    radius = np.sqrt(2.0 * float(eps))
    q = (u / norm) * radius / w
    bonds = b0 + V.T @ q
    return np.concatenate([[0.0], np.cumsum(bonds)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    chain = ("eq_bonds = np.array([1.0399, 1.7090, 1.0338])\n"
             "omegas = np.array([1.0465, 2.1509, 2.2333])\n"
             "modes = np.array([[-0.0152, 0.9990, -0.0410], [-0.8270, -0.0356, -0.5611], [-0.5620, 0.0253, 0.8267]])\n")
    return [
        # three-mode chain with an unnormalized mixed direction
        {
            "setup": "import numpy as np\nimport copy\n" + chain + "eps = 0.012\ndirection = np.array([2.0, -1.0, 0.5])\n",
            "call": "displace_along_modes(*copy.deepcopy((eq_bonds, modes, omegas, eps, direction)))",
            "gold_call": "_oracle_displace_along_modes(eq_bonds, modes, omegas, eps, direction)",
        },
        # diatomic compressed along its single mode
        {
            "setup": "import numpy as np\nimport copy\neq_bonds = np.array([1.2752])\nomegas = np.array([1.5935])\nmodes = np.array([[1.0]])\neps = 0.019\ndirection = np.array([-1.0])\n",
            "call": "displace_along_modes(*copy.deepcopy((eq_bonds, modes, omegas, eps, direction)))",
            "gold_call": "_oracle_displace_along_modes(eq_bonds, modes, omegas, eps, direction)",
        },
        # zero excess energy returns the equilibrium chain
        {
            "setup": "import numpy as np\nimport copy\n" + chain + "eps = 0.0\ndirection = np.array([0.0, -3.0, -4.0])\n",
            "call": "displace_along_modes(*copy.deepcopy((eq_bonds, modes, omegas, eps, direction)))",
            "gold_call": "_oracle_displace_along_modes(eq_bonds, modes, omegas, eps, direction)",
        },
        # direction along the softest mode only, large scaled direction
        {
            "setup": "import numpy as np\nimport copy\n" + chain + "eps = 0.018\ndirection = np.array([-50.0, 0.0, 0.0])\n",
            "call": "displace_along_modes(*copy.deepcopy((eq_bonds, modes, omegas, eps, direction)))",
            "gold_call": "_oracle_displace_along_modes(eq_bonds, modes, omegas, eps, direction)",
        },
        # negative excess energy
        {
            "setup": "import numpy as np\nimport copy\n" + chain + """eps = -0.01
direction = np.array([1.0, 1.0, 1.0])
def run_model():
    try:
        displace_along_modes(eq_bonds, modes, omegas, eps, direction)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_displace_along_modes(eq_bonds, modes, omegas, eps, direction)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # zero direction vector
        {
            "setup": "import numpy as np\nimport copy\n" + chain + """eps = 0.01
direction = np.zeros(3)
def run_model():
    try:
        displace_along_modes(eq_bonds, modes, omegas, eps, direction)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_displace_along_modes(eq_bonds, modes, omegas, eps, direction)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
