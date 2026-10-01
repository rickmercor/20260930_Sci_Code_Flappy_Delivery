"""
Harmonic frequencies and normal modes of a chain molecule in bond-length coordinates.

The curvature matrix of the SCF energy at the equilibrium geometry is taken in bond-length coordinates with unit masses, so its eigenvalues are the squared harmonic frequencies and its eigenvectors are the normal modes. The curvature is obtained numerically by central differences of the analytic bond-length gradient with a fixed step, symmetrized afterwards, and the modes are ordered and signed by a fixed convention so that sampling directions expressed in mode coordinates are unambiguous.

Returns
-------
np.ndarray, shape (K + 1, K) float array with the K frequencies in row 0 and one unit mode vector per following row
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normal_modes(types: "np.ndarray", eq_bonds: "np.ndarray", params: dict, step: float = 1e-4) -> "np.ndarray":
    """Return the harmonic frequencies and unit normal-mode vectors.

    Parameters
    ----------
    types : np.ndarray
        Shape (K + 1,) element types of the chain, 0 or 1; K + 1 is even.
    eq_bonds : np.ndarray
        Shape (K,) equilibrium bond lengths.
    params : dict
        Model constants.
    step : float
        Finite-difference step for the curvature matrix.

    Returns
    -------
    modes : np.ndarray
        Shape (K + 1, K) float array. Row 0 holds the K harmonic frequencies
        in increasing order, each the square root of an eigenvalue of the
        curvature matrix. Row k (1-based) holds the unit-length eigenvector of
        the k-th smallest eigenvalue in bond-length coordinates, with its sign
        chosen so that the component of largest magnitude is positive. The
        curvature matrix entry (i, j) is the central difference of bond-gradient
        component i between bond j displaced by +step and by -step, divided by
        twice the step, and the matrix is symmetrized as half the sum of itself
        and its transpose before diagonalization. All eigenvalues are positive
        at an equilibrium geometry.

    Raises
    ------
    ValueError
        If the atom count is odd, eq_bonds does not have K entries, or any
        eigenvalue of the curvature matrix is not positive.
    """
    return modes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_normal_modes(types: "np.ndarray", eq_bonds: "np.ndarray", params: dict, step: float = 1e-4) -> "np.ndarray":
    typ = np.asarray(types, dtype=int)
    bonds = np.asarray(eq_bonds, dtype=float)
    if typ.shape[0] % 2:
        raise ValueError("the model needs an even number of atoms")
    if bonds.shape != (typ.shape[0] - 1,):
        raise ValueError("eq_bonds must hold one length per consecutive pair")
    hess = _bond_hessian_by_differences(bonds, typ, params, float(step))
    values, vectors = np.linalg.eigh(hess)
    if np.any(values <= 0.0):
        raise ValueError("the curvature matrix is not positive definite")
    modes = vectors.T.copy()
    for row in modes:
        if row[np.argmax(np.abs(row))] < 0.0:
            row *= -1.0
    return np.vstack([np.sqrt(values), modes])

def _bond_hessian_by_differences(bonds: "np.ndarray", types: "np.ndarray", params: dict, step: float) -> "np.ndarray":
    bonds = np.asarray(bonds, dtype=float)
    k = bonds.shape[0]
    hess = np.zeros((k, k))
    for col in range(k):
        shift = np.zeros(k)
        shift[col] = step
        hess[:, col] = (_bond_gradient(bonds + shift, types, params)
                        - _bond_gradient(bonds - shift, types, params)) / (2.0 * step)
    return 0.5 * (hess + hess.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # three modes of the chain at its equilibrium
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 1, 0, 1])\neq_bonds = equilibrium_bonds(types, np.array([1.2, 1.2, 1.2]), params)\n",
            "call": "normal_modes(*copy.deepcopy((types, eq_bonds, params)))",
            "gold_call": "_oracle_normal_modes(types, eq_bonds, params)",
            "tol": 1e-7,
        },
        # single mode of a diatomic
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 1])\neq_bonds = equilibrium_bonds(types, np.array([1.2]), params)\n",
            "call": "normal_modes(*copy.deepcopy((types, eq_bonds, params)))",
            "gold_call": "_oracle_normal_modes(types, eq_bonds, params)",
            "tol": 1e-7,
        },
        # a smaller step must give the same curvature within tolerance
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([1, 1])\neq_bonds = equilibrium_bonds(types, np.array([1.2]), params)\n",
            "call": "normal_modes(*copy.deepcopy((types, eq_bonds, params, 2e-5)))",
            "gold_call": "_oracle_normal_modes(types, eq_bonds, params, 2e-5)",
            "tol": 1e-7,
        },
        # a chain whose atom order is types 1,0,1,0
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([1, 0, 1, 0])\neq_bonds = equilibrium_bonds(types, np.array([1.2, 1.2, 1.2]), params)\n",
            "call": "normal_modes(*copy.deepcopy((types, eq_bonds, params)))",
            "gold_call": "_oracle_normal_modes(types, eq_bonds, params)",
            "tol": 1e-7,
        },
        # wrong number of bonds is rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + """types = np.array([0, 1, 0, 1])
eq_bonds = np.array([1.04, 1.71])
def run_model():
    try:
        normal_modes(types, eq_bonds, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_normal_modes(types, eq_bonds, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
