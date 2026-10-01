"""
Equilibrium bond lengths of a chain molecule of the site model.

The molecule's geometry is described by the distances between consecutive atoms, with the first atom fixed at the origin. The equilibrium geometry is the minimum of the converged SCF total energy in these bond-length coordinates. Bond lengths must stay positive throughout the search, and starting guesses may lie far from the minimum on either side, where the energy surface is far from quadratic, so the search has to control its steps rather than trust a local model of the surface. The gradient and curvature with respect to bond lengths follow from the position derivatives, since lengthening one bond moves every atom beyond it by the same amount.

Returns
-------
np.ndarray, shape (K,) float bond lengths at the minimum of the converged SCF energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_bonds(types: "np.ndarray", start_bonds: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the bond lengths that minimize the converged SCF total energy.

    Parameters
    ----------
    types : np.ndarray
        Shape (K + 1,) element types of the chain, 0 or 1; K + 1 is even.
    start_bonds : np.ndarray
        Shape (K,) positive starting bond lengths, anywhere between roughly
        half and twice the equilibrium values. Every molecule of this task has
        exactly one minimum with all bonds positive, and it must be reached
        from any such start.
    params : dict
        Model constants.

    Returns
    -------
    bonds : np.ndarray
        Shape (K,) float bond lengths at which every component of the energy
        gradient with respect to the bond lengths has magnitude below 1e-10,
        all bonds being positive. The bond-length gradient is obtained from the
        position gradient: the derivative with respect to bond k is the sum of
        the position derivatives of all atoms that follow bond k.

    Raises
    ------
    ValueError
        If the atom count is odd, if start_bonds does not have K entries, or if
        any starting bond is not positive.
    """
    return bonds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize

def _oracle_equilibrium_bonds(types: "np.ndarray", start_bonds: "np.ndarray", params: dict) -> "np.ndarray":
    typ = np.asarray(types, dtype=int)
    b0 = np.asarray(start_bonds, dtype=float)
    if typ.shape[0] % 2:
        raise ValueError("the model needs an even number of atoms")
    if b0.shape != (typ.shape[0] - 1,) or np.any(b0 <= 0.0):
        raise ValueError("start_bonds must hold one positive length per consecutive pair")
    energy = lambda x: _oracle_scf_reference_energy(_bonds_to_positions(np.exp(x)), typ, params, 1e-12, 20000, 0.3)
    gradient = lambda x: _bond_gradient(np.exp(x), typ, params) * np.exp(x)
    result = minimize(energy, np.log(b0), jac=gradient, method="BFGS", options={"gtol": 1e-11, "maxiter": 500})
    bonds = np.exp(np.asarray(result.x, dtype=float))
    for _ in range(20):
        g = _bond_gradient(bonds, typ, params)
        if np.max(np.abs(g)) < 1e-13:
            break
        step = np.linalg.solve(_bond_hessian(bonds, typ, params), g)
        while np.any(bonds - step <= 0.0):
            step = 0.5 * step
        bonds = bonds - step
    return bonds

def _bonds_to_positions(bonds: "np.ndarray") -> "np.ndarray":
    return np.concatenate([[0.0], np.cumsum(np.asarray(bonds, dtype=float))])

def _bond_gradient(bonds: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    g = _oracle_scf_energy_gradient(_bonds_to_positions(bonds), types, params)
    return np.array([g[k + 1:].sum() for k in range(len(bonds))])

def _bond_hessian(bonds: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    hess = _oracle_scf_energy_hessian(_bonds_to_positions(bonds), types, params)
    k = len(bonds)
    return np.array([[hess[i + 1:, j + 1:].sum() for j in range(k)] for i in range(k)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # the four-atom chain from a uniform start
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 1, 0, 1])\nstart_bonds = np.array([1.2, 1.2, 1.2])\n",
            "call": "equilibrium_bonds(*copy.deepcopy((types, start_bonds, params)))",
            "gold_call": "_oracle_equilibrium_bonds(types, start_bonds, params)",
        },
        # stretched uniform start where a raw Newton step overshoots into negative bonds
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 1, 0, 1])\nstart_bonds = np.array([1.6, 1.6, 1.6])\n",
            "call": "equilibrium_bonds(*copy.deepcopy((types, start_bonds, params)))",
            "gold_call": "_oracle_equilibrium_bonds(types, start_bonds, params)",
        },
        # short outer bonds and a long central bond
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 1, 0, 1])\nstart_bonds = np.array([0.8, 2.6, 0.8])\n",
            "call": "equilibrium_bonds(*copy.deepcopy((types, start_bonds, params)))",
            "gold_call": "_oracle_equilibrium_bonds(types, start_bonds, params)",
        },
        # far stretched homonuclear start; the mirrored negative-bond solution is not acceptable
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 0])\nstart_bonds = np.array([2.8])\n",
            "call": "equilibrium_bonds(*copy.deepcopy((types, start_bonds, params)))",
            "gold_call": "_oracle_equilibrium_bonds(types, start_bonds, params)",
        },
        # compressed heteronuclear start
        {
            "setup": "import numpy as np\nimport copy\n" + params + "types = np.array([0, 1])\nstart_bonds = np.array([0.6])\n",
            "call": "equilibrium_bonds(*copy.deepcopy((types, start_bonds, params)))",
            "gold_call": "_oracle_equilibrium_bonds(types, start_bonds, params)",
        },
        # a non-positive starting bond is rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + """types = np.array([0, 1, 0, 1])
start_bonds = np.array([1.2, 0.0, 1.2])
def run_model():
    try:
        equilibrium_bonds(types, start_bonds, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_equilibrium_bonds(types, start_bonds, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
