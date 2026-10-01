"""
Set up the overlap, core Hamiltonian and site-site Coulomb matrices of the site model.

Every atom carries one basis function, one electron and a unit core charge, and belongs to one of two element types that differ in on-site energy and on-site repulsion. The overlap between two different sites falls off as a Gaussian in their separation. The site-site Coulomb interaction uses the Ohno form, which equals the on-site repulsion when the two sites coincide and turns into the bare Coulomb law at large separation. A site's diagonal core-Hamiltonian entry is its on-site energy reduced by the screened attraction to every other core, and the off-diagonal entries follow the Wolfsberg-Helmholz rule of extended Hueckel theory.

Returns
-------
np.ndarray, shape (3, n, n) float array holding the overlap, the core Hamiltonian and the Ohno matrix in this order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_model_matrices(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    """Return the overlap, core-Hamiltonian and Coulomb matrices stacked together.

    Parameters
    ----------
    positions : np.ndarray
        Shape (n,), atom positions along the line.
    types : np.ndarray
        Shape (n,), element type of each atom, 0 or 1.
    params : dict
        Model constants: ``onsite_energy`` (two floats, one per type),
        ``onsite_repulsion`` (two floats, one per type), ``kappa`` (the
        Wolfsberg-Helmholz factor) and ``sigma`` (the standard deviation of
        the Gaussian overlap). Other keys are ignored.

    Returns
    -------
    matrices : np.ndarray
        Shape (3, n, n) float array holding, in this order, the overlap S (unit
        diagonal, Gaussian off-diagonal with standard deviation sigma in the
        separation), the core Hamiltonian H (diagonal: on-site energy minus the
        sum of the Ohno interactions with all other sites; off-diagonal:
        kappa times the overlap times the mean of the two diagonal entries) and
        the Ohno interaction matrix Gamma (on-site repulsion on the diagonal;
        for distinct sites the inverse of the root of the squared separation
        plus the squared Ohno length, the Ohno length being twice the inverse
        of the summed on-site repulsions of the pair).

    Raises
    ------
    ValueError
        If positions and types are not one-dimensional arrays of the same
        length, or if a type is not 0 or 1.
    """
    return matrices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_model_matrices(positions: "np.ndarray", types: "np.ndarray", params: dict) -> "np.ndarray":
    pos = np.asarray(positions, dtype=float)
    typ = np.asarray(types, dtype=int)
    if pos.ndim != 1 or typ.ndim != 1 or pos.shape[0] != typ.shape[0]:
        raise ValueError("positions and types must be 1-D arrays of the same length")
    if typ.size and (typ.min() < 0 or typ.max() > 1):
        raise ValueError("types must be 0 or 1")
    alpha = np.asarray(params["onsite_energy"], dtype=float)
    u_onsite = np.asarray(params["onsite_repulsion"], dtype=float)
    kappa = float(params["kappa"])
    sigma = float(params["sigma"])
    n = pos.shape[0]
    R = np.abs(pos[:, None] - pos[None, :])
    off = ~np.eye(n, dtype=bool)
    S = np.where(off, np.exp(-R ** 2 / (2.0 * sigma ** 2)), 1.0)
    a = 2.0 / (u_onsite[typ][:, None] + u_onsite[typ][None, :])
    gamma = 1.0 / np.sqrt(R ** 2 + a ** 2)
    hdiag = alpha[typ] - np.sum(gamma * off, axis=1)
    H = np.diag(hdiag) + kappa * S * off * 0.5 * (hdiag[:, None] + hdiag[None, :])
    return np.stack([S, H, gamma])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Test cases for this step."""
    params = ("params = {'onsite_energy': [-1.0, -1.6], 'onsite_repulsion': [0.9, 1.3], "
              "'kappa': 1.75, 'sigma': 1.1, 'core_repulsion_strength': 6.0, 'core_repulsion_range': 0.45}\n")
    return [
        # heteronuclear diatomic
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.1659])\ntypes = np.array([0, 1])\n",
            "call": "build_model_matrices(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_build_model_matrices(positions, types, params)",
        },
        # four-atom chain with unequal bonds
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 1.0399, 2.7489, 3.7827])\ntypes = np.array([0, 1, 0, 1])\n",
            "call": "build_model_matrices(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_build_model_matrices(positions, types, params)",
        },
        # a single atom has no neighbours
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([2.5])\ntypes = np.array([1])\n",
            "call": "build_model_matrices(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_build_model_matrices(positions, types, params)",
        },
        # far-apart atoms: overlap and coupling vanish, Coulomb tail survives
        {
            "setup": "import numpy as np\nimport copy\n" + params + "positions = np.array([0.0, 12.0, 30.0])\ntypes = np.array([1, 0, 0])\n",
            "call": "build_model_matrices(*copy.deepcopy((positions, types, params)))",
            "gold_call": "_oracle_build_model_matrices(positions, types, params)",
        },
        # mismatched lengths are rejected
        {
            "setup": "import numpy as np\nimport copy\n" + params + """positions = np.array([0.0, 1.2, 2.4])
types = np.array([0, 1])
def run_model():
    try:
        build_model_matrices(positions, types, params)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_build_model_matrices(positions, types, params)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
