"""
Computes the nonlocal deformation gradient of every point from the shape-function derivatives of step 3 and from it the bond-associated (quadrature-point) deformation gradient of every bond pair for a given displacement field on the families of step 1. Validates the bond-level construction of the framework the source builds on, which reproduces homogeneous deformations exactly; the returned (P, 3, 3) array feeds the stress evaluation of step 5 inside every internal-force evaluation of step 7. Deliberately excluded: damage and constitutive response.

The nonlocal deformation gradient of a point is built from the family displacement differences and the shape-function derivatives of step 3. Plain correspondence is polluted by zero-energy modes; the bond-associated (quadrature-point) formulation evaluates a separate deformation gradient on every bond, built from the endpoint nonlocal gradients and the bond's own kinematics, and reproduces homogeneous deformations exactly even near boundaries. The exact bond-level construction must be taken from the bond-associated correspondence framework that the source builds on.

Returns
-------
np.ndarray: (P,3,3) bond-associated deformation gradient of every bond pair (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bond_deformation_gradients(fam: dict, ops: dict, u: np.ndarray) -> np.ndarray:
    """Bond-associated deformation gradient of every bond pair.

    From the bond families of step 1, the kinematic operators of step 3
    and a displacement field u, computes the nonlocal deformation gradient
    of every point and from it the bond-associated (quadrature-point)
    deformation gradient of every bond pair, exactly as the bond-associated
    correspondence framework the source builds on prescribes; the
    construction must be taken from that framework. It reproduces
    homogeneous deformations exactly. The returned gradients are consumed
    by the stress evaluation of step 5 inside every internal-force
    evaluation of step 7.

    Parameters
    ----------
    fam : dict
        Bond-family dictionary returned by build_families (step 1).
    ops : dict
        Kinematic operators returned by kinematic_operators (step 3) for
        the current bond phase-field state.
    u : np.ndarray
        (N, 3) displacement of every point in m.

    Returns
    -------
    np.ndarray
        (P, 3, 3) array holding the bond-associated deformation gradient of
        every bond pair (dimensionless), in the pair order of fam['pk'] /
        fam['pn'].

    Raises
    ------
    ValueError
        If u is not an (N, 3) array.
    """
    return np.zeros((0, 3, 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bond_deformation_gradients(fam, ops, u):
    pk = fam['pk']; pn = fam['pn']; dX = fam['dX']; r = fam['r']
    N = fam['X'].shape[0]
    u = np.asarray(u, dtype=float)
    if u.shape != (N, 3):
        raise ValueError("u must be an (N, 3) displacement array")
    dphi_kn = ops['dphi_kn']; dphi_nk = ops['dphi_nk']
    dU = u[pn] - u[pk]
    Fb = np.tile(np.eye(3), (N, 1, 1))
    np.add.at(Fb, pk, dU[:, :, None]*dphi_kn[:, None, :])
    np.add.at(Fb, pn, (-dU)[:, :, None]*dphi_nk[:, None, :])
    Fav = 0.5*(Fb[pk] + Fb[pn])
    corr = (dX + dU) - np.einsum('pij,pj->pi', Fav, dX)
    Ft = Fav + corr[:, :, None]*dX[:, None, :]/(r**2)[:, None, None]
    return Ft

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\ndef _bond_norm(build, kin, grad, A, curv, i0, j0, k0, di, dj, dk):\n    fam = build(16, 10, 2, 1.0e-3, 2.015e-3)\n    X = fam['X']\n    u = X @ A.T + curv*np.stack([X[:, 1]**2, X[:, 0]**2, np.zeros(len(X))], axis=1)\n    ops = kin(fam, np.zeros(len(fam['pk'])), 0.95)\n    F = grad(fam, ops, u)\n    ijk = fam['ijk']\n    a = int(np.flatnonzero((ijk[:, 0] == i0) & (ijk[:, 1] == j0) & (ijk[:, 2] == k0))[0])\n    b = int(np.flatnonzero((ijk[:, 0] == i0+di) & (ijk[:, 1] == j0+dj) & (ijk[:, 2] == k0+dk))[0])\n    m = ((fam['pk'] == a) & (fam['pn'] == b)) | ((fam['pk'] == b) & (fam['pn'] == a))\n    return float(np.linalg.norm(F[m][0], 'fro'))"
    return [
        {"name": "normal_mixed_affine_plus_quadratic_field",
         "setup": setup + "\nA = np.array([[2.0e-3, 1.0e-3, 0.0], [-5.0e-4, 3.0e-3, 0.0], [0.0, 0.0, -1.0e-3]])",
         "call": "_bond_norm(build_families, kinematic_operators, bond_deformation_gradients, A, 20.0, 8, 5, 0, 1, 1, 1)",
         "gold_call": "_bond_norm(_oracle_build_families, _oracle_kinematic_operators, _oracle_bond_deformation_gradients, A, 20.0, 8, 5, 0, 1, 1, 1)"},
        {"name": "boundary_rigid_rotation_exactness",
         "setup": setup + "\nth = 0.3\nA = np.array([[np.cos(th) - 1.0, -np.sin(th), 0.0], [np.sin(th), np.cos(th) - 1.0, 0.0], [0.0, 0.0, 0.0]])",
         "call": "_bond_norm(build_families, kinematic_operators, bond_deformation_gradients, A, 0.0, 8, 5, 0, 1, 0, 0)",
         "gold_call": "_bond_norm(_oracle_build_families, _oracle_kinematic_operators, _oracle_bond_deformation_gradients, A, 0.0, 8, 5, 0, 1, 0, 0)"},
        {"name": "edge_pure_quadratic_field_axial_bond",
         "setup": setup + "\nA = np.zeros((3, 3))",
         "call": "_bond_norm(build_families, kinematic_operators, bond_deformation_gradients, A, 60.0, 2, 2, 1, 0, 1, 0)",
         "gold_call": "_bond_norm(_oracle_build_families, _oracle_kinematic_operators, _oracle_bond_deformation_gradients, A, 60.0, 2, 2, 1, 0, 1, 0)"},
    ]
