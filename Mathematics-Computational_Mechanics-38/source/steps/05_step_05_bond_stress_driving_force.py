"""
Evaluates the Saint Venant-Kirchhoff response of one or many bond deformation gradients and returns both the undamaged first Piola-Kirchhoff stress and the benchmark's stress-based crack driving force in the form the source adopts. Validates the constitutive law, the choice of stress measure for the driving force, and its scaling. Both outputs are consumed by every internal-force evaluation of step 7. Deliberately excluded: mesh, families, and evolution - this is a pure pointwise constitutive functional.

Under Saint Venant-Kirchhoff the second Piola-Kirchhoff stress is linear in the Green strain and the undamaged first Piola-Kirchhoff stress is P0 = F S. Stress-based damage criteria start a crack where a suitable measure of tensile stress reaches a critical value tied to Gc, while compressive states should not drive damage; the benchmark selects the maximum-principal-stress-based variant offered by the source. Which bond-wise stress tensor is used, which part of it, and how it is scaled to a driving force in J/m^3 must be taken from the source.

Returns
-------
dict: 'P0' undamaged first Piola-Kirchhoff stress (Pa) and 'Y' stress-based crack driving force (J/m^3), single bond or batch
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bond_stress_and_driving_force(F: np.ndarray, E: float, nu: float) -> dict:
    """Undamaged bond stress and stress-based crack driving force.

    Evaluates the Saint Venant-Kirchhoff response of one or many bond
    deformation gradients: the undamaged first Piola-Kirchhoff stress
    P0 = F S with S = lambda tr(E_G) I + 2 mu E_G, E_G = (F^T F - I)/2
    (the benchmark material), and the benchmark's stress-based crack
    driving force in the exact form the source adopts for the
    maximum-principal-stress-based variant that the benchmark selects.
    Which stress tensor is used, which part of it, and how it is scaled
    to a driving force must be taken from the source. Both outputs are
    consumed by every internal-force evaluation of step 7.

    Parameters
    ----------
    F : np.ndarray
        Bond deformation gradient(s), shape (3, 3) for a single bond or
        (P, 3, 3) for a batch (dimensionless), with positive determinant.
    E : float
        Young's modulus in Pa (> 0).
    nu : float
        Poisson's ratio, in (-1, 0.5).

    Returns
    -------
    dict
        'P0' : undamaged first Piola-Kirchhoff stress in Pa, shape (3, 3)
            for a single bond or (P, 3, 3) for a batch;
        'Y' : crack driving force in J/m^3, a float for a single bond or a
            (P,) array for a batch.

    Raises
    ------
    ValueError
        If F does not end in a 3x3 block, any det(F) <= 0, E <= 0, or nu
        is outside (-1, 0.5).
    """
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bond_stress_and_driving_force(F, E, nu):
    F = np.asarray(F, dtype=float)
    if F.shape[-2:] != (3, 3):
        raise ValueError("F must have shape (3, 3) or (P, 3, 3)")
    if not (E > 0.0 and -1.0 < nu < 0.5):
        raise ValueError("require E > 0 and -1 < nu < 0.5")
    single = (F.ndim == 2)
    Ft = F.reshape(-1, 3, 3)
    J = np.linalg.det(Ft)
    if np.any(J <= 0.0):
        raise ValueError("deformation gradient must have positive determinant")
    lam = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    mu = E/(2.0*(1.0 + nu))
    Ftt = np.swapaxes(Ft, -1, -2)
    Egr = 0.5*(Ftt @ Ft - np.eye(3))
    trE = np.trace(Egr, axis1=-2, axis2=-1)[..., None, None]
    S = lam*trE*np.eye(3) + 2.0*mu*Egr
    P0 = Ft @ S
    sig = (P0 @ Ftt)/J[..., None, None]
    sig = 0.5*(sig + np.swapaxes(sig, -1, -2))
    s1 = np.linalg.eigvalsh(sig)[..., -1]
    Y = np.maximum(s1, 0.0)**2/(2.0*E)
    if single:
        return {'P0': P0[0], 'Y': float(Y[0])}
    return {'P0': P0, 'Y': Y}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"name": "normal_general_deformation",
         "setup": "import numpy as np\nF = np.array([[1.0012, 0.0008, -0.0002], [0.0005, 0.9991, 0.0004], [-0.0001, 0.0003, 1.0006]])",
         "call": "bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']",
         "gold_call": "_oracle_bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']"},
        {"name": "normal_general_deformation_undamaged_stress_component",
         "setup": "import numpy as np\nF = np.array([[1.0012, 0.0008, -0.0002], [0.0005, 0.9991, 0.0004], [-0.0001, 0.0003, 1.0006]])",
         "call": "float(bond_stress_and_driving_force(F, 3.2e10, 0.25)['P0'][0, 1])",
         "gold_call": "float(_oracle_bond_stress_and_driving_force(F, 3.2e10, 0.25)['P0'][0, 1])"},
        {"name": "boundary_uniaxial_stretch_closed_form",
         "setup": "import numpy as np\nF = np.diag([1.002, 1.0, 1.0])",
         "call": "bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']",
         "gold_call": "_oracle_bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']"},
        {"name": "edge_pure_contraction_exact_zero",
         "setup": "import numpy as np\nF = np.diag([0.997, 1.0, 1.0])",
         "call": "bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']",
         "gold_call": "_oracle_bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']"},
        {"name": "edge_batched_bonds_sum_of_driving_forces",
         "setup": "import numpy as np\nF = np.stack([np.diag([1.002, 1.0, 1.0]), np.diag([0.997, 1.0, 1.0]), np.array([[1.0012, 0.0008, -0.0002], [0.0005, 0.9991, 0.0004], [-0.0001, 0.0003, 1.0006]])])",
         "call": "float(np.sum(bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']))",
         "gold_call": "float(np.sum(_oracle_bond_stress_and_driving_force(F, 3.2e10, 0.25)['Y']))"},
    ]
