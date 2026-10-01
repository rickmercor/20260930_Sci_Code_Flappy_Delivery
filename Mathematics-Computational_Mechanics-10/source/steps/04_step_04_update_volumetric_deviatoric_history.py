"""
Update one cell's tensile history from its reconstructed strain field.

For the volumetric-deviatoric split,



$$\psi_0^+(\boldsymbol{\varepsilon})=\frac{K}{2}[\operatorname{tr}(\boldsymbol{\varepsilon})]_+^2+\mu\,\boldsymbol{\varepsilon}^{\mathrm{dev}}:\boldsymbol{\varepsilon}^{\mathrm{dev}},\qquad K=\lambda+\frac{2\mu}{3},\qquad \boldsymbol{\varepsilon}^{\mathrm{dev}}=\boldsymbol{\varepsilon}-\frac{\operatorname{tr}(\boldsymbol{\varepsilon})}{3}I.$$



Plane strain is embedded in three dimensions with $\varepsilon_{zz}=0$, and the tensor norm counts the $\varepsilon_{xy}$ component twice. The entire current strain field is retained only when its maximum quadrature-node energy strictly exceeds the stored maximum.

Returns
-------
np.ndarray of nonnegative history values with shape previous_history.shape
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_volumetric_deviatoric_history(
    strain_reconstruction: np.ndarray,
    displacement: np.ndarray,
    previous_history: np.ndarray,
    lame_lambda: float,
    shear_modulus: float,
) -> np.ndarray:
    r"""Return the selected quadrature-node tensile-energy field.

    Parameters
    ----------
    strain_reconstruction : np.ndarray, shape (n_q, 4, n_dof)
        Strain matrices in $(xx,yy,zz,xy)$ order.
    displacement : np.ndarray, shape (n_dof,)
        Local displacement coefficients.
    previous_history : np.ndarray, shape (n_q,)
        Finite nonnegative stored energy values.
    lame_lambda : float
        Finite Lamé first parameter.
    shear_modulus : float
        Finite positive shear modulus.

    Returns
    -------
    np.ndarray, shape (n_q,)
        Either all current tensile-energy values or the unchanged previous
        field, according to the strict maximum comparison.

    Raises
    ------
    ValueError
        If dimensions, finiteness, nonnegativity, or elastic stability are
        invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_update_volumetric_deviatoric_history(
    strain_reconstruction, displacement, previous_history, lame_lambda, shear_modulus
):
    """Reference three-dimensional volumetric-deviatoric history update."""
   

    strain_operator = np.asarray(strain_reconstruction, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    previous = np.asarray(previous_history, dtype=float)
    if strain_operator.ndim != 3 or strain_operator.shape[1] != 4:
        raise ValueError("strain_reconstruction must have shape (n_q, 4, n_dof)")
    if strain_operator.shape[0] < 1 or strain_operator.shape[2] < 1:
        raise ValueError("strain_reconstruction must be nonempty")
    if displacement.shape != (strain_operator.shape[2],):
        raise ValueError("displacement length must match the local dof count")
    if previous.shape != (strain_operator.shape[0],):
        raise ValueError("previous_history length must match the quadrature count")
    if (
        not np.all(np.isfinite(strain_operator))
        or not np.all(np.isfinite(displacement))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError("array inputs must be finite")
    if np.any(previous < 0.0):
        raise ValueError("previous_history must be nonnegative")
    if not np.isfinite(lame_lambda) or not np.isfinite(shear_modulus):
        raise ValueError("elastic parameters must be finite")
    if shear_modulus <= 0.0 or lame_lambda + 2.0 * shear_modulus / 3.0 <= 0.0:
        raise ValueError(
            "elastic parameters must define positive bulk and shear moduli"
        )

    strains = np.einsum("qij,j->qi", strain_operator, displacement)
    bulk_modulus = lame_lambda + 2.0 * shear_modulus / 3.0
    current = np.empty(strains.shape[0])
    for node, strain in enumerate(strains):
        trace = strain[0] + strain[1] + strain[2]
        deviatoric_normal = strain[0:3] - trace / 3.0
        deviatoric_norm_squared = (
            deviatoric_normal @ deviatoric_normal + 2.0 * strain[3] ** 2
        )
        current[node] = (
            0.5 * bulk_modulus * max(trace, 0.0) ** 2
            + shear_modulus * deviatoric_norm_squared
        )
    return current if np.max(current) > np.max(previous) else previous.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return updating, unloading, tie, and invalid-history cases."""
    return [
        {
            "setup": """import numpy as np
B = np.zeros((2, 4, 2))
B[0,0,0] = 1.0
B[0,1,1] = -0.2
B[1,0,0] = 0.5
B[1,3,1] = 0.25
u = np.array([0.02, 0.01])
old = np.zeros(2)
lam = 121.15
mu = 80.77
""",
            "call": "update_volumetric_deviatoric_history(B, u, old, lam, mu)",
            "gold_call": "_oracle_update_volumetric_deviatoric_history(B, u, old, lam, mu)",
        },
        {
            "setup": """import numpy as np
B = np.zeros((3, 4, 1))
B[:,0,0] = np.array([1.0, 0.8, 0.6])
u = np.array([0.001])
old = np.array([0.2, 0.1, 0.15])
lam = 121.15
mu = 80.77
""",
            "call": "update_volumetric_deviatoric_history(B, u, old, lam, mu)",
            "gold_call": "_oracle_update_volumetric_deviatoric_history(B, u, old, lam, mu)",
        },
        {
            "setup": """import numpy as np
B = np.zeros((1, 4, 1))
u = np.array([0.0])
old = np.array([0.0])
lam = 1.0
mu = 1.0
""",
            "call": "update_volumetric_deviatoric_history(B, u, old, lam, mu)",
            "gold_call": "_oracle_update_volumetric_deviatoric_history(B, u, old, lam, mu)",
        },
        {
            "setup": """import numpy as np
B = np.zeros((1, 4, 1))
u = np.array([0.0])
old = np.array([-1.0])
def run_model():
    try:
        update_volumetric_deviatoric_history(B, u, old, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_gold():
    try:
        _oracle_update_volumetric_deviatoric_history(B, u, old, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
