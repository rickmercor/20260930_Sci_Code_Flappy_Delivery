"""
Evaluate the branchwise corotational Cauchy stress and its consistent Mandel tangent.

Loading and unloading use different derivatives because the internal state evolves only when the history reaches a new maximum. On an active branch, radial alignment reduces the deviatoric elastic energy to



$$

\\Psi_{dev}=\\dfrac{3}{2}\\mu(q-h)^2.

$$



With $g_p=dh/dq$ and $G=d[\\exp(-Ch)W_p]/dh$, its complete scalar derivative is



$$

t=3\\mu(q-h)(1-g_p)+Gg_p.

$$



Using $\\partial q/\\partial e=2e/(3q)$ gives the active stress



$$

\\sigma_{cr}^{active}

=K\\operatorname{tr}(\\epsilon)I+\\dfrac{2t}{3q}e.

$$



The consistent tangent is the exact derivative of the selected stress with respect to symmetric strain while holding the previous state fixed. On the active branch this requires differentiating $t$ through both $h(q)$ and $G(h(q))$ and differentiating the radial factor $2t/(3q)$. Use $m=(1,1,1,0,0,0)^T$, the deviatoric Mandel projector $\\mathbb P_{dev}=I_6-m\\otimes m/3$, and $e_M=(e_{xx},e_{yy},e_{zz},\\sqrt2e_{yz},\\sqrt2e_{xz},\\sqrt2e_{xy})^T$ in order $(xx,yy,zz,yz,xz,xy)$.



On unloading or sub-maximum reloading, the history and residual tensor are fixed, so the stress is



$$

\\sigma_{cr}^{inactive}

=K\\operatorname{tr}(\\epsilon)I+2\\mu(e-E_p).

$$



On the inactive branch, differentiate with $h$ and $E_p$ fixed; no radial division by $q$ is permitted.



The inactive expression also applies when the current equivalent strain is zero and the residual tensor is nonzero. It avoids division by $q$ and preserves residual stress. Applying the radial active formula during unloading would discard the stored tensor direction. Engineering-shear Voigt scaling is incompatible with the declared Mandel map and gives clean but incorrect shear blocks.

Returns
-------
Return the corotational Cauchy stress and consistent Mandel tangent in MPa as arrays of shapes (n_elements, 3, 3) and (n_elements, 6, 6).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def branch_stress(
    strain: np.ndarray,
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    plastic: np.ndarray,
    history: np.ndarray,
    branch_derivative: np.ndarray,
    active: np.ndarray,
    bulk_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    attenuated_work_derivative: np.ndarray,
    activation_curvature: np.ndarray,
    attenuated_work_second_derivative: np.ndarray,
) -> tuple:
    r"""Evaluate branchwise stress and the consistent symmetric-strain tangent.

    All inputs must be finite with matching shapes. Tensors must be symmetric within
    absolute tolerance 1e-10; q,h >= 0; K,mu > 0; g_p in [0,1]; flags exactly 0 or 1.
    Inactive g_p and activation curvature must be zero, while active q must be
    positive. Deviator and plastic tensors must be traceless and their equivalent
    norms must agree with q and h within atol=1e-10 and rtol=1e-8. The tangent uses
    Mandel order (xx, yy, zz, yz, xz, xy), with square-root-of-two scaling on shear
    entries, and differentiates the selected branch while holding the previous state
    fixed. Invalid inputs raise ValueError.

    Parameters
    ----------
    strain : np.ndarray
        Symmetric polar-stretch strain, shape (n_elements, 3, 3).
    deviator : np.ndarray
        Symmetric deviator of strain, same shape.
    equivalent_strain : np.ndarray
        Nonnegative q, shape (n_elements,).
    plastic : np.ndarray
        Updated symmetric residual tensor, shape (n_elements, 3, 3).
    history : np.ndarray
        Updated nonnegative h, shape (n_elements,).
    branch_derivative : np.ndarray
        g_p in [0, 1], zero on inactive cells, shape (n_elements,).
    active : np.ndarray
        Numerical branch flags 0 or 1, shape (n_elements,).
    bulk_modulus : np.ndarray
        Positive K in MPa, shape (n_elements,).
    shear_modulus : np.ndarray
        Positive mu in MPa, shape (n_elements,).
    attenuated_work_derivative : np.ndarray
        G in MPa, shape (n_elements,); may be negative.
    activation_curvature : np.ndarray
        Second candidate derivative with respect to q, shape (n_elements,); it is
        supplied as zero on inactive cells.
    attenuated_work_second_derivative : np.ndarray
        Derivative dG/dh in MPa, shape (n_elements,); may be negative.

    Raises
    ------
    ValueError
        If inputs are nonfinite or have inconsistent shapes; a tensor is nonsymmetric
        or nondeviatoric where required; scalar and tensor equivalent measures
        disagree; strain or history is negative; a modulus is nonpositive; a branch
        flag is not 0 or 1; an inactive derivative or curvature is nonzero; or an
        active state has zero equivalent strain.

    Returns
    -------
    tuple
        (stress, tangent): float arrays of shapes (n_elements, 3, 3) and
        (n_elements, 6, 6), in MPa. The tangent maps Mandel strain increments to
        Mandel stress increments.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _mandel_vectors(tensors):
    root_two = np.sqrt(2.0)
    return np.column_stack(
        (
            tensors[:, 0, 0],
            tensors[:, 1, 1],
            tensors[:, 2, 2],
            root_two * tensors[:, 1, 2],
            root_two * tensors[:, 0, 2],
            root_two * tensors[:, 0, 1],
        )
    )


def _oracle_branch_stress(
    strain: np.ndarray,
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    plastic: np.ndarray,
    history: np.ndarray,
    branch_derivative: np.ndarray,
    active: np.ndarray,
    bulk_modulus: np.ndarray,
    shear_modulus: np.ndarray,
    attenuated_work_derivative: np.ndarray,
    activation_curvature: np.ndarray,
    attenuated_work_second_derivative: np.ndarray,
) -> tuple:
    q, h, gp, flags, bulk, mu, derivative, curvature, derivative_h = _matched_vectors(
        equivalent_strain,
        history,
        branch_derivative,
        active,
        bulk_modulus,
        shear_modulus,
        attenuated_work_derivative,
        activation_curvature,
        attenuated_work_second_derivative,
    )
    strain = _symmetric_tensors(strain, len(q))
    deviator = _symmetric_tensors(deviator, len(q))
    plastic = _symmetric_tensors(plastic, len(q))
    if np.any(q < 0) or np.any(h < 0) or np.any(bulk <= 0) or np.any(mu <= 0):
        raise ValueError("invalid strain, history, or moduli")
    if (
        np.any((flags != 0) & (flags != 1))
        or np.any(gp < 0)
        or np.any(gp > 1)
        or np.any(curvature < 0)
    ):
        raise ValueError("invalid branch flags or slopes")
    active_mask = flags == 1
    if (
        np.any(gp[~active_mask] != 0)
        or np.any(curvature[~active_mask] != 0)
        or np.any(q[active_mask] <= 0)
    ):
        raise ValueError("branch slope or equivalent strain inconsistent with branch")
    for tensor, magnitude in ((deviator, q), (plastic, h)):
        if not np.allclose(
            np.trace(tensor, axis1=1, axis2=2), 0.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError("deviatoric tensors must be traceless")
        norm = np.sqrt(2.0 / 3.0 * np.sum(tensor**2, axis=(1, 2)))
        if not np.allclose(norm, magnitude, atol=1e-10, rtol=1e-8):
            raise ValueError("tensor equivalent norm disagrees with scalar measure")
    stress = 2.0 * mu[:, None, None] * (deviator - plastic)
    derivative_q = 3.0 * mu * (q - h) * (1.0 - gp) + derivative * gp
    stress[active_mask] = (2.0 * derivative_q[active_mask] / (3.0 * q[active_mask]))[
        :, None, None
    ] * deviator[active_mask]
    stress += (bulk * np.trace(strain, axis1=1, axis2=2))[:, None, None] * np.eye(3)

    volumetric = np.outer(
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    deviatoric = np.eye(6) - volumetric / 3.0
    tangent = bulk[:, None, None] * volumetric + 2.0 * mu[:, None, None] * deviatoric
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size:
        q_active = q[active_mask]
        h_active = h[active_mask]
        gp_active = gp[active_mask]
        curvature_active = curvature[active_mask]
        derivative_q_prime = (
            3.0
            * mu[active_mask]
            * ((1.0 - gp_active) ** 2 - (q_active - h_active) * curvature_active)
            + derivative_h[active_mask] * gp_active**2
            + derivative[active_mask] * curvature_active
        )
        radial = 2.0 * derivative_q[active_mask] / (3.0 * q_active)
        radial_prime = (2.0 / 3.0) * (
            derivative_q_prime / q_active - derivative_q[active_mask] / q_active**2
        )
        deviator_mandel = _mandel_vectors(deviator[active_mask])
        tangent[active_mask] = (
            bulk[active_mask, None, None] * volumetric
            + radial[:, None, None] * deviatoric
            + (2.0 * radial_prime / (3.0 * q_active))[:, None, None]
            * np.einsum("ni,nj->nij", deviator_mandel, deviator_mandel)
        )
    return stress, tangent

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.13, .06])
e = q[:, None, None] * B
strain = e + .02 * np.eye(3)
h = np.array([.05, .1])
Ep = h[:, None, None] * B
Ep[1] = h[1] * np.diag([-.5, 1., -.5])
gp = np.array([.9, 0.])
active = np.array([1., 0.])
K = np.full(2, 20 / 1.2)
mu = np.full(2, 20 / 2.6)
G = np.array([1.3, 1.1])
g2 = np.array([1.7, 0.])
G2 = np.array([-.5, .2])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
        },
        {
            "setup": """
import numpy as np
def _equivalent(tensor):
    return np.sqrt(2. / 3. * np.sum(tensor * tensor))
e = np.array([
    [[.08, .03, -.02], [.03, -.05, .015], [-.02, .015, -.03]],
    [[-.04, .025, .01], [.025, .07, -.03], [.01, -.03, -.03]],
])
q = np.array([_equivalent(tensor) for tensor in e])
strain = e + np.array([.015, -.01])[:, None, None] * np.eye(3)
h = np.array([.6 * q[0], .8 * q[1]])
Ep = np.empty_like(e)
Ep[0] = h[0] / q[0] * e[0]
old_direction = np.array([[.02, -.01, .03], [-.01, -.05, .015], [.03, .015, .03]])
Ep[1] = h[1] / _equivalent(old_direction) * old_direction
gp = np.array([.43, 0.])
active = np.array([1., 0.])
K = np.array([17., 23.])
mu = np.array([8., 11.])
G = np.array([1.6, -.4])
g2 = np.array([21., 0.])
G2 = np.array([-3.2, 1.7])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
        },
        {
            "setup": """
import numpy as np
def _equivalent(tensor):
    return np.sqrt(2. / 3. * np.sum(tensor * tensor))
templates = np.array([
    [[.08, .03, -.02], [.03, -.05, .015], [-.02, .015, -.03]],
    [[-.04, .025, .01], [.025, .07, -.03], [.01, -.03, -.03]],
])
mu = np.array([20. / 2.6, 10.])
sy = np.array([2., 1.2])
beta = np.array([12., 1000.])
q = sy / (3. * mu)
e = np.array([q_i / _equivalent(tensor) * tensor for q_i, tensor in zip(q, templates)])
strain = e + np.array([.012, -.008])[:, None, None] * np.eye(3)
candidate, slope, g2 = _oracle_normalized_activation(q, mu, sy, beta)
h, Ep, gp, active = _oracle_irreversible_state(
    e, q, candidate, slope, np.zeros(2), np.zeros((2, 3, 3))
)
H = np.array([.5, 2.4])
C = np.array([2.2, 4.4])
_, _, G, G2 = _oracle_inelastic_work(h, sy, H, C)
K = np.array([20. / 1.2, 30. / 1.5])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
        },
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.13, .06])
e = q[:, None, None] * B
strain = e + .02 * np.eye(3)
h = np.array([.05, .1])
Ep = h[:, None, None] * B
Ep[1] = h[1] * np.diag([-.5, 1., -.5])
gp = np.array([.9, 0.])
active = np.array([1., 0.])
K = np.full(2, 20 / 1.2)
mu = np.full(2, 20 / 2.6)
G = np.array([1.3, 1.1])
g2 = np.zeros(2)
G2 = np.array([-.5, .2])
q[:] = 0
e[:] = 0
strain[:] = .02 * np.eye(3)
gp[:] = 0
active[:] = 0
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
        },
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.13, .06])
e = q[:, None, None] * B
strain = e + .02 * np.eye(3)
h = np.array([.05, .1])
Ep = h[:, None, None] * B
Ep[1] = h[1] * np.diag([-.5, 1., -.5])
gp = np.array([.9, 0.])
active = np.array([1., 0.])
K = np.full(2, 20 / 1.2)
mu = np.full(2, 20 / 2.6)
G = np.array([1.3, 1.1])
g2 = np.array([2.1, 3.4])
G2 = np.array([-.5, .2])
active[:] = 1
gp[:] = .95
G[:] = -2.
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_branch_stress(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)])",
        },
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.13, .06])
e = q[:, None, None] * B
strain = e + .02 * np.eye(3)
h = np.array([.05, .1])
Ep = h[:, None, None] * B
Ep[1] = h[1] * np.diag([-.5, 1., -.5])
gp = np.array([.9, 0.])
active = np.array([1., 0.])
K = np.full(2, 20 / 1.2)
mu = np.full(2, 20 / 2.6)
G = np.array([1.3, 1.1])
g2 = np.array([1.7, 0.])
G2 = np.array([-.5, .2])
active[0] = 2
def _invalid_status(function):
    try:
        function(strain, e, q, Ep, h, gp, active, K, mu, G, g2, G2)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(branch_stress)",
            "gold_call": "_invalid_status(_oracle_branch_stress)",
        },
    ]
