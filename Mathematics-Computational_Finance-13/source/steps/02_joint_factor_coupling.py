"""
Combine the one-step transition laws of two square-root factors into a joint transition table over the nine successor pairs. The table must reproduce both marginal transition laws exactly and impose a prescribed correlation between the two factors' standardised innovations, and every entry must be non-negative. Inputs are the marginal probabilities and the centred unit-variance innovations of each factor's stencil, together with the target correlation. Return the 3-by-3 table indexed by the variance successor along the first axis and the rate successor along the second.

Two square-root factors are discretised on separate recombining lattices, each carrying a three-point transition at every node. Advancing them jointly requires a table over the nine successor pairs that leaves both marginals untouched while reproducing the dependence between them.



For a factor at parent level z with successors Z_i, exact conditional mean m(z) and exact conditional variance v(z), the standardised innovation of successor i is



    z_i = (Z_i - m(z)) / sqrt(v(z))



so that under the marginal probabilities p_i the innovation is centred with unit variance:

sum_i p_i z_i = 0 and sum_i p_i z_i^2 = 1.

Given marginals (p^V, z^V) and (p^X, z^X) and a target innovation correlation rho, a one-parameter family of joint tables is available. Because the innovations are centred, summing over one index returns the other marginal unchanged; because they have unit variance, the innovation cross-moment of the table equals rho exactly. Both properties are consequences of the standardisation and neither requires an iterative fit.

Admissibility is a genuine restriction rather than a formality. The correction to the product form is signed, so for a large enough correlation, or for stencils whose innovations reach far from zero, some entries turn negative and the table is not a probability distribution. Sufficient ranges quoted for a given pair of lattice scale parameters are asymptotic in the time step and can fail at moderate step sizes, so admissibility should be checked on the actual pair of stencils rather than assumed from a scale-parameter bound.

Returns
-------
np.ndarray of shape (3, 3), the joint transition probabilities with the first factor's successors along axis 0 and the second factor's along axis 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def joint_factor_coupling(p_v: np.ndarray, z_v: np.ndarray,
                          p_x: np.ndarray, z_x: np.ndarray, rho: float) -> np.ndarray:
    '''Build the joint transition table over the nine successor pairs.

    Parameters
    ----------
    p_v : np.ndarray
        Shape (3,), transition probabilities of the first factor's stencil.
    z_v : np.ndarray
        Shape (3,), standardised innovations of the first factor's successors,
        centred with unit variance under p_v.
    p_x : np.ndarray
        Shape (3,), transition probabilities of the second factor's stencil.
    z_x : np.ndarray
        Shape (3,), standardised innovations of the second factor's successors,
        centred with unit variance under p_x.
    rho : float
        Target correlation between the two standardised innovations, in [-1, 1].

    Returns
    -------
    table : np.ndarray
        Shape (3, 3), joint transition probabilities. The first axis indexes the
        first factor's successors, the second axis the second factor's.

    Raises
    ------
    ValueError
        If the inputs do not describe a valid pair of standardised stencils, or
        if the requested correlation is not admissible on that pair.
    '''
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_joint_factor_coupling(p_v: np.ndarray, z_v: np.ndarray,
                                  p_x: np.ndarray, z_x: np.ndarray, rho: float) -> np.ndarray:
    arrs = []
    for a in (p_v, z_v, p_x, z_x):
        b = np.asarray(a, dtype=float)
        if b.ndim != 1 or b.shape[0] != 3 or not np.all(np.isfinite(b)):
            raise ValueError("each marginal input must be a finite 1-D array of length 3")
        arrs.append(b)
    pv, zv, px, zx = arrs

    if not isinstance(rho, (int, float, np.integer, np.floating)) or not np.isfinite(rho):
        raise ValueError("rho must be a finite real number")
    rho = float(rho)
    if abs(rho) > 1.0:
        raise ValueError("rho must lie in [-1, 1]")

    for p, z in ((pv, zv), (px, zx)):
        if p.min() < 0.0 or abs(p.sum() - 1.0) > 1e-12:
            raise ValueError("marginal probabilities must be non-negative and sum to one")
        if abs(float((p * z).sum())) > 1e-9 or abs(float((p * z * z).sum()) - 1.0) > 1e-9:
            raise ValueError("innovations must be centred with unit variance under their marginal")

    table = pv[:, None] * px[None, :] * (1.0 + rho * zv[:, None] * zx[None, :])
    active = (pv[:, None] * px[None, :]) > 0.0
    if np.any(table[active] < 0.0):
        raise ValueError("the requested correlation is not admissible on this pair of stencils")
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = """import numpy as np
pv0 = np.array([0.388997266971, 0.344846708259, 0.266156024770])
zv0 = np.array([-1.021935325935, 0.0, 1.493597784059])
pv2 = np.array([0.335825549311, 0.101453072118, 0.562721378571])
zv2 = np.array([-1.311867471905, -0.377886283266, 0.851036333364])
pvm = np.array([0.262314879312, 0.529562739078, 0.208122381610])
zvm = np.array([-1.144581741887, -0.126450038492, 1.764365982098])
px0 = np.array([0.220513258109, 0.539705234627, 0.239781507264])
zx0 = np.array([-1.388712127817, -0.116070394717, 1.538372327804])
px4 = np.array([0.434210919095, 0.445128996305, 0.120660084599])
zx4 = np.array([-0.992314231034, 0.419422893778, 2.023671567648])
pxc = np.array([0.166660684899, 0.476556779221, 0.356782535880])
zxc = np.array([-1.670948529006, -0.295697852262, 1.175501041096])
"""
    err = pre + """
def run_model(**kw):
    a = dict(p_v=pv0, z_v=zv0, p_x=px0, z_x=zx0, rho=0.02)
    a.update(kw)
    try:
        joint_factor_coupling(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(p_v=pv0, z_v=zv0, p_x=px0, z_x=zx0, rho=0.02)
    a.update(kw)
    try:
        _oracle_joint_factor_coupling(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": pre, "call": "joint_factor_coupling(pv0, zv0, px0, zx0, 0.02)",
         "gold_call": "_oracle_joint_factor_coupling(pv0, zv0, px0, zx0, 0.02)"},
        {"setup": pre, "call": "joint_factor_coupling(pv2, zv2, px4, zx4, 0.02)",
         "gold_call": "_oracle_joint_factor_coupling(pv2, zv2, px4, zx4, 0.02)"},
        {"setup": pre, "call": "joint_factor_coupling(pvm, zvm, pxc, zxc, 0.02)",
         "gold_call": "_oracle_joint_factor_coupling(pvm, zvm, pxc, zxc, 0.02)"},
        {"setup": pre, "call": "joint_factor_coupling(pv0, zv0, px0, zx0, 0.0)",
         "gold_call": "_oracle_joint_factor_coupling(pv0, zv0, px0, zx0, 0.0)"},
        {"setup": pre, "call": "joint_factor_coupling(pv0, zv0, px0, zx0, -0.25)",
         "gold_call": "_oracle_joint_factor_coupling(pv0, zv0, px0, zx0, -0.25)"},
        {"setup": pre, "call": "joint_factor_coupling(pv2, zv2, pxc, zxc, 0.30)",
         "gold_call": "_oracle_joint_factor_coupling(pv2, zv2, pxc, zxc, 0.30)"},
        {"setup": err, "call": "run_model(rho=1.4)", "gold_call": "run_gold(rho=1.4)"},
        {"setup": err, "call": "run_model(p_v=np.array([0.2, 0.2, 0.2]))",
         "gold_call": "run_gold(p_v=np.array([0.2, 0.2, 0.2]))"},
        {"setup": err, "call": "run_model(z_x=np.array([-1.0, 0.0, 1.0]))",
         "gold_call": "run_gold(z_x=np.array([-1.0, 0.0, 1.0]))"},
        {"setup": err, "call": "run_model(p_v=pvm, z_v=zvm, p_x=pxc, z_x=zxc, rho=0.95)",
         "gold_call": "run_gold(p_v=pvm, z_v=zvm, p_x=pxc, z_x=zxc, rho=0.95)"},
    ]
