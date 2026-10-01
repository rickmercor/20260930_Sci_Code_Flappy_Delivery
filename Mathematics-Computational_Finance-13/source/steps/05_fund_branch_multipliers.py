"""
Build the one-step fund transition from a single joint factor node. Each of the nine joint factor successors is combined with a three-point quadrature over the independent residual of the equity innovation, giving 27 branches. For each branch return its probability, its discount factor, and its fund multiplier. The integrated short rate over the step is formed from the rate factor at both endpoints of the branch together with the deterministic shift increment. The multipliers must be rescaled by a single node-dependent factor so that the probability-weighted sum of discounted multipliers equals the one-step dividend discount factor exactly. Branches are ordered with the variance successor varying slowest, then the rate successor, then the quadrature point.

Between contract dates the fund has the same proportional return as the equity index, so over one step it is multiplied by the exponential of an integrated drift plus a diffusion term. Conditional on a joint factor branch, the variance level is taken at the parent node and the integrated short rate is formed from the rate factor at the two endpoints of the branch together with the deterministic shift increment for that step. Writing R for that integrated rate, q for the dividend yield, v for the parent variance and xi_S for the standardised equity innovation, the uncorrected multiplier over a step of length h is



    M0 = exp( R - q h - 0.5 v h + sqrt(v h) * xi_S )



with the Ito correction -0.5 v h accompanying the diffusion term.



The equity innovation is expressed through the weights of the previous step as xi_S = w_V xi_V + w_r xi_r + w_perp eta, where xi_V and xi_r are the standardised innovations already carried by the two factor successors and eta is independent. The two factor innovations take only three values each and are supplied by the joint table, but eta is continuous and must be integrated numerically. The rule used here is the three-point quadrature for the standard normal that is exact for polynomials up to degree five; this is the unique symmetric three-point rule with that property, and its nodes and weights follow from requiring the first six moments to be reproduced. A cruder rule matching only the lower moments leaves an error in the fourth moment of the innovation. Because the multiplier is exponential in eta and the contract payoff is convex, that error is material at the fixed step size used here, although it vanishes at first order as the step is refined.



Discretising a lognormal transition on a finite branch set does not preserve the martingale property of the discounted fund: the probability-weighted sum of discounted multipliers misses the correct value by a small amount that depends on the node. Since a valuation that fails this property misprices even a claim with no optionality, the multipliers are rescaled at each parent node by a single positive factor chosen so that the identity holds exactly. The factor is the ratio of the required value to the achieved sum and applies uniformly to all branches from that node, so it changes the level of the multipliers without disturbing their relative dispersion.

Returns
-------
np.ndarray of shape (3, 27): row 0 branch probabilities, row 1 branch discount factors, row 2 fund multipliers, ordered by variance successor, then rate successor, then residual quadrature point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fund_branch_multipliers(v_parent: float, x_parent: float, x_children: np.ndarray,
                            joint: np.ndarray, z_v: np.ndarray, z_x: np.ndarray,
                            weights: np.ndarray, shift: float, q: float, h: float) -> np.ndarray:
    '''Probabilities, discount factors and fund multipliers on the 27 branches.

    Parameters
    ----------
    v_parent : float
        Non-negative variance level at the parent node.
    x_parent : float
        Rate factor level at the parent node.
    x_children : np.ndarray
        Shape (3,), rate factor levels at the three rate successors.
    joint : np.ndarray
        Shape (3, 3), joint transition probabilities over the factor successors,
        with the variance successor along axis 0.
    z_v : np.ndarray
        Shape (3,), standardised innovations of the variance successors.
    z_x : np.ndarray
        Shape (3,), standardised innovations of the rate successors.
    weights : np.ndarray
        Shape (3,), weights on the variance innovation, the rate innovation and
        the independent residual, in that order.
    shift : float
        Integral of the deterministic shift over this step.
    q : float
        Dividend yield.
    h : float
        Strictly positive time step.

    Returns
    -------
    branches : np.ndarray
        Shape (3, 27). Row 0 holds branch probabilities, row 1 the branch
        discount factors, row 2 the fund multipliers. Branches are ordered with
        the variance successor varying slowest, then the rate successor, then
        the residual quadrature point.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid joint factor node and step.
    '''
    return branches

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fund_branch_multipliers(v_parent: float, x_parent: float, x_children: np.ndarray,
                                    joint: np.ndarray, z_v: np.ndarray, z_x: np.ndarray,
                                    weights: np.ndarray, shift: float, q: float, h: float) -> np.ndarray:
    for name, val in (("v_parent", v_parent), ("x_parent", x_parent),
                      ("shift", shift), ("q", q), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite real number")
    if float(h) <= 0.0:
        raise ValueError("h must be a positive time step")
    if float(v_parent) < 0.0:
        raise ValueError("v_parent must be non-negative")

    xc = np.asarray(x_children, dtype=float)
    zv = np.asarray(z_v, dtype=float)
    zx = np.asarray(z_x, dtype=float)
    w = np.asarray(weights, dtype=float)
    J = np.asarray(joint, dtype=float)
    for arr, n in ((xc, 3), (zv, 3), (zx, 3), (w, 3)):
        if arr.ndim != 1 or arr.shape[0] != n or not np.all(np.isfinite(arr)):
            raise ValueError("x_children, z_v, z_x and weights must be finite 1-D arrays of length 3")
    if J.shape != (3, 3) or not np.all(np.isfinite(J)):
        raise ValueError("joint must be a finite 3-by-3 array")
    if J.min() < 0.0 or abs(J.sum() - 1.0) > 1e-12:
        raise ValueError("joint must be a probability table summing to one")
    if w[2] < 0.0:
        raise ValueError("the residual weight must be non-negative")

    v_parent, x_parent = float(v_parent), float(x_parent)
    shift, q, h = float(shift), float(q), float(h)

    eta = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
    wq = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])

    R = 0.5 * h * (x_parent + xc) + shift
    R9 = np.repeat(R[None, :], 3, axis=0).ravel()
    base = (w[0] * zv[:, None] + w[1] * zx[None, :]).ravel()
    prob9 = J.ravel()

    sig = np.sqrt(v_parent * h)
    m0 = np.exp(R9[:, None] - q * h - 0.5 * v_parent * h
                + sig * (base[:, None] + w[2] * eta[None, :]))
    pr = prob9[:, None] * wq[None, :]
    disc = np.repeat(np.exp(-R9)[:, None], 3, axis=1)

    denom = float((pr * disc * m0).sum())
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("the branch set does not admit a positive martingale correction")
    m = m0 * (np.exp(-q * h) / denom)

    return np.vstack([pr.ravel(), disc.ravel(), m.ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = """import numpy as np
zv0 = np.array([-1.021935325935, 0.0, 1.493597784059])
zx0 = np.array([-1.388712127817, -0.116070394717, 1.538372327804])
zv2 = np.array([-1.311867471905, -0.377886283266, 0.851036333364])
zx4 = np.array([-0.992314231034, 0.419422893778, 2.023671567648])
pv0 = np.array([0.388997266971, 0.344846708259, 0.266156024770])
px0 = np.array([0.220513258109, 0.539705234627, 0.239781507264])
pv2 = np.array([0.335825549311, 0.101453072118, 0.562721378571])
px4 = np.array([0.434210919095, 0.445128996305, 0.120660084599])
J00 = pv0[:, None] * px0[None, :] * (1.0 + 0.02 * zv0[:, None] * zx0[None, :])
J24 = pv2[:, None] * px4[None, :] * (1.0 + 0.02 * zv2[:, None] * zx4[None, :])
W = np.array([-0.696278511405, -0.186074429772, 0.689485428463])
xc0 = np.array([0.009833513692, 0.018000000000, 0.028616487399])
xc4 = np.array([0.028616487399, 0.041682974798, 0.057199462197])
h, q = 0.25, 0.015
"""
    err = pre + """
def run_model(**kw):
    a = dict(v_parent=0.04, x_parent=0.018, x_children=xc0, joint=J00, z_v=zv0, z_x=zx0,
             weights=W, shift=0.000557157441, q=q, h=h)
    a.update(kw)
    try:
        fund_branch_multipliers(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(v_parent=0.04, x_parent=0.018, x_children=xc0, joint=J00, z_v=zv0, z_x=zx0,
             weights=W, shift=0.000557157441, q=q, h=h)
    a.update(kw)
    try:
        _oracle_fund_branch_multipliers(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": pre,
         "call": "fund_branch_multipliers(0.04, 0.018, xc0, J00, zv0, zx0, W, 0.000557157441, q, h)",
         "gold_call": "_oracle_fund_branch_multipliers(0.04, 0.018, xc0, J00, zv0, zx0, W, 0.000557157441, q, h)"},
        {"setup": pre,
         "call": "fund_branch_multipliers(0.1225, 0.041682974798, xc4, J24, zv2, zx4, W, 0.001203428, q, h)",
         "gold_call": "_oracle_fund_branch_multipliers(0.1225, 0.041682974798, xc4, J24, zv2, zx4, W, 0.001203428, q, h)"},
        {"setup": pre,
         "call": "fund_branch_multipliers(0.04, 0.018, xc0, J00, zv0, zx0, W, 0.000557157441, 0.0, h)",
         "gold_call": "_oracle_fund_branch_multipliers(0.04, 0.018, xc0, J00, zv0, zx0, W, 0.000557157441, 0.0, h)"},
        {"setup": pre,
         "call": "fund_branch_multipliers(0.0, 0.018, xc0, J00, zv0, zx0, W, 0.000557157441, q, h)",
         "gold_call": "_oracle_fund_branch_multipliers(0.0, 0.018, xc0, J00, zv0, zx0, W, 0.000557157441, q, h)"},
        {"setup": pre,
         "call": "fund_branch_multipliers(0.04, 0.018, xc0, J00, zv0, zx0, np.array([-0.7, -0.2, 0.0]), 0.000557157441, q, h)",
         "gold_call": "_oracle_fund_branch_multipliers(0.04, 0.018, xc0, J00, zv0, zx0, np.array([-0.7, -0.2, 0.0]), 0.000557157441, q, h)"},
        {"setup": pre,
         "call": "fund_branch_multipliers(0.075625, 0.018, xc0, J00, zv0, zx0, W, -0.0008, q, 1.0)",
         "gold_call": "_oracle_fund_branch_multipliers(0.075625, 0.018, xc0, J00, zv0, zx0, W, -0.0008, q, 1.0)"},
        {"setup": err, "call": "run_model(h=0.0)", "gold_call": "run_gold(h=0.0)"},
        {"setup": err, "call": "run_model(joint=J00 * 2.0)", "gold_call": "run_gold(joint=J00 * 2.0)"},
        {"setup": err, "call": "run_model(v_parent=-0.01)", "gold_call": "run_gold(v_parent=-0.01)"},
    ]
