"""
Solve the regularized least-squares fit of Eqs (64) to (68) at one node and return the relation it yields between this node's weight and its neighbours' weights. Stack the collocation block (the basis at the neighbours, whose targets are the neighbours' own weights and are therefore left symbolic), the derivative block and the energy block into one matrix; scale every row of that matrix, together with its target, to unit Euclidean norm; form the Tikhonov-regularized generalized inverse with parameter lam; and read off the row that gives the weight field's value at the node. Return a length n_F + 1 array whose first entry is the contribution of the supplied targets and whose remaining entries are the coefficients multiplying the weights of the n_F neighbours, in family order. Raise ValueError if the blocks are not 2-D with a common basis dimension, if the target length does not match the derivative and energy rows, or if lam is negative.

The local fit does not determine a node's weight on its own: its collocation rows involve the neighbours' weights, which are unknown. What the fit yields is one linear relation per node between that node's weight, its neighbours' weights and the analytic targets.

Returns
-------
ndarray of float64 with shape (n_F + 1,): the target term, then one coefficient per neighbour.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def local_coupling_row(basis, m_deriv, m_energy, targets, lam):
    """ndarray of float64 with shape (n_F + 1,): the target term, then one coefficient per neighbour."""
    return np.zeros(np.shape(basis)[0] + 1, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_local_coupling_row(basis, m_deriv, m_energy, targets, lam):
    """Eqs (64)-(68) plus the source's per-row normalization: one node's row of (69).

    M = [M_c; M_D; M_E] with M_c the basis at the neighbours (collocation, target = the
    neighbours' weights w_j, which are UNKNOWN) and the operator blocks with the analytic
    targets. Every row of M, and its target, is scaled to unit Euclidean norm; then
    a = (M~^T M~ + lam I)^-1 M~^T b~ and the node's own weight is w_i = p(0)^T a = a_0.
    Reading off the first row of the generalized inverse gives
        w_i = sum_j gamma_j w_j + f_i,
    where gamma_j multiplies neighbour j's weight and f_i collects the target
    contributions. Returns the (n_F + 1,) array [f_i, gamma_1, ..., gamma_nF].
    """
    import numpy as np
    Mc = np.asarray(basis, dtype=np.float64)
    MD = np.asarray(m_deriv, dtype=np.float64)
    ME = np.asarray(m_energy, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64).reshape(-1)
    if Mc.ndim != 2 or MD.ndim != 2 or ME.ndim != 2:
        raise ValueError("blocks must be 2-D")
    n_p = Mc.shape[1]
    if MD.shape[1] != n_p or ME.shape[1] != n_p:
        raise ValueError("all blocks must share the basis dimension")
    if t.size != MD.shape[0] + ME.shape[0]:
        raise ValueError("targets must hold one entry per derivative and energy row")
    if lam < 0.0:
        raise ValueError("lam must be nonnegative")
    n_F = Mc.shape[0]
    M = np.vstack([Mc, MD, ME])
    nr = np.linalg.norm(M, axis=1)
    nr = np.where(nr > 1e-20, nr, 1.0)          # an all-zero row stays zero, untouched
    Mn = M / nr[:, None]
    A = Mn.T @ Mn + float(lam) * np.eye(n_p)
    Mplus = np.linalg.solve(A, Mn.T)            # (n_p, n_rows) regularized generalized inverse
    row0 = Mplus[0] / nr                         # scaling the target by 1/nr as well
    gamma = row0[:n_F]
    f_i = float(row0[n_F:] @ t)
    return np.concatenate([[f_i], gamma])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "basis = np.array([[1.0, 0.5, 0.0, 0.25, 0.0, 0.0], [1.0, 0.0, 0.5, 0.0, 0.0, 0.25], [1.0, -0.5, 0.0, 0.25, 0.0, 0.0], [1.0, 0.0, -0.5, 0.0, 0.0, 0.25], [1.0, 0.5, 0.5, 0.25, 0.25, 0.25], [1.0, -0.5, 0.5, 0.25, -0.25, 0.25], [1.0, 0.5, -0.5, 0.25, -0.25, 0.25]])\nm_deriv = np.eye(10, 6)\nm_energy = 0.5 * np.eye(10, 6) + 0.1\ntargets = np.linspace(0.1, 2.0, 20)\nlam = 0.0",
            "call": "local_coupling_row(basis, m_deriv, m_energy, targets, lam)",
            "gold_call": "_oracle_local_coupling_row(basis, m_deriv, m_energy, targets, lam)",
        },
        {
            "setup": "basis = np.ones((1, 6))\nm_deriv = np.zeros((10, 6))\nm_energy = np.zeros((10, 6))\ntargets = np.zeros(20)\nlam = 1e-6",
            "call": "local_coupling_row(basis, m_deriv, m_energy, targets, lam)",
            "gold_call": "_oracle_local_coupling_row(basis, m_deriv, m_energy, targets, lam)",
        },
        {
            "setup": "basis = np.array([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0, 0.0, 0.0]])\nm_deriv = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (10, 1))\nm_energy = np.tile(np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), (10, 1)) * np.arange(1, 11)[:, None]\ntargets = np.ones(20)\nlam = 0.01",
            "call": "local_coupling_row(basis, m_deriv, m_energy, targets, lam)",
            "gold_call": "_oracle_local_coupling_row(basis, m_deriv, m_energy, targets, lam)",
        },
    ]
