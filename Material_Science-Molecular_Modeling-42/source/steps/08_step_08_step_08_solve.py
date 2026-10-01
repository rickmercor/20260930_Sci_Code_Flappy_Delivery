"""
The threshold is the extra Einstein stiffness needed for nonnegative curvature in the specified internal space. The geometry is fixed; the calculation does not locate an equilibrium phase transition.

The reported coefficient concerns one displacement direction at a fixed finite-bead configuration. It does not establish a thermodynamic transition, a relaxed structure, or stability in every direction.

Returns
-------
Return one finite float in reduced energy per squared reduced length.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(q: "np.ndarray", beta: float, j0: float, g: float, alpha: float, mass: float) -> float:
    """Return the exact local internal-curvature Einstein threshold.
    
    Parameters
    ----------
    q : finite real array (p, n), p=1,...,8; n=4,6,8
    beta, j0, mass : positive floats
    g, alpha : nonnegative floats
        Same domain and reduced units as rped_force_constants.
    
    Returns
    -------
    k_star : float
        Finite stiffness in reduced energy per squared reduced length.
        Construct the periodic Sz=0 spin operators and all total-spin
        projectors. Obtain the analytic Hessian of F_p=-log(Z)/(beta/p)
        for quadratic exchange J=j0*(1-g*Delta q+alpha*Delta q**2),
        retaining thermal multiplicities and all mixed bead derivatives.
        Add the physical bead-spring Hessian and restrict to the
        per-bead zero-site-sum space using internal_curvature.
        Return max(0,-lambda_min) before adding the Einstein k*I term.
        Use the full chain at the supplied p; no finite stencil, fitting,
        relaxation or continuum limit. NumPy only. Inputs are not mutated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve(q: "np.ndarray", beta: float, j0: float, g: float, alpha: float, mass: float) -> float:
    """Return the exact local internal-curvature Einstein threshold.

    Parameters
    ----------
    q : finite real array (p, n), p=1,...,8; n=4,6,8
    beta, j0, mass : positive floats
    g, alpha : nonnegative floats
        Same domain and reduced units as rped_force_constants.

    Returns
    -------
    k_star : float
        Construct the periodic Sz=0 spin operators and all total-spin
        projectors. Obtain the analytic Hessian of F_p=-log(Z)/(beta/p)
        for quadratic exchange J=j0*(1-g*Delta q+alpha*Delta q**2),
        retaining thermal multiplicities and all mixed bead derivatives.
        Add the physical bead-spring Hessian and restrict to the
        per-bead zero-site-sum space using internal_curvature.
        Return max(0,-lambda_min) before adding the Einstein k*I term.
        Use the full chain at the supplied p; no finite stencil, fitting,
        relaxation or continuum limit. NumPy only. Inputs are not mutated.
    """
    n = np.shape(q)[1]
    bonds,s2 = _oracle_spin_operators(n)
    projectors = _oracle_spin_projectors(s2,n)
    _,_,hessian,_ = _oracle_rped_force_constants(q,beta,j0,g,alpha,bonds,projectors)
    return _oracle_internal_curvature(hessian,np.shape(q)[0],n,beta,mass)[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = """import numpy as np
q = np.array([
    [0.36, -0.24, 0.12, -0.33, 0.21, -0.12],
    [-0.09, 0.30, -0.18, 0.27, -0.36, 0.06],
    [0.24, -0.06, -0.30, 0.15, 0.09, -0.12],
    [-0.21, 0.12, 0.33, -0.09, -0.27, 0.12]
], dtype=float)
"""
    return [
        {
            "setup": setup,
            "call": "solve(q, 3.7, 1.0, 0.8, 0.3, 0.2)",
            "gold_call": "_oracle_solve(q, 3.7, 1.0, 0.8, 0.3, 0.2)",
            "tol": 1e-7,
        },
        {
            "setup": setup + "q = q[:, :4].copy()\n",
            "call": "solve(q, 2.1, 0.9, 0.6, 0.2, 0.15)",
            "gold_call": "_oracle_solve(q, 2.1, 0.9, 0.6, 0.2, 0.15)",
            "tol": 1e-7,
        },
        {
            "setup": setup + "q = q[:1].copy()\n",
            "call": "solve(q, 3.7, 1.0, 0.8, 0.3, 0.2)",
            "gold_call": "_oracle_solve(q, 3.7, 1.0, 0.8, 0.3, 0.2)",
            "tol": 1e-7,
        },
        {
            "setup": setup + "q = q[:2].copy()\n",
            "call": "solve(q, 3.7, 1.0, 0.8, 0.3, 0.2)",
            "gold_call": "_oracle_solve(q, 3.7, 1.0, 0.8, 0.3, 0.2)",
            "tol": 1e-7,
        },
        {
            "setup": setup,
            "call": "solve(q, 3.7, 1.0, 0.0, 0.0, 0.2)",
            "gold_call": "_oracle_solve(q, 3.7, 1.0, 0.0, 0.0, 0.2)",
            "tol": 1e-10,
        },
    ]
