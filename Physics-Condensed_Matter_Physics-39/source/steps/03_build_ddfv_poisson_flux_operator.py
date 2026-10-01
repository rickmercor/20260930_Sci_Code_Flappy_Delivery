"""
Construct local linear maps for the two oriented electrostatic fluxes.

The input geometry follows the column contract of
``construct_diamond_geometry``.  Local scalar data are ordered as
``[u_K,u_L,u_Kstar,u_Lstar]``.  Each output row is a ``2``-by-``4`` map whose
two results are ordered as the primal and dual integrated fluxes.  Diamond rows
remain independent and retain their supplied orientation.

Returns
-------
np.ndarray of shape (m, 2, 4), the local primal and dual integrated Poisson-flux maps.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_ddfv_poisson_flux_operator(
    geometry: "NDArray[np.float64]",
) -> "NDArray[np.float64]":
    """Build the local integrated Poisson-flux operator for each diamond.

    Parameters
    ----------
    geometry : np.ndarray
        Finite array with shape (m,11) produced by
        ``construct_diamond_geometry``. Columns are
        [|sigma|, |sigma*|, |D|, nKL_x, nKL_y, nstar_x, nstar_y,
        eta, a, b, c].

    Returns
    -------
    operators : np.ndarray
        Float array with shape (m,2,4). For local values
        q=[u_K,u_L,u_Kstar,u_Lstar], ``operators[i] @ q`` equals
        [F_KL,F_KstarLstar], the two integrated oriented Poisson fluxes.

    Raises
    ------
    ValueError
        If geometry is not finite with shape (m,11), m < 1, or contains
        nonpositive lengths/areas or coefficients inconsistent with a valid
        nondegenerate DDFV diamond.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def _oracle_build_ddfv_poisson_flux_operator(
    geometry: NDArray[np.float64],
) -> NDArray[np.float64]:
    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 11 or g.shape[0] < 1 or not np.all(np.isfinite(g)):
        raise ValueError("geometry must be finite with shape (m,11), m >= 1")
    if np.any(g[:, :3] <= 0.0) or np.any(g[:, 8:11] <= 0.0):
        raise ValueError("geometry contains nonpositive lengths, areas, or coefficients")
    if np.any(np.abs(g[:, 7]) > 1.0 + 1.0e-12):
        raise ValueError("eta must be a valid normal dot product")
    # Cross-check the three metric coefficients against the stored lengths/area.
    sigma, sigstar, area = g[:,0], g[:,1], g[:,2]
    ref = np.column_stack((sigma*sigma/(2*area), sigma*sigstar/(2*area), sigstar*sigstar/(2*area)))
    if not np.allclose(g[:,8:11], ref, rtol=2e-12, atol=2e-13):
        raise ValueError("geometry coefficients are inconsistent with lengths and area")
    a,b,c,eta = g[:,8],g[:,9],g[:,10],g[:,7]
    op = np.empty((g.shape[0],2,4),dtype=np.float64)
    op[:,0,0] = -a
    op[:,0,1] =  a
    op[:,0,2] = -b*eta
    op[:,0,3] =  b*eta
    op[:,1,0] = -b*eta
    op[:,1,1] =  b*eta
    op[:,1,2] = -c
    op[:,1,3] =  c
    return op

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return orthogonal, skew, reversal, nullspace, and affine-flux cases."""
    return [
        {
            "setup": "import numpy as np\ng=np.array([[1.,2.,1.,1.,0.,0.,1.,0.,0.5,1.,2.]])",
            "call": "np.round(build_ddfv_poisson_flux_operator(g.copy()),13)",
            "gold_call": "np.round(_oracle_build_ddfv_poisson_flux_operator(g.copy()),13)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nx=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[2,3,0,2],[1,5,1,2]],dtype=int)\ng_model=construct_diamond_geometry(v.copy(),x.copy(),d.copy())\ng_gold=_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())",
            "call": "np.round(build_ddfv_poisson_flux_operator(g_model.copy()),12)",
            "gold_call": "np.round(_oracle_build_ddfv_poisson_flux_operator(g_gold.copy()),12)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.,0.],[1.7,1.3]])\nx=np.array([[0.99,0.61],[0.5,0.7333333333333333]])\nd=np.array([[0,1,0,1],[1,0,1,0]],dtype=int)\ng_model=construct_diamond_geometry(v.copy(),x.copy(),d.copy())\ng_gold=_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())",
            "call": "np.round(build_ddfv_poisson_flux_operator(g_model.copy()),12)",
            "gold_call": "np.round(_oracle_build_ddfv_poisson_flux_operator(g_gold.copy()),12)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.2,-0.1],[1.3,0.7],[-0.4,1.2]])\nx=np.array([[0.,0.],[1.1,0.2],[0.25,1.4]])\nd=np.array([[1,0,2,0],[2,1,0,1]],dtype=int)\ng_model=construct_diamond_geometry(v.copy(),x.copy(),d.copy())\ng_gold=_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())\nq=np.ones((2,4))",
            "call": "np.round(np.einsum('mij,mj->mi',build_ddfv_poisson_flux_operator(g_model.copy()),q.copy()),13)",
            "gold_call": "np.round(np.einsum('mij,mj->mi',_oracle_build_ddfv_poisson_flux_operator(g_gold.copy()),q.copy()),13)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nx=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[2,3,0,2],[1,5,1,2]],dtype=int)\ng_model=construct_diamond_geometry(v.copy(),x.copy(),d.copy())\ng_gold=_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())\nuc=1.7*x[:,0]-0.8*x[:,1]+0.3\nuv=1.7*v[:,0]-0.8*v[:,1]+0.3\nq=np.column_stack((uc[d[:,0]],uc[d[:,1]],uv[d[:,2]],uv[d[:,3]]))",
            "call": "np.round(np.einsum('mij,mj->mi',build_ddfv_poisson_flux_operator(g_model.copy()),q.copy()),12)",
            "gold_call": "np.round(np.einsum('mij,mj->mi',_oracle_build_ddfv_poisson_flux_operator(g_gold.copy()),q.copy()),12)",
        },
    ]
