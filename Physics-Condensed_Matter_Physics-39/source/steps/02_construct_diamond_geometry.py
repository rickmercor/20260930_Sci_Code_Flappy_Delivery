"""
Construct orientation-aware geometric data for primal-dual diamonds.

Each connectivity row is ordered as ``(K,L,Kstar,Lstar)``.  The primal normal

points from the first primal cell toward the second, and the dual normal points

from the first dual vertex toward the second.  The returned columns are ordered

as ``[sigma_length, dual_length, area, nKL_x, nKL_y, nstar_x, nstar_y,

eta, a, b, c]``, where `$a = sigma_length**2/(2*area)$`, `$b = sigma_length*dual_length/(2*area)$` (unsigned; ``eta`` is stored separately), and `$c = dual_length**2/(2*area)$`.  The sign of ``eta`` is retained.  Diamond rows are independent

and may be supplied in any order or orientation.

Returns
-------
np.ndarray of shape (m, 11), one oriented geometry row per diamond in the documented column order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_diamond_geometry(
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
) -> "NDArray[np.float64]":
    """Construct oriented DDFV geometry for every diamond.

    Parameters
    ----------
    vertices : np.ndarray
        Finite array with shape (nv,2) containing dual-mesh vertices.
    cell_centers : np.ndarray
        Finite array with shape (nc,2) containing primal cell centers and
        boundary-edge midpoint cells.
    diamonds : np.ndarray
        Integer array with shape (m,4). Each row is (K,L,Kstar,Lstar).

    Returns
    -------
    geometry : np.ndarray
        Float array with shape (m,11) and columns
        [sigma_length, dual_length, area, nKL_x, nKL_y, nstar_x, nstar_y,
        eta, a, b, c].

    Raises
    ------
    ValueError
        If shapes or indices are invalid or any diamond is degenerate.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def _oracle_construct_diamond_geometry(
    vertices: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64],
) -> NDArray[np.float64]:
    v = np.asarray(vertices, dtype=np.float64)
    x = np.asarray(cell_centers, dtype=np.float64)
    d = np.asarray(diamonds)
    if v.ndim != 2 or v.shape[1] != 2 or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("vertices and cell_centers must have shape (n,2)")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(x)):
        raise ValueError("coordinates must be finite")
    if d.ndim != 2 or d.shape[1] != 4 or d.shape[0] == 0:
        raise ValueError("diamonds must have shape (m,4) with m >= 1")
    if not np.issubdtype(d.dtype, np.integer):
        raise ValueError("diamonds must contain integer indices")
    d = d.astype(np.int64, copy=False)
    if np.any(d[:, :2] < 0) or np.any(d[:, :2] >= x.shape[0]):
        raise ValueError("primal diamond indices are out of range")
    if np.any(d[:, 2:] < 0) or np.any(d[:, 2:] >= v.shape[0]):
        raise ValueError("dual diamond indices are out of range")

    geometry = np.empty((d.shape[0], 11), dtype=np.float64)
    for i, (K, L, Kstar, Lstar) in enumerate(d):
        sigma_vec = v[Lstar] - v[Kstar]
        sigmastar_vec = x[L] - x[K]
        sigma = float(np.linalg.norm(sigma_vec))
        sigmastar = float(np.linalg.norm(sigmastar_vec))
        if sigma <= 0.0 or sigmastar <= 0.0:
            raise ValueError("diamond edges must have positive length")
        area = 0.5 * abs(float(np.linalg.det(np.stack((sigmastar_vec, sigma_vec)))))
        scale = sigma * sigmastar
        if area <= 1.0e-14 * max(1.0, scale):
            raise ValueError("diamond geometry is degenerate")

        n_kl = np.array([-sigma_vec[1], sigma_vec[0]], dtype=np.float64) / sigma
        if float(np.dot(n_kl, sigmastar_vec)) < 0.0:
            n_kl = -n_kl
        n_star = np.array([-sigmastar_vec[1], sigmastar_vec[0]], dtype=np.float64) / sigmastar
        if float(np.dot(n_star, sigma_vec)) < 0.0:
            n_star = -n_star

        eta = float(np.dot(n_kl, n_star))
        a = sigma * sigma / (2.0 * area)
        b = sigma * sigmastar / (2.0 * area)
        c = sigmastar * sigmastar / (2.0 * area)
        geometry[i] = [sigma, sigmastar, area, n_kl[0], n_kl[1], n_star[0], n_star[1], eta, a, b, c]
    return geometry

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return orthogonal, benchmark, and orientation-stress geometry cases."""
    return [
        {
            "setup": "import numpy as np\nv=np.array([[0.,0.],[0.,1.]])\nx=np.array([[-1.,0.5],[1.,0.5]])\nd=np.array([[0,1,0,1]],dtype=int)",
            "call": "np.round(construct_diamond_geometry(v.copy(),x.copy(),d.copy()), 13)",
            "gold_call": "np.round(_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy()), 13)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nx=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[0,1,1,4],[1,2,2,4],[2,3,0,2],[0,4,0,1],[1,5,1,2],[3,6,2,3],[3,7,3,0]],dtype=int)",
            "call": "np.round(construct_diamond_geometry(v.copy(),x.copy(),d.copy()), 12)",
            "gold_call": "np.round(_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy()), 12)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.2,-0.1],[1.3,0.7],[-0.4,1.2],[1.7,-0.6]])\nx=np.array([[0.,0.],[1.1,0.2],[0.25,1.4],[-0.3,0.8]])\nd=np.array([[1,0,2,0],[2,1,0,1],[0,3,1,3]],dtype=int)",
            "call": "np.round(construct_diamond_geometry(v.copy(),x.copy(),d.copy()), 12)",
            "gold_call": "np.round(_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy()), 12)",
        },
        {
            "setup": "import numpy as np\nv=np.array([[0.,0.],[1.7,1.3],[1.27,0.53]])\nx=np.array([[0.99,0.61],[0.5,0.7333333333333333],[1.09,0.21]])\nd=np.array([[0,1,0,1],[1,0,0,1],[0,1,1,0],[1,0,1,0]],dtype=int)",
            "call": "np.round(construct_diamond_geometry(v.copy(),x.copy(),d.copy())[:,[0,1,2,7,8,9,10]], 12)",
            "gold_call": "np.round(_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())[:,[0,1,2,7,8,9,10]], 12)",
        },
    ]
