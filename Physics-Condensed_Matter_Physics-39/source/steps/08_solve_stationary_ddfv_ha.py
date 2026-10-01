"""
Return the outward right-contact current for the converged stationary branch.

The reported observable is the primal carrier current through the right physical
contact; no dual-direction contact contribution is included.  The corresponding
left-contact current is used for the conservation check controlled by
``conservation_tolerance``.  Diamond rows may be supplied in any order, and the
physical contact diamonds are identified by connectivity rather than fixed row
numbers.

Returns
-------
float, the outward right-contact primal carrier current at the final voltage.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_stationary_ddfv_ha(
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]",
    doping: "NDArray[np.float64]",
    left_doping: float,
    right_doping: float,
    left_voltage: float,
    voltages: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]",
    D_n: float,
    D_p: float,
    gamma_p: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 20,
    jacobian_check_step: float = 2.0e-6,
    conservation_tolerance: float = 1.0e-10,
) -> float:
    """Run the complete stationary DDFV harmonic-average calculation.
    
    Parameters
    ----------
    vertices : np.ndarray
        Finite vertex-coordinate array of shape ``(5,2)`` for ``v0,...,v4``.
    cell_centers : np.ndarray
        Finite primal/boundary cell-point array of shape ``(8,2)`` for
        ``x0,...,x7``.
    diamonds : np.ndarray
        Integer array of shape ``(8,4)`` with the prescribed benchmark
        ``(K,L,Kstar,Lstar)`` connectivities. Rows may be permuted; contact
        diamonds are identified from connectivity rather than row number.
    control_volumes : np.ndarray
        Finite strictly positive length-5 control-volume array ordered as
        ``[K0,K1,K2,K3,v4]``.
    doping : np.ndarray
        Finite length-5 active-cell doping array in the same order.
    left_doping : float
        Finite left-contact doping.
    right_doping : float
        Finite right-contact doping.
    left_voltage : float
        Finite fixed left-contact voltage.
    voltages : np.ndarray
        Nonempty finite one-dimensional right-contact continuation sequence that
        begins at zero and increases strictly. Its final value is the terminal
        voltage at which the current is returned.
    neumann_diamonds : np.ndarray
        One-dimensional integer array containing the current row indices of the
        insulating boundary diamonds, remapped consistently under row permutation.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.
    tolerance : float, optional
        Finite strictly positive residual infinity-norm tolerance used by voltage
        continuation.
    max_iterations : int, optional
        Positive integer Newton-iteration limit per continuation voltage.
    jacobian_check_step : float, optional
        Finite strictly positive consistency-check scale passed to the
        Newton-system assembly.
    conservation_tolerance : float, optional
        Finite strictly positive upper bound on ``abs(I_right + I_left)`` at the
        final stationary solution.
    
    Returns
    -------
    current : float
        Outward right-contact primal carrier current at the final voltage, with
        no dual-direction contact contribution.
    
    Raises
    ------
    ValueError
        If continuation fails, the required physical contact diamonds are absent,
        a terminal current is nonfinite, or terminal-current imbalance exceeds
        ``conservation_tolerance``.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def _oracle_solve_stationary_ddfv_ha(
    vertices: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64],
    control_volumes: NDArray[np.float64],
    doping: NDArray[np.float64],
    left_doping: float,
    right_doping: float,
    left_voltage: float,
    voltages: NDArray[np.float64],
    neumann_diamonds: NDArray[np.int64],
    D_n: float,
    D_p: float,
    gamma_p: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 20,
    jacobian_check_step: float = 2.0e-6,
    conservation_tolerance: float = 1.0e-10,
) -> float:
    state = _oracle_run_voltage_continuation(
        vertices, cell_centers, diamonds, control_volumes, doping,
        left_doping, right_doping, left_voltage, voltages,
        neumann_diamonds, D_n, D_p, gamma_p, tolerance,
        max_iterations, jacobian_check_step,
    )
    seq = np.asarray(voltages, dtype=np.float64)
    if seq.ndim != 1 or seq.size == 0:
        raise ValueError("voltages must be a nonempty one-dimensional array")
    final_voltage = float(seq[-1])
    cons_tol = float(conservation_tolerance)
    if not np.isfinite(cons_tol) or cons_tol <= 0.0:
        raise ValueError("conservation_tolerance must be finite and positive")

    left = _step07_contact_state(left_doping, left_voltage)
    right = _step07_contact_state(right_doping, final_voltage)
    psi_active = state[:5]; n_active = state[5:10]; p_active = state[10:15]
    psi_c = np.full(8, np.nan, dtype=np.float64); n_c = np.full(8, np.nan, dtype=np.float64); p_c = np.full(8, np.nan, dtype=np.float64)
    psi_v = np.full(5, np.nan, dtype=np.float64); n_v = np.full(5, np.nan, dtype=np.float64); p_v = np.full(5, np.nan, dtype=np.float64)
    psi_c[:4] = psi_active[:4]; n_c[:4] = n_active[:4]; p_c[:4] = p_active[:4]
    psi_v[4] = psi_active[4]; n_v[4] = n_active[4]; p_v[4] = p_active[4]
    psi_c[5], n_c[5], p_c[5] = right
    psi_c[7], n_c[7], p_c[7] = left
    for idx in (0, 3):
        psi_v[idx], n_v[idx], p_v[idx] = left
    for idx in (1, 2):
        psi_v[idx], n_v[idx], p_v[idx] = right

    d = np.asarray(diamonds, dtype=np.int64)
    geometry = _oracle_construct_diamond_geometry(np.asarray(vertices, dtype=np.float64), np.asarray(cell_centers, dtype=np.float64), d)
    rows = [tuple(map(int, row)) for row in d]
    try:
        right_row = rows.index((1,5,1,2))
        left_row = rows.index((3,7,3,0))
    except ValueError as exc:
        raise ValueError("required physical contact diamonds are missing") from exc
    contact_rows = np.array([right_row, left_row], dtype=np.int64)
    flux = _oracle_compute_ddfv_ha_flux_jets(
        psi_c, psi_v, n_c, n_v, p_c, p_v,
        d[contact_rows], geometry[contact_rows], D_n, D_p,
    )
    right_current = float(flux[0, 0, 0] + flux[0, 2, 0])
    left_current = float(flux[1, 0, 0] + flux[1, 2, 0])
    if not np.isfinite(right_current) or not np.isfinite(left_current):
        raise ValueError("terminal current is nonfinite")
    if abs(right_current + left_current) > cons_tol:
        raise ValueError("terminal-current conservation check failed")
    return right_current

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, alternate-path, permuted, and parameter-variation orchestrator cases."""
    setup = 'import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nc=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[0,1,1,4],[1,2,2,4],[2,3,0,2],[0,4,0,1],[1,5,1,2],[3,6,2,3],[3,7,3,0]],dtype=int)\nvol=np.array([0.4665,0.3735,0.375,0.895,0.135])\ndop=np.array([-0.6,0.6,-0.6,-0.6,0.6])\nneu=np.array([4,6],dtype=int)'
    return [
        {
            "setup": setup + "\nseq=np.array([0.0,0.125,0.25])",
            "call": "round(solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6,1e-10), 10)",
            "gold_call": "round(_oracle_solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6,1e-10), 10)",
        },
        {
            "setup": setup + "\nseq=np.array([0.0,0.05,0.13,0.19,0.25])",
            "call": "round(solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6,1e-10), 10)",
            "gold_call": "round(_oracle_solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6,1e-10), 10)",
        },
        {
            "setup": setup + "\nperm=np.array([5,2,7,0,6,1,4,3]); d=d[perm]; neu=np.flatnonzero(np.isin(perm,[4,6])).astype(int); seq=np.array([0.0,0.125,0.25])",
            "call": "round(solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6,1e-10), 10)",
            "gold_call": "round(_oracle_solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6,1e-10), 10)",
        },
        {
            "setup": setup + "\nseq=np.array([0.0,0.10,0.20])",
            "call": "round(solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.9,0.2,1e-12,20,2e-6,1e-10), 10)",
            "gold_call": "round(_oracle_solve_stationary_ddfv_ha(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.9,0.2,1e-12,20,2e-6,1e-10), 10)",
        },
    ]
