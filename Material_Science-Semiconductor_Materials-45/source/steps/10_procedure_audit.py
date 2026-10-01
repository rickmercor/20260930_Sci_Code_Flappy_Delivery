"""
Run the complete chain for the cell described by device = (d, eps, T, N_c, N_v, E_g, phi_an, phi_cat, mu_n, mu_p, zeta, G_ex) in the units of the first step, and assemble the audit table: one row per zero-field yield with the columns [V_0, J_sc/(q G_ex d), V_oc, V_mpp, FF, P_gen(F_sc), short-circuit collection efficiency, -J_sat,exp/(q G_ex d) and P_diss at the first reverse bias, the same two at the second reverse bias, first-order bulk loss, second-order bulk loss, anode extraction loss, cathode extraction loss at short circuit for that yield's generation], and a head row [P_diss at the first reverse bias of the unity-yield cell, the ideal-transport collection ceiling of the seventh step at the barrier of the grid nearest the cell's anode barrier and at the reduction factor nearest the cell's, the largest ceiling on the barrier grid for that reduction factor and the barrier at which it occurs, the Sokel-Hughes value at the nearest grid barrier, log10 of the crossover mobility of the eighth step for the cell's barrier and reduction factor, the collection efficiency at the lowest intensity for the design mobility and for the ideal mobility, the unity-yield cell's J_sc and its photocurrents -J_ph at the first and at the second reverse bias in mA/cm^2 obtained from the sweep step at the voltages [second reverse bias, first reverse bias, 0] under unity generation and in the dark, and the Onsager-Braun yield of the second step at the nominal short-circuit field V_bi/d for the smallest zero-field yield (1 if that yield is 1)] followed by zeros. Raise ValueError if the yields are empty or do not include 1, if there are not exactly two reverse biases, if device does not have 12 entries or N is not an even integer of at least 4.

The head scalar, the dissociation probability that the photocurrent-versus-effective-voltage procedure returns for a cell whose free-charge generation is complete and field independent, measures how far the procedure's reading sits from the yield it is taken to report.

Returns
-------
A (1 + len(zero_field_yields), 15) float64 array: the head row and one row per yield.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def procedure_audit(zero_field_yields: "np.ndarray", reverse_biases: "np.ndarray", barriers: "np.ndarray", reduction_factors: "np.ndarray", mobility_ideal: float, intensities: "np.ndarray", intervals: int, device: "tuple[float, ...]") -> "np.ndarray":
    """Run the complete chain for the cell described by device = (d, eps, T, N_c, N_v, E_g,
        phi_an, phi_cat, mu_n, mu_p, zeta, G_ex) in the units of the first step, and assemble
        the audit table: one row per zero-field yield with the columns [V_0, J_sc/(q G_ex d),
        V_oc, V_mpp, FF, P_gen(F_sc), short-circuit collection efficiency, -J_sat,exp/(q G_ex d)
        and P_diss at the first reverse bias, the same two at the second reverse bias, first-
        order bulk loss, second-order bulk loss, anode extraction loss, cathode extraction loss
        at short circuit for that yield's generation], and a head row [P_diss at the first
        reverse bias of the unity-yield cell, the ideal-transport collection ceiling of the
        seventh step at the barrier of the grid nearest the cell's anode barrier and at the
        reduction factor nearest the cell's, the largest ceiling on the barrier grid for that
        reduction factor and the barrier at which it occurs, the Sokel-Hughes value at the
        nearest grid barrier, log10 of the crossover mobility of the eighth step for the cell's
        barrier and reduction factor, the collection efficiency at the lowest intensity for the
        design mobility and for the ideal mobility, the unity-yield cell's J_sc and its
        photocurrents -J_ph at the first and at the second reverse bias in mA/cm^2 obtained from
        the sweep step at the voltages [second reverse bias, first reverse bias, 0] under unity
        generation and in the dark, and the Onsager-Braun yield of the second step at the
        nominal short-circuit field V_bi/d for the smallest zero-field yield (1 if that yield is
        1)] followed by zeros. A (1 + len(zero_field_yields), 15) float64 array: the head row
        and one row per yield.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19
def _KB():
    return 1.380649e-23
def _E0():
    return 8.8541878128e-12


def _NAMES():
    return ["V0", "eta_sc", "Voc", "Vmpp", "FF", "Pgen_sc", "eta_coll", "Jsat_1", "Papp_1", "Jsat_2", "Papp_2",
             "loss_first", "loss_second", "loss_anode", "loss_cathode"]

def _oracle_procedure_audit(zero_field_yields: "np.ndarray", reverse_biases: "np.ndarray", barriers: "np.ndarray", reduction_factors: "np.ndarray", mobility_ideal: float, intensities: "np.ndarray", intervals: int, device: "tuple[float, ...]") -> "np.ndarray":
    """Run the chain for every zero-field yield; head row [apparent dissociation probability of the unity-yield device
    at the first reverse bias, the ideal-transport ceiling at the design barrier, the largest ceiling on the barrier grid
    and its barrier, the Sokel-Hughes value at the design barrier, log10 of the crossover mobility, the low-intensity
    collection efficiency at the design and ideal mobilities, J_sc and the two saturation currents of the unity cell in
    mA/cm^2 from the sweep step, the short-circuit yield of the smallest zero-field yield] followed by one row per yield."""
    P0s = np.asarray(zero_field_yields, dtype=float).ravel(); Vrs = np.asarray(reverse_biases, dtype=float).ravel()
    dev = [float(v) for v in np.asarray(device, dtype=float).ravel()]; N = int(intervals)
    if P0s.size == 0 or Vrs.size != 2 or len(dev) != 12 or N < 4 or N % 2 != 0:
        raise ValueError("yields must be non-empty, two reverse biases, device the 12-entry tuple of step 1 and intervals an even integer of at least 4")
    if not np.any(np.abs(P0s - 1.0) < 1e-12):
        raise ValueError("the unity yield must be among the zero-field yields")
    par = _oracle_device_parameters(*dev)
    d_nm, eps, T, Nc, Nv, Eg, pa, pc, mun, mup, gam, Gex = dev
    material = (d_nm, eps, T, Nc, Nv, Eg, Gex)
    rows = []
    for P0 in P0s:
        m = _oracle_procedure_metrics(P0, Vrs, N, par)
        s = m[5]
        sl = _oracle_steady_state(0.0, s, N, par); sd = _oracle_steady_state(0.0, 0.0, N, par)
        lb = _oracle_loss_budget(sl, sd, 0.0, s, par)
        rows.append(list(m[:11]) + list(lb[3:7]))
    ceil = _oracle_collection_ceiling(barriers, reduction_factors, mobility_ideal, N, material)
    gs = np.asarray(reduction_factors, dtype=float).ravel(); Ph = np.asarray(barriers, dtype=float).ravel()
    gcol = int(np.argmin(np.abs(gs - gam))); col = ceil[:, gcol]; im = int(np.argmax(col)); ip = int(np.argmin(np.abs(Ph - pa)))
    cx = _oracle_transport_crossover(mobility_ideal, mun, pa, gam, N, material)
    inten = _oracle_intensity_dependence(intensities, [mun, mobility_ideal], pa, gam, N, material)
    iu = int(np.argmin(np.abs(P0s - 1.0)))
    jl = _oracle_jv_characteristics([Vrs[1], Vrs[0], 0.0], 1.0, N, par); jd = _oracle_jv_characteristics([Vrs[1], Vrs[0], 0.0], 0.0, N, par)
    Jsc = -jl[2]; Jsat1 = -(jl[1] - jd[1]); Jsat2 = -(jl[0] - jd[0])
    pmin = float(P0s.min()); VT = par[0]; T = VT * _Q() / _KB(); eps = par[3] / _E0()
    pg_min = float(_oracle_onsager_braun_yield(par[1] / par[2], pmin, eps, T)[0]) if pmin < 1.0 else 1.0
    head = [rows[iu][8], col[ip], col[im], Ph[im], ceil[ip, gs.size], cx[0], inten[0, 0], inten[1, 0], Jsc, Jsat1, Jsat2, pg_min]
    T_ = np.zeros((len(rows) + 1, len(_NAMES())))
    T_[0, :len(head)] = head
    T_[1:] = np.array(rows)
    return T_

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nzero_field_yields = np.array([1.0, 0.8, 0.6])\nreverse_biases = np.array([-1.0, -3.0])\nbarriers = np.array([0.15, 0.20, 0.25, 0.35])\nreduction_factors = np.array([0.0, 1.0])\nmobility_ideal = 1.0\nintensities = np.array([0.01, 1.0])\nintervals = 200\ndevice = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\n',
         'call': 'procedure_audit(zero_field_yields, reverse_biases, barriers, reduction_factors, mobility_ideal, intensities, intervals, device)',
         'gold_call': '_oracle_procedure_audit(zero_field_yields, reverse_biases, barriers, reduction_factors, mobility_ideal, intensities, intervals, device)'},
        {'setup': 'import numpy as np\nzero_field_yields = np.array([0.6, 1.0])\nreverse_biases = np.array([-2.0, -0.5])\nbarriers = np.array([0.10, 0.30])\nreduction_factors = np.array([0.2, 1.0])\nmobility_ideal = 0.5\nintensities = np.array([0.1, 1.0])\nintervals = 200\ndevice = (150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)\n',
         'call': 'procedure_audit(zero_field_yields, reverse_biases, barriers, reduction_factors, mobility_ideal, intensities, intervals, device)',
         'gold_call': '_oracle_procedure_audit(zero_field_yields, reverse_biases, barriers, reduction_factors, mobility_ideal, intensities, intervals, device)'},
        {'setup': 'import numpy as np\nzero_field_yields = np.array([1.0])\nreverse_biases = np.array([-1.0, -3.0])\nbarriers = np.array([0.25])\nreduction_factors = np.array([1.0])\nmobility_ideal = 1.0\nintensities = np.array([0.01])\nintervals = 400\ndevice = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\n',
         'call': 'float(np.asarray(procedure_audit(zero_field_yields, reverse_biases, barriers, reduction_factors, mobility_ideal, intensities, intervals, device))[0, 0])',
         'gold_call': 'float(np.asarray(_oracle_procedure_audit(zero_field_yields, reverse_biases, barriers, reduction_factors, mobility_ideal, intensities, intervals, device))[0, 0])'},
    ]
