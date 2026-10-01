"""
Return the chirality-resolved, Fano-corrected peak separation.

## The end-to-end calculation chains the preceding steps: mass weighting, the curvature

construction, the unperturbed degenerate analysis, the projected first-order pair with

its displacement basis, the full quadratic solution, projection matching, and chirality

labelling. The intrinsic frequencies are not directly observable when a phonon

interferes with an electronic continuum: the measured line shape becomes asymmetric and

its peak is displaced from the bare frequency by an amount set by that branch's

linewidth and its Fano asymmetry parameter. Applying the branch-resolved correction with

the parameters indexed in ${(+,-)}$ order, taking the absolute separation of the

corrected peaks and converting it to $\mathrm{cm}^{-1}$ gives the reported observable.

Returns
-------
separation : float     Absolute corrected peak separation in $\mathrm{cm}^{-1}$, rounded at the final step     to 11 decimal places.  Raises ------ ValueError     Under any invalid-input condition declared by the preceding seven     steps; if gamma or inv_q is not a finite one-dimensional array of     length two; if either linewidth is negative; or if conversion is not     strictly positive and finite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_corrected_peak_separation(
    phi: "np.ndarray",
    masses: "np.ndarray",
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
    omega0: float,
    xy_pairs: "np.ndarray",
    gamma: "np.ndarray",
    inv_q: "np.ndarray",
    conversion: float,
) -> float:
    r"""Return the chirality-resolved, Fano-corrected peak separation.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.
    eps_v : np.ndarray
        One-dimensional valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex coupling array with shape $(N_k, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    xy_pairs : np.ndarray
        Integer Cartesian-pair array covering all $N$ coordinates exactly once.
    gamma : np.ndarray
        Two finite nonnegative linewidths ordered [plus, minus].
    inv_q : np.ndarray
        Two finite inverse Fano parameters ordered [plus, minus].
    conversion : float
        Strictly positive finite conversion factor from meV to $\mathrm{cm}^{-1}$.

    Returns
    -------
    separation : float
        Absolute corrected peak separation in $\mathrm{cm}^{-1}$, rounded at the final step
        to 11 decimal places.

    Raises
    ------
    ValueError
        Under any invalid-input condition declared by the preceding seven
        steps; if gamma or inv_q is not a finite one-dimensional array of
        length two; if either linewidth is negative; or if conversion is not
        strictly positive and finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_corrected_peak_separation(
    phi: "np.ndarray",
    masses: "np.ndarray",
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
    omega0: float,
    xy_pairs: "np.ndarray",
    gamma: "np.ndarray",
    inv_q: "np.ndarray",
    conversion: float,
) -> float:
    r"""Return the chirality-resolved, Fano-corrected peak separation.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.
    eps_v : np.ndarray
        One-dimensional valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex coupling array with shape $(N_k, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    xy_pairs : np.ndarray
        Integer Cartesian-pair array covering all $N$ coordinates exactly once.
    gamma : np.ndarray
        Two finite nonnegative linewidths ordered [plus, minus].
    inv_q : np.ndarray
        Two finite inverse Fano parameters ordered [plus, minus].
    conversion : float
        Strictly positive finite conversion factor from meV to $\mathrm{cm}^{-1}$.

    Returns
    -------
    separation : float
        Absolute corrected peak separation in $\mathrm{cm}^{-1}$, rounded at the final step
        to 11 decimal places.

    Raises
    ------
    ValueError
        Under any invalid-input condition declared by the preceding seven
        steps; if gamma or inv_q is not a finite one-dimensional array of
        length two; if either linewidth is negative; or if conversion is not
        strictly positive and finite.
    """
    try:
        linewidths = np.asarray(gamma, dtype=float)
        inverse_q = np.asarray(inv_q, dtype=float)
        unit_conversion = float(conversion)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gamma, inv_q, and conversion must be real numeric values") from exc
    if linewidths.ndim != 1 or linewidths.shape != (2,):
        raise ValueError("gamma must be a one-dimensional array of length two")
    if inverse_q.ndim != 1 or inverse_q.shape != (2,):
        raise ValueError("inv_q must be a one-dimensional array of length two")
    if not np.all(np.isfinite(linewidths)) or not np.all(np.isfinite(inverse_q)):
        raise ValueError("gamma and inv_q must be finite")
    if np.any(linewidths < 0.0):
        raise ValueError("linewidths must be nonnegative")
    if not np.isfinite(unit_conversion) or unit_conversion <= 0.0:
        raise ValueError("conversion must be strictly positive and finite")

    mass_weighted = _oracle_build_mass_weighted_dynamical_matrix(phi, masses)
    curvature = _oracle_compute_molecular_berry_curvature(eps_v, eps_c, g_cv)
    unperturbed = _oracle_analyze_unperturbed_subspace(mass_weighted, omega0)
    first_order_modes = _oracle_compute_first_order_frequencies(
        unperturbed,
        curvature,
        omega0,
    )
    full_modes = _oracle_solve_full_mbc_eigenproblem(mass_weighted, curvature)
    matched = _oracle_match_target_modes(first_order_modes, full_modes)
    branch_data = _oracle_label_modes_by_chirality(matched, xy_pairs)

    corrected_peaks = branch_data[0] + 0.5 * linewidths * inverse_q
    separation = abs(float(corrected_peaks[1] - corrected_peaks[0]))
    return float(np.round(separation * unit_conversion, 11))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    main_setup = '''phi=np.array([
[59679.484666666664,0.0,-29804.330987819010,0.0,-22269.663077830097,0.0],
[0.0,82329.401000000000,0.0,2341.968010268314,0.0,-34773.709425369052],
[-29804.330987819010,0.0,120213.660000000000,0.0,-31606.617761896014,0.0],
[0.0,2341.968010268314,0.0,165837.870000000000,0.0,-49353.209257352260],
[-22269.663077830097,0.0,-31606.617761896014,0.0,37467.453333333338,0.0],
[0.0,-34773.709425369052,0.0,-49353.209257352260,0.0,71333.500000000015]
],dtype=float)
masses=np.array([58.933,58.933,118.710,118.710,32.060,32.060])
eps_v=np.array([-1.20,-1.05,-1.35,-1.10,-1.28])
eps_c=np.array([0.95,1.10,0.88,1.22,1.03])
g=np.array([
[0.653499585001107+0.049682995078799j,0.063874877690685-0.374064565267564j,-0.519956847935370+0.049682995078799j,0.063874877690685+0.799391867668913j,0.473268600995773-0.099365990157599j,0.063874877690685-0.109146657832457j],
[0.451756613101524+0.026843836803511j,0.034511743703323-0.719329299328893j,-0.917275891991032+0.026843836803511j,0.034511743703323+0.649703205763663j,-0.029511023213302-0.053687673607021j,0.034511743703323+0.227716415849677j],
[0.860830739376333-0.020675421232230j,-0.026581328285814-0.640894033485497j,-0.573393789768250-0.020675421232230j,-0.026581328285814+0.793330495659086j,0.143718474804042+0.041350842464459j,-0.026581328285814-0.152436462173589j],
[0.808183469783874-0.049185792336648j,-0.063235649635050-0.508027567868974j,-0.300080939100576-0.049185792336648j,-0.063235649635050+0.600236841015477j,0.050802649110197+0.098371584673296j,-0.063235649635050-0.250299595430949j],
[0.633055030255410-0.032474972798655j,-0.041751406335960-0.839177814692008j,-0.670785450785120-0.032474972798655j,-0.041751406335960+0.464662666348522j,-0.425362442727759+0.064949945597311j,-0.041751406335960+0.058334503774594j]
],dtype=complex)
omega0=37.0
pairs=np.array([[0,1],[2,3],[4,5]])
gamma=np.array([0.095,0.095])
inv_q=np.array([0.775,0.0])
conversion=8.065543937
'''
    return [
        {
            "setup": main_setup,
            "call": "compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
            "gold_call": "_oracle_compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
        },
        {
            "setup": "phi=9.0*np.eye(2); masses=np.ones(2); eps_v=np.array([-1.0]); eps_c=np.array([1.0]); g=np.array([[1.0+0.0j,0.0+1.0j]]); omega0=3.0; pairs=np.array([[0,1]]); gamma=np.zeros(2); inv_q=np.zeros(2); conversion=1.0",
            "call": "compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
            "gold_call": "_oracle_compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
        },
        {
            "setup": "phi=25.0*np.eye(2); masses=np.ones(2); eps_v=np.array([-2.0,-1.0]); eps_c=np.array([1.0,2.0]); g=np.array([[0.7+0.2j,-0.1+0.8j],[0.3-0.4j,0.9+0.1j]]); omega0=5.0; pairs=np.array([[0,1]]); gamma=np.array([0.02,0.11]); inv_q=np.array([-0.4,0.6]); conversion=8.0",
            "call": "compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
            "gold_call": "_oracle_compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
        },
        {
            "setup": "phi=9.0*np.eye(2); masses=np.ones(2); eps_v=np.array([-1.0]); eps_c=np.array([1.0]); g=np.array([[1.0+0.0j,0.0+1.0j]]); omega0=3.0; pairs=np.array([[0,1]]); gamma=np.zeros(2); inv_q=np.zeros(2); conversion=1.0e-12",
            "call": "compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
            "gold_call": "_oracle_compute_corrected_peak_separation(phi,masses,eps_v,eps_c,g,omega0,pairs,gamma,inv_q,conversion)",
        },
    ]
