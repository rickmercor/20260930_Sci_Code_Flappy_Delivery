"""
Assembling the interaction energy of the specified pair

This step runs the whole pipeline at the operating point given in the problem statement and

returns the single number the task asks for. Nothing new is derived here; every ingredient comes

from the earlier steps, and the only work is to feed them the right parameters in the right order

and on the grid the problem statement fixes.



The two molecules are identical and identically driven, so one set of Green's functions serves for

both, and the same charge-fluctuation propagator enters the energy in both of its slots. The

mean-field shift is obtained once, from the electrode-only description, and is then held fixed

while the correlation dressing is iterated to convergence. The propagators that enter the energy

are the converged ones, not the mean-field ones: stopping short of self-consistency at this

operating point changes the answer by more than an order of magnitude, and using the equilibrium

form of the propagators changes it by more still.

Returns
-------
#     float: the dispersion interaction energy of the specified pair, in eV  # ============================================================================
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def final_answer(eps: float = -0.75, u_coupling: float = 0.90, bias_v: float = 2.0,
                 temp_k: float = 300.0, gamma_left: float = 0.050, gamma_right: float = 0.030,
                 frac_left: float = 0.70, e_fermi: float = 0.0, q_core: float = 1.0,
                 w_max: float = 10.0, n: int = 4096, tol: float = 1e-11,
                 max_iter: int = 4000) -> float:
    '''Dispersion interaction energy of the pair of current-carrying molecules specified in the
    problem statement.

    Every parameter defaults to the value the problem statement fixes, so calling this with no
    arguments returns the answer the task asks for.

    Parameters
    ----------
    eps : float
        Bare energy of the single spinless level of each molecule, in eV.
    u_coupling : float
        Intermolecular density-density Coulomb coupling in eV. Must be non-negative.
    bias_v : float
        Applied bias across each junction in volts. May be zero or negative.
    temp_k : float
        Electronic temperature of all four electrodes in kelvin. Must not be zero; a negative
        value describes population-inverted reservoirs.
    gamma_left, gamma_right : float
        Wide-band hybridisation of the left and right electrode in eV. Both must be positive.
    frac_left : float
        Fraction of the applied bias dropped on the left side, in [0, 1].
    e_fermi : float
        Common equilibrium Fermi energy in eV.
    q_core : float
        Positive ionic core charge on each molecule.
    w_max : float
        Half-width of the frequency window in eV. Must be positive.
    n : int
        Number of grid points. Must be even and at least 2.
    tol : float
        Convergence threshold on the largest change in any Green's function component, in eV^-1.
    max_iter : int
        Maximum number of self-consistency passes.

    Returns
    -------
    result : float
        The dispersion interaction energy in eV.

    Raises
    ------
    ValueError
        If either hybridisation is not positive, the coupling is negative, the temperature is
        zero, or the grid does not have a positive half-width and an even number of points.
    RuntimeError
        If the converged solution produces a non-finite Green's function or spectral weight, or
        if the converged spectral weights violate the contract documented in the noise and
        response step.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_final_answer(eps=-0.75, u_coupling=0.90, bias_v=2.0, temp_k=300.0,
                         gamma_left=0.050, gamma_right=0.030, frac_left=0.70, e_fermi=0.0,
                         q_core=1.0, w_max=10.0, n=4096, tol=1e-11, max_iter=4000):
    if not (w_max > 0.0) or int(n) < 2 or int(n) % 2 != 0:
        raise ValueError("the frequency grid needs a positive half-width and an even point count")
    if not (gamma_left > 0.0 and gamma_right > 0.0) or u_coupling < 0.0 or temp_k == 0.0:
        raise ValueError("hybridisations must be positive, the coupling non-negative and the temperature non-zero")

    dw = 2.0 * w_max / n
    w = (np.arange(n) - n // 2) * dw
    gamma_total = gamma_left + gamma_right

    sig = _oracle_lead_selfenergies(w, bias_v, temp_k, gamma_left, gamma_right, frac_left, e_fermi)
    sig_lesser, sig_greater = sig[0], sig[1]

    mf = _oracle_mean_field_occupation(w, dw, eps, gamma_total, sig_lesser, u_coupling, q_core)
    shift = float(mf[1])
    eps_shifted = eps + shift

    g = _oracle_self_consistent_greens(w, dw, eps_shifted, gamma_total, sig_lesser, sig_greater,
                                       u_coupling, tol, max_iter)
    g_lesser, g_greater = g[0], g[1]

    pi = _oracle_polarisation_bubbles(g_lesser, g_greater, dw)
    pi_lesser, pi_greater = pi[0], pi[1]

    # Re-derive the correlation self-energy and the Green's functions from the converged
    # solution, and read off the spectral weights. These exercise the remaining steps on the
    # converged state and are checked against the contracts those steps document.
    sc = _oracle_correlation_selfenergies(g_lesser, g_greater, pi_lesser, pi_greater,
                                          u_coupling, dw)
    sc_ret = _oracle_retarded_from_keldysh(sc[0], sc[1], w, dw)
    chk = _oracle_keldysh_greens(w, eps_shifted, gamma_total, sig_lesser, sig_greater,
                                 sc_ret, sc[0], sc[1])
    weights = _oracle_noise_response_kms(pi_lesser, pi_greater)
    if not (np.all(np.isfinite(chk)) and np.all(np.isfinite(weights))):
        raise RuntimeError("the converged solution produced a non-finite Green's function or spectral weight")
    if weights[:2].min() < -1e-12 or np.abs(weights[2] + weights[3] - weights[0]).max() > 1e-9:
        raise RuntimeError("the converged spectral weights violate the step 08 contract")

    return _oracle_dispersion_energy(pi_lesser, pi_greater, pi_lesser, pi_greater,
                                     w, dw, u_coupling)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SETUP = "import numpy as np"
    _RAISES = ("import numpy as np\n"
               "def _raises(fn):\n"
               "    try:\n"
               "        fn()\n"
               "    except ValueError:\n"
               "        return 1.0\n"
               "    return 0.0\n")

    return [
        {
            "setup": _SETUP,
            "call": 'np.array([final_answer()])',
            "gold_call": 'np.array([_oracle_final_answer()])',
            "note": 'normal, the dispersion interaction energy of the specified pair',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([final_answer(u_coupling=0.0)])',
            "gold_call": 'np.array([_oracle_final_answer(u_coupling=0.0)])',
            "note": 'boundary, zero intermolecular coupling must give exactly zero energy',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([float(np.sign(final_answer(temp_k=-300.0)))])',
            "gold_call": 'np.array([float(np.sign(_oracle_final_answer(temp_k=-300.0)))])',
            "note": 'edge, population-inverted reservoirs must turn the interaction repulsive',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([final_answer(bias_v=0.0)])',
            "gold_call": 'np.array([_oracle_final_answer(bias_v=0.0)])',
            "note": 'boundary, zero bias recovers the equilibrium dispersion energy',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([final_answer(n=1024)])',
            "gold_call": 'np.array([_oracle_final_answer(n=1024)])',
            "note": 'normal, the same operating point on a coarser frequency grid',
        },
        {
            "setup": _SETUP,
            "call": 'np.array([final_answer() / final_answer(bias_v=0.0)])',
            "gold_call": 'np.array([_oracle_final_answer() / _oracle_final_answer(bias_v=0.0)])',
            "note": 'contract, the drive sets the scale: driven over equilibrium, two orders of magnitude',
        },
        {
            "setup": _RAISES,
            "call": ('np.array([_raises(lambda: final_answer(gamma_left=-0.05)),'
                     ' _raises(lambda: final_answer(temp_k=0.0)),'
                     ' _raises(lambda: final_answer(n=4095))])'),
            "gold_call": ('np.array([_raises(lambda: _oracle_final_answer(gamma_left=-0.05)),'
                          ' _raises(lambda: _oracle_final_answer(temp_k=0.0)),'
                          ' _raises(lambda: _oracle_final_answer(n=4095))])'),
            "note": 'contract, a negative hybridisation, zero temperature and an odd grid must raise ValueError',
        },
    ]
