"""
Subtract the plateau from the rate profile and integrate what is left over the

waiting time, giving the correction that is added to the lowest-order rate.



Only the part of the profile that decays carries genuine fourth-order transfer. The constant

part is a pair of independent lowest-order hops and belongs to the quadratic-in-time growth

of the acceptor population, not to a rate, so integrating it would give a number that simply

grows with the waiting window rather than converging. Removing the plateau first leaves an

integrand that decays to zero on the timescale of the bath memory, and its integral is finite.



The waiting-time integral uses the trapezoidal rule on the same grid as the profile, but the

waiting times must be expressed in picoseconds so that a profile in ps^-2 integrates to a rate

in ps^-1. The window must be long enough that the integrand has reached zero; when it has not,

the result drifts with tw_max_fs, which is the practical signal that the window is too short.



The correction is negative whenever the coupling is non-zero, so including it always lowers

the predicted transfer rate. Its magnitude grows as the fourth power of the coupling and

increases with the bath memory, so it matters most when the segments are close together and

the environment is slow.



Formulas

--------

k4 = trapz( profile(tw) - C, tw/1000 )               [ps^-1]



Returns

-------

float, the fourth-order rate correction in ps^-1

Returns
-------
float, the fourth-order rate correction in ps^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fourth_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                      gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float,
                      tw_max_fs: float, dtw_fs: float) -> float:
    '''Fourth-order correction to the transfer rate at one frozen gap.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of both coherence-interval integrals, in fs.
    tw_max_fs, dtw_fs : float
        Upper limit and spacing of the waiting-time grid, in fs.

    Returns
    -------
    float
        The fourth-order rate correction in ps^-1; negative for any non-zero coupling.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if either grid limit is non-positive, either spacing is
        non-positive, or a spacing exceeds its own limit.
    '''
    return k4_ps


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fourth_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                              gap_cm, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs):
    profile = _oracle_waiting_time_profile(sigma_donor, sigma_acceptor, tau_fs,
                                           tau_static_fs, gap_cm, j_cm,
                                           t_max_fs, dt_fs, tw_max_fs, dtw_fs)
    plateau = _oracle_plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                       gap_cm, j_cm, t_max_fs, dt_fs)
    twmax, dtw = float(tw_max_fs), float(dtw_fs)
    tw_ps = np.linspace(0.0, twmax, int(round(twmax / dtw)) + 1) * 1.0e-3
    return float(np.trapezoid(profile - plateau, tw_ps))


#

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = (
        'import copy\n'
        'def _fp(a):\n'
        '    a = np.asarray(a)\n'
        '    if np.iscomplexobj(a):\n'
        '        a = np.concatenate([np.real(a).ravel(), np.imag(a).ravel()])\n'
        '    a = np.asarray(a, dtype=float).ravel()\n'
        '    w = 1.0 + (np.arange(a.size) % 97) / 97.0\n'
        '    return float(np.sum(w * a / (1.0 + np.abs(a))))\n'
        'def _v(fn):\n'
        '    try:\n'
        '        out = fn()\n'
        '        if isinstance(out, (tuple, list)):\n'
        '            return float(sum(_fp(x) * (i + 1) for i, x in enumerate(out)))\n'
        '        return _fp(out)\n'
        '    except ValueError:\n'
        '        return -1.2345e6\n'
        '    except Exception:\n'
        '        return -9.8765e6\n'
        'SD = [90.0, 110.0, 70.0]\n'
        'SA = [90.0, 110.0, 70.0]\n'
        'TAU = [40.0, 180.0, 3000.0]\n'
    )
    base = "import numpy as np\n" + guard
    return [
        # normal: the main problem at its mean gap on the production grid
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 2.0, 2400.0, 4.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 2.0, 2400.0, 4.0))'},
        # normal: a gap one static standard deviation above the mean
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 218.99, 26.0, 160.0, 4.0, 1600.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 218.99, 26.0, 160.0, 4.0, 1600.0, 8.0))'},
        # boundary: flipping the gap sign leaves the correction unchanged
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0, 26.0, 160.0, 4.0, 1600.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0, 26.0, 160.0, 4.0, 1600.0, 8.0))'},
        # boundary: a zero coupling gives exactly zero correction
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 0.0, 160.0, 4.0, 1600.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 0.0, 160.0, 4.0, 1600.0, 8.0))'},
        # edge: quartic scaling, half the coupling gives a sixteenth of the correction
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 13.0, 160.0, 4.0, 1600.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 13.0, 160.0, 4.0, 1600.0, 8.0))'},
        # edge: a truncated waiting window, which leaves the integral short of convergence
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 160.0, 4.0, 400.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 160.0, 4.0, 400.0, 8.0))'},
        # edge: one fast component only, where the memory and the correction are both small
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate([140.0], [140.0], [30.0], 1000.0, 120.0, 26.0, 160.0, 4.0, 800.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate([140.0], [140.0], [30.0], 1000.0, 120.0, 26.0, 160.0, 4.0, 800.0, 8.0))'},
        # invalid: a non-positive waiting-time window
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 160.0, 4.0, 0.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 160.0, 4.0, 0.0, 8.0))'},
        # invalid: a negative bath correlation time
        {"setup": base,
         "call": '_v(lambda: fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), [40.0, -180.0, 3000.0], 1000.0, 120.0, 26.0, 160.0, 4.0, 800.0, 8.0))',
         "gold_call": '_v(lambda: _oracle_fourth_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), [40.0, -180.0, 3000.0], 1000.0, 120.0, 26.0, 160.0, 4.0, 800.0, 8.0))'},
    ]
