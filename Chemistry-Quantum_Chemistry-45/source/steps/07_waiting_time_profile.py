"""
Integrate the two kernels over both coherence intervals to obtain the fourth-order

rate density as a function of the waiting time alone.



At each waiting time the two coherence intervals are integrated out over the same uniform

grid used for the second-order rate, the two kernels are summed, and the result is scaled into

a rate density. The scale factor is a signed integer A times the fourth power of the coupling.

A is not given here and must be worked out. Its sign follows from the fact that this order of

perturbation theory removes population the lowest order has already counted. Its magnitude

follows from how many distinct time orderings of the four interaction vertices survive in

total, given that the two kernels summed inside F cover only those orderings in which the

waiting interval sits on the acceptor, and that for a pair of single chromophores the

remaining orderings contribute equally to those two.



The two coherence integrals both use the trapezoidal rule on the grid of

n = round(t_max_fs / dt_fs) + 1 points from 0 to t_max_fs, and the waiting times are the

n_w = round(tw_max_fs / dtw_fs) + 1 points from 0 to tw_max_fs. Only the real part survives.

Units are chosen so that the profile is a rate per unit time in ps^-2: multiply the doubly

integrated kernel, which carries fs^2, by 1e6.



This profile does not decay to zero. It approaches a non-zero constant, and what that

constant is, and why it has to be removed before the waiting-time integral is taken, is the

subject of the next two steps.



Formulas

--------

J_ang = 2 pi c * j_cm

F(tw) = trapz_t1 trapz_t3 [ Phi_NR(t1,tw,t3) + Phi_R(t1,tw,t3) ]

profile(tw) = A * J_ang^4 * Re F(tw) * 1e6           [ps^-2],  A a signed integer to determine



Returns

-------

np.ndarray of shape (round(tw_max_fs/dtw_fs) + 1,), float dtype, in ps^-2

Returns
-------
np.ndarray of shape (round(tw_max_fs/dtw_fs) + 1,), float dtype, in ps^-2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def waiting_time_profile(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                         gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float,
                         tw_max_fs: float, dtw_fs: float) -> np.ndarray:
    '''Fourth-order rate density as a function of the waiting time.

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
    np.ndarray
        Real array over the waiting-time grid, in ps^-2.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if either grid limit is non-positive, either spacing is
        non-positive, or a spacing exceeds its own limit.
    '''
    return profile_ps2


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_S07_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s07_grid(t_max_fs, dt_fs, what):
    tmax, dt = float(t_max_fs), float(dt_fs)
    if not np.isfinite(tmax) or tmax <= 0.0:
        raise ValueError("%s upper limit must be finite and strictly positive" % what)
    if not np.isfinite(dt) or dt <= 0.0 or dt > tmax:
        raise ValueError("%s spacing must be finite, positive and at most its limit" % what)
    return np.linspace(0.0, tmax, int(round(tmax / dt)) + 1)


def _oracle_waiting_time_profile(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                 gap_cm, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs):
    j = float(j_cm)
    if not np.isfinite(j):
        raise ValueError("j_cm must be finite")
    t = _s07_grid(t_max_fs, dt_fs, "coherence")
    tw = _s07_grid(tw_max_fs, dtw_fs, "waiting")
    t1 = t[:, None, None]
    t3 = t[None, None, :]
    nt = t.size
    chunk = max(1, int(4.0e6 // (nt * nt)))
    out = np.empty(tw.size, dtype=float)
    for a in range(0, tw.size, chunk):
        b = min(a + chunk, tw.size)
        twb = tw[None, a:b, None]
        block = (_oracle_nonrephasing_kernel(t1, twb, t3, sigma_donor, sigma_acceptor,
                                             tau_fs, tau_static_fs, gap_cm)
                 + _oracle_rephasing_kernel(t1, twb, t3, sigma_donor, sigma_acceptor,
                                            tau_fs, tau_static_fs, gap_cm))
        out[a:b] = np.trapezoid(np.trapezoid(block, t, axis=2), t, axis=0).real
    j_ang = j * _S07_TWO_PI_C
    return -4.0 * j_ang ** 4 * out * 1.0e6


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
        # normal: the main problem's grid at its mean gap
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 2.0, 2400.0, 4.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 200.0, 2.0, 2400.0, 4.0))'},
        # normal: a short waiting window on a coarse grid, cheap but structurally identical
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 120.0, 10.0, 300.0, 25.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 120.0, 10.0, 300.0, 25.0))'},
        # boundary: a gap displaced into the static wing changes the whole profile
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 366.5, 26.0, 120.0, 10.0, 300.0, 25.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 366.5, 26.0, 120.0, 10.0, 300.0, 25.0))'},
        # boundary: a single waiting point, tw = 0 only
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 120.0, 10.0, 50.0, 50.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 120.0, 10.0, 50.0, 50.0))'},
        # edge: quartic in the coupling, so halving j divides the profile by sixteen
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 13.0, 120.0, 10.0, 300.0, 25.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 13.0, 120.0, 10.0, 300.0, 25.0))'},
        # edge: a single fast bath component, where the profile is nearly flat
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile([140.0], [140.0], [30.0], 1000.0, 120.0, 26.0, 120.0, 10.0, 300.0, 25.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile([140.0], [140.0], [30.0], 1000.0, 120.0, 26.0, 120.0, 10.0, 300.0, 25.0))'},
        # edge: a zero gap
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0, 26.0, 120.0, 10.0, 300.0, 25.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0, 26.0, 120.0, 10.0, 300.0, 25.0))'},
        # invalid: a waiting spacing wider than the waiting window
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 120.0, 10.0, 300.0, 500.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 120.0, 10.0, 300.0, 500.0))'},
        # invalid: a non-positive coherence limit
        {"setup": base,
         "call": '_v(lambda: waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, -120.0, 10.0, 300.0, 25.0))',
         "gold_call": '_v(lambda: _oracle_waiting_time_profile(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, -120.0, 10.0, 300.0, 25.0))'},
    ]
