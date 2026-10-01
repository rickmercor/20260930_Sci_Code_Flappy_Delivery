"""
Turn the response function into the lowest-order incoherent transfer rate for one

frozen value of the donor-acceptor gap.



The rate is twice the real part of the time integral of the response, scaled by the square

of the inter-site electronic coupling. Working in units where hbar = 1 and every energy is

an angular frequency, the coupling is converted from cm^-1 with the same 2 pi c factor used

for the gap, so the bracket has units of inverse femtoseconds; the result is reported in

inverse picoseconds.



The time integral runs over a uniform grid from 0 to t_max_fs with spacing dt_fs, evaluated

with the trapezoidal rule. The grid is fixed by n = round(t_max_fs / dt_fs) + 1 points placed

by numpy.linspace between 0 and t_max_fs inclusive, so that the endpoint is hit exactly.

t_max_fs must be long enough that the response has decayed; that is a property of the bath,

not something this routine checks.



At infinite temperature the backward rate is obtained by flipping the sign of the gap, and

because that conjugates the response it leaves the real part unchanged. The equilibration

rate between the pair is therefore exactly twice this number.



Formulas

--------

J_ang = 2 pi c * j_cm;  S = trapz(R(t), t) over the grid

k2 = 2 * J_ang^2 * Re(S) * 1e3          [ps^-1]



Returns

-------

float, the second-order transfer rate in ps^-1

Returns
-------
float, the second-order transfer rate in ps^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def second_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                      gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float) -> float:
    '''Lowest-order incoherent transfer rate at one frozen gap.

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
        Electronic coupling between the two chromophores, in cm^-1. Any finite real
        value is allowed; only its square enters.
    t_max_fs : float
        Upper limit of the coherence-time integral, in fs; must be finite and > 0.
    dt_fs : float
        Grid spacing of that integral, in fs; must be finite, > 0 and <= t_max_fs.

    Returns
    -------
    float
        Transfer rate in ps^-1.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, or if t_max_fs or dt_fs is non-positive or dt_fs exceeds t_max_fs.
    '''
    return rate_ps


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_S04_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s04_grid(t_max_fs, dt_fs):
    tmax, dt = float(t_max_fs), float(dt_fs)
    if not np.isfinite(tmax) or tmax <= 0.0:
        raise ValueError("t_max_fs must be finite and strictly positive")
    if not np.isfinite(dt) or dt <= 0.0 or dt > tmax:
        raise ValueError("dt_fs must be finite, strictly positive and at most t_max_fs")
    return np.linspace(0.0, tmax, int(round(tmax / dt)) + 1)


def _oracle_second_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                              gap_cm, j_cm, t_max_fs, dt_fs):
    j = float(j_cm)
    if not np.isfinite(j):
        raise ValueError("j_cm must be finite")
    t = _s04_grid(t_max_fs, dt_fs)
    resp = _oracle_second_order_response(t, sigma_donor, sigma_acceptor, tau_fs,
                                         tau_static_fs, gap_cm)
    s = np.trapezoid(resp, t)
    j_ang = j * _S04_TWO_PI_C
    return float(2.0 * j_ang * j_ang * s.real * 1.0e3)


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
        # normal: the main problem at its mean gap
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 1.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 1.0))'},
        # normal: the same gap on a finer grid, which must barely move the answer
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 0.5))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 0.5))'},
        # boundary: flipping the sign of the gap must give the same rate
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0, 26.0, 220.0, 1.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, -120.0, 26.0, 220.0, 1.0))'},
        # boundary: quadratic in the coupling, so halving j must quarter the rate
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 13.0, 220.0, 1.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 13.0, 220.0, 1.0))'},
        # edge: a gap far out in the static wing, where the rate is strongly suppressed
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 566.6, 26.0, 220.0, 1.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 566.6, 26.0, 220.0, 1.0))'},
        # edge: a single grid interval, the coarsest legal trapezoid
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 220.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 220.0))'},
        # edge: a zero coupling gives exactly zero rate
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 0.0, 220.0, 1.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 0.0, 220.0, 1.0))'},
        # invalid: a spacing larger than the whole interval
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 400.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 220.0, 400.0))'},
        # invalid: a non-positive upper limit
        {"setup": base,
         "call": '_v(lambda: second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 0.0, 1.0))',
         "gold_call": '_v(lambda: _oracle_second_order_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0, 26.0, 0.0, 1.0))'},
    ]
