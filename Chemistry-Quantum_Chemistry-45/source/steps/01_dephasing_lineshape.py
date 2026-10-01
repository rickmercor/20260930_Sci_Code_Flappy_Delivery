"""
Build the lineshape function g(t) that controls how fast the donor-acceptor

coherence dephases, keeping only the bath components that carry genuine memory.



The two chromophores each sit in their own overdamped Brownian-oscillator bath, so every

site energy fluctuates with an autocorrelation function that is a sum of exponentials,

<dE(t) dE(0)> = sum_j sigma_j^2 exp(-t / tau_j). What controls the transfer is the

difference of the two site energies, so when the two baths are uncorrelated their

variances add component by component and the second-order cumulant gives one lineshape

function per component.



A component whose correlation time is comparable to, or longer than, the population

transfer itself is not memory at all: on the timescale of the transfer that component is

frozen, and folding it into g(t) makes the higher-order waiting-time integral fail to

converge. Such components must be excluded here and carried separately as static

inhomogeneous broadening of the donor-acceptor gap. The threshold is supplied as

tau_static_fs, and a component is treated as static when tau_j >= tau_static_fs.



Energies arrive as angular wavenumbers in cm^-1 and must be converted to rad/fs with

2 * pi * c, c = 2.99792458e-5 cm/fs, before g(t) is assembled; g(t) itself is dimensionless.



Formulas

--------

Lambda_j = (2 pi c sigma_D,j)^2 + (2 pi c sigma_A,j)^2          [rad^2 fs^-2]

g(t) = sum_{j : tau_j < tau_static} Lambda_j tau_j^2 (exp(-t/tau_j) + t/tau_j - 1)



Returns

-------

np.ndarray of shape (len(t_fs),): the dimensionless lineshape g(t)

Returns
-------
np.ndarray of shape (len(t_fs),): the dimensionless lineshape g(t)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dephasing_lineshape(t_fs, sigma_donor, sigma_acceptor, tau_fs,
                        tau_static_fs: float) -> np.ndarray:
    '''Second-order cumulant lineshape for the donor-acceptor energy gap.

    Parameters
    ----------
    t_fs : array_like
        Times in fs at which g is wanted; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitude of each bath component, in cm^-1,
        for the donor and the acceptor. Same length as tau_fs; entries must be
        finite and >= 0.
    tau_fs : array_like
        Bath correlation time of each component, in fs; entries must be finite and > 0.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are treated as static disorder and
        contribute nothing to g(t). Must be finite and > 0.

    Returns
    -------
    np.ndarray
        Shape (len(t_fs),) array of dimensionless g(t) values, float dtype.

    Raises
    ------
    ValueError
        If the three component lists differ in length, if any component list is
        empty, if any entry is non-finite, if any sigma is negative, if any tau is
        non-positive, if tau_static_fs is non-positive, or if any requested time is
        negative or non-finite.
    '''
    return g_values


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5     # rad fs^-1 per cm^-1


def _s01_check_bath(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd = np.atleast_1d(np.asarray(sigma_donor, dtype=float))
    sa = np.atleast_1d(np.asarray(sigma_acceptor, dtype=float))
    ta = np.atleast_1d(np.asarray(tau_fs, dtype=float))
    if sd.size == 0 or sd.size != sa.size or sd.size != ta.size:
        raise ValueError("sigma_donor, sigma_acceptor and tau_fs must be non-empty and equal length")
    if not (np.all(np.isfinite(sd)) and np.all(np.isfinite(sa)) and np.all(np.isfinite(ta))):
        raise ValueError("bath parameters must all be finite")
    if np.any(sd < 0.0) or np.any(sa < 0.0):
        raise ValueError("fluctuation amplitudes must be non-negative")
    if np.any(ta <= 0.0):
        raise ValueError("correlation times must be strictly positive")
    tstat = float(tau_static_fs)
    if not np.isfinite(tstat) or tstat <= 0.0:
        raise ValueError("tau_static_fs must be finite and strictly positive")
    return sd, sa, ta, tstat


def _oracle_dephasing_lineshape(t_fs, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    sd, sa, ta, tstat = _s01_check_bath(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    t = np.atleast_1d(np.asarray(t_fs, dtype=float))
    if not np.all(np.isfinite(t)) or np.any(t < 0.0):
        raise ValueError("times must be finite and non-negative")
    g = np.zeros_like(t)
    for s_d, s_a, tau in zip(sd, sa, ta):
        if tau >= tstat:
            continue
        lam = (s_d * _TWO_PI_C) ** 2 + (s_a * _TWO_PI_C) ** 2
        x = t / tau
        g = g + lam * tau * tau * (np.exp(-x) + x - 1.0)
    return g


#

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = (
        'import copy\n'
        'def _fp(a):\n'
        '    """Scale-aware fingerprint; every component carries O(1) weight."""\n'
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
        'TG = np.arange(0.0, 220.0 + 0.5, 1.0)\n'
    )
    base = "import numpy as np\n" + guard
    return [
        # normal: the three-component bath of the main problem on the coherence grid
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
        # normal: a single fast component, short times
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape([0.0, 5.0, 25.0, 60.0], [120.0], [95.0], [55.0], 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape([0.0, 5.0, 25.0, 60.0], [120.0], [95.0], [55.0], 1000.0))'},
        # boundary: g(0) must be exactly zero whatever the bath
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape([0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape([0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
        # boundary: threshold exactly at a component tau, so that component is static
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 180.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 180.0))'},
        # edge: a slow but sub-threshold mode, where g is still in its quadratic regime
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape(copy.deepcopy(TG), [200.0], [0.0], [900.0], 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape(copy.deepcopy(TG), [200.0], [0.0], [900.0], 1000.0))'},
        # edge: a zero-amplitude component contributes nothing but must not be rejected
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape(copy.deepcopy(TG), [90.0, 0.0, 55.0], [90.0, 0.0, 55.0], [40.0, 180.0, 600.0], 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape(copy.deepcopy(TG), [90.0, 0.0, 55.0], [90.0, 0.0, 55.0], [40.0, 180.0, 600.0], 1000.0))'},
        # edge: very long times, where g grows linearly with slope sum(Lambda_j tau_j)
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape([2000.0, 4000.0, 8000.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape([2000.0, 4000.0, 8000.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
        # invalid: mismatched component list lengths
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape(copy.deepcopy(TG), [90.0, 110.0], copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape(copy.deepcopy(TG), [90.0, 110.0], copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
        # invalid: a non-positive correlation time
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), [40.0, 0.0, 3000.0], 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape(copy.deepcopy(TG), copy.deepcopy(SD), copy.deepcopy(SA), [40.0, 0.0, 3000.0], 1000.0))'},
        # invalid: a negative time
        {"setup": base,
         "call": '_v(lambda: dephasing_lineshape([-1.0, 0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))',
         "gold_call": '_v(lambda: _oracle_dephasing_lineshape([-1.0, 0.0], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0))'},
    ]
