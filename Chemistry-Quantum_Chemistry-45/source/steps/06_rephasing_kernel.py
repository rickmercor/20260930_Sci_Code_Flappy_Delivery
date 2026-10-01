"""
Build the rephasing member of the fourth-order transfer response, the one whose two

coherence intervals accumulate phase in opposite directions.



In this diagram the phase acquired during the first coherence interval is undone during the

second, so the two intervals appear as a difference in the oscillatory factor. Every sign in

the waiting-time-dependent group of lineshape terms is also reversed relative to its

non-rephasing partner.



The three time arguments are the two coherence intervals t1 and t3, during which the donor

and acceptor manifolds are propagated in opposite directions and the pair loses its mutual

coherence, and the waiting interval tw between them, during which both sides of the diagram

sit in the same manifold. The three arguments broadcast against one another, so passing a

column, a row and a depth vector returns the full three-dimensional block in one call.



The same six lineshape evaluations enter as in the non-rephasing kernel, at the three

intervals on their own and at the three consecutive sums. The two terms that depend on t1 or

t3 alone keep the signs they carry there; every term that involves the waiting interval

carries the opposite sign. The signs are not given here and must be worked out, and the

long-waiting-time limit below fixes them uniquely.



Formulas

--------

omega = 2 pi c * gap_cm,  c = 2.99792458e-5 cm/fs

Phi = exp(-i omega (t3 - t1)) * exp( s1 g(t1) + s3 g(t3) + sw g(tw)

                                     + s1w g(t1+tw) + sw3 g(tw+t3) + s13 g(t1+tw+t3) )

with each s in {+1, -1}, determined by requiring, with R the second-order response of step 3,

    Phi -> conj(R(t1)) R(t3)   as tw -> infinity, for every t1 and t3

Unlike its non-rephasing partner this kernel is not a function of a single combined time at

tw = 0, so that limit gives no further constraint here.



Returns

-------

np.ndarray, complex, of the broadcast shape of t1_fs, tw_fs and t3_fs

Returns
-------
np.ndarray, complex, of the broadcast shape of t1_fs, tw_fs and t3_fs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rephasing_kernel(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs,
                     tau_static_fs: float, gap_cm: float) -> np.ndarray:
    '''rephasing kernel of the fourth-order transfer response.

    Parameters
    ----------
    t1_fs, tw_fs, t3_fs : array_like
        First coherence interval, waiting interval and second coherence interval,
        in fs. They must broadcast together; every entry must be finite and >= 0.
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static and excluded from g(t).
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.

    Returns
    -------
    np.ndarray
        Complex array of the broadcast shape.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm is not finite,
        if any time is negative or non-finite, or if the three time arguments do
        not broadcast together.
    '''
    return kernel


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_S06_TWO_PI_C = 2.0 * np.pi * 2.99792458e-5


def _s06_pieces(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs):
    t1 = np.asarray(t1_fs, dtype=float)
    tw = np.asarray(tw_fs, dtype=float)
    t3 = np.asarray(t3_fs, dtype=float)
    try:
        t1b, twb, t3b = np.broadcast_arrays(t1, tw, t3)
    except ValueError:
        raise ValueError("t1_fs, tw_fs and t3_fs must broadcast together")

    def g(x):
        return _oracle_dephasing_lineshape(x, sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)

    g1 = g(t1b)
    g3 = g(t3b)
    gw = g(twb)
    h = -gw + g(t1b + twb) + g(twb + t3b) - g(t1b + twb + t3b)
    return t1b, twb, t3b, g1, g3, h


def _oracle_rephasing_kernel(t1_fs, tw_fs, t3_fs, sigma_donor, sigma_acceptor, tau_fs,
                             tau_static_fs, gap_cm):
    gap = float(gap_cm)
    if not np.isfinite(gap):
        raise ValueError("gap_cm must be finite")
    t1b, twb, t3b, g1, g3, h = _s06_pieces(t1_fs, tw_fs, t3_fs, sigma_donor,
                                               sigma_acceptor, tau_fs, tau_static_fs)
    w = gap * _S06_TWO_PI_C
    return np.exp(-1j * w * (t3b - t1b)) * np.exp(-g1 - g3 - h)


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
        'TS = np.linspace(0.0, 220.0, 12)\n'
        'BLK1 = TS[:, None, None]\n'
        'BLKW = np.array([0.0, 60.0, 240.0, 900.0])[None, :, None]\n'
        'BLK3 = TS[None, None, :]\n'
    )
    base = "import numpy as np\n" + guard
    return [
        # normal: the full three-dimensional block on the main problem's bath
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(copy.deepcopy(BLK1), copy.deepcopy(BLKW), copy.deepcopy(BLK3), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(copy.deepcopy(BLK1), copy.deepcopy(BLKW), copy.deepcopy(BLK3), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # normal: a single interior point of that block
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(37.0, 150.0, 88.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(37.0, 150.0, 88.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # boundary: all three intervals zero, where the kernel must be exactly 1
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(0.0, 0.0, 0.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(0.0, 0.0, 0.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # boundary: zero waiting time, where the two coherence intervals are fully correlated
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(copy.deepcopy(TS)[:, None], 0.0, copy.deepcopy(TS)[None, :], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(copy.deepcopy(TS)[:, None], 0.0, copy.deepcopy(TS)[None, :], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # edge: a long waiting time, where the kernel must factorise into two responses
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(copy.deepcopy(TS)[:, None], 6000.0, copy.deepcopy(TS)[None, :], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(copy.deepcopy(TS)[:, None], 6000.0, copy.deepcopy(TS)[None, :], copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # edge: a zero gap removes the oscillatory factor entirely
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(copy.deepcopy(BLK1), copy.deepcopy(BLKW), copy.deepcopy(BLK3), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(copy.deepcopy(BLK1), copy.deepcopy(BLKW), copy.deepcopy(BLK3), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 0.0))'},
        # edge: with every mode static the kernel is a pure phase of unit modulus
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(copy.deepcopy(BLK1), copy.deepcopy(BLKW), copy.deepcopy(BLK3), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(copy.deepcopy(BLK1), copy.deepcopy(BLKW), copy.deepcopy(BLK3), copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1.0, 120.0))'},
        # invalid: a negative coherence interval
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(-10.0, 100.0, 50.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(-10.0, 100.0, 50.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
        # invalid: time arguments that cannot broadcast together
        {"setup": base,
         "call": '_v(lambda: rephasing_kernel(np.zeros(3), np.zeros(4), 0.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))',
         "gold_call": '_v(lambda: _oracle_rephasing_kernel(np.zeros(3), np.zeros(4), 0.0, copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), 1000.0, 120.0))'},
    ]
