"""
Update the positive importance weight using the physical overlap phase.

The overlap ratio controls the cosine projection. Its phase differs from that of the full importance factor when the force-biased Gaussian correction is complex.

Returns
-------
return updated_weight, phase
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phaseless_weight(old_overlap: "np.ndarray", new_overlap: "np.ndarray", gauge: "np.ndarray", log_gaussian: "np.ndarray", weight: float, dt: float, energy_shift: float) -> "tuple[float, float]":
    """Update the positive importance weight using the physical overlap phase.
    
    Packed complex arrays have a final axis [real, imaginary]. All inputs are finite and must not be modified. NumPy and SciPy are available; earlier public functions may be called.
    
    Parameters
    ----------
    old_overlap, new_overlap, gauge, log_gaussian : real ndarrays, shape (2,)
        Packed complex scalars. new_overlap is measured on normalized Q walkers;
        gauge converts Q to the raw propagated determinant and may be complex for this standalone function.
    weight : float
        Current nonnegative weight.
    dt : float
        Nonnegative time step, inverse Eh.
    energy_shift : float
        Real constant in Eh.
    Returns
    -------
    updated_weight : float
        weight*abs(I)*max(0,cos(theta)), where S=gauge*new_overlap/old_overlap,
        I=S*exp(log_gaussian+dt*energy_shift), theta=arg(S).
    phase : float
        theta in radians, the principal argument from numpy.angle.
    Raises
    ------
    ValueError
        If abs(old_overlap)<=1e-12, weight<0, or dt<0.
    """
    return updated_weight, phase

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_phaseless_weight(old_overlap: "np.ndarray", new_overlap: "np.ndarray", gauge: "np.ndarray", log_gaussian: "np.ndarray", weight: float, dt: float, energy_shift: float) -> "tuple[float, float]":
    def _unpack(a):
        a = np.asarray(a,dtype=float)
        return a[0]+1j*a[1]
    old,new,g,lg = [_unpack(x) for x in (old_overlap,new_overlap,gauge,log_gaussian)]
    if abs(old)<=1e-12 or weight < 0 or dt < 0:
        raise ValueError('Nonzero old overlap, nonnegative weight and dt required.')
    ratio = g*new/old
    phase = float(np.angle(ratio))
    importance = ratio*np.exp(lg+dt*energy_shift)
    updated = weight*abs(importance)*max(0.0,np.cos(phase))
    return float(updated),phase

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge comparisons against the oracle."""
    checks = '''
def _checked(fn, *args):
    copied = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in args]
    before = [arg.copy() if isinstance(arg, np.ndarray) else arg for arg in copied]
    try:
        result = fn(*copied)
    finally:
        for original, current in zip(before, copied):
            if isinstance(original, np.ndarray) and not np.array_equal(original, current):
                raise AssertionError('Input arrays must not be modified')
    expected_shapes = ((), ())
    if not isinstance(result, (tuple, list)) or len(result) != len(expected_shapes):
        raise AssertionError('Return structure does not match the documented tuple')
    arrays = [np.asarray(value, dtype=float) for value in result]
    if any(value.shape != shape for value, shape in zip(arrays, expected_shapes)):
        raise AssertionError('Return shapes do not match the documented contract')
    return np.concatenate([value.ravel() for value in arrays])
'''
    setup_0 = '''import numpy as np
def pack(z):
    z=np.asarray(z)
    return np.stack((z.real,z.imag),axis=-1)

'''
    setup_1 = '''import numpy as np
def pack(z):
    z=np.asarray(z)
    return np.stack((z.real,z.imag),axis=-1)


def _raises(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError('Expected ValueError')
'''
    return [
        {
            "setup": setup_0 + checks,
            "call": '_checked(phaseless_weight, pack(0.8 + 0.2j), pack(0.6 + 0.7j), pack(0.7 - 0.3j), pack(0.12 + 0.4j), 1.3, 0.08, -1.2)',
            "gold_call": '_checked(_oracle_phaseless_weight, pack(0.8 + 0.2j), pack(0.6 + 0.7j), pack(0.7 - 0.3j), pack(0.12 + 0.4j), 1.3, 0.08, -1.2)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(phaseless_weight, pack(1.0), pack(-0.7 + 0.2j), pack(1.0), pack(0.2 - 1j), 0.9, 0.1, -0.8)',
            "gold_call": '_checked(_oracle_phaseless_weight, pack(1.0), pack(-0.7 + 0.2j), pack(1.0), pack(0.2 - 1j), 0.9, 0.1, -0.8)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(phaseless_weight, pack(1.0 + 0.1j), pack(0.8 - 0.3j), pack(1.2), pack(0.1 + 0.2j), 0.0, 0.2, 0.3)',
            "gold_call": '_checked(_oracle_phaseless_weight, pack(1.0 + 0.1j), pack(0.8 - 0.3j), pack(1.2), pack(0.1 + 0.2j), 0.0, 0.2, 0.3)',
            "tol": 2e-08,
        },
        {
            "setup": setup_0 + checks,
            "call": '_checked(phaseless_weight, pack(2.0), pack(1.0), pack(1.0), pack(0.0), 3.0, 0.0, 2.0)',
            "gold_call": '_checked(_oracle_phaseless_weight, pack(2.0), pack(1.0), pack(1.0), pack(0.0), 3.0, 0.0, 2.0)',
            "tol": 2e-08,
        },
        {
            "setup": setup_1 + checks,
            "call": '_raises(_checked, phaseless_weight, pack(0.0), pack(1.0), pack(1.0), pack(0.0), 1.0, 0.1, 0.0)',
            "gold_call": '_raises(_checked, _oracle_phaseless_weight, pack(0.0), pack(1.0), pack(1.0), pack(0.0), 1.0, 0.1, 0.0)',
            "tol": 2e-08,
        },
    ]
