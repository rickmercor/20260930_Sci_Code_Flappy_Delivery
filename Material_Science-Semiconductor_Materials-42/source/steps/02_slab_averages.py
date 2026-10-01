"""
Evaluate the distinct orbital and uniform-pair slab integrals in the symmetric Q2D construction (SI S.25–S.26).

Evaluate the distinct orbital and uniform-pair slab integrals in the symmetric Q2D construction (SI S.25–S.26).

The two dimensionless slab averages are defined by
\[
F_a(Q,d)=\frac1d\int_{-d/2}^{d/2}e^{-Q|z-z_a|}\,dz,\qquad B(Q,d)=\frac1{d^2}\int_{-d/2}^{d/2}\int_{-d/2}^{d/2}e^{-Q|z-z'|}\,dz\,dz'.
\]
Both extend continuously to one at \(Qd=0\), with fixed fractional heights.

Returns
-------
return result  # real ndarray, shape S+(A+1,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def slab_averages(lengths: np.ndarray, thickness: float, z_fractions: np.ndarray) -> np.ndarray:
    """Parameters
    ----------
    lengths : finite nonnegative real ndarray, shape S, rank>=1, nonempty
        Q=|q+G| in inverse angstroms.
    thickness : finite nonnegative float
        Slab thickness d in angstroms.
    z_fractions : finite real ndarray, shape (A,), A>=1
        Orbital heights divided by d, each in [-0.5,0.5].
    Returns
    -------
    real ndarray, shape S+(A+1,)
        Dimensionless single-site integrals F_a in orbital order, followed by the
        uniform-pair integral B in the last entry. Continuous Q*d=0 values are one.
    Raises
    ------
    ValueError : malformed inputs, negative Q or d, or heights outside the slab."""
    return np.zeros(np.shape(lengths)+(len(z_fractions)+1,))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_slab_averages(lengths, thickness, z_fractions):
    q = np.asarray(lengths, dtype=float)
    z = np.asarray(z_fractions, dtype=float)
    if q.ndim < 1 or not q.size or z.ndim != 1 or not z.size or not np.isfinite(q).all() or np.any(q < 0) or not np.isfinite(thickness) or thickness < 0 or not np.isfinite(z).all() or np.any(np.abs(z) > .5):
        raise ValueError('nonnegative momenta/thickness and slab-contained sites required')
    x = q * thickness
    result = np.ones(x.shape + (len(z)+1,), dtype=float)
    regular = x > 1e-4
    a, b = .5-z, .5+z
    xx = x[regular, None]
    result[regular, :-1] = (-np.expm1(-xx*a)-np.expm1(-xx*b))/xx
    result[regular, -1] = 2*(np.expm1(-x[regular])+x[regular])/x[regular]**2
    xx = x[~regular, None]
    result[~regular, :-1] = (1-xx*(a*a+b*b)/2 + xx**2*(a**3+b**3)/6
                            -xx**3*(a**4+b**4)/24 + xx**4*(a**5+b**5)/120)
    xx = x[~regular]
    result[~regular, -1] = 1-xx/3+xx**2/12-xx**3/60+xx**4/360
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nq=np.array([0.1, 0.7, 2.0]); z=np.array([0, 0.29, -0.23])\n', 'call': 'slab_averages(q,2.4,z)', 'gold_call': '_oracle_slab_averages(q,2.4,z)'}, {'setup': 'import numpy as np\nq=np.array([0.0, 0.8, 4.0]); z=np.array([0, 0.29, -0.23])\n', 'call': 'slab_averages(q,0.0,z)', 'gold_call': '_oracle_slab_averages(q,0.0,z)'}, {'setup': 'import numpy as np\nq=np.array([0.0, 0.0]); z=np.array([-0.5, 0, 0.5])\n', 'call': 'slab_averages(q,3.0,z)', 'gold_call': '_oracle_slab_averages(q,3.0,z)'}, {'setup': 'import numpy as np\nq=np.array([0.0, 0.01, 1.2]); z=np.array([0, 0.41])\n', 'call': 'slab_averages(q,1.1,z)', 'gold_call': '_oracle_slab_averages(q,1.1,z)'}, {'setup': 'import numpy as np\nq=np.array([1e-12, 2e-10]); z=np.array([0, 0.1, 0.49])\n', 'call': 'slab_averages(q,1.0,z)', 'gold_call': '_oracle_slab_averages(q,1.0,z)'}, {'setup': 'import numpy as np\nq=np.array([9.99e-05, 0.0001001]); z=np.array([-0.4, 0, 0.3])\n', 'call': 'slab_averages(q,1.0,z)', 'gold_call': '_oracle_slab_averages(q,1.0,z)'}, {'setup': 'import numpy as np\nq=np.array([20.0, 500.0, 2000.0]); z=np.array([-0.2, 0, 0.3])\n', 'call': 'slab_averages(q,1.0,z)', 'gold_call': '_oracle_slab_averages(q,1.0,z)'}, {'setup': 'import numpy as np\nq=np.array([20.0, 500.0, 2000.0]); z=np.array([-0.5, 0, 0.5])\n', 'call': 'slab_averages(q,1.0,z)', 'gold_call': '_oracle_slab_averages(q,1.0,z)'}, {'setup': 'import numpy as np\nq=np.array([10000.0, 1000000.0]); z=np.array([0.499999, 0.5, 0])\n', 'call': 'slab_averages(q,1.0,z)', 'gold_call': '_oracle_slab_averages(q,1.0,z)'}, {'setup': 'import numpy as np\nq=np.array([0.33, 1.2, 3.7]); z=np.array([-0.29, 0.23, 0])\n', 'call': 'slab_averages(q,2.0,z)', 'gold_call': '_oracle_slab_averages(q,2.0,z)'}, {'setup': 'import numpy as np\nq=np.array([0.15, 1.5, 15.0]); z=np.array([0.17])\n', 'call': 'slab_averages(q,0.9,z)', 'gold_call': '_oracle_slab_averages(q,0.9,z)'}, {'setup': 'import numpy as np\nq=np.array([[0.0, 1e-08, 0.2], [0.7, 5.0, 500.0]]); z=np.array([0, 0.29, -0.23])\n', 'call': 'slab_averages(q,1.7,z)', 'gold_call': '_oracle_slab_averages(q,1.7,z)'}, {'setup': 'import numpy as np\n\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: slab_averages(np.array([1.]),-1.,np.array([0.])))', 'gold_call': '_exception_code(lambda: _oracle_slab_averages(np.array([1.]),-1.,np.array([0.])))'}]
