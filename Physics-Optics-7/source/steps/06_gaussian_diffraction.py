"""
Evaluate the full Gaussian-screen diffraction amplitude with controlled cancellation.

Section 2, equations (12)–(19). Use the convergent Gaussian-lens diffraction expansion or an equivalently converged evaluation of the same Abel integral. The normalization is omega/(2*pi*i) in two dimensions. The complex Gaussian factors use the branch continuous from the empty screen.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gaussian_diffraction(precision: "np.ndarray", alpha: float, position: "np.ndarray", frequency: float) -> "np.ndarray":
    """Evaluate the full Gaussian-screen diffraction amplitude with controlled cancellation.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A.
    alpha : float
        Nonnegative reduced Gaussian strength, 0 through 3.
    position : ndarray, shape (2,), float
        Source position y.
    frequency : float
        Positive dimensionless frequency, 0.1 through 180.
    Returns
    -------
    ndarray, shape (2,), float
        [Re(Psi), Im(Psi)] for the Abel-regulated full two-dimensional lens field.
        Both components have required absolute accuracy 1e-8. All entries are
        dimensionless; Psi=1 for alpha=0.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def _oracle_gaussian_diffraction(precision: "np.ndarray", alpha: float, position: "np.ndarray", frequency: float) -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    vals, V = np.linalg.eigh(A)
    y = V.T @ np.asarray(position, dtype=float)
    # The absolute coefficients peak at frequency*alpha. Guard the ensuing cancellation.
    dps = max(60, int(abs(frequency*alpha)/np.log(10)) + 45)
    if alpha == 0:
        return np.array([1., 0.])
    with localcontext() as ctx:
        ctx.prec = dps
        one, zero = Decimal(1), Decimal(0)
        w, al = Decimal(float(frequency)), Decimal(float(alpha))
        vals = [Decimal(float(v)) for v in vals]
        ys = [Decimal(float(v)) for v in y]
        tolerance = one.scaleb(-dps + 5)

        def _atan_reciprocal(q):
            x = one / q
            term, answer, k = x, x, 1
            while abs(term) > tolerance:
                term *= -x*x
                answer += term / (2*k+1)
                k += 1
            return answer

        pi = 16*_atan_reciprocal(Decimal(5))-4*_atan_reciprocal(Decimal(239))

        def _sincos(angle):
            x = angle % (2*pi)
            if x > pi:
                x -= 2*pi
            sine, cosine, st, ct, k = x, one, x, one, 1
            while max(abs(st), abs(ct)) > tolerance:
                st *= -x*x / ((2*k)*(2*k+1))
                ct *= -x*x / ((2*k-1)*(2*k))
                sine += st
                cosine += ct
                k += 1
            return sine, cosine

        factor = one
        total_re, total_im = one, zero
        for n in range(1, int(4*abs(frequency*alpha)) + 201):
            factor *= w*al/n
            gr, gi, er, ei = one, zero, zero, zero
            for aj, yj in zip(vals, ys):
                a = n*aj
                den = a*a+w*w
                norm = w/den.sqrt()
                sr = ((norm+w*w/den)/2).sqrt()
                si = -w*a/(2*den*sr)
                gr, gi = gr*sr-gi*si, gr*si+gi*sr
                er -= w*w*yj*yj*a/(2*den)
                ei += w*yj*yj*a*a/(2*den)
            sine, cosine = _sincos(ei)
            magnitude = factor*er.exp()
            tr, ti = magnitude*(gr*cosine-gi*sine), magnitude*(gr*sine+gi*cosine)
            if n % 4 == 1:
                tr, ti = -ti, tr
            elif n % 4 == 2:
                tr, ti = -tr, -ti
            elif n % 4 == 3:
                tr, ti = ti, -tr
            total_re += tr
            total_im += ti
        return np.array([float(total_re), float(total_im)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\nA=np.array([[1, 0.1], [0.1, 0.5]]);y=np.array([0.3, -0.2])',
      'call': 'gaussian_diffraction(A.copy(),0,y.copy(),25)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),0,y.copy(),25)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0], [0, 1]]);y=np.array([0, 0])',
      'call': 'gaussian_diffraction(A.copy(),0.3,y.copy(),0.2)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),0.3,y.copy(),0.2)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0.25], [0.25, 0.7]]);y=np.array([0.2, -0.5])',
      'call': 'gaussian_diffraction(A.copy(),1.4,y.copy(),2)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),1.4,y.copy(),2)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[0.5, 0], [0, 1.2]]);y=np.array([-0.3, 0.1])',
      'call': 'gaussian_diffraction(A.copy(),2.0,y.copy(),10)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),2.0,y.copy(),10)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, -0.15], [-0.15, 0.6]]);y=np.array([-0.2, -0.1])',
      'call': 'gaussian_diffraction(A.copy(),1.8,y.copy(),35)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),1.8,y.copy(),35)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0.18], [0.18, 0.58]]);y=np.array([-0.42, -0.1])',
      'call': 'gaussian_diffraction(A.copy(),1.9011537047695237,y.copy(),70)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),1.9011537047695237,y.copy(),70)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[1, 0], [0, 0.4]]);y=np.array([1.1, -0.8])',
      'call': 'gaussian_diffraction(A.copy(),2.2,y.copy(),110)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),2.2,y.copy(),110)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\nA=np.array([[0.9, 0.1], [0.1, 0.7]]);y=np.array([-1.4, 0.9])',
      'call': 'gaussian_diffraction(A.copy(),0.7,y.copy(),180)',
      'gold_call': '_oracle_gaussian_diffraction(A.copy(),0.7,y.copy(),180)',
      'tol': 1e-08}]
