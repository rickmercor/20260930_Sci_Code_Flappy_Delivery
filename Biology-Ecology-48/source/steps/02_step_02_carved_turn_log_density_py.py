"""
Evaluate normalized carved von Mises density and its concentration score.

Small apparent heading changes are often excluded when real turns are detected from high-frequency movement data. The source models surviving turning angles with a von Mises factor multiplied by a central-carving factor. For concentration \(\kappa\) and carving parameter \(w\), the normalized density is

\[

f(\phi\mid\kappa,w)

=

\frac{\exp(\kappa\cos\phi)

\left[1-\exp\{-w(1-\cos\phi)\}\right]}

{Z(\kappa,w)},

\qquad -\pi\leq\phi<\pi,

\]

where

\[

Z(\kappa,w)

=

2\pi\left[I_0(\kappa)-e^{-w}I_0(\kappa+w)\right].

\]

Here \(I_0\) is the modified Bessel function of the first kind. The density is zero at \(\phi=0\), while ordinary nonzero turns have positive density.



The concentration score differentiates the normalized log density:

\[

\frac{\partial\log f(\phi\mid\kappa,w)}{\partial\kappa}

=

\cos\phi-

\frac{I_1(\kappa)-e^{-w}I_1(\kappa+w)}

{I_0(\kappa)-e^{-w}I_0(\kappa+w)}.

\]

The normalizer's derivative is essential to likelihood sensitivity. When terms in the normalizer are close, scaled Bessel functions and cancellation-aware arithmetic preserve precision.

Returns
-------
Tuple of two np.ndarray matrices, each shape (n, 2): normalized angular log densities and their state-specific concentration scores.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def carved_turn_log_density(angles: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Return the log angle density and its derivative with respect to kappa.
 
    The source density is exp(kappa*cos(phi))*(1-exp(-width*(1-cos(phi))))
    on [-pi,pi), normalized over that interval. The exact normalization is
    2*pi*[I0(kappa)-exp(-width)*I0(kappa+width)].
 
    Parameters
    ----------
    angles : np.ndarray
        Finite (n,) angles in [-pi,pi). Values at exactly zero yield log density -inf.
    kappa, width : np.ndarray
        Positive finite (2,) state parameters, with width >= 1e-5 and kappa <= 50.
 
    Returns
    -------
    log_density, concentration_score : tuple[np.ndarray, np.ndarray]
        Both (n,2). Score differentiates log density in each state's kappa.
        Use sufficient stability for near-zero angles and moderate widths.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import i0e, i1e
 
def _oracle_carved_turn_log_density(angles: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    phi=np.asarray(angles,float); k=np.asarray(kappa,float); w=np.asarray(width,float)
    if phi.ndim!=1 or k.shape!=(2,) or w.shape!=(2,) or not all(np.all(np.isfinite(a)) for a in (phi,k,w)) or np.any(k<=0) or np.any(k>50) or np.any(w<1e-5) or np.any(phi< -np.pi) or np.any(phi>=np.pi):
        raise ValueError("invalid carved density parameters")
    a=i0e(k); z=i0e(k+w); D=a-z
    # The scaled Bessel difference removes the large exponential factor.
    logZ=np.log(2*np.pi)+k+np.log(D)
    dlogZ=1+((i1e(k)-a)-(i1e(k+w)-z))/D
    u=1-np.cos(phi[:,None]); factor=-np.expm1(-w[None,:]*u)
    with np.errstate(divide='ignore'):
        logf=k[None,:]*np.cos(phi[:,None])+np.log(factor)-logZ[None,:]
    score=np.cos(phi[:,None])-dlogZ[None,:]
    return logf,score

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\nphi=np.array([.2,-1.3,2.4]);k=np.array([2.4,5.7]);w=np.array([1.3,.42])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(carved_turn_log_density(phi,k,w))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_carved_turn_log_density(phi,k,w))', "tol": 1e-07},
        {"setup": 'import numpy as np\nphi=np.array([-np.pi,0.,.0001]);k=np.array([.2,8.]);w=np.array([.001,.12])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(carved_turn_log_density(phi,k,w))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_carved_turn_log_density(phi,k,w))', "tol": 1e-07},
        {"setup": 'import numpy as np\nphi=np.array([.004,-2.8]);k=np.array([12.,20.]);w=np.array([.05,7.])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(carved_turn_log_density(phi,k,w))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_carved_turn_log_density(phi,k,w))', "tol": 1e-07},
    ]
