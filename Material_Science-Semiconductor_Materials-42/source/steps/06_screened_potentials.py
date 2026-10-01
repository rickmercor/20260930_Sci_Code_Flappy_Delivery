"""
Combine the independently averaged Coulomb pair potential with the inverse symmetric dielectric matrix and regularize head, wings and body according to the specified Q2D cell prescription.

Combine the independently averaged Coulomb pair potential with the inverse symmetric dielectric matrix and regularize head, wings and body according to the specified Q2D cell prescription.

Write \(Q_G=|\mathbf q+\mathbf G|\), \(c=2\pi(14.3996454784255)/(a^2\kappa)\), and \(v_G=c/Q_G\). The Coulomb constant in parentheses has units eV Å. The two dimensionless slab averages are defined by
\[
F_a(Q,d)=\frac1d\int_{-d/2}^{d/2}e^{-Q|z-z_a|}\,dz,\qquad B(Q,d)=\frac1{d^2}\int_{-d/2}^{d/2}\int_{-d/2}^{d/2}e^{-Q|z-z'|}\,dz\,dz'.
\]
Both extend continuously to one at \(Qd=0\), with fixed fractional heights. The symmetric orbital prescription of SI Eq. S.25 uses vertices
\[
J^{nb}_G(\mathbf k,\mathbf q)=\sum_a U_{an}(\mathbf k)^*U_{ab}(\mathbf k+\mathbf q)e^{-i(\mathbf q+\mathbf G)\cdot\boldsymbol\tau_a}\sqrt{F_a(Q_G,d)}.
\]
The static occupation-difference response and symmetric dielectric matrix are
\[
\chi_{GG'}(\mathbf q)=\frac{2}{49}\sum_{\mathbf k\in\mathcal K_7}\sum_{n,b:f_n\ne f_b}\frac{f_n-f_b}{e_n(\mathbf k)-e_b(\mathbf k+\mathbf q)}J^{nb}_G(J^{nb}_{G'})^*,\qquad \bar\varepsilon=I-\sqrt v\,\chi\sqrt v.
\]
The effective screened potential is \(\bar W=\sqrt{vB}\,\bar\varepsilon^{-1}\sqrt{vB}\); diagonal square roots act on the reciprocal indices. The macroscopic response is \(\varepsilon_M=1/(\bar\varepsilon^{-1})_{00}\). At exactly \(Q_G=0\), use the value zero for the bare factor during matrix construction: the dielectric head is then one and its wings vanish, while the body is computed normally.

The following finite-grid origin prescription fixes the benchmark and extends the paper's first-order circular-cell rule to its pair-averaged Q2D potential. Let \(\delta=0.002/a\) and \(s_\alpha(t)=[\varepsilon_M(t\hat{\boldsymbol\alpha})-1]/t\); use \(r_\alpha=2s_\alpha(\delta/2)-s_\alpha(\delta)\) for \(\alpha=x,y\). With \(q_0=0.35(2\pi/an)\), replace \(\bar W_{00}(0)\) by \(c[2/q_0-(r_x+r_y)/2-d/3]\), set the origin wings to zero, and retain the origin body. This is the stipulated first-order cell prescription at the fixed \(\delta,q_0\); the synthetic targets refer to that prescription.

Returns
-------
return result  # complex ndarray, shape (Q,G,G)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def screened_potentials(dielectrics: np.ndarray, bare: np.ndarray, pair_averages: np.ndarray, zero_index: int, q0: float, lengths: np.ndarray, kappa: float, thickness: float) -> np.ndarray:
    """Parameters
    ----------
    dielectrics : finite positive Hermitian complex ndarray, shape (Q,G,G)
        Symmetric dimensionless Q2D dielectric matrices; reciprocal index 0 is G=0.
    bare : finite nonnegative real ndarray, shape (Q,G)
        Bare Coulomb factors in eV.
    pair_averages : finite nonnegative real ndarray, shape (Q,G)
        Dimensionless uniform-pair slab factors B.
    zero_index : int in {-1,0,...,Q-1}
        Transfer index of q=0; -1 means that this batch has no zero transfer.
    q0 : finite positive float
        Circular-cell radius in inverse angstroms.
    lengths : finite real ndarray, shape (2,)
        Screening lengths [r_x,r_y], in angstroms.
    kappa : finite positive float
        Dimensionless environmental permittivity.
    thickness : finite nonnegative float
        Slab thickness d, in angstroms.
    Returns
    -------
    complex ndarray, shape (Q,G,G)
        W in eV. Use the full dielectric inverse between square roots of v*B.
        At zero_index>=0, the head is c*(2/q0-(r_x+r_y)/2-d/3), origin wings
        are zero, and the body retains its matrix-inverse value; a=3.2 angstroms.
    Raises
    ------
    ValueError : incompatible/nonfinite data, invalid scalar domains, or a
        dielectric matrix that is not positive Hermitian."""
    return np.zeros(np.shape(dielectrics),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_screened_potentials(dielectrics, bare, pair_averages, zero_index,
                                q0, lengths, kappa, thickness):
    eps = np.asarray(dielectrics, dtype=complex)
    b = np.asarray(bare, dtype=float)
    f = np.asarray(pair_averages, dtype=float)
    if eps.ndim != 3 or eps.shape[1] != eps.shape[2] or b.shape != eps.shape[:2] or f.shape != b.shape or not all(np.isfinite(x).all() for x in (eps,b,f)) or np.any(b < 0) or np.any(f < 0) or zero_index < -1 or zero_index >= len(eps) or q0 <= 0 or kappa <= 0 or thickness < 0 or np.asarray(lengths).shape != (2,) or not np.isfinite([q0,kappa,thickness,*lengths]).all():
        raise ValueError('invalid screened-potential data')
    if not np.allclose(eps,eps.conj().swapaxes(-1,-2),atol=1e-10) or np.any(np.linalg.eigvalsh(eps) <= 0):
        raise ValueError('positive Hermitian dielectric matrices required')
    sq = np.sqrt(b*f)
    screened = sq[:, :, None]*np.linalg.inv(eps)*sq[:, None, :]
    if zero_index >= 0:
        c = 2*np.pi*14.3996454784255/(3.2**2*kappa)
        screened[zero_index, 0, :] = 0
        screened[zero_index, :, 0] = 0
        screened[zero_index, 0, 0] = c*(2/q0-np.sum(lengths)/2-thickness/3)
    return screened

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\ne=np.array([np.diag([2.,3.,4.]),np.diag([1.,2.,5.])])\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\ne=np.tile(np.eye(3),(2,1,1))\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nf=np.ones_like(f); d=0.\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nzi=0; b[0,0]=0.\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nzi=1\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nzi=0; e[0,1,2]=.2j; e[0,2,1]=-.2j\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nzi=0; rr=np.array([.01,3.7])\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nzi=0; d=6.\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\nzi=0; q0=.002\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\np=[2,0,1]; e=e[:,p][:,:,p]; b=b[:,p]; f=f[:,p]\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\ne=np.eye(3)[None]+1e-10*(e-np.eye(3)[None])\n', 'call': 'screened_potentials(e,b,f,zi,q0,rr,kap,d)', 'gold_call': '_oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d)'}, {'setup': 'import numpy as np\nx=np.array([[[1,.2j,.4],[.3,1.,-.1j],[.2j,.4,1.]],[[.6,.1,.2j],[.3j,1.,.1],[.2,.1j,.9]]],dtype=complex)\ne=np.eye(3)[None]+x@x.conj().swapaxes(-1,-2)\nb=np.array([[3.,1.,.4],[2.,.8,.3]])\nf=np.array([[.9,.7,.2],[.8,.6,.3]])\nzi=-1; q0=.07; rr=np.array([.2,.4]); kap=7.; d=2.\ne[0,0,0]=-1.\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: screened_potentials(e,b,f,zi,q0,rr,kap,d))', 'gold_call': '_exception_code(lambda: _oracle_screened_potentials(e,b,f,zi,q0,rr,kap,d))'}]
