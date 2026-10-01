"""
Construct the complete finite-grid static RPA response using main Eq. 5 and the symmetric Q2D dielectric prescription.

Construct the complete finite-grid static RPA response using main Eq. 5 and the symmetric Q2D dielectric prescription.

The static occupation-difference response and symmetric dielectric matrix are
\[
\chi_{GG'}(\mathbf q)=\frac{2}{49}\sum_{\mathbf k\in\mathcal K_7}\sum_{n,b:f_n\ne f_b}\frac{f_n-f_b}{e_n(\mathbf k)-e_b(\mathbf k+\mathbf q)}J^{nb}_G(J^{nb}_{G'})^*,\qquad \bar\varepsilon=I-\sqrt v\,\chi\sqrt v.
\]
The effective screened potential is \(\bar W=\sqrt{vB}\,\bar\varepsilon^{-1}\sqrt{vB}\); diagonal square roots act on the reciprocal indices. The macroscopic response is \(\varepsilon_M=1/(\bar\varepsilon^{-1})_{00}\). At exactly \(Q_G=0\), use the value zero for the bare factor during matrix construction: the dielectric head is then one and its wings vanish, while the body is computed normally.
For these supplied arrays, use K=len(left_energies) in the k average and the supplied spin parameter in place of two.

Returns
-------
return result  # complex ndarray, shape (Q,G,G)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def dielectric_matrices(left_energies: np.ndarray, right_energies: np.ndarray, vertices: np.ndarray, bare: np.ndarray, spin: float = 2.) -> np.ndarray:
    """Parameters
    ----------
    left_energies : finite real ndarray, shape (K,3)
        Energies at k in eV; band 0 occupied and bands 1,2 empty.
    right_energies : finite real ndarray, shape (Q,K,3)
        Energies at k+q in eV. Occupied/empty spectra remain globally separated.
    vertices : finite complex ndarray, shape (Q,K,G,3,3)
        Dimensionless density vertices in left-band/right-band order.
    bare : finite nonnegative real ndarray, shape (Q,G)
        Bare Coulomb factors in eV, with zero placeholders at singular Q_G=0.
    spin : finite nonnegative float
        Spin degeneracy multiplying the k average; default 2.
    Returns
    -------
    complex ndarray, shape (Q,G,G)
        Dimensionless symmetric dielectric matrices I-sqrt(v)*chi*sqrt(v).
        chi uses the complete static occupation-difference sum, normalized by K.
    Raises
    ------
    ValueError : incompatible/nonfinite arrays, negative bare/spin, or overlapping
        occupied and empty spectra."""
    return np.zeros((len(right_energies),np.shape(bare)[1],np.shape(bare)[1]),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_dielectric_matrices(left_energies, right_energies, vertices, bare, spin=2.):
    el = np.asarray(left_energies, dtype=float)
    er = np.asarray(right_energies, dtype=float)
    vv = np.asarray(vertices, dtype=complex)
    potential = np.asarray(bare, dtype=float)
    if el.ndim != 2 or el.shape[1] != 3 or er.ndim != 3 or er.shape[1:] != el.shape or potential.ndim != 2 or potential.shape[0] != len(er) or vv.shape != (len(er),len(el),potential.shape[1],3,3) or np.any(potential < 0) or spin < 0 or not np.isfinite(spin) or not all(np.isfinite(x).all() for x in (el,er,vv,potential)):
        raise ValueError('incompatible response inputs')
    if np.any(np.max(el[:,0]) >= er[:,:,1:]) or np.any(np.max(er[:,:,0]) >= el[:,1:]):
        raise ValueError('occupied and empty spectra must be separated')
    occ = np.array([1., 0., 0.])
    weights = np.zeros(er.shape[:2] + (3, 3), dtype=float)
    for n in range(3):
        for b in range(3):
            if occ[n] != occ[b]:
                weights[:, :, n, b] = (spin/len(el)*(occ[n]-occ[b])
                                       /(el[None, :, n]-er[:, :, b]))
    chi = np.einsum('qkgnb,qkhnb,qknb->qgh', vv, vv.conj(), weights, optimize=True)
    sq = np.sqrt(potential)
    return np.eye(potential.shape[1])[None, :, :] - sq[:, :, None]*chi*sq[:, None, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\ns=0.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nb[:,0]=0.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nv[:,:,:,1:,0]=0.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nv[:,:,:,0,1:]=0.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nv[:,:,:,0,1]=0.; v[:,:,:,1,0]=0.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nv=v*np.exp(1j*np.array([.2,.9,-.7]))[None,None,:,None,None]\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nel=np.repeat(el,2,axis=0); er=np.repeat(er,2,axis=1); v=np.repeat(v,2,axis=1)\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nel=el+17.; er=er+17.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nel[:,1:]+=5.; er[:,:,1:]+=5.\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nv[:]=0.; v[:,:,:,0,0]=1.e3\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\nel[:,0]=-.001; el[:,1]=.001; er[:,:,0]=-.002; er[:,:,1]=.003\n', 'call': 'dielectric_matrices(el,er,v,b,s)', 'gold_call': '_oracle_dielectric_matrices(el,er,v,b,s)'}, {'setup': 'import numpy as np\nel=np.array([[-1.1,1.2,2.7],[-.8,1.7,2.3]])\ner=np.array([[[-.9,1.5,3.],[-1.3,1.6,2.5]],[[-1.4,1.8,2.6],[-.6,1.3,2.9]]])\na=np.arange(2*2*3*3*3).reshape(2,2,3,3,3)\nv=(np.sin(a*.7)+1j*np.cos(a*.31))/5\nb=np.array([[2.,.7,.3],[1.3,.6,.2]])\ns=2.\ns=-1.\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: dielectric_matrices(el,er,v,b,s))', 'gold_call': '_exception_code(lambda: _oracle_dielectric_matrices(el,er,v,b,s))'}]
