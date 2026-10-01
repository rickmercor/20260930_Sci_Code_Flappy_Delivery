"""
Apply the paper’s microscopic direct kernel with reciprocal local fields and both conduction channels; the real-space Eq. 20 fixes the stated conjugation convention.

Apply the paper’s microscopic direct kernel with reciprocal local fields and both conduction channels; the real-space Eq. 20 fixes the stated conjugation convention.

For the direct term define the unweighted point-orbital vertex
\[
I^{nb}_{ij,G}=\sum_a U_{an}(\mathbf k_j)^*e^{-i(\mathbf k_i-\mathbf k_j+\mathbf G)\cdot\boldsymbol\tau_a}U_{ab}(\mathbf k_i).
\]
With \(N=n^2\), electron bands \(c,c'=1,2\), and pair order \((i,c)\),
\[
D_{ic,jc'}=\frac1N\sum_{GG'}(I^{c'c}_{ij,G})^*\bar W_{GG'}(\mathbf k_i-\mathbf k_j)I^{00}_{ij,G'},\qquad H^X_{ic,jc'}=(e_c(\mathbf k_i)-e_0(\mathbf k_i))\delta_{ij}\delta_{cc'}-D_{ic,jc'}.
\]
\(E_0\leq E_1\leq\cdots\) are the eigenvalues of this Hermitian matrix. \(E_0^{\mathrm{head}}\) uses the same parameters and the same full dielectric inversion, with only the \(G=G'=0\) screened-potential entry retained in the direct term. Full-precision arithmetic is used except for the prescribed six-decimal rounding of fitted parameters. The exact-target equations define all admissible calibration roots; the supplied box has isolated roots. The root calculation has coordinate convergence \(10^{-8}\) in \(\kappa\) and \(d\), together with absolute energy residual below \(10^{-9}\) eV before rounding.

Here N=len(energies).

Returns
-------
return result  # complex ndarray, shape (2*N,2*N)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def exciton_hamiltonian(energies: np.ndarray, pair_vertices: np.ndarray, screened: np.ndarray, q_indices: np.ndarray, head_only: bool = False) -> np.ndarray:
    """Parameters
    ----------
    energies : finite real ndarray, shape (N,3), N>=1
        Bloch energies in eV, band 0 valence and bands 1,2 conduction.
    pair_vertices : finite complex ndarray, shape (N,N,G,3,3)
        Unweighted dimensionless I[i,j,G,n,b], with left Bloch state at k_j
        and right state at k_i, and transfer k_i-k_j.
    screened : finite complex ndarray, shape (Q,G,G)
        Screened potentials in eV, with the origin prescription already applied.
    q_indices : integer ndarray, shape (N,N), values in [0,Q)
        Lookup q_indices[i,j] for each raw momentum difference.
    head_only : bool
        If true, use only screened[:,0,0] in the direct contraction. The supplied
        full dielectric inversion underlying that entry is retained.
    Returns
    -------
    complex ndarray, shape (2*N,2*N)
        Exciton Hamiltonian in eV, ordered by momentum i then conduction c=1,2.
        It equals the vertical-gap diagonal minus the direct attraction with 1/N
        normalization and the stated I^{c'c} conjugation. Physically consistent
        paired transfers and vertices yield a Hermitian matrix.
    Raises
    ------
    ValueError : incompatible/nonfinite arrays or invalid transfer indices."""
    return np.zeros((2*len(energies),2*len(energies)),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_exciton_hamiltonian(energies, pair_vertices, screened, q_indices,
                                head_only=False):
    e = np.asarray(energies, dtype=float)
    vv = np.asarray(pair_vertices, dtype=complex)
    ws = np.asarray(screened, dtype=complex)
    qi = np.asarray(q_indices)
    if e.ndim != 2 or e.shape[1] != 3 or not len(e) or ws.ndim != 3 or ws.shape[1] != ws.shape[2] or vv.shape != (len(e),len(e),ws.shape[1],3,3) or qi.shape != (len(e),len(e)) or not np.issubdtype(qi.dtype,np.integer) or np.any(qi < 0) or np.any(qi >= len(ws)) or not all(np.isfinite(x).all() for x in (e,vv,ws)):
        raise ValueError('incompatible BSE data')
    w = ws[qi]
    if head_only:
        w = w.copy()
        w[:, :, 1:, :] = 0
        w[:, :, :, 1:] = 0
    # Each vertex has left state at k_j and right state at k_i.
    electron = vv[:, :, :, 1:, 1:]
    hole = vv[:, :, :, 0, 0]
    direct = np.einsum('ijgdc,ijgh,ijh->icjd', electron.conj(), w, hole, optimize=True)/len(e)
    gaps = (e[:, 1:]-e[:, 0, None]).ravel()
    return np.diag(gaps) - direct.reshape(2*len(e), 2*len(e))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nw[:]=0\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nhead=True\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nw[:,0,:]=0; w[:,:,0]=0\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nw[:,1:,1:]=0\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nv[:,:,:,1:,:]*=np.exp(-1j*np.array([.3,.7]))[None,None,None,:,None]; v[:,:,:,:,1:]*=np.exp(1j*np.array([.3,.7]))\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nphase=np.exp(1j*np.array([.2,.9])); v[:,:,:,0,:]*=phase.conj()[None,:,None,None]; v[:,:,:,:,0]*=phase[:,None,None,None]\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\ne[:,1:]+=1.3\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\ne=e[:,[0,2,1]]; v=v[:,:,:,[0,2,1]][:,:,:,:,[0,2,1]]\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\ne=e[::-1]; v=v[::-1,::-1]; qi=qi[::-1,::-1]\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\ne=e[:1]; v=v[:1,:1]; qi=qi[:1,:1]\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nw=np.concatenate([w,.3*w]); qi=np.array([[0,1],[1,0]])\n', 'call': 'exciton_hamiltonian(e,v,w,qi,head)', 'gold_call': '_oracle_exciton_hamiltonian(e,v,w,qi,head)'}, {'setup': 'import numpy as np\nN,G=2,3\ne=np.array([[-1.,1.2,2.4],[-.7,1.7,2.1]])\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nframes=np.stack([np.eye(3),u]).astype(complex)\np=np.exp(-1j*np.array([[0.,0.,0.],[0.,.7,1.1],[0.,-.7,-1.1]]))\nv=np.empty((N,N,G,3,3),complex)\nfor i in range(N):\n for j in range(N):\n  for g in range(G):v[i,j,g]=frames[j].conj().T@np.diag(p[g])@frames[i]\nw=np.array([[[1.4,.1j,-.1j],[-.1j,.7,.03j],[.1j,-.03j,.7]]])\nqi=np.zeros((N,N),int); head=False\nqi[0,0]=8\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: exciton_hamiltonian(e,v,w,qi,head))', 'gold_call': '_exception_code(lambda: _oracle_exciton_hamiltonian(e,v,w,qi,head))'}]
