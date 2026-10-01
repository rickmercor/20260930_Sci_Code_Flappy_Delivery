"""
Form point-orbital density vertices with the symmetric orbital factors of SI S.25, keeping the Fourier and Bloch-gauge phases.

Form point-orbital density vertices with the symmetric orbital factors of SI S.25, keeping the Fourier and Bloch-gauge phases.

The symmetric orbital prescription of SI Eq. S.25 uses vertices
\[
J^{nb}_G(\mathbf k,\mathbf q)=\sum_a U_{an}(\mathbf k)^*U_{ab}(\mathbf k+\mathbf q)e^{-i(\mathbf q+\mathbf G)\cdot\boldsymbol\tau_a}\sqrt{F_a(Q_G,d)}.
\]

Returns
-------
return result  # complex ndarray, shape (Q,K,G,3,3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def density_vertices(left_vectors: np.ndarray, right_vectors: np.ndarray, phases: np.ndarray, orbital_averages: np.ndarray) -> np.ndarray:
    """Parameters
    ----------
    left_vectors : finite complex ndarray, shape (K,3,3)
        Bloch coefficients U[k,orbital,band] at the left momenta.
    right_vectors : finite complex ndarray, shape (Q,K,3,3)
        Coefficients at right momenta k+q for each transfer.
    phases : finite complex ndarray, shape (Q,G,3)
        exp[-i(q+G).tau_a] in transfer, reciprocal, orbital order.
    orbital_averages : finite nonnegative real ndarray, shape (Q,G,3)
        Single-site F_a, dimensionless; ones define the unweighted direct vertex.
    Returns
    -------
    complex ndarray, shape (Q,K,G,3,3)
        Dimensionless vertices, indexed [q,k,G,left_band,right_band], with left
        coefficients conjugated and symmetric square-root orbital factors.
    Raises
    ------
    ValueError : incompatible shapes, nonfinite inputs, or negative form factors."""
    return np.zeros((np.shape(right_vectors)[0],len(left_vectors),np.shape(phases)[1],3,3),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_density_vertices(left_vectors, right_vectors, phases, orbital_averages):
    left = np.asarray(left_vectors, dtype=complex)
    right = np.asarray(right_vectors, dtype=complex)
    phase = np.asarray(phases, dtype=complex)
    averages = np.asarray(orbital_averages, dtype=float)
    if left.ndim != 3 or left.shape[1:] != (3,3) or right.ndim != 4 or right.shape[1:] != left.shape or phase.ndim != 3 or phase.shape[0] != len(right) or phase.shape[2] != 3 or averages.shape != phase.shape or np.any(averages < 0) or not all(np.isfinite(x).all() for x in (left,right,phase,averages)):
        raise ValueError('incompatible vertex tensors or negative form factors')
    form = np.sqrt(averages)
    return np.einsum('kan,qkab,qga->qkgnb', left.conj(), right, phase*form, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nl=np.tile(np.eye(3),(K,1,1)); r=np.tile(np.eye(3),(Q,K,1,1))\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nf=np.ones_like(f)\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nf[:,:,1:]=0\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nl=l*np.exp(1j*np.array([.2,-.7,1.4]))\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nr=r*np.exp(1j*np.array([-.3,.8,-1.1]))\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\np=p[:,[2,0,1]]; f=f[:,[2,0,1]]\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\np=p*np.exp(1j*np.array([[.2,.5,.9],[-.1,.3,.7]]))[:,:,None]\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nl=np.tile(u,(K,1,1)); r=np.tile(u,(Q,K,1,1)); p=np.ones_like(p); f=np.ones_like(f)\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nl=l[:,[2,0,1]]; r=r[:,:,[2,0,1]]; p=p[:,:,[2,0,1]]; f=f[:,:,[2,0,1]]\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nf[:]=1e-16; f[:,:,0]=1\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nl=l[:1]; r=r[:,:1]; p=p[:,:2]; f=f[:,:2]\n', 'call': 'density_vertices(l,r,p,f)', 'gold_call': '_oracle_density_vertices(l,r,p,f)'}, {'setup': 'import numpy as np\nK,Q,G=2,2,3\nu=np.array([[1,1,1],[1,np.exp(2j*np.pi/3),np.exp(4j*np.pi/3)],[1,np.exp(4j*np.pi/3),np.exp(2j*np.pi/3)]])/np.sqrt(3)\nl=np.stack([np.eye(3),u]).astype(complex)\nr=np.stack([np.stack([u,np.eye(3)]),np.stack([u.conj(),u])])\np=np.exp(-1j*np.arange(Q*G*3).reshape(Q,G,3)*.37)\nf=np.array([[[1.,.7,.3],[.9,.6,.2],[.7,.4,.1]],[[.8,.4,.2],[.6,.3,.1],[.5,.2,.05]]])\nf[0,0,0]=-1\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: density_vertices(l,r,p,f))', 'gold_call': '_exception_code(lambda: _oracle_density_vertices(l,r,p,f))'}]
