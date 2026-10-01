"""
Find all elastic natural modes of a specimen with positive specimen inertia and exactly zero buffer inertia. Buffer displacements are unconstrained and satisfy instantaneous mechanical equilibrium. Elastic displacement is measured relative to the specimen centre of mass; exclude the three rigid translations. Return specimen mode columns normalised in the specimen mass inner product, with positive frequencies in ascending order. Signs and orthonormal bases within repeated eigenspaces are unrestricted. The condensed stiffness maps specimen displacements to specimen forces when the buffer is in equilibrium.

A massless exterior introduces algebraic equilibrium constraints. Rigid translations span the zero-frequency subspace and carry no elastic deformation.

Returns
-------
dict, positive frequencies, mass-normalised modes and condensed specimen stiffness.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def elastic_modes(stiffness, mass, n_specimen) -> dict:
    r"""Find all elastic natural modes of a specimen with positive specimen inertia and exactly zero buffer inertia. Buffer displacements are unconstrained and satisfy instantaneous mechanical equilibrium. Elastic displacement is measured relative to the specimen centre of mass; exclude the three rigid translations. Return specimen mode columns normalised in the specimen mass inner product, with positive frequencies in ascending order. Signs and orthonormal bases within repeated eigenspaces are unrestricted. The condensed stiffness maps specimen displacements to specimen forces when the buffer is in equilibrium.

    Parameters
    ----------
    stiffness
        Finite real symmetric stiffness matrix, ordered voxel then component.
    mass
        Diagonal inertia vector, positive on specimen entries and exactly zero on buffer entries.
    n_specimen
        Number of specimen voxels, at least two and smaller than the total.

    Returns
    -------
    dict, keys frequencies (3*n_specimen-3,) in hertz, modes (3*n_specimen, 3*n_specimen-3) and condensed_stiffness (3*n_specimen, 3*n_specimen).

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def _oracle_elastic_modes(stiffness, mass, n_specimen):
    stiffness = np.asarray(stiffness, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1 or mass.size % 3 or stiffness.shape != (mass.size, mass.size):
        raise ValueError('stiffness and mass dimensions disagree')
    if not isinstance(n_specimen, (int, np.integer)) or not 2 <= n_specimen < mass.size//3:
        raise ValueError('n_specimen must leave a nonempty buffer')
    n = 3*int(n_specimen)
    if not np.isfinite(stiffness).all() or not np.isfinite(mass).all() or np.any(mass[:n] <= 0) or np.any(mass[n:] != 0):
        raise ValueError('mass is positive on the specimen and zero on the buffer')
    if not np.allclose(stiffness, stiffness.T, rtol=1e-12, atol=1e-12*np.abs(stiffness).max()):
        raise ValueError('stiffness must be symmetric')
    condensed = stiffness[:n, :n] - stiffness[:n, n:] @ np.linalg.solve(stiffness[n:, n:], stiffness[n:, :n])
    condensed = (condensed+condensed.T)/2
    root_mass = np.sqrt(mass[:n])
    weighted = condensed / root_mass[:, None] / root_mass[None, :]
    translations = np.tile(np.eye(3), (n_specimen, 1))
    basis = null_space((root_mass[:, None]*translations).T)
    reduced = basis.T @ weighted @ basis
    eigenvalues, eigenvectors = eigh((reduced+reduced.T)/2)
    if np.any(eigenvalues <= 0):
        raise ValueError('elastic subspace must be positive definite')
    modes = (basis @ eigenvectors) / root_mass[:, None]
    return {'frequencies': np.sqrt(eigenvalues)/(2*np.pi), 'modes': modes,
            'condensed_stiffness': condensed}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return numerical test specifications."""
    return [
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
D=(np.roll(np.eye(7),1,axis=1)-np.eye(7))*6000
A=np.array([[[2.3,.2,-.1],[.2,.8,.05],[-.1,.05,.7]],[[1.9,-.1,.15],[-.1,.9,-.02],[.15,-.02,.6]],[[2.1,.12,.02],[.12,.7,.04],[.02,.04,.8]],[[1.8,-.04,.08],[-.04,.85,.01],[.08,.01,.65]],[[2.,.1,.03],[.1,.8,.04],[.03,.04,.75]],np.eye(3)*1e-6,np.eye(3)*1e-6])*1e11
def assemble():
    k=np.zeros((21,21))
    for i in range(7):
        b=np.kron(D[i:i+1],np.eye(3)); k+=b.T@A[i]@b
    return k
K=assemble()
M=np.repeat(np.array([4000.,4400.,5100.,4700.,4200.,0.,0.]),3)
def summary(out):
    f=out['frequencies']; v=out['modes']; kc=out['condensed_stiffness']; mass=M[:15]
    residual=kc@v-mass[:,None]*v*(2*np.pi*f)**2
    return (tuple(np.round(f[:6]/1e6,6)),v.shape,int(np.linalg.norm(v.T@(mass[:,None]*v)-np.eye(len(f)))<1e-9),int(np.linalg.norm(residual)<1e-9*np.linalg.norm(kc@v)),int(np.linalg.norm(np.tile(np.eye(3),(5,1)).T@(mass[:,None]*v))<1e-8),round(float(np.linalg.norm(kc))/1e19,6))
""",
            "call": 'summary(elastic_modes(K,M,5))',
            "gold_call": 'summary(_oracle_elastic_modes(K,M,5))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
D=(np.roll(np.eye(7),1,axis=1)-np.eye(7))*6000
A=np.array([[[2.3,.2,-.1],[.2,.8,.05],[-.1,.05,.7]],[[1.9,-.1,.15],[-.1,.9,-.02],[.15,-.02,.6]],[[2.1,.12,.02],[.12,.7,.04],[.02,.04,.8]],[[1.8,-.04,.08],[-.04,.85,.01],[.08,.01,.65]],[[2.,.1,.03],[.1,.8,.04],[.03,.04,.75]],np.eye(3)*1e-6,np.eye(3)*1e-6])*1e11
def assemble():
    k=np.zeros((21,21))
    for i in range(7):
        b=np.kron(D[i:i+1],np.eye(3)); k+=b.T@A[i]@b
    return k
K=assemble()
M=np.repeat(np.array([4000.,4400.,5100.,4700.,4200.,0.,0.]),3)
def summary(out):
    f=out['frequencies']; v=out['modes']; kc=out['condensed_stiffness']; mass=M[:15]
    residual=kc@v-mass[:,None]*v*(2*np.pi*f)**2
    return (tuple(np.round(f[:6]/1e6,6)),v.shape,int(np.linalg.norm(v.T@(mass[:,None]*v)-np.eye(len(f)))<1e-9),int(np.linalg.norm(residual)<1e-9*np.linalg.norm(kc@v)),int(np.linalg.norm(np.tile(np.eye(3),(5,1)).T@(mass[:,None]*v))<1e-8),round(float(np.linalg.norm(kc))/1e19,6))
A[:5]=np.diag([2.,.7,.7])*1e11
K=assemble()
M[:15]=4506.3
""",
            "call": 'summary(elastic_modes(K,M,5))',
            "gold_call": 'summary(_oracle_elastic_modes(K,M,5))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
D=(np.roll(np.eye(7),1,axis=1)-np.eye(7))*6000
A=np.array([[[2.3,.2,-.1],[.2,.8,.05],[-.1,.05,.7]],[[1.9,-.1,.15],[-.1,.9,-.02],[.15,-.02,.6]],[[2.1,.12,.02],[.12,.7,.04],[.02,.04,.8]],[[1.8,-.04,.08],[-.04,.85,.01],[.08,.01,.65]],[[2.,.1,.03],[.1,.8,.04],[.03,.04,.75]],np.eye(3)*1e-6,np.eye(3)*1e-6])*1e11
def assemble():
    k=np.zeros((21,21))
    for i in range(7):
        b=np.kron(D[i:i+1],np.eye(3)); k+=b.T@A[i]@b
    return k
K=assemble()
M=np.repeat(np.array([4000.,4400.,5100.,4700.,4200.,0.,0.]),3)
def summary(out):
    f=out['frequencies']; v=out['modes']; kc=out['condensed_stiffness']; mass=M[:15]
    residual=kc@v-mass[:,None]*v*(2*np.pi*f)**2
    return (tuple(np.round(f[:6]/1e6,6)),v.shape,int(np.linalg.norm(v.T@(mass[:,None]*v)-np.eye(len(f)))<1e-9),int(np.linalg.norm(residual)<1e-9*np.linalg.norm(kc@v)),int(np.linalg.norm(np.tile(np.eye(3),(5,1)).T@(mass[:,None]*v))<1e-8),round(float(np.linalg.norm(kc))/1e19,6))
K*=2.7
M[:15]*=.8
""",
            "call": 'summary(elastic_modes(K,M,5))',
            "gold_call": 'summary(_oracle_elastic_modes(K,M,5))',
        },
    ]
