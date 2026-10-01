"""
Determine the vector receptor displacement residue for unit scalar traction at the source in each distinct elastic eigenspace. Normalise the prescribed traction direction and represent surface traction as a voxel force density. Mode columns contain three consecutive components per specimen voxel and are mass-normalised. Starting at the lowest unused frequency, a cluster contains consecutive frequencies no greater than (1 + cluster_rtol) times that first frequency; represent it by the arithmetic-mean frequency. Residues must be invariant under sign changes and orthogonal basis changes within repeated eigenspaces.

Eigenvector signs and bases within repeated eigenspaces are arbitrary. A physical source-to-receptor response must be independent of those choices.

Returns
-------
dict, cluster frequencies, vector receptor residues and cluster sizes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def modal_residues(frequencies, modes, source, receptor, direction, voxel_size, cluster_rtol) -> dict:
    r"""Determine the vector receptor displacement residue for unit scalar traction at the source in each distinct elastic eigenspace. Normalise the prescribed traction direction and represent surface traction as a voxel force density. Mode columns contain three consecutive components per specimen voxel and are mass-normalised. Starting at the lowest unused frequency, a cluster contains consecutive frequencies no greater than (1 + cluster_rtol) times that first frequency; represent it by the arithmetic-mean frequency. Residues must be invariant under sign changes and orthogonal basis changes within repeated eigenspaces.

    Parameters
    ----------
    frequencies
        Finite positive sorted frequencies in hertz.
    modes
        Mass-normalised specimen mode columns.
    source
        Integer source voxel index inside the specimen.
    receptor
        Integer receptor voxel index inside the specimen.
    direction
        Finite nonzero vector (3,), normalised internally.
    voxel_size
        Positive voxel size in metre.
    cluster_rtol
        Relative clustering threshold in [0, 1), measured from the first frequency of each cluster.

    Returns
    -------
    dict, keys frequencies (n_clusters,) in hertz, residues (n_clusters, 3) and integer sizes (n_clusters,).

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


def _oracle_modal_residues(frequencies, modes, source, receptor, direction, voxel_size, cluster_rtol):
    frequencies = np.asarray(frequencies, dtype=float)
    modes = np.asarray(modes, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0) or np.any(np.diff(frequencies) < 0):
        raise ValueError('frequencies must be positive and sorted')
    if modes.ndim != 2 or modes.shape[1] != frequencies.size or modes.shape[0] % 3:
        raise ValueError('modes must have shape (3*n_specimen, n_modes)')
    n = modes.shape[0]//3
    if not all(isinstance(x, (int, np.integer)) and 0 <= x < n for x in (source, receptor)):
        raise ValueError('source and receptor must index specimen voxels')
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError('direction must be a finite nonzero three-vector')
    if not np.isfinite(frequencies).all() or not np.isfinite(modes).all() or voxel_size <= 0 or not 0 <= cluster_rtol < 1:
        raise ValueError('invalid numerical inputs')
    traction = direction / np.linalg.norm(direction) / voxel_size
    excitation = modes[3*source:3*source+3].T @ traction
    residues = modes[3*receptor:3*receptor+3].T * excitation[:, None]
    groups, centres, sizes = [], [], []
    start = 0
    while start < frequencies.size:
        end = start+1
        while end < frequencies.size and frequencies[end]-frequencies[start] <= cluster_rtol*frequencies[start]:
            end += 1
        groups.append(residues[start:end].sum(axis=0))
        centres.append(frequencies[start:end].mean())
        sizes.append(end-start)
        start = end
    return {'frequencies': np.array(centres), 'residues': np.array(groups), 'sizes': np.array(sizes, dtype=int)}

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
Q=np.linalg.qr(np.vstack([np.eye(6),np.array([[.2,-.3,.4,.1,-.2,.5],[-.4,.2,.3,-.5,.1,.2],[.3,.1,-.2,.4,.5,-.3]])]))[0]
F=np.array([1.,1.,2.,2.,3.,4.])*1e6
def summary(out):
    return (tuple(np.round(out['frequencies']/1e6,8)),tuple(out['sizes']),tuple(np.round(out['residues'].ravel()/1e4,7)))
""",
            "call": 'summary(modal_residues(F,Q,0,2,np.array([1.,2.,-1.]),1e-4,1e-8))',
            "gold_call": 'summary(_oracle_modal_residues(F,Q,0,2,np.array([1.,2.,-1.]),1e-4,1e-8))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
Q=np.linalg.qr(np.vstack([np.eye(6),np.array([[.2,-.3,.4,.1,-.2,.5],[-.4,.2,.3,-.5,.1,.2],[.3,.1,-.2,.4,.5,-.3]])]))[0]
F=np.array([1.,1.,2.,2.,3.,4.])*1e6
def summary(out):
    return (tuple(np.round(out['frequencies']/1e6,8)),tuple(out['sizes']),tuple(np.round(out['residues'].ravel()/1e4,7)))
R=np.array([[.6,-.8],[.8,.6]])
Q[:,:2]=Q[:,:2]@R
Q[:,2:4]=Q[:,2:4]@R.T
Q[:,4]*=-1
""",
            "call": 'summary(modal_residues(F,Q,0,2,np.array([1.,2.,-1.]),1e-4,1e-8))',
            "gold_call": 'summary(_oracle_modal_residues(F,Q,0,2,np.array([1.,2.,-1.]),1e-4,1e-8))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
Q=np.linalg.qr(np.vstack([np.eye(6),np.array([[.2,-.3,.4,.1,-.2,.5],[-.4,.2,.3,-.5,.1,.2],[.3,.1,-.2,.4,.5,-.3]])]))[0]
F=np.array([1.,1.,2.,2.,3.,4.])*1e6
def summary(out):
    return (tuple(np.round(out['frequencies']/1e6,8)),tuple(out['sizes']),tuple(np.round(out['residues'].ravel()/1e4,7)))
F=np.array([1.,1.000000009,1.000000018,2.,3.,4.])*1e6
""",
            "call": 'summary(modal_residues(F,Q,1,1,np.array([2.,-1.,3.]),2e-4,1e-8))',
            "gold_call": 'summary(_oracle_modal_residues(F,Q,1,1,np.array([2.,-1.,3.]),2e-4,1e-8))',
        },
    ]
