"""
The one-dimensional displacement field retains three coupled components. Each hexagonal grain retains its full anisotropic stiffness. Crystal axes in the specimen frame are the columns of the active rotation $R=Z(\varphi_1)X(\Phi)Z(\varphi_2)$, with right-handed rotations and angles in degrees. The acoustic tensor $A$ is defined by $\sigma_{i1}=A_{ik}\partial_1 u_k$. The engineering Voigt ordering is (11, 22, 33, 23, 13, 12); the sixfold axis is the third crystal axis. The independent constants are c11, c33, c12, c13 and c44; c22=c11, c23=c13, c55=c44 and c66=(c11-c12)/2.

Anisotropic elasticity couples longitudinal and transverse displacement. Grain orientation determines the acoustic response along the specimen axis.

Returns
-------
np.ndarray, acoustic tensors of shape (n_grains, 3, 3), in pascal.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def crystal_acoustic_tensors(euler_angles, constants) -> np.ndarray:
    r"""The one-dimensional displacement field retains three coupled components. Each hexagonal grain retains its full anisotropic stiffness. Crystal axes in the specimen frame are the columns of the active rotation $R=Z(\varphi_1)X(\Phi)Z(\varphi_2)$, with right-handed rotations and angles in degrees. The acoustic tensor $A$ is defined by $\sigma_{i1}=A_{ik}\partial_1 u_k$. The engineering Voigt ordering is (11, 22, 33, 23, 13, 12); the sixfold axis is the third crystal axis. The independent constants are c11, c33, c12, c13 and c44; c22=c11, c23=c13, c55=c44 and c66=(c11-c12)/2.

    Parameters
    ----------
    euler_angles
        Finite array (n_grains, 3) of Euler angles in degrees.
    constants
        Mapping c11, c33, c12, c13, c44 in pascal, defining a positive definite hexagonal stiffness.

    Returns
    -------
    np.ndarray, acoustic tensors of shape (n_grains, 3, 3), in pascal.

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


def _oracle_crystal_acoustic_tensors(euler_angles, constants):
    angles = np.asarray(euler_angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3 or not np.isfinite(angles).all():
        raise ValueError('euler_angles must be a finite (n_grains, 3) array')
    c11, c33, c12, c13, c44 = [float(constants[key]) for key in ('c11', 'c33', 'c12', 'c13', 'c44')]
    voigt = np.array([[c11, c12, c13, 0, 0, 0], [c12, c11, c13, 0, 0, 0],
                      [c13, c13, c33, 0, 0, 0], [0, 0, 0, c44, 0, 0],
                      [0, 0, 0, 0, c44, 0], [0, 0, 0, 0, 0, (c11-c12)/2]])
    if not np.isfinite(voigt).all() or np.linalg.eigvalsh(voigt).min() <= 0:
        raise ValueError('crystal stiffness must be positive definite')
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    tensor = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    tensor[ii, jj, kk, ll] = voigt[a, b]
    result = []
    for a, b, c in np.deg2rad(angles):
        za = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        xb = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        zc = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        rotation = za @ xb @ zc
        acoustic = np.einsum('ia,b,kc,d,abcd->ik', rotation, rotation[0], rotation, rotation[0], tensor)
        result.append((acoustic + acoustic.T) / 2)
    return np.asarray(result).reshape((-1, 3, 3))

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
def summary(out):
    return tuple(np.round(out.ravel()/1e9,5))
""",
            "call": 'summary(crystal_acoustic_tensors(ANGLES,TI))',
            "gold_call": 'summary(_oracle_crystal_acoustic_tensors(ANGLES,TI))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
def summary(out):
    return tuple(np.round(out.ravel()/1e9,5))
""",
            "call": 'summary(crystal_acoustic_tensors(ANGLES,ISO))',
            "gold_call": 'summary(_oracle_crystal_acoustic_tensors(ANGLES,ISO))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
def summary(out):
    return tuple(np.round(out.ravel()/1e9,5))
""",
            "call": 'summary(crystal_acoustic_tensors(np.array([[0.,0.,0.],[0.,90.,90.]]),TI))',
            "gold_call": 'summary(_oracle_crystal_acoustic_tensors(np.array([[0.,0.,0.],[0.,90.,90.]]),TI))',
        },
    ]
