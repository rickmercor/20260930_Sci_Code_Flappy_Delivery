"""
The specimen occupies the first n_specimen voxels of an odd periodic grid, with three displacement components stored consecutively per voxel. A forward voxel difference defines strain and its adjoint defines the positive stiffness operator. Buffer voxels have zero density and acoustic tensor buffer_ratio times diag(mean_A11, mean_Atransverse, mean_Atransverse), where the means are voxel-weighted specimen averages and mean_Atransverse is half the sum of the transverse diagonal means. Construct the real stiffness matrix of this Fourier discretisation and its diagonal inertia coefficients. Neither matrix includes a voxel-volume factor.

A compliant exterior permits a periodic discretisation of a specimen with approximately traction-free surfaces. Its stiffness and inertia have distinct physical roles.

Returns
-------
dict, stiffness matrix, diagonal inertia, specimen voxel count and voxel size.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def buffered_stiffness(acoustic, grain_voxels, density, length, n_buffer, buffer_ratio) -> dict:
    r"""The specimen occupies the first n_specimen voxels of an odd periodic grid, with three displacement components stored consecutively per voxel. A forward voxel difference defines strain and its adjoint defines the positive stiffness operator. Buffer voxels have zero density and acoustic tensor buffer_ratio times diag(mean_A11, mean_Atransverse, mean_Atransverse), where the means are voxel-weighted specimen averages and mean_Atransverse is half the sum of the transverse diagonal means. Construct the real stiffness matrix of this Fourier discretisation and its diagonal inertia coefficients. Neither matrix includes a voxel-volume factor.

    Parameters
    ----------
    acoustic
        Finite symmetric positive definite tensors (n_grains, 3, 3), in pascal.
    grain_voxels
        Positive integer array (n_grains,), in spatial order.
    density
        Positive specimen density in kg per cubic metre.
    length
        Positive specimen length in metre.
    n_buffer
        Positive integer buffer voxel count; total grid length is odd.
    buffer_ratio
        Positive buffer stiffness multiplier.

    Returns
    -------
    dict, keys stiffness (3*n_total, 3*n_total), mass (3*n_total,), n_specimen and voxel_size.

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


def _oracle_buffered_stiffness(acoustic, grain_voxels, density, length, n_buffer, buffer_ratio):
    acoustic = np.asarray(acoustic, dtype=float)
    counts = np.asarray(grain_voxels)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer) or counts.size == 0 or np.any(counts < 1):
        raise ValueError('grain_voxels must contain positive integers')
    if acoustic.shape != (counts.size, 3, 3) or not np.isfinite(acoustic).all():
        raise ValueError('acoustic must have shape (n_grains, 3, 3)')
    if not np.allclose(acoustic, acoustic.swapaxes(-1, -2), rtol=1e-12, atol=0) or np.linalg.eigvalsh(acoustic).min() <= 0:
        raise ValueError('acoustic tensors must be symmetric positive definite')
    if not isinstance(n_buffer, (int, np.integer)) or n_buffer < 1:
        raise ValueError('n_buffer must be a positive integer')
    if not all(np.isfinite(x) and x > 0 for x in (density, length, buffer_ratio)):
        raise ValueError('density, length and buffer_ratio must be positive')
    n_specimen = int(counts.sum())
    n_total = n_specimen + int(n_buffer)
    if n_total % 2 == 0:
        raise ValueError('the periodic grid must have odd length')
    h = float(length) / n_specimen
    specimen = np.repeat(acoustic, counts, axis=0)
    average = specimen.mean(axis=0)
    buffer = np.diag([average[0, 0], (average[1, 1]+average[2, 2])/2,
                      (average[1, 1]+average[2, 2])/2]) * buffer_ratio
    material = np.concatenate([specimen, np.repeat(buffer[None], n_buffer, axis=0)])
    symbol = (np.exp(2j*np.pi*np.fft.fftfreq(n_total)) - 1) / h
    derivative = np.fft.ifft(symbol[:, None] * np.fft.fft(np.eye(n_total), axis=0), axis=0).real
    gradient = np.kron(derivative, np.eye(3))
    stiffness = gradient.T @ block_diag(*material) @ gradient
    mass = np.repeat(np.r_[np.full(n_specimen, density), np.zeros(n_buffer)], 3)
    return {'stiffness': (stiffness+stiffness.T)/2, 'mass': mass,
            'n_specimen': n_specimen, 'voxel_size': h}

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
A=np.array([[[160.,12.,-7.],[12.,49.,3.],[-7.,3.,43.]],[[180.,-8.,6.],[-8.,46.,-4.],[6.,-4.,52.]]])*1e9
def summary(out):
    k=out['stiffness']; m=out['mass']; n=m.size; p=np.sin(np.arange(n)*.37)
    return (k.shape,tuple(np.round((k@p)[[0,2,4,7,-4]]/1e18,6)),round(float(p@k@p)/1e18,6),round(float(m.sum()),5),out['n_specimen'],round(out['voxel_size']*1e6,5),int(np.linalg.norm(k@np.tile(np.eye(3),(n//3,1))) < 1e-12*np.linalg.norm(k)))
""",
            "call": 'summary(buffered_stiffness(A,np.array([3,5]),4506.3,1e-3,3,1e-7))',
            "gold_call": 'summary(_oracle_buffered_stiffness(A,np.array([3,5]),4506.3,1e-3,3,1e-7))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
A=np.array([[[160.,12.,-7.],[12.,49.,3.],[-7.,3.,43.]],[[180.,-8.,6.],[-8.,46.,-4.],[6.,-4.,52.]]])*1e9
def summary(out):
    k=out['stiffness']; m=out['mass']; n=m.size; p=np.sin(np.arange(n)*.37)
    return (k.shape,tuple(np.round((k@p)[[0,2,4,7,-4]]/1e18,6)),round(float(p@k@p)/1e18,6),round(float(m.sum()),5),out['n_specimen'],round(out['voxel_size']*1e6,5),int(np.linalg.norm(k@np.tile(np.eye(3),(n//3,1))) < 1e-12*np.linalg.norm(k)))
""",
            "call": 'summary(buffered_stiffness(A,np.array([2,2]),2700.,2e-3,1,.01))',
            "gold_call": 'summary(_oracle_buffered_stiffness(A,np.array([2,2]),2700.,2e-3,1,.01))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
A=np.array([[[160.,12.,-7.],[12.,49.,3.],[-7.,3.,43.]],[[180.,-8.,6.],[-8.,46.,-4.],[6.,-4.,52.]]])*1e9
def summary(out):
    k=out['stiffness']; m=out['mass']; n=m.size; p=np.sin(np.arange(n)*.37)
    return (k.shape,tuple(np.round((k@p)[[0,2,4,7,-4]]/1e18,6)),round(float(p@k@p)/1e18,6),round(float(m.sum()),5),out['n_specimen'],round(out['voxel_size']*1e6,5),int(np.linalg.norm(k@np.tile(np.eye(3),(n//3,1))) < 1e-12*np.linalg.norm(k)))
""",
            "call": 'summary(buffered_stiffness(A[:1],np.array([6]),4506.3,1e-3,3,1e-5))',
            "gold_call": 'summary(_oracle_buffered_stiffness(A[:1],np.array([6]),4506.3,1e-3,3,1e-5))',
        },
    ]
