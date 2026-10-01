"""
Compute the finite-record interference ratio from the grain orientations and physical configuration. Retain the rotated anisotropic acoustic tensors, the odd periodic grid, the massless compliant buffer, every elastic eigenspace and the prescribed finite-duration pulse experiment. The source is voxel zero. Compare the total receptor vector displacement power with the sum of isolated-eigenspace powers in the band. Recompute every intermediate quantity from the top-level inputs.

The measured spectral power depends jointly on anisotropy, exterior equilibrium, modal excitation and the finite observation interval.

Returns
-------
dict, keys ratio, coherent_power, incoherent_power and integer n_bins.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polycrystal_coherence(euler_angles, constants, grain_voxels, density, length, n_buffer, buffer_ratio, direction, receptor, dt, n_steps, pulse_amplitude, pulse_width, pulse_centre, frequency_band, cluster_rtol) -> dict:
    r"""Compute the finite-record interference ratio from the grain orientations and physical configuration. Retain the rotated anisotropic acoustic tensors, the odd periodic grid, the massless compliant buffer, every elastic eigenspace and the prescribed finite-duration pulse experiment. The source is voxel zero. Compare the total receptor vector displacement power with the sum of isolated-eigenspace powers in the band. Recompute every intermediate quantity from the top-level inputs.

    Parameters
    ----------
    euler_angles
        Finite array (n_grains, 3) of Euler angles in degrees.
    constants
        Mapping c11, c33, c12, c13, c44 in pascal, defining a positive definite hexagonal stiffness.
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
    direction
        Finite nonzero vector (3,), normalised internally.
    receptor
        Integer receptor voxel index inside the specimen.
    dt
        Positive time increment in second.
    n_steps
        Positive integer number of time increments.
    pulse_amplitude
        Finite scalar traction amplitude in pascal.
    pulse_width
        Positive Gaussian width in second.
    pulse_centre
        Finite first-pulse centre in second.
    frequency_band
        Pair (low, high) in hertz, with 0 < low < high <= Nyquist and at least one retained bin.
    cluster_rtol
        Relative clustering threshold in [0, 1), measured from the first frequency of each cluster.

    Returns
    -------
    dict, keys ratio, coherent_power, incoherent_power and integer n_bins.

    Raises
    ------
    ValueError
        If input dimensions or stated physical constraints are violated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_polycrystal_coherence(euler_angles, constants, grain_voxels, density, length, n_buffer,
                         buffer_ratio, direction, receptor, dt, n_steps, pulse_amplitude,
                         pulse_width, pulse_centre, frequency_band, cluster_rtol):
    acoustic = _oracle_crystal_acoustic_tensors(euler_angles, constants)  # noqa: F821
    domain = _oracle_buffered_stiffness(acoustic, grain_voxels, density, length, n_buffer, buffer_ratio)  # noqa: F821
    eigensystem = _oracle_elastic_modes(domain['stiffness'], domain['mass'], domain['n_specimen'])  # noqa: F821
    grouped = _oracle_modal_residues(eigensystem['frequencies'], eigensystem['modes'], 0, receptor,  # noqa: F821
                             direction, domain['voxel_size'], cluster_rtol)
    history = _oracle_modal_impact_record(grouped['frequencies'], grouped['residues'], dt, n_steps,  # noqa: F821
                                  pulse_amplitude, pulse_width, pulse_centre)
    result = _oracle_finite_record_coherence(history, dt, frequency_band)  # noqa: F821
    return result

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
CFG=dict(euler_angles=ANGLES,constants=TI,grain_voxels=np.array([4,3,5]),density=4506.3,
length=1e-3,n_buffer=3,buffer_ratio=1e-7,direction=np.array([1.,2.,-1.]),receptor=11,
dt=4e-9,n_steps=192,pulse_amplitude=1e6,pulse_width=2e-8,pulse_centre=8e-8,
frequency_band=np.array([2e6,9e6]),cluster_rtol=1e-8)
def summary(out):
    return (round(out['ratio'],6),round(out['coherent_power']*1e15,6),round(out['incoherent_power']*1e15,6),out['n_bins'])
""",
            "call": 'summary(polycrystal_coherence(**CFG))',
            "gold_call": 'summary(_oracle_polycrystal_coherence(**CFG))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
CFG=dict(euler_angles=ANGLES,constants=TI,grain_voxels=np.array([4,3,5]),density=4506.3,
length=1e-3,n_buffer=3,buffer_ratio=1e-7,direction=np.array([1.,2.,-1.]),receptor=11,
dt=4e-9,n_steps=192,pulse_amplitude=1e6,pulse_width=2e-8,pulse_centre=8e-8,
frequency_band=np.array([2e6,9e6]),cluster_rtol=1e-8)
def summary(out):
    return (round(out['ratio'],6),round(out['coherent_power']*1e15,6),round(out['incoherent_power']*1e15,6),out['n_bins'])
CFG.update(constants=ISO,direction=np.array([1.,1.,1.]),n_steps=160)
""",
            "call": 'summary(polycrystal_coherence(**CFG))',
            "gold_call": 'summary(_oracle_polycrystal_coherence(**CFG))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
CFG=dict(euler_angles=ANGLES,constants=TI,grain_voxels=np.array([4,3,5]),density=4506.3,
length=1e-3,n_buffer=3,buffer_ratio=1e-7,direction=np.array([1.,2.,-1.]),receptor=11,
dt=4e-9,n_steps=192,pulse_amplitude=1e6,pulse_width=2e-8,pulse_centre=8e-8,
frequency_band=np.array([2e6,9e6]),cluster_rtol=1e-8)
def summary(out):
    return (round(out['ratio'],6),round(out['coherent_power']*1e15,6),round(out['incoherent_power']*1e15,6),out['n_bins'])
CFG.update(grain_voxels=np.array([3,5,4]),direction=np.array([2.,-1.,.3]),n_buffer=5,n_steps=256,receptor=8)
""",
            "call": 'summary(polycrystal_coherence(**CFG))',
            "gold_call": 'summary(_oracle_polycrystal_coherence(**CFG))',
        },
    ]
