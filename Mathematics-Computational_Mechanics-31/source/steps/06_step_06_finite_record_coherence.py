"""
Measure interference between elastic eigenspaces in a frequency band. The numerator is the band sum of squared vector Fourier amplitudes of the total receptor displacement. The denominator is the sum of the corresponding powers of the individual eigenspace histories. Use the complete record, NumPy's default Fourier normalisation, no window and no zero padding, retaining non-negative bin centres in the closed band. The powers are unnormalised DFT sums. Their dimensionless ratio can exceed one.

Finite records can leave neighbouring resonances unresolved. Cross terms between modal Fourier amplitudes then contribute to the observed vector displacement power.

Returns
-------
dict, keys ratio, coherent_power, incoherent_power and integer n_bins.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_record_coherence(history, dt, frequency_band) -> dict:
    r"""Measure interference between elastic eigenspaces in a frequency band. The numerator is the band sum of squared vector Fourier amplitudes of the total receptor displacement. The denominator is the sum of the corresponding powers of the individual eigenspace histories. Use the complete record, NumPy's default Fourier normalisation, no window and no zero padding, retaining non-negative bin centres in the closed band. The powers are unnormalised DFT sums. Their dimensionless ratio can exceed one.

    Parameters
    ----------
    history
        Finite displacement array (n_samples, n_clusters, 3), including the initial sample.
    dt
        Positive time increment in second.
    frequency_band
        Pair (low, high) in hertz, with 0 < low < high <= Nyquist and at least one retained bin.

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

import numpy as np
from scipy.linalg import block_diag, eigh, null_space


def _oracle_finite_record_coherence(history, dt, frequency_band):
    history = np.asarray(history, dtype=float)
    band = np.asarray(frequency_band, dtype=float)
    if history.ndim != 3 or history.shape[0] < 2 or history.shape[1] < 1 or history.shape[2] != 3 or not np.isfinite(history).all():
        raise ValueError('history must be finite with shape (n_samples, n_clusters, 3)')
    if not np.isfinite(dt) or dt <= 0 or band.shape != (2,) or not np.isfinite(band).all() or not 0 < band[0] < band[1] <= 0.5/dt:
        raise ValueError('frequency_band must lie above zero and at or below Nyquist')
    frequencies = np.fft.rfftfreq(history.shape[0], dt)
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    if not selected.any():
        raise ValueError('the frequency band contains no DFT bin')
    transformed = np.fft.rfft(history, axis=0)[selected]
    coherent = float(np.sum(np.abs(transformed.sum(axis=1))**2))
    incoherent = float(np.sum(np.abs(transformed)**2))
    if incoherent == 0:
        raise ValueError('the selected modal response has zero power')
    return {'ratio': coherent/incoherent, 'coherent_power': coherent,
            'incoherent_power': incoherent, 'n_bins': int(selected.sum())}

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
dt=2e-9
t=np.arange(257)*dt
H=np.zeros((257,3,3))
H[:,0,0]=np.sin(2*np.pi*3.2e6*t)
H[:,1,0]=.7*np.sin(2*np.pi*4.1e6*t+.6)
H[:,1,1]=.3*np.cos(2*np.pi*5.2e6*t)
H[:,2,2]=.8*np.sin(2*np.pi*8.9e6*t-.3)
def summary(out):
    return (round(out['ratio'],7),round(out['coherent_power']/1e4,7),round(out['incoherent_power']/1e4,7),out['n_bins'])
""",
            "call": 'summary(finite_record_coherence(H,dt,np.array([2e6,9e6])))',
            "gold_call": 'summary(_oracle_finite_record_coherence(H,dt,np.array([2e6,9e6])))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
dt=2e-9
t=np.arange(257)*dt
H=np.zeros((257,3,3))
H[:,0,0]=np.sin(2*np.pi*3.2e6*t)
H[:,1,0]=.7*np.sin(2*np.pi*4.1e6*t+.6)
H[:,1,1]=.3*np.cos(2*np.pi*5.2e6*t)
H[:,2,2]=.8*np.sin(2*np.pi*8.9e6*t-.3)
def summary(out):
    return (round(out['ratio'],7),round(out['coherent_power']/1e4,7),round(out['incoherent_power']/1e4,7),out['n_bins'])
H[:,1]=-H[:,0]
H[:,2]=0
""",
            "call": 'summary(finite_record_coherence(H,dt,np.array([1e6,9e6])))',
            "gold_call": 'summary(_oracle_finite_record_coherence(H,dt,np.array([1e6,9e6])))',
        },
        {
            "setup": """import numpy as np
TI=dict(c11=162.4e9,c33=180.7e9,c12=92e9,c13=69e9,c44=46.7e9)
ISO=dict(c11=170e9,c33=170e9,c12=70e9,c13=70e9,c44=50e9)
ANGLES=np.array([[17.,63.,29.],[124.,37.,204.],[279.,112.,81.]])
dt=2e-9
t=np.arange(257)*dt
H=np.zeros((257,3,3))
H[:,0,0]=np.sin(2*np.pi*3.2e6*t)
H[:,1,0]=.7*np.sin(2*np.pi*4.1e6*t+.6)
H[:,1,1]=.3*np.cos(2*np.pi*5.2e6*t)
H[:,2,2]=.8*np.sin(2*np.pi*8.9e6*t-.3)
def summary(out):
    return (round(out['ratio'],7),round(out['coherent_power']/1e4,7),round(out['incoherent_power']/1e4,7),out['n_bins'])
H[:,1]=H[:,0]
H[:,2]=0
""",
            "call": 'summary(finite_record_coherence(H,dt,np.array([1e6,9e6])))',
            "gold_call": 'summary(_oracle_finite_record_coherence(H,dt,np.array([1e6,9e6])))',
        },
    ]
