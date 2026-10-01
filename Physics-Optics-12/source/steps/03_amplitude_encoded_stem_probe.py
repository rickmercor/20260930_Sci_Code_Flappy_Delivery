"""
Form a displaced QuScope STEM input state on two log2(N)-qubit registers. With unshifted fftfreq and ij indexing, use A(k)=1 for |k|<=alpha*1e-3/lambda+1e-15; chi=pi*lambda*defocus*k^2+(pi/2)*(cs_mm*1e7)*lambda^3*k^4; multiply by exp(-2*pi*i*(kx*x_scan+ky*y_scan)), compute the orthonormal IFFT2, then normalize once. Array entry [a,b] is the amplitude of |a>row tensor |b>column in row-major flattening.

QuScope amplitude-encodes the focused electron probe rather than treating it only as an intensity image.

Returns
-------
np.ndarray, (N,N) unit-L2-normalized complex incident amplitudes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def amplitude_encoded_stem_probe(n: int, pixel_size_a: float, wavelength_a: float, convergence_mrad: float, defocus_a: float, cs_mm: float=0.05, scan_position_a: tuple=(0.0, 0.0)) -> 'np.ndarray':
    """Prepare one displaced aberrated STEM probe on an n-by-n grid.

Parameters
----------
n : int
    Power-of-two square-grid dimension, at least 2.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
wavelength_a : float
    Positive finite electron wavelength, in Å.
convergence_mrad : float
    Positive finite circular-aperture semi-angle, in mrad.
defocus_a : float
    Finite defocus, in Å; either sign is allowed.
cs_mm : float
    Finite nonnegative spherical-aberration coefficient, in mm.
scan_position_a : tuple
    Finite (x,y) probe displacement, in Å.

Returns
-------
result : np.ndarray
    (N,N) unit-L2-normalized complex incident amplitudes.

Notes
-----
Prepare one displaced aberrated STEM probe on an n-by-n grid.

Sampling and optical arguments are finite scalars and scan_position_a is
an (x,y) pair in angstrom. Returns a normalized complex ndarray of shape
(n,n), using the stated Fourier sign and register order."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_amplitude_encoded_stem_probe(n: int, pixel_size_a: float, wavelength_a: float, convergence_mrad: float, defocus_a: float, cs_mm: float=0.05, scan_position_a: tuple=(0.0, 0.0)) -> 'np.ndarray':
    """Return the normalized N-by-N amplitudes stored on 2 log2(N) qubits."""
    n = int(n)
    values = [float(pixel_size_a), float(wavelength_a), float(convergence_mrad), float(defocus_a), float(cs_mm)]
    position = np.asarray(scan_position_a, dtype=float)
    if n < 2 or n & n - 1:
        raise ValueError('n must be a power of two at least 2')
    if any((not math.isfinite(x) for x in values)):
        raise ValueError('probe parameters must be finite')
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError('scan_position_a must contain two finite coordinates')
    if values[0] <= 0.0 or values[1] <= 0.0 or values[2] <= 0.0 or (values[4] < 0.0):
        raise ValueError('invalid probe scale')
    frequency = np.fft.fftfreq(n, d=values[0])
    (kx, ky) = np.meshgrid(frequency, frequency, indexing='ij')
    k2 = kx * kx + ky * ky
    aperture = np.sqrt(k2) <= values[2] * 0.001 / values[1] + 1e-15
    cs_a = values[4] * 10000000.0
    chi = np.pi * values[1] * values[3] * k2 + 0.5 * np.pi * cs_a * values[1] ** 3 * k2 ** 2
    shift = np.exp(-2j * np.pi * (kx * position[0] + ky * position[1]))
    reciprocal = aperture.astype(float) * np.exp(-1j * chi) * shift
    probe = np.fft.ifft2(reciprocal, norm='ortho')
    norm = float(np.linalg.norm(probe))
    if norm == 0.0:
        raise ValueError('empty aperture')
    return np.asarray(probe / norm, dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np',
      'call': 'float(np.linalg.norm(amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,18.0,0.0)))',
      'gold_call': 'float(np.linalg.norm(_oracle_amplitude_encoded_stem_probe(16,0.4,0.025079340436272274,18.0,0.0)))'},
     {'setup': '',
      'call': 'float(amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,10.0,-30.0)[7,21].imag)',
      'gold_call': 'float(_oracle_amplitude_encoded_stem_probe(32,0.2,0.01968748899648993,10.0,-30.0)[7,21].imag)'},
     {'setup': '',
      'call': 'float(abs(amplitude_encoded_stem_probe(8,0.8,0.03701436611487085,12.0,30.0)[3,4]))',
      'gold_call': 'float(abs(_oracle_amplitude_encoded_stem_probe(8,0.8,0.03701436611487085,12.0,30.0)[3,4]))'},
     {'setup': '',
      'call': 'float(amplitude_encoded_stem_probe(32,0.2,0.025079340436272274,14.0,15.0,0.05,(0.18,0.18))[11,6].real)',
      'gold_call': 'float(_oracle_amplitude_encoded_stem_probe(32,0.2,0.025079340436272274,14.0,15.0,0.05,(0.18,0.18))[11,6].real)'},
     {'setup': '',
      'call': 'float(amplitude_encoded_stem_probe(16,0.4,0.030,16.0,-45.0,0.05,(0.31,-0.13))[4,12].real)',
      'gold_call': 'float(_oracle_amplitude_encoded_stem_probe(16,0.4,0.030,16.0,-45.0,0.05,(0.31,-0.13))[4,12].real)'},
     {'setup': '',
      'call': 'float(amplitude_encoded_stem_probe(8,0.8,0.025,10.0,45.0,0.0,(0.18,0.0))[1,6].imag)',
      'gold_call': 'float(_oracle_amplitude_encoded_stem_probe(8,0.8,0.025,10.0,45.0,0.0,(0.18,0.0))[1,6].imag)'},
     {'setup': 'import numpy as np',
      'call': 'float(np.sum(np.abs(amplitude_encoded_stem_probe(32,0.2,0.025079340436272274,13.0,15.0))**2))',
      'gold_call': 'float(np.sum(np.abs(_oracle_amplitude_encoded_stem_probe(32,0.2,0.025079340436272274,13.0,15.0))**2))'}]
