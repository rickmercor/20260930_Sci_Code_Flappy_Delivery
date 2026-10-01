"""
Return the detector-restricted coherence of scattering-order amplitudes.

The order labels describe coherent alternatives, not an incoherent ensemble.
The requested matrices retain their complex cross terms and their absolute
norms. They permit the normalized detector probability of any reconstructed
exit wave to be evaluated against the full-space norm, including interference.
This is a task-derived representation of the paper's momentum-plane observable.

The source physics is the coherent ordered multislice map and momentum-plane detection. The scattering-order representation is task-derived; it preserves the existing linearized diagnostic and does not alter the released STEM resource boundary.

Returns
-------
np.ndarray, (4,K,K) complex unnormalized coherence matrices for three annuli and the full reciprocal grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def annular_order_coherence(order_amplitudes: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    """Return unnormalized detector-restricted complex coherence matrices.

Parameters
----------
order_amplitudes : array_like
    Finite complex (K,N,N) coefficient amplitudes; K>=1, N>=2 a power of two. Zero coefficients are allowed.
wavelength_a : float
    Positive finite electron wavelength, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
detector_edges_mrad : array_like
    Four finite increasing nonnegative angles in mrad; each half-open annulus must contain a grid point.

Returns
-------
result : np.ndarray
    (4,K,K) complex unnormalized coherence matrices for three annuli and the full reciprocal grid.

Notes
-----
Return unnormalized detector-restricted complex coherence matrices.

order_amplitudes: finite complex (K,N,N), K>=1, N>=2 a power of two.
These are amplitude coefficients, with neither independent nor joint
normalization imposed. Zero coefficient arrays are valid.
wavelength_a and pixel_size_a: positive finite Å scalars.
detector_edges_mrad: four finite strictly increasing nonnegative angles,
defining three consecutive half-open annuli. Every mask must be nonempty.
Within each (N,N) coefficient, axis 0=x and axis 1=y; these are the last
two input axes. Use unshifted frequencies and an orthonormal forward
transform with kernel exp(-2*pi*i*k.r).

Returns complex128 (4,K,K). If z_a is the transformed coefficient a,
H[c,a,b]=sum_mask_c(conj(z_a)*z_b). Channels 0,1,2 are the annuli;
channel 3 is the entire reciprocal grid, not just their union. There is
no normalization or deletion of complex off-diagonal entries. For a
coherent superposition with coefficient weights w, its probability is
(w.conj() @ H[c] @ w)/(w.conj() @ H[3] @ w).
Invalid shapes, sampling, nonfinite data or empty masks raise ValueError."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_annular_order_coherence(order_amplitudes: 'np.ndarray', wavelength_a: float, pixel_size_a: float, detector_edges_mrad: 'np.ndarray') -> 'np.ndarray':
    modes = np.asarray(order_amplitudes, dtype=complex)
    edges = np.asarray(detector_edges_mrad, dtype=float)
    if modes.ndim != 3 or modes.shape[0] < 1 or (not np.all(np.isfinite(modes))):
        raise ValueError('finite (K,N,N) amplitude array required')
    (count, n, m) = modes.shape
    if n != m or n < 2 or n & n - 1:
        raise ValueError('square power-of-two spatial axes required')
    (lam, dx) = (float(wavelength_a), float(pixel_size_a))
    if not np.all(np.isfinite([lam, dx])) or min(lam, dx) <= 0:
        raise ValueError('invalid sampling')
    if edges.shape != (4,) or not np.all(np.isfinite(edges)) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('four ordered nonnegative detector edges required')
    f = np.fft.fftfreq(n, d=dx)
    theta = 1000 * lam * np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    masks = [(theta >= lo) & (theta < hi) for (lo, hi) in zip(edges[:-1], edges[1:])]
    if any((not np.any(mask) for mask in masks)):
        raise ValueError('every annulus must contain a grid point')
    masks.append(np.ones((n, n), dtype=bool))
    spectral = np.fft.fft2(modes, axes=(-2, -1), norm='ortho')
    return np.asarray([spectral[:, mask].conj() @ spectral[:, mask].T for mask in masks])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(417); modes=rng.normal(size=(3,8,8))+1j*rng.normal(size=(3,8,8))\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'},
     {'setup': 'import numpy as np\n'
               'modes=np.zeros((1,8,8),complex); modes[0,2,5]=2+3j\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'},
     {'setup': 'import numpy as np\n'
               'modes=np.asarray([1.,1j,-2.])[:,None,None]*np.ones((1,8,8))/8\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'},
     {'setup': 'import numpy as np\n'
               'spectral=np.zeros((3,4,4),complex); spectral[0,0,0]=1; spectral[1,1,0]=1j; spectral[2,2,0]=-2; '
               'modes=np.fft.ifft2(spectral,norm="ortho")\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.032,0.8,[0.,10.,20.,30.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.032, 0.8, [0.0, 10.0, 20.0, 30.0])'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(417); modes=rng.normal(size=(3,8,8))+1j*rng.normal(size=(3,8,8))\n'
               'modes[1]=-modes[0]+.03*modes[1]\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'},
     {'setup': 'import numpy as np\n'
               'modes=np.zeros((6,8,8),complex)\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(118); '
               'modes=(rng.normal(size=(28,8,8))+1j*rng.normal(size=(28,8,8)))*np.exp(-np.arange(28)[:,None,None]/7)\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(modes,0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(417); modes=rng.normal(size=(3,8,8))+1j*rng.normal(size=(3,8,8))\n'
               'import copy\n'
               'modes_gold = copy.deepcopy(modes)',
      'call': 'annular_order_coherence(np.roll(modes,(2,-3),axis=(-2,-1)),0.025,0.8,[0.,5.,11.,24.])',
      'gold_call': '_oracle_annular_order_coherence(modes_gold, 0.025, 0.8, [0.0, 5.0, 11.0, 24.0])'}]
