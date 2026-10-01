"""
Construct the explicit six-slice periodic benchmark, then sum consecutive groups. On L=N*dx, wrap each coordinate displacement to [-L/2,L/2). Axis 0 is x and axis 1 is y: a center (cx,cy) sits at array index [cx/pixel_size_a,cy/pixel_size_a] modulo N, equivalently meshgrid indexing='ij'. Use centers (0,0), (L/2,L/2), (L/4,3L/4), w=0.42 Å, and A*exp(-r_periodic^2/(2*width^2)). In slice j=0..5, the central amplitude is (1500 for hypothesis 1, else 1050)*(1+0.08*((j%3)-1)); the other amplitudes are 650*(1+0.05*(-1)^j) and 400*(1-0.04*(-1)^j), with widths w,1.25w,0.9w. For g in {1,2,3,6}, return 6/g slices formed by summing base[j:j+g].

The periodic specimen and grouping are task-authored test data for the source multislice method. Summed projected potential is preserved; the scattering evolution need not be.

Returns
-------
np.ndarray, (6//group_factor,N,N) real grouped projected potentials in V Å.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_grouped_strong_scattering_stack(n: int, pixel_size_a: float, hypothesis: int, group_factor: int=1) -> 'np.ndarray':
    """Build the six-slice periodic potential for one specimen hypothesis.

Parameters
----------
n : int
    Power-of-two square-grid dimension, at least 2.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.
hypothesis : int
    Specimen class: 0 denotes H0 and 1 denotes H1.
group_factor : int
    Number of consecutive base slices per retained grating; one of 1, 2, 3, 6.

Returns
-------
result : np.ndarray
    (6//group_factor,N,N) real grouped projected potentials in V Å.

Notes
-----
Build the six-slice periodic potential for one specimen hypothesis.

n and pixel_size_a define the square grid; hypothesis is class H0 (0) or
class H1 (1); group_factor must divide six. Axis 0 is x and axis 1 is y,
so (cx,cy) maps to [cx/pixel_size_a,cy/pixel_size_a] modulo n. Returns a
float ndarray of shape (6//group_factor,n,n), with consecutive slices
summed in order."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_build_grouped_strong_scattering_stack(n: int, pixel_size_a: float, hypothesis: int, group_factor: int=1) -> 'np.ndarray':
    """Return grouped projected potentials for the fixed six-slice specimen."""
    n = int(n)
    pixel_size_a = float(pixel_size_a)
    hypothesis = int(hypothesis)
    group_factor = int(group_factor)
    if n < 2 or n & n - 1:
        raise ValueError('n must be a power of two at least 2')
    if not math.isfinite(pixel_size_a) or pixel_size_a <= 0.0:
        raise ValueError('pixel_size_a must be finite and positive')
    if hypothesis not in (0, 1):
        raise ValueError('hypothesis must be 0 or 1')
    if group_factor not in (1, 2, 3, 6):
        raise ValueError('group_factor must divide the six base slices')
    length = n * pixel_size_a
    axis = np.arange(n, dtype=float) * pixel_size_a
    (xx, yy) = np.meshgrid(axis, axis, indexing='ij')

    def _periodic_r2(cx, cy):
        dx = (xx - cx + 0.5 * length) % length - 0.5 * length
        dy = (yy - cy + 0.5 * length) % length - 0.5 * length
        return dx * dx + dy * dy
    width = 0.42
    d0 = _periodic_r2(0.0, 0.0)
    d1 = _periodic_r2(0.5 * length, 0.5 * length)
    d2 = _periodic_r2(0.25 * length, 0.75 * length)
    base = []
    for j in range(6):
        central = (1500.0 if hypothesis == 1 else 1050.0) * (1.0 + 0.08 * (j % 3 - 1))
        corner = 650.0 * (1.0 + 0.05 * (-1) ** j)
        offset = 400.0 * (1.0 - 0.04 * (-1) ** j)
        base.append(central * np.exp(-d0 / (2.0 * width ** 2)) + corner * np.exp(-d1 / (2.0 * (1.25 * width) ** 2)) + offset * np.exp(-d2 / (2.0 * (0.9 * width) ** 2)))
    base = np.asarray(base, dtype=float)
    return np.asarray([np.sum(base[j:j + group_factor], axis=0) for j in range(0, 6, group_factor)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'float(build_grouped_strong_scattering_stack(16,0.4,1,1)[0,0,0])',
      'gold_call': 'float(_oracle_build_grouped_strong_scattering_stack(16,0.4,1,1)[0,0,0])'},
     {'setup': 'import numpy as np',
      'call': 'float(np.sum(build_grouped_strong_scattering_stack(16,0.4,0,3)[0]))',
      'gold_call': 'float(np.sum(_oracle_build_grouped_strong_scattering_stack(16,0.4,0,3)[0]))'},
     {'setup': '',
      'call': 'float(build_grouped_strong_scattering_stack(32,0.2,1,6)[0,17,29])',
      'gold_call': 'float(_oracle_build_grouped_strong_scattering_stack(32,0.2,1,6)[0,17,29])'},
     {'setup': 'import numpy as np',
      'call': 'float(np.linalg.norm(build_grouped_strong_scattering_stack(8,0.8,0,2)[1]))',
      'gold_call': 'float(np.linalg.norm(_oracle_build_grouped_strong_scattering_stack(8,0.8,0,2)[1]))'},
     {'setup': '',
      'call': 'float(build_grouped_strong_scattering_stack(8,0.8,1,2)[2,0,0])',
      'gold_call': 'float(_oracle_build_grouped_strong_scattering_stack(8,0.8,1,2)[2,0,0])'},
     {'setup': 'import numpy as np',
      'call': 'float(np.mean(build_grouped_strong_scattering_stack(32,0.2,0,1)[5]))',
      'gold_call': 'float(np.mean(_oracle_build_grouped_strong_scattering_stack(32,0.2,0,1)[5]))'},
     {'setup': '',
      'call': 'float(build_grouped_strong_scattering_stack(16,0.4,1,3)[1,8,8])',
      'gold_call': 'float(_oracle_build_grouped_strong_scattering_stack(16,0.4,1,3)[1,8,8])'}]
