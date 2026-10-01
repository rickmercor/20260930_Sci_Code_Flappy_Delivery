"""
Implement solenoidal_projection, the operator that returns the divergence-free

part of a vector field on the periodic unit cube.

Contract

--------

The input F is a (3, N, N, N) float array on the grid x_i = i/N (likewise y_j,

z_k), component-first, with N >= 2 and a cubic grid. The return G has the same

shape, the same layout, and is real-valued.



The separation is made in Fourier space, on the wavevector grid of step 02 and

the coefficients of ``np.fft.fftn`` taken over the three spatial axes, not by

finite differences on the grid. G is required to be divergence-free to

floating-point round-off when its divergence is measured the same way, by the

spectral diagnostic of step 06. That is the physical requirement. The test

criterion is separate and stricter: the returned array is compared

elementwise against the reference projection to a tolerance of 1e-9, so a

field that is divergence-free but built by a different convention will not

pass.



Inputs

------

F: (3, N, N, N) float array, component-first vector field on the grid

   x_i = i/N (likewise y_j, z_k), with N >= 2 and a cubic grid.



Returns

-------

G: (3, N, N, N) float array, the solenoidal part of F on the same grid.

Returns
-------
G : numpy.ndarray of shape (3, N, N, N), float64, the solenoidal part of F.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solenoidal_projection(F: np.ndarray) -> np.ndarray:
    '''Return the solenoidal (divergence-free) part of a periodic vector field.

    The separation is made spectrally, on the wavevector grid of step 02 and
    the coefficients of np.fft.fftn taken over the three spatial axes. The
    returned field is real-valued and is required to be divergence-free to
    floating-point round-off when measured by the spectral diagnostic of
    step 06.

    Parameters
    ----------
    F : np.ndarray
        (3, N, N, N) float array, component-first vector field on the periodic
        unit cube with grid x_i = i/N, and N >= 2.

    Returns
    -------
    G : np.ndarray
        (3, N, N, N) float array, the solenoidal part of F.

    Raises
    ------
    ValueError
        If F is not a (3, N, N, N) float array with equal trailing
        dimensions and N >= 2.
    '''
    return G  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solenoidal_projection(F: np.ndarray) -> np.ndarray:
    arr = np.asarray(F, dtype=float)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError("F must be an array of shape (3, N, N, N)")
    n = int(arr.shape[1])
    if arr.shape[2] != n or arr.shape[3] != n:
        raise ValueError("F must be defined on a cubic (N, N, N) grid")
    if n < 2:
        raise ValueError("N must be >= 2")

    kgrid = _oracle_wavevector_grid(n)
    KX, KY, KZ = kgrid[0], kgrid[1], kgrid[2]

    Fh = np.fft.fftn(arr, axes=(1, 2, 3))

    nyq = n // 2
    if n % 2 == 0:
        Fh[:, nyq, :, :] = 0.0
        Fh[:, :, nyq, :] = 0.0
        Fh[:, :, :, nyq] = 0.0

    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0

    dot = KX * Fh[0] + KY * Fh[1] + KZ * Fh[2]
    Gh = np.empty_like(Fh)
    Gh[0] = Fh[0] - KX * dot / K2
    Gh[1] = Fh[1] - KY * dot / K2
    Gh[2] = Fh[2] - KZ * dot / K2

    Gh[:, 0, 0, 0] = Fh[:, 0, 0, 0]

    if n % 2 == 0:
        Gh[:, nyq, :, :] = 0.0
        Gh[:, :, nyq, :] = 0.0
        Gh[:, :, :, nyq] = 0.0

    return np.real(np.fft.ifftn(Gh, axes=(1, 2, 3)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the seed field on an N = 8 grid, smooth envelope ---
        {
            "setup": """import numpy as np
n = 8
amp = 2.0
sig = 0.15
kx = 2
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (XX - 0.5) ** 2 + (YY - 0.5) ** 2 + (ZZ - 0.5) ** 2
env = amp * np.exp(-dr2 / (2 * sig ** 2))
phi = 2 * np.pi * kx * XX
F_in = np.empty((3, n, n, n))
F_in[0] = 1.0
F_in[1] = np.cos(phi) * env
F_in[2] = np.sin(phi) * env
""",
            "call": "solenoidal_projection(F_in)",
            "gold_call": "_oracle_solenoidal_projection(F_in)",
        },
        # --- Normal: the seed field on N = 8 with the narrow reference envelope,
        #     which puts real power on the Nyquist planes ---
        {
            "setup": """import numpy as np
n = 8
amp = 20.0
sig = 1.0 / 30.0
kx = 4
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (XX - 0.5) ** 2 + (YY - 0.5) ** 2 + (ZZ - 0.5) ** 2
env = amp * np.exp(-dr2 / (2 * sig ** 2))
phi = 2 * np.pi * kx * XX
F_in = np.empty((3, n, n, n))
F_in[0] = 1.0
F_in[1] = np.cos(phi) * env
F_in[2] = np.sin(phi) * env
""",
            "call": "solenoidal_projection(F_in)",
            "gold_call": "_oracle_solenoidal_projection(F_in)",
        },
        # --- Boundary: an already-solenoidal field, a single-mode transverse
        #     wave on a uniform mean field, which must come back unchanged ---
        {
            "setup": """import numpy as np
n = 8
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
phi = 2 * np.pi * XX
F_in = np.empty((3, n, n, n))
F_in[0] = 1.0
F_in[1] = np.cos(phi)
F_in[2] = np.sin(phi)
""",
            "call": "solenoidal_projection(F_in)",
            "gold_call": "_oracle_solenoidal_projection(F_in)",
        },
        # --- Edge: odd grid N = 5, where no Nyquist plane exists ---
        {
            "setup": """import numpy as np
n = 5
amp = 1.5
sig = 0.2
kx = 1
coord = np.arange(n) / n
XX, YY, ZZ = np.meshgrid(coord, coord, coord, indexing="ij")
dr2 = (XX - 0.5) ** 2 + (YY - 0.5) ** 2 + (ZZ - 0.5) ** 2
env = amp * np.exp(-dr2 / (2 * sig ** 2))
phi = 2 * np.pi * kx * XX
F_in = np.empty((3, n, n, n))
F_in[0] = 1.0
F_in[1] = np.cos(phi) * env
F_in[2] = np.sin(phi) * env
""",
            "call": "solenoidal_projection(F_in)",
            "gold_call": "_oracle_solenoidal_projection(F_in)",
        },
        # --- Invalid: only two components, shape (2, 8, 8, 8) ---
        {
            "setup": """import numpy as np
F_bad = np.zeros((2, 8, 8, 8))
def run_model():
    try:
        solenoidal_projection(F_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solenoidal_projection(F_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-cubic grid, shape (3, 8, 8, 4) ---
        {
            "setup": """import numpy as np
F_bad = np.zeros((3, 8, 8, 4))
def run_model():
    try:
        solenoidal_projection(F_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solenoidal_projection(F_bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
