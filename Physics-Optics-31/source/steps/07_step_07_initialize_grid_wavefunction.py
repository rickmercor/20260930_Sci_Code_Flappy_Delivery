"""
Evaluate the same two-dimensional Gaussian used by the mesh-free expansion.

A fair comparison uses the target packet's width and phase for the grid field rather than the narrower representing width.

Returns
-------
np.ndarray: finite complex Cartesian field with unit cell-area-weighted L2 norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np

def initialize_grid_wavefunction(
    x: np.ndarray,
    y: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    gamma0: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return the discretely normalized target Gaussian on uniform axes.

    ``q0`` and ``p0`` have length two and ``gamma0`` is a real symmetric
    positive-definite 2-by-2 matrix. The finite one-dimensional axes must
    each contain at least two strictly monotone, uniformly spaced points with
    nonzero spacing. The sampled field must have finite nonzero cell norm.

    Returns
    -------
    np.ndarray
        Complex field of shape ``(len(x),len(y))``.

    Raises
    ------
    ValueError
        If axes are invalid, centers or widths have invalid shapes, inputs
        are non-finite, ``gamma0`` is not symmetric positive-definite, ``hbar``
        is not positive finite, or the sampled field has zero or non-finite norm.
    """
    return wavefunction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np

def _oracle_initialize_grid_wavefunction(
    x: np.ndarray,
    y: np.ndarray,
    q0: np.ndarray,
    p0: np.ndarray,
    gamma0: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference normalized Gaussian sampling."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    q0, p0, g0 = np.asarray(q0, float), np.asarray(p0, float), np.asarray(gamma0, float)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("x and y must be one-dimensional axes")
    if q0.shape != (2,) or p0.shape != (2,) or g0.shape != (2, 2):
        raise ValueError("center and width shapes are invalid")
    for axis in (x, y):
        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must be finite")
        spacing = np.diff(axis)
        if (not np.all(np.isfinite(spacing))
                or not (np.all(spacing > 0) or np.all(spacing < 0))
                or not np.allclose(spacing, spacing[0])):
            raise ValueError("axes must be strictly monotone and uniform with nonzero spacing")
    if not np.allclose(g0, g0.T):
        raise ValueError("gamma0 must be symmetric")
    try:
        np.linalg.cholesky(g0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("gamma0 must be positive definite") from exc
    if isinstance(hbar, bool) or not isinstance(hbar, Real) or not np.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be positive and finite")
    if not np.all(np.isfinite(np.r_[x, y, q0, p0, g0.ravel()])):
        raise ValueError("inputs must be finite")
    gx, gy = np.meshgrid(x, y, indexing="ij")
    dr = np.stack((gx - q0[0], gy - q0[1]), axis=-1)
    decay = np.einsum("...i,ij,...j->...", dr, g0, dr)
    phase = np.einsum("i,...i->...", p0, dr)
    pref = np.linalg.det(g0) ** .25 / (np.pi * float(hbar)) ** .5
    wave = pref * np.exp((-0.5 * decay + 1j * phase) / float(hbar))
    norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(np.diff(x)[0] * np.diff(y)[0]))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("sampled field has zero or non-finite norm")
    return wave / norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-5,3,48);y=np.linspace(-2,2,40);q=np.array([-2.4,.65]);p=np.array([4.,0.]);g=np.diag([.8,4.])',
      'call': '(lambda '
              'a:float(np.dot(a.real.ravel(),np.linspace(.2,1.,1920))+2*np.dot(a.imag.ravel(),np.linspace(1.,.2,1920))))(initialize_grid_wavefunction(x,y,q,p,g))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real.ravel(),np.linspace(.2,1.,1920))+2*np.dot(a.imag.ravel(),np.linspace(1.,.2,1920))))(_oracle_initialize_grid_wavefunction(x,y,q,p,g))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-2,2,5);y=np.linspace(-3,3,7);q=np.zeros(2);p=np.zeros(2);g=np.eye(2)*.2',
      'call': '(lambda '
              'a:float(np.dot(a.real.ravel(),np.linspace(.2,1.,35))+2*np.dot(a.imag.ravel(),np.linspace(1.,.2,35))))(initialize_grid_wavefunction(x,y,q,p,g,.5))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real.ravel(),np.linspace(.2,1.,35))+2*np.dot(a.imag.ravel(),np.linspace(1.,.2,35))))(_oracle_initialize_grid_wavefunction(x,y,q,p,g,.5))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5);y=np.linspace(-1,1,5);q=np.zeros(2);p=np.zeros(2);g=np.eye(2)\n'
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': 'status(lambda: initialize_grid_wavefunction(x,y,q,p,g,0.0))',
      'gold_call': 'status(lambda: _oracle_initialize_grid_wavefunction(x,y,q,p,g,0.0))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); q=np.zeros(2); p=np.zeros(2); '
               'g=np.eye(2)\n'
               'x=np.zeros(5)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: initialize_grid_wavefunction(x,y,q,p,g))',
      'gold_call': 'status(lambda: _oracle_initialize_grid_wavefunction(x,y,q,p,g))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); q=np.zeros(2); p=np.zeros(2); '
               'g=np.eye(2)\n'
               'y=np.zeros(5)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: initialize_grid_wavefunction(x,y,q,p,g))',
      'gold_call': 'status(lambda: _oracle_initialize_grid_wavefunction(x,y,q,p,g))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); q=np.zeros(2); p=np.zeros(2); '
               'g=np.eye(2)\n'
               'q=np.array([1e6,1e6])\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: initialize_grid_wavefunction(x,y,q,p,g))',
      'gold_call': 'status(lambda: _oracle_initialize_grid_wavefunction(x,y,q,p,g))'}]
