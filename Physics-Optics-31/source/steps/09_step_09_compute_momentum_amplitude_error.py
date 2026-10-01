"""
Reduce two endpoint fields to one unit-normalized Fourier-amplitude L2 error.

The paper motivates an absolute-value L2 comparison; endpoint normalization isolates momentum-shape error in this reduced benchmark.

Returns
-------
float: finite nonnegative unit-normalized momentum-shape error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np

def compute_momentum_amplitude_error(
    meshfree_momentum: np.ndarray,
    reference_wave: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """Return the L2 distance between unit-normalized momentum amplitudes.

    The first field is already in shifted momentum order; the second is a
    real-space grid field transformed with the aligned continuous FFT convention.

    Returns
    -------
    float
        Nonnegative unit-normalized momentum-amplitude shape discrepancy.

    Raises
    ------
    ValueError
        If the fields are misaligned or non-finite, either spacing is not
        positive finite, or either momentum field has zero or non-finite norm.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np

def _oracle_compute_momentum_amplitude_error(
    meshfree_momentum: np.ndarray,
    reference_wave: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """Reference unit-normalized Fourier-amplitude shape comparison."""
    mesh = np.asarray(meshfree_momentum, complex)
    ref = np.asarray(reference_wave, complex)
    if mesh.ndim != 2 or mesh.shape != ref.shape or min(mesh.shape) < 2:
        raise ValueError("fields must be aligned two-dimensional arrays")
    if not np.all(np.isfinite(np.r_[mesh.real.ravel(), mesh.imag.ravel(), ref.real.ravel(), ref.imag.ravel()])):
        raise ValueError("fields must be finite")
    for spacing in (dx, dy):
        if isinstance(spacing, bool) or not isinstance(spacing, Real) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacings must be positive finite reals")
    ref_k = float(dx) * float(dy) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ref)))
    mesh_norm = np.linalg.norm(mesh)
    ref_norm = np.linalg.norm(ref_k)
    if not np.isfinite(mesh_norm) or mesh_norm <= 0:
        raise ValueError("mesh-free momentum field has zero or non-finite norm")
    if not np.isfinite(ref_norm) or ref_norm <= 0:
        raise ValueError("reference field has zero or non-finite Fourier norm")
    mesh_k = mesh / mesh_norm
    ref_k = ref_k / ref_norm
    denominator = np.linalg.norm(np.abs(ref_k))
    return float(np.linalg.norm(np.abs(mesh_k) - np.abs(ref_k)) / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               "x=np.linspace(-1,1,8);y=np.linspace(-2,2,6);X,Y=np.meshgrid(x,y,indexing='ij');b=np.exp(-.8*X**2-1.1*Y**2);a=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(b)))*(1+.1j*X)",
      'call': 'float(compute_momentum_amplitude_error(a,b,x[1]-x[0],y[1]-y[0]))',
      'gold_call': 'float(_oracle_compute_momentum_amplitude_error(a,b,x[1]-x[0],y[1]-y[0]))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'rng=np.random.default_rng(2);b=rng.normal(size=(7,5))+1j*rng.normal(size=(7,5));a=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(b)))',
      'call': 'float(compute_momentum_amplitude_error(a,b,.2,.3))',
      'gold_call': 'float(_oracle_compute_momentum_amplitude_error(a,b,.2,.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'a=np.ones((4,4),complex);b=np.ones((4,4),complex)\n'
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': 'status(lambda: compute_momentum_amplitude_error(a,b,0.0,1.0))',
      'gold_call': 'status(lambda: _oracle_compute_momentum_amplitude_error(a,b,0.0,1.0))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'a=np.ones((4,4),complex); b=np.ones((4,4),complex)\n'
               'a=np.zeros((4,4),complex)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: compute_momentum_amplitude_error(a,b,.2,.3))',
      'gold_call': 'status(lambda: _oracle_compute_momentum_amplitude_error(a,b,.2,.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Real\n'
               'a=np.ones((4,4),complex); b=np.ones((4,4),complex)\n'
               'b=np.zeros((4,4),complex)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: compute_momentum_amplitude_error(a,b,.2,.3))',
      'gold_call': 'status(lambda: _oracle_compute_momentum_amplitude_error(a,b,.2,.3))'}]
