"""
Alternate potential half-steps and an exact spectral kinetic step.

The paper benchmarks the mesh-free approximation against a grid-based Schrodinger solution propagated by Strang splitting. The reference is normalized to unit cell-area-weighted L2 norm after propagation, including a zero-step call. Its input and output norms must be finite and nonzero.

Returns
-------
np.ndarray: final complex field of shape (len(x), len(y)), normalized to unit cell-area-weighted L2 norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def propagate_split_step_reference(
    psi: np.ndarray, x: np.ndarray, y: np.ndarray,
    dt: float, n_steps: int, mass: float,
    field_amplitude: float, radius: float, dielectric: complex,
    omega: float, charge: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Propagate a two-dimensional field with midpoint-potential Strang steps.

    The potential is sampled at each temporal midpoint. Apply, in order, a
    potential half-step, a Fourier kinetic full-step, and the same potential
    half-step. The finite one-dimensional axes must each contain at least
    two strictly monotone, uniformly spaced points with nonzero spacing.
    The input field shape must equal ``(len(x),len(y))`` and its cell norm
    must be finite and nonzero. After propagation, normalize the output to
    unit cell-area-weighted L2 norm, including when ``n_steps=0``. The input
    field is copied and remains unchanged.

    Returns
    -------
    np.ndarray
        Final complex field with the input shape and unit cell-area-weighted norm.

    Raises
    ------
    ValueError
        If the field and axes are misaligned, non-finite or invalid as above,
        ``n_steps`` is not a nonnegative integer, ``dt``, ``mass``, or ``hbar``
        is not positive finite, or the input/output cell norm is zero or
        non-finite. During propagation, invalid potential parameters also raise
        ValueError under the contract of ``evaluate_quasistatic_dipole``.
    """
    return propagated_wavefunction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Integral, Real
import numpy as np

def _oracle_propagate_split_step_reference(
    psi: np.ndarray, x: np.ndarray, y: np.ndarray,
    dt: float, n_steps: int, mass: float,
    field_amplitude: float, radius: float, dielectric: complex,
    omega: float, charge: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Reference Fourier Strang propagator."""
    wave = np.array(psi, complex, copy=True)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or wave.shape != (len(x), len(y)) or min(len(x), len(y)) < 2:
        raise ValueError("field and axes have incompatible shapes")
    for axis in (x, y):
        if not np.all(np.isfinite(axis)):
            raise ValueError("axes must be finite")
        spacing = np.diff(axis)
        if (not np.all(np.isfinite(spacing))
                or not (np.all(spacing > 0) or np.all(spacing < 0))
                or not np.allclose(spacing, spacing[0])):
            raise ValueError("axes must be strictly monotone and uniform with nonzero spacing")
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 0:
        raise ValueError("n_steps must be a nonnegative integer")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in (dt, mass, hbar)):
        raise ValueError("dt, mass, and hbar must be finite reals")
    if dt <= 0 or mass <= 0 or hbar <= 0 or not np.all(np.isfinite(np.r_[wave.real.ravel(), wave.imag.ravel()])):
        raise ValueError("positive scalars and a finite field are required")
    dx, dy = np.diff(x)[0], np.diff(y)[0]
    input_norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(dx * dy))
    if not np.isfinite(input_norm) or input_norm <= 0:
        raise ValueError("input field has zero or non-finite cell norm")
    kx = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(len(y), d=dy)
    kinetic = np.exp(-.5j * float(hbar) * float(dt) * (kx[:, None] ** 2 + ky[None, :] ** 2) / float(mass))
    gx, gy = np.meshgrid(x, y, indexing="ij")
    points = np.stack((gx, gy), axis=-1)
    for k in range(int(n_steps)):
        tmid = (k + .5) * float(dt)
        potential = _oracle_evaluate_quasistatic_dipole(tmid, points, field_amplitude, radius, dielectric, omega, charge)[0]
        half = np.exp(-.5j * float(dt) * potential / float(hbar))
        wave *= half
        wave = np.fft.ifft2(np.fft.fft2(wave) * kinetic)
        wave *= half
    norm = np.sqrt(np.sum(np.abs(wave) ** 2) * abs(dx * dy))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("propagated field has zero or non-finite cell norm")
    return wave / norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               "x=np.linspace(-3,3,16);y=np.linspace(-2,2,12);X,Y=np.meshgrid(x,y,indexing='ij');psi=np.exp(-.5*((X+1)**2+3*(Y-.5)**2)+2j*(X+1));psi/=np.sqrt(np.sum(np.abs(psi)**2)*abs((x[1]-x[0])*(y[1]-y[0])))",
      'call': '(lambda '
              'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,192))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,192))))(propagate_split_step_reference(psi,x,y,.02,8,1.,.12,.4,-24.061+1.5068j,1.3))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,192))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,192))))(_oracle_propagate_split_step_reference(psi,x,y,.02,8,1.,.12,.4,-24.061+1.5068j,1.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               "x=np.linspace(-3,3,16);y=np.linspace(-2,2,12);X,Y=np.meshgrid(x,y,indexing='ij');psi=np.exp(-.5*((X+1)**2+3*(Y-.5)**2)+2j*(X+1));psi/=np.sqrt(np.sum(np.abs(psi)**2)*abs((x[1]-x[0])*(y[1]-y[0])))",
      'call': '(lambda '
              'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,192))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,192))))(propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3))',
      'gold_call': '(lambda '
                   'a:float(np.dot(a.real.ravel(),np.linspace(.1,1.,192))+2*np.dot(a.imag.ravel(),np.linspace(1.,.1,192))))(_oracle_propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               "x=np.linspace(-3,3,16);y=np.linspace(-2,2,12);X,Y=np.meshgrid(x,y,indexing='ij');psi=np.exp(-.5*((X+1)**2+3*(Y-.5)**2)+2j*(X+1));psi/=np.sqrt(np.sum(np.abs(psi)**2)*abs((x[1]-x[0])*(y[1]-y[0])))\n"
               'def status(fn):\n'
               '    try: fn(); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': 'status(lambda: '
              'propagate_split_step_reference(psi,x,y,0.,1,1.,.12,.4,-24.061+1.5068j,1.3))',
      'gold_call': 'status(lambda: '
                   '_oracle_propagate_split_step_reference(psi,x,y,0.,1,1.,.12,.4,-24.061+1.5068j,1.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); psi=np.ones((5,5),complex)\n'
               'psi=np.zeros((5,5),complex)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: '
              'propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3))',
      'gold_call': 'status(lambda: '
                   '_oracle_propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); psi=np.ones((5,5),complex)\n'
               'x=np.zeros(5)\n'
               'def status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': 'status(lambda: '
              'propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3))',
      'gold_call': 'status(lambda: '
                   '_oracle_propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'x=np.linspace(-1,1,5); y=np.linspace(-1,1,5); psi=np.ones((5,5),complex)\n'
               'psi*=2.',
      'call': 'float(np.sum(propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3).real))',
      'gold_call': 'float(np.sum(_oracle_propagate_split_step_reference(psi,x,y,.02,0,1.,.12,.4,-24.061+1.5068j,1.3).real))'}]
