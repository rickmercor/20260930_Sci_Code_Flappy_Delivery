"""
Solve the semi-infinite anisotropic Voigt boundary problem.

Return the signed TM surface contrast with shape (...,A,E).



The final two axes of ``tensors`` are (E,6), ordered Re xx, Im xx,

Re zz, Im zz, Re xz, Im xz, for the x-z gyrotropic block with

eps_zx=-eps_xz. For each positive angle in degrees set q=sin(theta),

c=cos(theta), and solve the transmitted TM mode



  kz^2 = eps_x + eps_xz^2/eps_z - eps_x*q^2/eps_z.



Select the square root with Im(kz)>0, or Re(kz)>0 when Im(kz)=0.

Its normalized tangential admittance is

Y=(eps_z*kz-q*eps_xz)/(eps_z-q^2), and the tangential-electric-field

reflection amplitude is r=(1-c*Y)/(1+c*Y). Evaluate this expression

separately at +q and -q and return |r(+q)|^2-|r(-q)|^2. Treat

abs(eps_z)<1e-14 or abs(eps_z-q^2)<1e-14 as singular. Reject malformed,

nonfinite, nonpositive-angle, grazing, or singular input with

``ValueError``. Absolute accuracy: 2e-10.

Returns
-------
A (...,A,E) reconstructed signed-response array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def voigt_surface_responses(
    tensors: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Return the signed TM surface contrast with shape (...,A,E).

    The final two axes of ``tensors`` are (E,6), ordered Re xx, Im xx,
    Re zz, Im zz, Re xz, Im xz, for the x-z gyrotropic block with
    eps_zx=-eps_xz. For each positive angle in degrees set q=sin(theta),
    c=cos(theta), and solve the transmitted TM mode

      kz^2 = eps_x + eps_xz^2/eps_z - eps_x*q^2/eps_z.

    Select the square root with Im(kz)>0, or Re(kz)>0 when Im(kz)=0.
    Its normalized tangential admittance is
    Y=(eps_z*kz-q*eps_xz)/(eps_z-q^2), and the tangential-electric-field
    reflection amplitude is r=(1-c*Y)/(1+c*Y). Evaluate this expression
    separately at +q and -q and return |r(+q)|^2-|r(-q)|^2. Treat
    abs(eps_z)<1e-14 or abs(eps_z-q^2)<1e-14 as singular. Reject malformed,
    nonfinite, nonpositive-angle, grazing, or singular input with
    ``ValueError``. Absolute accuracy: 2e-10.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Solve the semi-infinite anisotropic Voigt boundary problem."""

import numpy as np

def _outgoing_root(argument):
    root = np.sqrt(argument)
    reverse = (np.imag(root) < 0) | ((np.imag(root) == 0) & (np.real(root) < 0))
    return np.where(reverse, -root, root)

def _reflection(eps_x, eps_z, eps_xz, q, c):
    kz = _outgoing_root(eps_x + eps_xz**2 / eps_z - eps_x * q**2 / eps_z)
    denominator = eps_z - q**2
    admittance = (eps_z * kz - q * eps_xz) / denominator
    return (1 - c * admittance) / (1 + c * admittance)

def _oracle_voigt_surface_responses(
    tensors: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(tensors, float)
    theta = np.asarray(angles, float)
    if values.ndim < 2 or values.shape[-1] != 6 or theta.ndim != 1:
        raise ValueError("expected tensor shape (...,E,6) and angle shape (A,)")
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(theta))
            or np.any(theta <= 0) or np.any(theta >= 90)):
        raise ValueError("finite angles must lie strictly between 0 and 90 degrees")
    eps_x = values[..., 0] + 1j * values[..., 1]
    eps_z = values[..., 2] + 1j * values[..., 3]
    eps_xz = values[..., 4] + 1j * values[..., 5]
    q = np.sin(np.deg2rad(theta)).reshape((1,) * (eps_x.ndim - 1) + (len(theta), 1))
    c = np.cos(np.deg2rad(theta)).reshape((1,) * (eps_x.ndim - 1) + (len(theta), 1))
    ex = np.expand_dims(eps_x, -2)
    ez = np.expand_dims(eps_z, -2)
    exz = np.expand_dims(eps_xz, -2)
    if np.any(np.abs(ez) < 1e-14) or np.any(np.abs(ez - q**2) < 1e-14):
        raise ValueError("singular Voigt boundary problem")
    positive = _reflection(ex, ez, exz, q, c)
    negative = _reflection(ex, ez, exz, -q, c)
    result = np.abs(positive)**2 - np.abs(negative)**2
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite surface response")
    return np.asarray(result, float)

def _voigt_fixture(shape, count, phase):
    lead = int(np.prod(shape))
    u = np.linspace(-1, 1, lead * count).reshape(shape + (count,))
    ex = -1.4 + .8 * u + 1j * (1.1 + .15 * np.cos((phase + 2) * u))
    ez = -1.1 + .6 * u + 1j * (1.3 + .12 * np.sin((phase + 3) * u))
    exz = .04 * np.sin(3 * u) + 1j * (.28 + .05 * np.cos(5 * u))
    return np.stack((ex.real, ex.imag, ez.real, ez.imag, exz.real, exz.imag), axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((2,), 7, 1)\n'
               'a = np.array([5.0, 19.0, 43.0, 71.0, 85.0])\n',
      'call': '_isolated_call(voigt_surface_responses, t, a)',
      'gold_call': '_isolated_call(_oracle_voigt_surface_responses, t, a)',
      'tol': 2e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((2, 3), 5, 2)\n'
               'a = np.array([8.0, 27.5, 54.0, 83.0])\n',
      'call': '_isolated_call(voigt_surface_responses, t, a)',
      'gold_call': '_isolated_call(_oracle_voigt_surface_responses, t, a)',
      'tol': 2e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'a = np.array([11.0, 35.0, 68.0])\n',
      'call': '_isolated_call(voigt_surface_responses, t, a)',
      'gold_call': '_isolated_call(_oracle_voigt_surface_responses, t, a)',
      'tol': 2e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'reversed_tensor = t.copy()\n'
               'reversed_tensor[..., 4:6] *= -1\n'
               'both = np.concatenate((t, reversed_tensor), axis=0)\n'
               'a = np.array([11.0, 35.0, 68.0])\n',
      'call': '_isolated_call(voigt_surface_responses, both, a)',
      'gold_call': '_isolated_call(_oracle_voigt_surface_responses, both, a)',
      'tol': 2e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'a = np.array([11.0, 35.0, 68.0])\n'
               'bad = t.copy()\n'
               'bad[0, 2, 1] = np.nan\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(voigt_surface_responses, bad, a)',
      'gold_call': '_value_error_code(_oracle_voigt_surface_responses, bad, a)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'a = np.array([11.0, 35.0, 68.0])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(voigt_surface_responses, t, np.array([0.0, 30.0]))',
      'gold_call': '_value_error_code(_oracle_voigt_surface_responses, t, np.array([0.0, 30.0]))',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'a = np.array([11.0, 35.0, 68.0])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(voigt_surface_responses, t, np.array([30.0, 90.0]))',
      'gold_call': '_value_error_code(_oracle_voigt_surface_responses, t, np.array([30.0, 90.0]))',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'a = np.array([11.0, 35.0, 68.0])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(voigt_surface_responses, t[..., :5], a)',
      'gold_call': '_value_error_code(_oracle_voigt_surface_responses, t[..., :5], a)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               't = _voigt_fixture((1,), 6, 4)\n'
               'a = np.array([11.0, 35.0, 68.0])\n'
               'singular = t.copy()\n'
               'singular[..., 2] = np.sin(np.deg2rad(a[0])) ** 2\n'
               'singular[..., 3] = 0.0\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(voigt_surface_responses, singular, np.array([a[0]]))',
      'gold_call': '_value_error_code(_oracle_voigt_surface_responses, singular, np.array([a[0]]))',
      'tol': 0}]
