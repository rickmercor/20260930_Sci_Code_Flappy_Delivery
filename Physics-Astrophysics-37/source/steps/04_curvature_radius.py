"""
Local field-line curvature radius of the multimode field.

Local field-line curvature radius of the multimode field.

The curvature radius of a field line is set by how fast the field direction turns
along itself. It is the length scale that has to be compared against the ion
gyroradius when deciding whether the magnetic moment survives.

This step returns one specific convention, which is its output contract: the
radius is Rc = |B|^2 / |(B . grad) B|, where (B . grad) B is the FULL directional
derivative of the field vector along the field, the vector whose component i is
the sum over j of field[j] times gradient[j, i]. Do not project (B . grad) B
perpendicular to B before taking its magnitude. The textbook curvature radius,
built from that perpendicular part only, is a different quantity and is not what
this step returns.

Returns
-------
The local field-line curvature radius |B|^2 / |(B . grad) B| in units of the background gyroradius scale.

Returns
-------
The local field-line curvature radius |B|^2 / |(B . grad) B| in units of the background gyroradius scale.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def curvature_radius(field: np.ndarray, gradient: np.ndarray) -> float:
    """Return the local field-line curvature radius.

    The radius is ``|B|**2 / |(B . grad) B|`` with the FULL directional
    derivative ``(B . grad) B``, whose component ``i`` is
    ``sum_j field[j] * gradient[j, i]``. It is not projected perpendicular to
    ``B`` before its magnitude is taken.

    Parameters
    ----------
    field : numpy.ndarray
        Total magnetic field at the point, shape ``(3,)``.
    gradient : numpy.ndarray
        Spatial gradient tensor at the same point, shape ``(3, 3)``, with the
        derivative index first.

    Returns
    -------
    float
        The curvature radius ``|B|**2 / |(B . grad) B|``.

    Raises
    ------
    ValueError
        If ``field`` does not have shape ``(3,)``, if ``gradient`` does not
        have shape ``(3, 3)``, if the field magnitude is zero, or if
        ``(B . grad) B`` vanishes so that the field line is locally straight.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_curvature_radius(field: np.ndarray, gradient: np.ndarray) -> float:
    field = np.asarray(field, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if field.shape != (3,):
        raise ValueError("field must have shape (3,)")
    if gradient.shape != (3, 3):
        raise ValueError("gradient must have shape (3, 3)")
    magnitude_squared = float(field @ field)
    if magnitude_squared <= 0.0:
        raise ValueError("field magnitude must be positive")
    directional = field @ gradient
    bend = float(np.linalg.norm(directional))
    if bend <= 0.0:
        raise ValueError("field line is locally straight, curvature radius undefined")
    return magnitude_squared / bend

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, field_gradient_tensor, mode_spectrum, total_field):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    p = np.linspace(0.3, 5.1, s.shape[1])\n'
               '    bf = total_field(p.copy(), s[1].copy())\n'
               '    gt = field_gradient_tensor(p.copy(), s[0].copy(), s[1].copy())\n'
               '    return _tested_function(bf.copy(), gt.copy())\n',
      'call': '_case_run(curvature_radius, field_gradient_tensor, mode_spectrum, total_field)',
      'gold_call': '_case_run(_oracle_curvature_radius, _oracle_field_gradient_tensor, '
                   '_oracle_mode_spectrum, _oracle_total_field)'},
     {'setup': 'def _case_run(_tested_function, field_gradient_tensor, mode_spectrum, total_field):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    z = np.zeros(s.shape[1])\n'
               '    bf0 = total_field(z.copy(), s[1].copy())\n'
               '    gt0 = field_gradient_tensor(z.copy(), s[0].copy(), s[1].copy())\n'
               '    return _tested_function(bf0.copy(), gt0.copy())\n',
      'call': '_case_run(curvature_radius, field_gradient_tensor, mode_spectrum, total_field)',
      'gold_call': '_case_run(_oracle_curvature_radius, _oracle_field_gradient_tensor, '
                   '_oracle_mode_spectrum, _oracle_total_field)'},
     {'setup': 'def _case_run(_tested_function, field_gradient_tensor, mode_spectrum, total_field):\n'
               '    import numpy as np\n'
               '    s5 = mode_spectrum(num_modes=2, bw2=0.45, omega1=0.31, q=0.4, tan_alpha=0.6)\n'
               '    p5 = np.array([0.77, 4.02])\n'
               '    bf5 = total_field(p5.copy(), s5[1].copy(), tan_alpha=0.6)\n'
               '    gt5 = field_gradient_tensor(p5.copy(), s5[0].copy(), s5[1].copy(), tan_alpha=0.6)\n'
               '    return _tested_function(bf5.copy(), gt5.copy())\n',
      'call': '_case_run(curvature_radius, field_gradient_tensor, mode_spectrum, total_field)',
      'gold_call': '_case_run(_oracle_curvature_radius, _oracle_field_gradient_tensor, '
                   '_oracle_mode_spectrum, _oracle_total_field)'},
     {'setup': 'import numpy as np\n'
               'def _probe_rc(fn):\n'
               '    try:\n'
               '        fn(np.array([0.0, 0.0, 1.0]), np.zeros((3, 3)))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_rc(curvature_radius)',
      'gold_call': '_probe_rc(_oracle_curvature_radius)'}]
