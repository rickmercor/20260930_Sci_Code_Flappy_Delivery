"""
Anisotropy of the field gradient about the background field direction.

Anisotropy of the field gradient about the background field direction.

The full gradient of the field and the part of it taken across the background field
direction are not the same, and their ratio measures how much of the field
variation is invisible to the gyromotion, which is primarily perpendicular to the
background field. The comparison is made against the uniform background direction,
not against the local field direction at the point.

Returns
-------
The ratio of the full gradient magnitude to the cross-field gradient magnitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gradient_anisotropy_ratio(gradient: np.ndarray) -> float:
    """Return the ratio of full to cross-field gradient magnitude.

    Both magnitudes are Frobenius norms of the corresponding tensors, the
    cross-field part being the gradient with the derivative along the
    background direction (the z axis) projected out.

    Parameters
    ----------
    gradient : numpy.ndarray
        Spatial gradient tensor at the same point, shape ``(3, 3)``, with the
        derivative index first.

    Returns
    -------
    float
        The dimensionless ratio, never smaller than one.

    Raises
    ------
    ValueError
        If ``gradient`` does not have shape ``(3, 3)``, if the gradient
        vanishes, or if its cross-field part vanishes.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_gradient_anisotropy_ratio(gradient: np.ndarray) -> float:
    gradient = np.asarray(gradient, dtype=float)
    if gradient.shape != (3, 3):
        raise ValueError("gradient must have shape (3, 3)")
    unit = np.array([0.0, 0.0, 1.0])
    full = float(np.linalg.norm(gradient))
    if full <= 0.0:
        raise ValueError("gradient vanishes, ratio undefined")
    across = float(np.linalg.norm(gradient - np.outer(unit, unit @ gradient)))
    if across <= 0.0:
        raise ValueError("cross-field gradient vanishes, ratio undefined")
    return full / across

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with independent candidate/reference dependencies."""
    return [{'setup': 'def _case_run(_tested_function, field_gradient_tensor, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    p = np.linspace(0.3, 5.1, s.shape[1])\n'
               '    gt = field_gradient_tensor(p.copy(), s[0].copy(), s[1].copy())\n'
               '    return _tested_function(gt.copy())\n',
      'call': '_case_run(gradient_anisotropy_ratio, field_gradient_tensor, mode_spectrum)',
      'gold_call': '_case_run(_oracle_gradient_anisotropy_ratio, _oracle_field_gradient_tensor, '
                   '_oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, field_gradient_tensor, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s = mode_spectrum()\n'
               '    z = np.zeros(s.shape[1])\n'
               '    gt0 = field_gradient_tensor(z.copy(), s[0].copy(), s[1].copy())\n'
               '    return _tested_function(gt0.copy())\n',
      'call': '_case_run(gradient_anisotropy_ratio, field_gradient_tensor, mode_spectrum)',
      'gold_call': '_case_run(_oracle_gradient_anisotropy_ratio, _oracle_field_gradient_tensor, '
                   '_oracle_mode_spectrum)'},
     {'setup': 'def _case_run(_tested_function, field_gradient_tensor, mode_spectrum):\n'
               '    import numpy as np\n'
               '    s2 = mode_spectrum(num_modes=3, bw2=0.4, omega1=0.3, tan_alpha=0.2)\n'
               '    p2 = np.array([5.9, 1.1, 3.3])\n'
               '    gt2 = field_gradient_tensor(p2.copy(), s2[0].copy(), s2[1].copy(), tan_alpha=0.2)\n'
               '    return _tested_function(gt2.copy())\n',
      'call': '_case_run(gradient_anisotropy_ratio, field_gradient_tensor, mode_spectrum)',
      'gold_call': '_case_run(_oracle_gradient_anisotropy_ratio, _oracle_field_gradient_tensor, '
                   '_oracle_mode_spectrum)'},
     {'setup': 'import numpy as np\n'
               'def _probe_ratio(fn):\n'
               '    try:\n'
               '        fn(np.zeros((3, 3)))\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_probe_ratio(gradient_anisotropy_ratio)',
      'gold_call': '_probe_ratio(_oracle_gradient_anisotropy_ratio)'}]
