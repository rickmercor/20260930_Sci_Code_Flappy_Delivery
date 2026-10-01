"""
Interpolate the first-principles dielectric-tensor field.

Return a (Q,E,6) tensor field in query order.



`$mu_nodes$` contains at least three nodes. ``tensors`` has shape

(J,E,6), ordered Re xx, Im xx, Re zz,

Im zz, Re xz, Im xz. Interpolate every scalar independently by PCHIP

along the strictly increasing chemical-potential axis. ``queries`` must

lie in the closed node range and may be unsorted. Reject malformed,

nonfinite, nonincreasing, or out-of-domain input with ``ValueError``.

Absolute accuracy: 1e-8.

Returns
-------
A (Q,E,6) interpolated tensor field
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.interpolate import PchipInterpolator

def material_fields(
    mu_nodes: np.ndarray,
    tensors: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Return a (Q,E,6) tensor field in query order.

    ``mu_nodes`` contains at least three nodes. ``tensors`` has shape
    (J,E,6), ordered Re xx, Im xx, Re zz,
    Im zz, Re xz, Im xz. Interpolate every scalar independently by PCHIP
    along the strictly increasing chemical-potential axis. ``queries`` must
    lie in the closed node range and may be unsorted. Reject malformed,
    nonfinite, nonincreasing, or out-of-domain input with ``ValueError``.
    Absolute accuracy: 1e-8.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Interpolate the first-principles dielectric-tensor field."""

import numpy as np

from scipy.interpolate import PchipInterpolator

def _oracle_material_fields(
    mu_nodes: np.ndarray,
    tensors: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    nodes = np.asarray(mu_nodes, float)
    values = np.asarray(tensors, float)
    points = np.asarray(queries, float)
    if (nodes.ndim != 1 or points.ndim != 1 or values.ndim != 3
            or values.shape[0] != len(nodes) or values.shape[2] != 6):
        raise ValueError("expected nodes (J,), tensors (J,E,6), queries (Q,)")
    if (len(nodes) < 3 or not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(points))
            or np.any(np.diff(nodes) <= 0)):
        raise ValueError("tensor interpolation inputs must be finite and ordered")
    if np.any(points < nodes[0]) or np.any(points > nodes[-1]):
        raise ValueError("queries lie outside the interpolation domain")
    return PchipInterpolator(nodes, values, axis=0, extrapolate=False)(points)

def _material_fixture(nodes, energies):
    m = nodes[:, None]
    w = energies[None, :]
    return np.stack([
        260 * (w - .18 - .07 * m) + 3 * np.sin(11 * m),
        1.7 + 2.2 * w + .3 * m * m,
        240 * (w - .176 - .065 * m) + 2 * np.cos(9 * m),
        1.9 + 1.8 * w + .2 * m * m,
        .15 * np.cos(7 * m) + .02 * w,
        1.1 + .35 * np.sin(8 * m) + .4 * w,
    ], axis=-1)

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
               'mu = np.array([-0.4, -0.3, -0.23, -0.15])\n'
               'e = np.array([0.08, 0.101, 0.14, 0.17, 0.21])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.35, -0.25, -0.19])\n',
      'call': '_isolated_call(material_fields, mu, t, q)',
      'gold_call': '_isolated_call(_oracle_material_fields, mu, t, q)',
      'tol': 1e-08},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.41, -0.29, -0.26, -0.11])\n'
               'e = np.array([0.07, 0.09, 0.13, 0.19])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.37, -0.24, -0.17, -0.12])\n',
      'call': '_isolated_call(material_fields, mu, t, q)',
      'gold_call': '_isolated_call(_oracle_material_fields, mu, t, q)',
      'tol': 1e-08},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.5, -0.37, -0.25, -0.1])\n'
               'e = np.array([0.08, 0.11, 0.17])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.1, -0.47, -0.5, -0.2, -0.37])\n',
      'call': '_isolated_call(material_fields, mu, t, q)',
      'gold_call': '_isolated_call(_oracle_material_fields, mu, t, q)',
      'tol': 1e-08},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.5, -0.37, -0.25, -0.1])\n'
               'e = np.array([0.08, 0.11, 0.17])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.1, -0.47, -0.5, -0.2, -0.37])\n'
               'bad = t.copy()\n'
               'bad[1, 1, 2] = np.nan\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(material_fields, mu, bad, q)',
      'gold_call': '_value_error_code(_oracle_material_fields, mu, bad, q)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.5, -0.37, -0.25, -0.1])\n'
               'e = np.array([0.08, 0.11, 0.17])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.1, -0.47, -0.5, -0.2, -0.37])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(material_fields, mu[::-1], t[::-1], q)',
      'gold_call': '_value_error_code(_oracle_material_fields, mu[::-1], t[::-1], q)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.5, -0.37, -0.25, -0.1])\n'
               'e = np.array([0.08, 0.11, 0.17])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.1, -0.47, -0.5, -0.2, -0.37])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(material_fields, mu, t, np.array([-0.51]))',
      'gold_call': '_value_error_code(_oracle_material_fields, mu, t, np.array([-0.51]))',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.5, -0.37, -0.25, -0.1])\n'
               'e = np.array([0.08, 0.11, 0.17])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.1, -0.47, -0.5, -0.2, -0.37])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(material_fields, mu, t[..., :5], q)',
      'gold_call': '_value_error_code(_oracle_material_fields, mu, t[..., :5], q)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'mu = np.array([-0.5, -0.37, -0.25, -0.1])\n'
               'e = np.array([0.08, 0.11, 0.17])\n'
               't = _material_fixture(mu, e)\n'
               'q = np.array([-0.1, -0.47, -0.5, -0.2, -0.37])\n'
               '\n'
               'def _value_error_code(fn, *args):\n'
               '    try:\n'
               '        _isolated_call(fn, *args)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return -1.0\n'
               '    return 0.0\n',
      'call': '_value_error_code(material_fields, mu[:2], t[:2], q)',
      'gold_call': '_value_error_code(_oracle_material_fields, mu[:2], t[:2], q)',
      'tol': 0}]
