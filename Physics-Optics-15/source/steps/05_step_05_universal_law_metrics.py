"""
Evaluate the anchor paper's universal strength and bandwidth laws.

Return an (M,A,3) array (Delta_law,Bpair_law,C_law).



``coordinates`` contains Step 4 rows (beta,loss,g,x,W,r_o), and

``angles`` contains finite observation angles strictly between 0 and 90

degrees. Require x=g/loss and the stated W(x) relation to satisfy

abs(actual-expected) <= 1e-12 + 1e-10*abs(expected). For each material

row set



  kappa=sqrt(2/(loss*(1+x*x))),

  G=sin(theta)*cos(theta)/(cos(theta)+kappa)^2,

  d=loss*W/2,

  f=-2*g*loss*d/((d*d+loss*loss-g*g)^2+4*g*g*loss*loss).



Return Delta_law=8*abs(f)*G, the universal signed-lobe peak spacing

Bpair_law=loss*W/beta, and their product. Reject malformed, nonfinite,

inconsistent, or nonphysical input with ``ValueError``. Absolute

accuracy: 1e-10.

Returns
-------
An (M,A,3) array (Delta_law,Bpair_law,C_law)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def universal_law_metrics(
    coordinates: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Return an (M,A,3) array (Delta_law,Bpair_law,C_law).

    ``coordinates`` contains Step 4 rows (beta,loss,g,x,W,r_o), and
    ``angles`` contains finite observation angles strictly between 0 and 90
    degrees. Require x=g/loss and the stated W(x) relation to satisfy
    abs(actual-expected) <= 1e-12 + 1e-10*abs(expected). For each material
    row set

      kappa=sqrt(2/(loss*(1+x*x))),
      G=sin(theta)*cos(theta)/(cos(theta)+kappa)^2,
      d=loss*W/2,
      f=-2*g*loss*d/((d*d+loss*loss-g*g)^2+4*g*g*loss*loss).

    Return Delta_law=8*abs(f)*G, the universal signed-lobe peak spacing
    Bpair_law=loss*W/beta, and their product. Reject malformed, nonfinite,
    inconsistent, or nonphysical input with ``ValueError``. Absolute
    accuracy: 1e-10.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Evaluate the anchor paper's universal strength and bandwidth laws."""

import numpy as np

def _oracle_universal_law_metrics(
    coordinates: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    values = np.asarray(coordinates, float)
    theta = np.asarray(angles, float)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("coordinates must have finite nonempty shape (M,6)")
    if (not np.all(np.isfinite(values)) or np.any(values[:, :5] <= 0)
            or np.any(values[:, 5] < 0)):
        raise ValueError("coordinates must be finite and physical")
    if (theta.ndim != 1 or len(theta) == 0 or not np.all(np.isfinite(theta))
            or np.any(theta <= 0) or np.any(theta >= 90)):
        raise ValueError("angles must be finite and strictly between 0 and 90 degrees")
    beta, loss, g, x, width_factor, _ = values.T
    expected_x = g / loss
    expected_width = 2 / np.sqrt(3) * np.sqrt(
        x*x - 1 + 2*np.sqrt(x**4 + x*x + 1)
    )
    if (not np.allclose(x, expected_x, rtol=1e-10, atol=1e-12)
            or not np.allclose(width_factor, expected_width, rtol=1e-10, atol=1e-12)):
        raise ValueError("x and W must be consistent with beta, loss, and g")

    radians = np.deg2rad(theta)[None, :]
    sine = np.sin(radians)
    cosine = np.cos(radians)
    dimensionless_denominator = (
        ((width_factor / 2)**2 + 1 - x*x)**2 + 4*x*x
    )
    inverse_kappa = np.sqrt(loss * (1 + x*x) / 2)[:, None]
    scaled_angular = (
        sine * cosine * (1 + x[:, None]*x[:, None]) / 2
        / (1 + cosine * inverse_kappa)**2
    )
    dimensionless_kernel = np.abs(x * width_factor) / dimensionless_denominator
    delta = 8 * dimensionless_kernel[:, None] * scaled_angular
    bandwidth = np.broadcast_to((loss * width_factor / beta)[:, None], delta.shape)
    output = np.stack((delta, bandwidth, delta * bandwidth), axis=2)
    if not np.all(np.isfinite(output)):
        raise ValueError("universal-law metrics must be finite")
    return output

def _law_fixture():
    beta = np.array([120., 480., 900.])
    loss = np.array([.7, 3.2, 11.])
    g = np.array([.14, 2.4, 14.3])
    x = g / loss
    width = 2 / np.sqrt(3) * np.sqrt(x*x - 1 + 2*np.sqrt(x**4 + x*x + 1))
    coordinates = np.column_stack((beta, loss, g, x, width, [.03, .08, .12]))
    return coordinates

def _value_error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0

def _law_contract_args(coordinates, angles, kind):
    if kind == "shape":
        return np.ones((2, 5)), angles
    if kind == "empty_coordinates":
        return np.empty((0, 6)), angles
    if kind == "empty_angles":
        return coordinates, np.array([])
    if kind == "zero_angle":
        return coordinates, np.array([0.0, 30.0])
    if kind == "grazing_angle":
        return coordinates, np.array([20.0, 90.0])
    if kind == "nan":
        bad = coordinates.copy(); bad[0, 2] = np.nan
        return bad, angles
    if kind.startswith("column_"):
        column, value = {
            "column_0": (0, 0.0), "column_1": (1, 0.0),
            "column_2": (2, 0.0), "column_3": (3, 0.0),
            "column_4": (4, 0.0), "column_5": (5, -0.1),
        }[kind]
        bad = coordinates.copy(); bad[0, column] = value
        return bad, angles
    if kind == "x_identity":
        bad = coordinates.copy(); bad[0, 3] *= 1.01
        return bad, angles
    if kind == "width_identity":
        bad = coordinates.copy(); bad[0, 4] *= 0.99
        return bad, angles
    raise ValueError("unknown law contract fixture")

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
               'c = _law_fixture()\n'
               'a = np.array([5.0, 27.0, 53.0, 79.0, 85.0])\n',
      'call': '_isolated_call(universal_law_metrics, c, a)',
      'gold_call': '_isolated_call(_oracle_universal_law_metrics, c, a)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'c[:, 0] *= [0.7, 1.2, 1.8]\n'
               'c[:, 1:3] *= np.array([1.1, 0.8, 1.3])[:, None]\n'
               'c[:, 3] = c[:, 2] / c[:, 1]\n'
               'x = c[:, 3]\n'
               'c[:, 4] = 2 / np.sqrt(3) * np.sqrt(x * x - 1 + 2 * np.sqrt(x ** 4 + x * x + 1))\n'
               'tiny = np.array([[1e-100, 1e-100, 1e-100, 1.0, 2 / np.sqrt(3) * np.sqrt(2 * '
               'np.sqrt(3)), 0.02]])\n'
               'c = np.vstack((c, tiny))\n'
               'a = np.array([11.0, 42.0, 68.0, 88.0])\n',
      'call': '_isolated_call(universal_law_metrics, c, a)',
      'gold_call': '_isolated_call(_oracle_universal_law_metrics, c, a)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n',
      'call': '_isolated_call(universal_law_metrics, c, a)',
      'gold_call': '_isolated_call(_oracle_universal_law_metrics, c, a)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'shape')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'empty_coordinates')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'empty_angles')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'zero_angle')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'grazing_angle')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'nan')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'column_0')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'column_1')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'column_2')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'column_3')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'column_4')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'column_5')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'x_identity')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'c = _law_fixture()\n'
               'a = np.array([7.0, 35.0, 66.0, 84.0])\n'
               "args = _law_contract_args(c, a, 'width_identity')\n",
      'call': '_isolated_call(_value_error_code, universal_law_metrics, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_universal_law_metrics, *args)',
      'tol': 0}]
