"""
Select the strongest law-screened fixed-geometry common passband.

Return the unique tie-broken best length-17 operating-point row.



``guarantees`` has finite shape (K,16), K>0, and the Step 6 column

contract, with positive Dmin, Bcommon, Delta_law, and Bpair_law and

nonnegative agreement errors. Maximize Dmin*Bcommon. Exact product ties

select smaller nominal chemical potential, then smaller nominal angle.

Return (nominal_mu,nominal_angle,worst_D_mu,worst_D_angle,left_mu,

left_angle,right_mu,right_angle,Dmin,common_left,common_right,Bcommon,

product,strength_error,spacing_error,Delta_law,Bpair_law). Reject

malformed, empty, nonfinite, or nonphysical input with ``ValueError``.

Absolute accuracy: 1e-10.

Returns
-------
A length-17 optimal law-screened common-passband row
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_operating_point(guarantees: np.ndarray) -> np.ndarray:
    """Return the unique tie-broken best length-17 operating-point row.

    ``guarantees`` has finite shape (K,16), K>0, and the Step 6 column
    contract, with positive Dmin, Bcommon, Delta_law, and Bpair_law and
    nonnegative agreement errors. Maximize Dmin*Bcommon. Exact product ties
    select smaller nominal chemical potential, then smaller nominal angle.
    Return (nominal_mu,nominal_angle,worst_D_mu,worst_D_angle,left_mu,
    left_angle,right_mu,right_angle,Dmin,common_left,common_right,Bcommon,
    product,strength_error,spacing_error,Delta_law,Bpair_law). Reject
    malformed, empty, nonfinite, or nonphysical input with ``ValueError``.
    Absolute accuracy: 1e-10.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Select the strongest law-screened fixed-geometry common passband."""

import numpy as np

def _oracle_select_operating_point(guarantees: np.ndarray) -> np.ndarray:
    values = np.asarray(guarantees, float)
    if values.ndim != 2 or values.shape[1] != 16 or len(values) == 0:
        raise ValueError("guarantee shape mismatch")
    if (not np.all(np.isfinite(values)) or np.any(values[:, 2:4] <= 0)
            or np.any(values[:, 12:14] < 0) or np.any(values[:, 14:16] <= 0)):
        raise ValueError("guarantees must be finite and physical")
    products = values[:, 2] * values[:, 3]
    best = float(np.max(products))
    indices = np.flatnonzero(products == best)
    index = min(indices, key=lambda k: (values[k, 0], values[k, 1]))
    row = values[index]
    return np.asarray((
        row[0], row[1], row[4], row[5], row[8], row[9], row[10], row[11],
        row[2], row[6], row[7], row[3], products[index], row[12], row[13],
        row[14], row[15],
    ))

def _value_error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0

def _selection_contract_table(guarantees, kind):
    if kind == "shape":
        return np.ones((2, 15))
    if kind == "empty":
        return np.empty((0, 16))
    bad = guarantees.copy()
    column, value = {
        "nan": (2, np.nan),
        "zero_dmin": (2, 0.0),
        "negative_error": (12, -0.1),
        "zero_law": (14, 0.0),
    }[kind]
    bad[0, column] = value
    return bad

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
               'g = np.array([[-0.1, 50, 0.8, 0.5, -0.2, 40, 0.1, 0.6, 0, 60, 0.1, 40, 0.2, 0.1, 0.9, '
               '0.07], [0, 30, 0.4, 0.3, -0.1, 20, 0.15, 0.45, 0.1, 40, -0.1, 20, 0.1, 0.2, 0.5, '
               '0.08]], float)\n',
      'call': '_isolated_call(select_operating_point, g)',
      'gold_call': '_isolated_call(_oracle_select_operating_point, g)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[-0.1, 50, 0.9, 0.8, -0.2, 40, 0.1, 0.9, -0.1, 60, 0, 70, 0.3, 0.2, 1.1, '
               '0.08], [-0.1, 40, 0.6, 0.6, -0.3, 45, 0.2, 0.8, -0.2, 65, 0.1, 75, 0.2, 0.1, 0.8, '
               '0.07], [0.0, 30, 0.72, 0.75, -0.1, 20, 0.15, 0.9, -0.1, 25, 0.1, 40, 0.1, 0.15, 0.9, '
               '0.06], [-0.2, 30, 0.72, 0.75, -0.2, 20, 0.14, 0.89, -0.2, 25, 0.1, 40, 0.12, 0.14, '
               '0.91, 0.061]], float)\n',
      'call': '_isolated_call(select_operating_point, g)',
      'gold_call': '_isolated_call(_oracle_select_operating_point, g)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n',
      'call': '_isolated_call(select_operating_point, g)',
      'gold_call': '_isolated_call(_oracle_select_operating_point, g)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n'
               "bad = _selection_contract_table(g, 'shape')\n",
      'call': '_isolated_call(_value_error_code, select_operating_point, bad)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_select_operating_point, bad)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n'
               "bad = _selection_contract_table(g, 'empty')\n",
      'call': '_isolated_call(_value_error_code, select_operating_point, bad)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_select_operating_point, bad)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n'
               "bad = _selection_contract_table(g, 'nan')\n",
      'call': '_isolated_call(_value_error_code, select_operating_point, bad)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_select_operating_point, bad)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n'
               "bad = _selection_contract_table(g, 'zero_dmin')\n",
      'call': '_isolated_call(_value_error_code, select_operating_point, bad)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_select_operating_point, bad)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n'
               "bad = _selection_contract_table(g, 'negative_error')\n",
      'call': '_isolated_call(_value_error_code, select_operating_point, bad)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_select_operating_point, bad)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'g = np.array([[0.5, 15, 0.72, 0.5, 0, 0, 0.1, 0.6, 0, 5, 1, 25, 0.1, 0.2, 0.8, 0.06], '
               '[-0.5, 45, 0.6, 0.6, -1, 15, 0.2, 0.8, -1, 20, 0, 70, 0.2, 0.1, 0.7, 0.08]], float)\n'
               "bad = _selection_contract_table(g, 'zero_law')\n",
      'call': '_isolated_call(_value_error_code, select_operating_point, bad)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_select_operating_point, bad)',
      'tol': 0}]
