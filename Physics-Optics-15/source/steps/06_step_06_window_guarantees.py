"""
Apply universal-law validity and guarantee one Maxwell passband.

Return finite common-band guarantees with nominal law agreement.



`$mu$` and ``angles`` are nonempty one-dimensional increasing uniform

grids with at least two nodes. ``metrics`` has shape

(N,A,7) with Step 3 columns (Delta,E0,Eleft,Eright,Bhalf,C,Bpair),

`$law_metrics$` has shape (N,A,3) with Step 5 columns

(Delta_law,Bpair_law,C_law), and ``validity`` is the length-N Step 4

real-to-imaginary gyrotropy ratio. ``drifts`` is a finite length-2 vector.

Each nonnegative drift is an exact

integer multiple of its grid spacing. `$validity_limit$` must be finite and

strictly positive. `$agreement_limits$` must be a finite length-2 vector,

and each of its two entries must lie strictly between zero and one; the

boundary values zero and one are outside this public contract.



A nominal node passes the universal-law fidelity screen only when its

relative strength error abs(Delta_law-Delta)/Delta is no greater than

`$agreement_limits[0]$` and its relative signed-lobe-spacing error

abs(Bpair_law-Bpair)/Bpair is no greater than `$agreement_limits[1]$`.

The screen is evaluated at the nominal material state and fixed angle;

gate and alignment robustness are then certified by the full Maxwell

rectangle. Every gate node must also satisfy `$validity_limit$`.



Within each full closed uncertainty rectangle, set Dmin to the minimum

Maxwell Delta, common_left to the largest Eleft, and common_right to the

smallest Eright. Omit nonpositive intersections. Ties for Dmin and the

two limiting endpoints select smaller chemical potential, then angle.

Return rows

(nominal_mu,nominal_angle,Dmin,Bcommon,worst_D_mu,worst_D_angle,

common_left,common_right,left_mu,left_angle,right_mu,right_angle,

strength_error,spacing_error,Delta_law,Bpair_law), ordered by nominal

chemical potential and then angle. Reject malformed, nonfinite,

inconsistent, or nonphysical input with ``ValueError``. Absolute

accuracy: 1e-10.

Returns
-------
A (K,16) table of law-screened target common-passband guarantees
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def window_guarantees(
    mu: np.ndarray,
    angles: np.ndarray,
    metrics: np.ndarray,
    law_metrics: np.ndarray,
    validity: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
    spectrum_valid: np.ndarray = None,
) -> np.ndarray:
    """Return finite common-band guarantees with nominal law agreement.

    ``mu`` and ``angles`` are nonempty one-dimensional increasing uniform
    grids with at least two nodes. ``metrics`` has shape
    (N,A,7) with Step 3 columns (Delta,E0,Eleft,Eright,Bhalf,C,Bpair),
    ``law_metrics`` has shape (N,A,3) with Step 5 columns
    (Delta_law,Bpair_law,C_law), and ``validity`` is the length-N Step 4
    real-to-imaginary gyrotropy ratio. Optional ``spectrum_valid`` is a Boolean
    (N,A) mask. A false cell may contain nonfinite metric placeholders and
    invalidates only uncertainty rectangles containing that cell; every true
    cell must satisfy the complete finite metric contract. If omitted, every
    spectrum is valid. ``drifts`` is a finite length-2 vector.
    Each nonnegative drift is an exact
    integer multiple of its grid spacing. ``validity_limit`` must be finite and
    strictly positive. ``agreement_limits`` must be a finite length-2 vector,
    and each of its two entries must lie strictly between zero and one; the
    boundary values zero and one are outside this public contract.

    A nominal node passes the universal-law fidelity screen only when its
    relative strength error abs(Delta_law-Delta)/Delta is no greater than
    ``agreement_limits[0]`` and its relative signed-lobe-spacing error
    abs(Bpair_law-Bpair)/Bpair is no greater than ``agreement_limits[1]``.
    The screen is evaluated at the nominal material state and fixed angle;
    gate and alignment robustness are then certified by the full Maxwell
    rectangle. Every gate node must also satisfy ``validity_limit``.

    Within each full closed uncertainty rectangle, set Dmin to the minimum
    Maxwell Delta, common_left to the largest Eleft, and common_right to the
    smallest Eright. Omit nonpositive intersections. Ties for Dmin and the
    two limiting endpoints select smaller chemical potential, then angle.
    Return rows
    (nominal_mu,nominal_angle,Dmin,Bcommon,worst_D_mu,worst_D_angle,
    common_left,common_right,left_mu,left_angle,right_mu,right_angle,
    strength_error,spacing_error,Delta_law,Bpair_law), ordered by nominal
    chemical potential and then angle. Reject malformed, nonfinite,
    inconsistent, or nonphysical input with ``ValueError``. Absolute
    accuracy: 1e-10.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Apply universal-law validity and guarantee one Maxwell passband."""

import numpy as np

def _window_radius(grid, drift):
    spacing = np.diff(grid)
    if len(spacing) == 0 or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-12):
        raise ValueError("design grids must be uniform")
    radius = int(round(float(drift) / spacing[0]))
    if radius < 0 or abs(radius * spacing[0] - drift) > 1e-10:
        raise ValueError("each drift must be a nonnegative integer grid multiple")
    return radius

def _first_window_location(mask):
    return tuple(np.argwhere(mask)[0])

def _oracle_window_guarantees(
    mu: np.ndarray,
    angles: np.ndarray,
    metrics: np.ndarray,
    law_metrics: np.ndarray,
    validity: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
    spectrum_valid: np.ndarray = None,
) -> np.ndarray:
    x = np.asarray(mu, float)
    theta = np.asarray(angles, float)
    values = np.asarray(metrics, float)
    laws = np.asarray(law_metrics, float)
    screens = np.asarray(validity, float)
    drift_values = np.asarray(drifts, float)
    agreement = np.asarray(agreement_limits, float)
    if (x.ndim != 1 or theta.ndim != 1 or len(x) < 2 or len(theta) < 2
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(theta))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(theta) <= 0)):
        raise ValueError("design grids must be finite increasing one-dimensional arrays")
    if (drift_values.shape != (2,) or not np.all(np.isfinite(drift_values))
            or np.any(drift_values < 0)):
        raise ValueError("drifts must be a finite nonnegative length-2 vector")
    if values.shape != (len(x), len(theta), 7):
        raise ValueError("Maxwell metric shape mismatch")
    if laws.shape != (len(x), len(theta), 3) or screens.shape != (len(x),):
        raise ValueError("law metric or validity shape mismatch")
    if spectrum_valid is None:
        valid = np.ones((len(x), len(theta)), dtype=bool)
    else:
        valid_input = np.asarray(spectrum_valid)
        if valid_input.shape != (len(x), len(theta)) or valid_input.dtype != np.bool_:
            raise ValueError("spectrum_valid must be a Boolean grid-shaped mask")
        valid = valid_input
    arrays = (laws, screens, agreement)
    if any(not np.all(np.isfinite(item)) for item in arrays) or not np.isfinite(validity_limit):
        raise ValueError("all inputs and limits must be finite")
    if (agreement.shape != (2,) or np.any(agreement <= 0) or np.any(agreement >= 1)
            or validity_limit <= 0 or np.any(screens < 0)):
        raise ValueError("validity and agreement limits must be physical")
    valid_values = values[valid]
    if (not np.all(np.isfinite(valid_values))
            or np.any(valid_values[:, [0, 4, 5, 6]] <= 0) or np.any(laws <= 0)):
        raise ValueError("Maxwell and universal-law metrics must be positive")
    if (np.any(valid_values[:, 2] > valid_values[:, 1])
            or np.any(valid_values[:, 1] > valid_values[:, 3])
            or not np.allclose(valid_values[:, 3] - valid_values[:, 2], valid_values[:, 4], rtol=0, atol=1e-10)
            or not np.allclose(valid_values[:, 0] * valid_values[:, 4], valid_values[:, 5], rtol=0, atol=1e-10)
            or not np.allclose(laws[:, :, 0] * laws[:, :, 1], laws[:, :, 2], rtol=0, atol=1e-10)):
        raise ValueError("metric identities or half-maximum supports are invalid")

    rx = _window_radius(x, drift_values[0])
    rt = _window_radius(theta, drift_values[1])
    rows = []
    for i in range(rx, len(x) - rx):
        if np.any(screens[i-rx:i+rx+1] > validity_limit):
            continue
        for j in range(rt, len(theta) - rt):
            valid_block = valid[i-rx:i+rx+1, j-rt:j+rt+1]
            if not np.all(valid_block):
                continue
            strength_error = abs(laws[i, j, 0] - values[i, j, 0]) / values[i, j, 0]
            spacing_error = abs(laws[i, j, 1] - values[i, j, 6]) / values[i, j, 6]
            if strength_error > agreement[0] or spacing_error > agreement[1]:
                continue
            block = values[i-rx:i+rx+1, j-rt:j+rt+1]
            magnitudes = block[:, :, 0]
            dmin = float(np.min(magnitudes))
            common_left = float(np.max(block[:, :, 2]))
            common_right = float(np.min(block[:, :, 3]))
            if common_right <= common_left:
                continue
            di, dj = _first_window_location(magnitudes == dmin)
            li, lj = _first_window_location(block[:, :, 2] == common_left)
            ri, rj = _first_window_location(block[:, :, 3] == common_right)
            rows.append((
                x[i], theta[j], dmin, common_right - common_left,
                x[i-rx+di], theta[j-rt+dj], common_left, common_right,
                x[i-rx+li], theta[j-rt+lj], x[i-rx+ri], theta[j-rt+rj],
                strength_error, spacing_error, laws[i, j, 0], laws[i, j, 1],
            ))
    if not rows:
        raise ValueError("no admissible nominal node")
    return np.asarray(rows)

def _window_fixture(phase, n=9, a=7):
    x = np.linspace(-.4, .4, n)
    theta = np.linspace(5., 65., a)
    ii, jj = np.indices((n, a))
    peak = .42 + .017*ii + .011*jj + .003*phase
    center = .14 + .0017*ii - .0011*jj
    left = center - (.026 + .0008*((ii+2*jj+phase) % 4))
    right = center + (.024 + .0007*((2*ii+jj+phase) % 5))
    width = right - left
    pair = .06 + .001*ii + .0005*jj
    metrics = np.stack((peak, center, left, right, width, peak*width, pair), axis=2)
    law_peak = peak * (1 + .08*np.sin(.8*ii+.3*jj+phase))
    law_pair = pair * (1 + .06*np.cos(.4*ii+.7*jj+phase))
    laws = np.stack((law_peak, law_pair, law_peak*law_pair), axis=2)
    validity = .07 + .004*(np.arange(n)-3)**2
    return x, theta, metrics, laws, validity

def _value_error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    except Exception:
        return -1.0
    return 0.0

def _window_contract_args(x, theta, metrics, laws, validity, drifts,
                          validity_limit, agreement, kind):
    args = [x, theta, metrics, laws, validity, drifts, validity_limit, agreement]
    if kind == "nonfinite_metric":
        bad = metrics.copy(); bad[3, 2, 0] = np.nan; args[2] = bad
    elif kind == "zero_pair":
        bad = metrics.copy(); bad[3, 2, 6] = 0.0; args[2] = bad
    elif kind == "width_identity":
        bad = metrics.copy(); bad[3, 2, 4] += 0.01; args[2] = bad
    elif kind == "law_product":
        bad = laws.copy(); bad[3, 2, 2] += 0.01; args[3] = bad
    elif kind == "nonfinite_validity":
        bad = validity.copy(); bad[2] = np.inf; args[4] = bad
    elif kind == "nonuniform_mu":
        bad = x.copy(); bad[4] += 0.01; args[0] = bad
    elif kind == "off_grid_drift":
        args[5] = (0.15, 10.0)
    elif kind == "zero_agreement":
        args[7] = (0.0, 0.1)
    elif kind == "unit_agreement":
        args[7] = (0.1, 1.0)
    elif kind == "strength_screen":
        bad = laws.copy(); bad[4, 3, 0] *= 1.5
        bad[4, 3, 2] = bad[4, 3, 0] * bad[4, 3, 1]; args[3] = bad
    elif kind == "spacing_screen":
        bad = laws.copy(); bad[4, 3, 1] *= 1.5
        bad[4, 3, 2] = bad[4, 3, 0] * bad[4, 3, 1]; args[3] = bad
    elif kind == "mu_shape":
        args[0] = x.reshape(1, -1)
    elif kind == "nonfinite_drift":
        args[5] = (0.1, np.nan)
    elif kind == "drift_shape":
        args[5] = (0.1,)
    else:
        raise ValueError("unknown window contract fixture")
    return tuple(args)

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
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n',
      'call': '_isolated_call(window_guarantees, x, t, m, l, v, drifts, limit, agreement)',
      'gold_call': '_isolated_call(_oracle_window_guarantees, x, t, m, l, v, drifts, limit, agreement)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(2, 11, 9)\n'
               'drifts = (0.08, 7.5)\n'
               'limit = 0.13\n'
               'agreement = (0.1, 0.08)\n',
      'call': '_isolated_call(window_guarantees, x, t, m, l, v, drifts, limit, agreement)',
      'gold_call': '_isolated_call(_oracle_window_guarantees, x, t, m, l, v, drifts, limit, agreement)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n',
      'call': '_isolated_call(window_guarantees, x, t, m, l, v, drifts, limit, agreement)',
      'gold_call': '_isolated_call(_oracle_window_guarantees, x, t, m, l, v, drifts, limit, agreement)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'nonfinite_metric')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               "args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, 'zero_pair')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'width_identity')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               "args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, 'law_product')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'nonfinite_validity')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'nonuniform_mu')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'off_grid_drift')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'zero_agreement')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'unit_agreement')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'strength_screen')\n",
      'call': '_isolated_call(window_guarantees, *args)',
      'gold_call': '_isolated_call(_oracle_window_guarantees, *args)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'spacing_screen')\n",
      'call': '_isolated_call(window_guarantees, *args)',
      'gold_call': '_isolated_call(_oracle_window_guarantees, *args)',
      'tol': 1e-10},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               "args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, 'mu_shape')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               'args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, '
               "'nonfinite_drift')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'x, t, m, l, v = _window_fixture(1)\n'
               'drifts = (0.1, 10.0)\n'
               'limit = 0.2\n'
               'agreement = (0.12, 0.1)\n'
               "args = _window_contract_args(x, t, m, l, v, drifts, limit, agreement, 'drift_shape')\n",
      'call': '_isolated_call(_value_error_code, window_guarantees, *args)',
      'gold_call': '_isolated_call(_value_error_code, _oracle_window_guarantees, *args)',
      'tol': 0}]
