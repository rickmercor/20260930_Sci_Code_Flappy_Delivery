"""
Extract signed Maxwell strength, support, and lobe separation.

Return (...,7) rows (Delta,E0,Eleft,Eright,Bhalf,C,Bpair).



``responses`` has shape (D1,...,Dk,E), where k>=1, every leading

dimension is nonempty, and the final axis is sampled on at least five

strictly increasing photon energies. Independently PCHIP every fixed-state,

fixed-angle signed spectrum and evaluate both endpoints and every

interior derivative root. The spectrum must have a strictly positive

maximum and a strictly negative minimum in its shape-preserving sampled

range, using a roundoff floor of 64*machine_epsilon*max(1,max(abs(row))).

``Bpair`` is the absolute

energy separation between those signed extrema, with exact same-sign

ties going to the smaller energy.



Select the largest absolute signed extremum for ``Delta`` and ``E0``;

an exact positive/negative magnitude tie also goes to the smaller energy.

Multiply the spectrum by that peak's sign. ``Eleft`` and ``Eright`` bound

the closed connected component containing ``E0`` on which the adjusted

response is at least `$Delta/2$`. Set `$Bhalf=Eright-Eleft$` and

`$C=Delta*Bhalf$`. Preserve every leading response dimension. Reject

malformed, nonfinite, non-double-lobed input with ``ValueError``.

Absolute accuracy: 2e-8 for values and 2e-7 eV for locations.

Returns
-------
A (...,7) array (Delta,E0,Eleft,Eright,Bhalf,C,Bpair)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.interpolate import PchipInterpolator

def full_maxwell_metrics(
    energies: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    """Return (...,7) rows (Delta,E0,Eleft,Eright,Bhalf,C,Bpair).

    ``responses`` has shape (D1,...,Dk,E), where k>=1, every leading
    dimension is nonempty, and the final axis is sampled on at least five
    strictly increasing photon energies. Independently PCHIP every fixed-state,
    fixed-angle signed spectrum and evaluate both endpoints and every
    interior derivative root. The spectrum must have a strictly positive
    maximum and a strictly negative minimum in its shape-preserving sampled
    range, using a roundoff floor of 64*machine_epsilon*max(1,max(abs(row))).
    ``Bpair`` is the absolute
    energy separation between those signed extrema, with exact same-sign
    ties going to the smaller energy.

    Select the largest absolute signed extremum for ``Delta`` and ``E0``;
    an exact positive/negative magnitude tie also goes to the smaller energy.
    Multiply the spectrum by that peak's sign. ``Eleft`` and ``Eright`` bound
    the closed connected component containing ``E0`` on which the adjusted
    response is at least ``Delta/2``. Set ``Bhalf=Eright-Eleft`` and
    ``C=Delta*Bhalf``. Preserve every leading response dimension. Reject
    malformed, nonfinite, non-double-lobed input with ``ValueError``.
    Absolute accuracy: 2e-8 for values and 2e-7 eV for locations.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Extract signed Maxwell strength, support, and lobe separation."""

import numpy as np

from scipy.interpolate import PchipInterpolator

def _surface_single_metric(energies, row):
    sampled = np.asarray(row, float)
    scale = float(np.max(np.abs(sampled)))
    sign_floor = 64 * np.finfo(float).eps * max(1.0, scale)
    if np.max(sampled) <= sign_floor or np.min(sampled) >= -sign_floor:
        raise ValueError("each spectrum must contain positive and negative lobes")

    interpolant = PchipInterpolator(energies, row)
    stationary = interpolant.derivative().roots(extrapolate=False)
    points = np.unique(np.concatenate(([energies[0]], stationary, [energies[-1]])))
    points = points[(points >= energies[0]) & (points <= energies[-1])]
    values = np.asarray(interpolant(points), float)
    knot_indices = np.searchsorted(energies, points)
    at_knots = (knot_indices < len(energies))
    at_knots[at_knots] &= energies[knot_indices[at_knots]] == points[at_knots]
    values[at_knots] = sampled[knot_indices[at_knots]]

    positive = float(np.max(values))
    negative = float(np.min(values))
    if positive <= sign_floor or negative >= -sign_floor:
        raise ValueError("each spectrum must contain positive and negative lobes")
    positive_energy = float(np.min(points[values == positive]))
    negative_energy = float(np.min(points[values == negative]))
    pair_width = abs(positive_energy - negative_energy)
    if pair_width <= 0:
        raise ValueError("signed lobe extrema must occur at distinct energies")

    delta = max(positive, -negative)
    tied = points[np.abs(values) == delta]
    center = float(tied.min())
    peak = float(values[np.flatnonzero(points == center)[0]])
    sign = 1.0 if peak >= 0 else -1.0
    threshold = PchipInterpolator(energies, sign * np.asarray(row) - delta / 2)
    roots = threshold.roots(extrapolate=False)
    bounds = np.unique(np.concatenate(([energies[0]], roots, [energies[-1]])))
    bounds = bounds[(bounds >= energies[0]) & (bounds <= energies[-1])]
    good = np.asarray([
        threshold((left + right) / 2) >= 0
        for left, right in zip(bounds[:-1], bounds[1:])
    ])
    seeds = [
        index for index, (left, right) in enumerate(zip(bounds[:-1], bounds[1:]))
        if good[index] and left - 1e-14 <= center <= right + 1e-14
    ]
    if not seeds:
        raise ValueError("selected peak has no half-maximum support component")
    first = min(seeds)
    last = max(seeds)
    while first > 0 and good[first - 1]:
        first -= 1
    while last + 1 < len(good) and good[last + 1]:
        last += 1
    left = float(bounds[first])
    right = float(bounds[last + 1])
    bandwidth = right - left
    return np.array((
        delta, center, left, right, bandwidth, delta * bandwidth, pair_width
    ))

def _oracle_full_maxwell_metrics(
    energies: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    x = np.asarray(energies, float)
    values = np.asarray(responses, float)
    if (x.ndim != 1 or len(x) < 5 or np.any(np.diff(x) <= 0)
            or not np.all(np.isfinite(x))):
        raise ValueError("energies must be finite, increasing, and length at least five")
    if (values.ndim < 2 or any(size == 0 for size in values.shape[:-1])
            or values.shape[-1] != len(x)
            or not np.all(np.isfinite(values))):
        raise ValueError("the finite final response axis must match energies")
    flat = values.reshape((-1, len(x)))
    result = np.asarray([_surface_single_metric(x, row) for row in flat])
    return result.reshape(values.shape[:-1] + (7,))

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
               'e = np.linspace(0.06, 0.24, 19)\n'
               'centers = np.array([0.12, 0.16])\n'
               'flat = np.array([(1 + 0.08 * j) * (e - centers[j % 2]) * np.exp(-((e - centers[j % 2]) '
               '/ (0.025 + 0.002 * (j % 3))) ** 2) for j in range(6)])\n'
               'r = flat.reshape(2, 3, len(e))\n',
      'call': '_isolated_call(full_maxwell_metrics, e, r)',
      'gold_call': '_isolated_call(_oracle_full_maxwell_metrics, e, r)',
      'tol': 2e-07},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'e = np.array([0.04, 0.06, 0.085, 0.11, 0.14, 0.18, 0.23, 0.29, 0.36])\n'
               'base = (e - 0.16) * np.exp(-((e - 0.16) / 0.055) ** 2)\n'
               'r = np.stack([base, -1.31 * base, base + 0.18 * (e - 0.16) * np.exp(-((e - 0.23) / '
               '0.08) ** 2), -base + 0.07 * (e - 0.12) * np.exp(-((e - 0.09) / 0.04) ** 2)])\n',
      'call': '_isolated_call(full_maxwell_metrics, e, r)',
      'gold_call': '_isolated_call(_oracle_full_maxwell_metrics, e, r)',
      'tol': 2e-07},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'e = np.array([0.0, 0.15, 0.6, 1.4, 2.9, 5.0, 8.0, 12.0, 17.0])\n'
               'r = np.array([[0, 1, 4, 1, 0, -1, -3, -1, 0], [0, -2, -4, -1, 0, 1, 3, 1, 0], [0, 4, '
               '2, 4, 0, -1, -3, -1, 0]], float)\n'
               '\n'
               'def _contract_check(fn):\n'
               '    base = np.asarray(_isolated_call(fn, e, r), float).ravel()\n'
               '    tied_e = np.array([0.05, 0.099, 0.111, 0.139, 0.175, 0.206, 0.246, 0.296, 0.358])\n'
               '    tied_r = np.array([[0, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.4]], float)\n'
               '    tied = np.asarray(_isolated_call(fn, tied_e, tied_r), float).ravel()\n'
               '    trials = [(e[:4], r[:, :4]), (e[[0, 1, 1, 3, 4, 5, 6, 7, 8]], r), (e, np.abs(r)), '
               '(e, -np.abs(r))]\n'
               '    bad = r.copy()\n'
               '    bad[0, 3] = np.nan\n'
               '    trials.append((e, bad))\n'
               '    flags = []\n'
               '    for args in trials:\n'
               '        try:\n'
               '            _isolated_call(fn, *args)\n'
               '        except ValueError:\n'
               '            flags.append(1.0)\n'
               '        except Exception:\n'
               '            flags.append(-1.0)\n'
               '        else:\n'
               '            flags.append(0.0)\n'
               '    return np.r_[base, tied, flags]\n',
      'call': '_contract_check(full_maxwell_metrics)',
      'gold_call': 'np.r_[np.asarray(_isolated_call(_oracle_full_maxwell_metrics, e, r), '
                   'float).ravel(), np.array([0.4, 0.099, 0.061500484328, 0.139, 0.077499515672, '
                   '0.030999806269, 0.259]), np.ones(5)]',
      'tol': 2e-07}]
