"""
Recover the anchor paper's local ENZ design coordinates.

Return rows (beta,loss,g,x,W,r_o).



``spectra`` must be nonempty. Each input is a finite (K,7) table with K>=5 containing energy, Re xx,

Im xx, Re zz, Im zz, Re xz and Im xz; K may differ between inputs and

each energy column is strictly increasing. Independently PCHIP every

tensor component in energy. Re xx must have exactly one crossing,

including a possible endpoint zero. At that crossing set

beta=abs(d Re xx/dE), loss=Im xx, g=abs(Im xz), x=g/loss,

W=2/sqrt(3)*sqrt(x*x-1+2*sqrt(x**4+x*x+1)) and

r_o=abs(Re xz)/g.

Reject a malformed or nonfinite table, a nonincreasing energy column,

Re xx with other than one crossing, or nonpositive beta, loss, or

gyrotropy with ``ValueError``. Absolute accuracy: 1e-8.

Returns
-------
An (M,6) array (beta,loss,g,x,W,r_o)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.interpolate import PchipInterpolator

def enz_design_coordinates(spectra: list) -> np.ndarray:
    """Return rows (beta,loss,g,x,W,r_o).

    ``spectra`` must be nonempty. Each input is a finite (K,7) table with K>=5 containing energy, Re xx,
    Im xx, Re zz, Im zz, Re xz and Im xz; K may differ between inputs and
    each energy column is strictly increasing. Independently PCHIP every
    tensor component in energy. Re xx must have exactly one crossing,
    including a possible endpoint zero. At that crossing set
    beta=abs(d Re xx/dE), loss=Im xx, g=abs(Im xz), x=g/loss,
    W=2/sqrt(3)*sqrt(x*x-1+2*sqrt(x**4+x*x+1)) and
    r_o=abs(Re xz)/g.
    Reject a malformed or nonfinite table, a nonincreasing energy column,
    Re xx with other than one crossing, or nonpositive beta, loss, or
    gyrotropy with ``ValueError``. Absolute accuracy: 1e-8.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Recover the anchor paper's local ENZ design coordinates."""

import numpy as np

from scipy.interpolate import PchipInterpolator

def _single_root(interpolant, x, y):
    roots = np.asarray(interpolant.roots(extrapolate=False), float)
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(x[0]), abs(x[-1]))
    roots = roots[np.isfinite(roots)]
    roots = roots[(roots >= x[0] - tolerance) & (roots <= x[-1] + tolerance)]
    roots = np.where(np.abs(roots - x[0]) <= tolerance, x[0], roots)
    roots = np.where(np.abs(roots - x[-1]) <= tolerance, x[-1], roots)
    if y[0] == 0:
        roots = np.append(roots, x[0])
    if y[-1] == 0:
        roots = np.append(roots, x[-1])
    roots = np.sort(roots)
    unique = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > tolerance:
            unique.append(float(root))
    if len(unique) != 1:
        raise ValueError("Re xx must have exactly one ENZ crossing")
    return unique[0]

def _oracle_enz_design_coordinates(spectra: list) -> np.ndarray:
    try:
        tables = list(spectra)
    except TypeError as error:
        raise ValueError("spectra must be a nonempty iterable") from error
    if not tables:
        raise ValueError("spectra must be a nonempty iterable")
    output = []
    for source in tables:
        table = np.asarray(source, float)
        if table.ndim != 2 or table.shape[0] < 5 or table.shape[1] != 7:
            raise ValueError("each spectrum must be a (K,7) table with K>=5")
        energy = table[:, 0]
        if np.any(np.diff(energy) <= 0) or not np.all(np.isfinite(table)):
            raise ValueError("spectra must be finite with increasing energies")
        columns = [PchipInterpolator(energy, table[:, j], extrapolate=False) for j in range(1, 7)]
        e0x = _single_root(columns[0], energy, table[:, 1])
        beta = abs(float(columns[0].derivative()(e0x)))
        loss = float(columns[1](e0x))
        real_off = float(columns[4](e0x))
        g = abs(float(columns[5](e0x)))
        primitives = np.asarray((beta, loss, real_off, g))
        if (not np.all(np.isfinite(primitives)) or beta <= 0
                or loss <= 0 or g <= 0):
            raise ValueError("beta, loss and gyrotropy must be positive")
        x = g / loss
        width_factor = 2 / np.sqrt(3) * np.sqrt(
            x*x - 1 + 2*np.sqrt(x**4 + x*x + 1)
        )
        derived = np.asarray((beta, loss, g, x, width_factor, abs(real_off) / g))
        if not np.all(np.isfinite(derived)):
            raise ValueError("derived ENZ coordinates must be finite")
        output.append(tuple(derived))
    return np.asarray(output)

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
               'ea = np.array([0.08, 0.11, 0.17, 0.26, 0.4])\n'
               'a = np.column_stack([ea, [2.1, 0.8, 0, -0.9, -2.2], 1.1 + 0.2 * ea, [-1.5, -0.7, 0.2, '
               '0.9, 1.8], 1.2 + 0.1 * ea, -0.03 + 0.01 * ea, 0.42 + 0.05 * ea])\n'
               'eb = np.array([0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.48])\n'
               'b = np.column_stack([eb, [0, -0.3, -0.8, -1.5, -2.4, -3.5, -4.9], 0.7 + 0.3 * eb, '
               '[1.8, 1.1, 0.45, 0, -0.6, -1.4, -2.5], 0.9 + 0.2 * eb, 0.015 * np.ones(7), -0.31 * '
               'np.ones(7)])\n'
               'ec = np.array([0.05, 0.08, 0.14, 0.23, 0.37, 0.56])\n'
               'c = np.column_stack([ec, 0.56 - ec, 0.8 + 0.2 * ec, [0, -0.2, -0.6, -1, -1.5, -2], 1 + '
               '0.1 * ec, -0.02 + 0.005 * ec, 0.28 + 0.03 * ec])\n'
               'spectra = [a, b, c]\n',
      'call': '_isolated_call(enz_design_coordinates, spectra)',
      'gold_call': '_isolated_call(_oracle_enz_design_coordinates, spectra)',
      'tol': 1e-08},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'e = np.array([0.08, 0.13, 0.21, 0.34, 0.55])\n'
               'a = np.column_stack([e, 5 * (e - 0.19) + 2 * (e - 0.19) ** 3, 2 - e, 5 * (e - 0.192), '
               '2.1 - e, -0.02 * np.ones(5), 0.7 + 0.1 * e])\n'
               'spectra = [a]\n',
      'call': '_isolated_call(enz_design_coordinates, spectra)',
      'gold_call': '_isolated_call(_oracle_enz_design_coordinates, spectra)',
      'tol': 1e-08},
     {'setup': 'import copy as _input_copy\n'
               '\n'
               'def _isolated_call(fn, *args, **kwargs):\n'
               '    copied_args, copied_kwargs = _input_copy.deepcopy((args, kwargs))\n'
               '    return fn(*copied_args, **copied_kwargs)\n'
               '\n'
               'import numpy as np\n'
               'e = np.array([0.07, 0.105, 0.16, 0.245, 0.36, 0.52])\n'
               'a = np.column_stack([e, [-2.4, -1, -0.25, 0.8, 1.4, 2.6], [0.25, 0.7, 0.32, 0.95, '
               '0.45, 1.2], [-1.8, -0.8, 0.35, 1.1, 2.1, 3], [0.4, 0.9, 0.5, 1, 0.6, 1.1], [-0.01, '
               '-0.08, -0.025, -0.12, -0.04, -0.15], [0.18, 0.55, 0.22, 0.75, 0.3, 0.9]])\n'
               'ee = e + 4e-15\n'
               'endpoint = np.column_stack([ee, [0.0, -0.2, -0.5, -0.9, -1.4, -2.0], [0.6, 0.7, 0.8, '
               '0.9, 1.0, 1.1], [-1.0, -0.7, -0.3, 0.2, 0.8, 1.4], [0.8, 0.85, 0.9, 0.95, 1.0, 1.05], '
               '-0.03 * np.ones(6), 0.4 * np.ones(6)])\n'
               'spectra = [a, endpoint]\n'
               '\n'
               'def _contract_check(fn):\n'
               '    base = np.asarray(_isolated_call(fn, spectra), float).ravel()\n'
               '    multiple = a.copy()\n'
               '    multiple[:, 1] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]\n'
               '    missing = a.copy()\n'
               '    missing[:, 1] = 1.0 + np.arange(len(e))\n'
               '    short = a[:4].copy()\n'
               '    nonfinite = a.copy()\n'
               '    nonfinite[2, 5] = np.nan\n'
               '    nonincreasing = a.copy()\n'
               '    nonincreasing[3, 0] = nonincreasing[2, 0]\n'
               '    zero_loss = a.copy()\n'
               '    zero_loss[:, 2] = 0.0\n'
               '    zero_g = a.copy()\n'
               '    zero_g[:, 6] = 0.0\n'
               '    trials = ([multiple], [missing], [short], [nonfinite], [], [nonincreasing], '
               '[zero_loss], [zero_g])\n'
               '    flags = []\n'
               '    for value in trials:\n'
               '        try:\n'
               '            _isolated_call(fn, value)\n'
               '        except ValueError:\n'
               '            flags.append(1.0)\n'
               '        except Exception:\n'
               '            flags.append(-1.0)\n'
               '        else:\n'
               '            flags.append(0.0)\n'
               '    return np.r_[base, flags]\n',
      'call': '_contract_check(enz_design_coordinates)',
      'gold_call': 'np.r_[np.asarray(_isolated_call(_oracle_enz_design_coordinates, spectra), '
                   'float).ravel(), np.ones(8)]',
      'tol': 1e-08}]
