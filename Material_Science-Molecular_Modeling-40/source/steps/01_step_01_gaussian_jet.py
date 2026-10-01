"""
Cartesian Gaussian interaction derivatives through fifth order.

Gaussian screening regularizes overlapping sites. Quadrupole interactions differentiate the scalar kernel four times; their nuclear forces require a fifth derivative. Coincident-center limits must remain finite.

Returns
-------
scalar and Cartesian arrays of orders 1 through 5
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def gaussian_jet(d, ui, uj):
    """Return Cartesian derivatives of the screened Coulomb kernel through order five.

    Parameters
    ----------
    d : finite real array (3,)
        Pair displacement R_i-R_j in atomic units.
    ui, uj : finite positive real scalars
        Site hardnesses in atomic units.

    Returns
    -------
    (f, D1, D2, D3, D4, D5) : tuple
        Let u=2*ui*uj/(ui+uj), a=sqrt(pi)*u/2 and r=||d||_2.
        The scalar is f(d)=erf(a*r)/r, continued to f(0)=u.
        Dk[i1,...,ik] is the k-th partial derivative of f with respect
        to d[i1],...,d[ik]. Dk has shape (3,)*k, for k=1,...,5.
        All odd-order tensors vanish at d=0. The even-order tensors
        there follow from the analytic power series
        f(d)=u*sum_{j>=0} (-a*a*dot(d,d))**j/(j!*(2*j+1)).
        Use analytic derivatives, including the continuous coincident
        limit. Finite differences are not the defined calculation.
        Return Cartesian tensors with no factorial or symmetry scaling.
        Inputs are not mutated.

    Raises
    ------
    ValueError
        If d is not a finite real (3,) array, or either hardness is not
        a finite positive real scalar.
    """
    return (0.0, np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3, 3)), np.zeros((3, 3, 3, 3)), np.zeros((3, 3, 3, 3, 3)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math
import numpy as np
from scipy.special import gammainc, gammaln
import math
import numpy as np

def _positive_scalar(value, name):
    result = float(_real(value, name, ()))
    if result <= 0:
        raise ValueError(name + ' must be positive')
    return result

def _derivative_specification():
    """Group Cartesian tensor entries by their derivative multi-index."""
    all_orders = []
    for order in range(6):
        grouped = {}
        for flat, index in enumerate(itertools.product(range(3), repeat=order)):
            counts = tuple((index.count(axis) for axis in range(3)))
            grouped.setdefault(counts, []).append(flat)
        entries = []
        for counts, positions in grouped.items():
            terms = []
            for pairs in itertools.product(*(range(n // 2 + 1) for n in counts)):
                powers = tuple((n - 2 * p for n, p in zip(counts, pairs)))
                coefficient = 1
                for n, p, power in zip(counts, pairs, powers):
                    coefficient *= math.factorial(n) // (2 ** p * math.factorial(p) * math.factorial(power))
                terms.append((order - sum(pairs), powers, coefficient))
            entries.append((np.array(positions, dtype=int), tuple(terms)))
        all_orders.append(tuple(entries))
    return tuple(all_orders)
_DERIVATIVE_SPEC = _derivative_specification()

def _gaussian_moments(argument):
    """Return integral_0^1 t**(2*n) exp(-argument*t*t) dt, n=0,...,5."""
    indices = np.arange(6, dtype=float)
    if argument < 0.5:
        values = 1.0 / (2 * indices + 1)
        power_over_factorial = 1.0
        for j in range(1, 80):
            power_over_factorial *= -argument / j
            increment = power_over_factorial / (2 * indices + 2 * j + 1)
            values += increment
            if np.max(np.abs(increment)) <= 2e-17 * np.max(np.abs(values)):
                break
        return values
    exponents = indices + 0.5
    return 0.5 * np.exp(gammaln(exponents) - exponents * math.log(argument)) * gammainc(exponents, argument)

def _real(value, name, shape=None):
    try:
        if np.iscomplexobj(value):
            raise ValueError(name + ' must be real')
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be finite and real') from exc
    if not np.all(np.isfinite(out)) or (shape is not None and out.shape != shape):
        raise ValueError(name + ' has an invalid shape or nonfinite entry')
    return out

def _oracle_gaussian_jet(d, ui, uj):
    """Return Cartesian derivatives of the screened Coulomb kernel through order five.

    Parameters
    ----------
    d : finite real array (3,)
        Pair displacement R_i-R_j in atomic units.
    ui, uj : finite positive real scalars
        Site hardnesses in atomic units.

    Returns
    -------
    (f, D1, D2, D3, D4, D5) : tuple
        Let u=2*ui*uj/(ui+uj), a=sqrt(pi)*u/2 and r=||d||_2.
        The scalar is f(d)=erf(a*r)/r, continued to f(0)=u.
        Dk[i1,...,ik] is the k-th partial derivative of f with respect
        to d[i1],...,d[ik]. Dk has shape (3,)*k, for k=1,...,5.
        All odd-order tensors vanish at d=0. The even-order tensors
        there follow from the analytic power series
        f(d)=u*sum_{j>=0} (-a*a*dot(d,d))**j/(j!*(2*j+1)).
        Use analytic derivatives, including the continuous coincident
        limit. Finite differences are not the defined calculation.
        Return Cartesian tensors with no factorial or symmetry scaling.
        Inputs are not mutated.

    Raises
    ------
    ValueError
        If d is not a finite real (3,) array, or either hardness is not
        a finite positive real scalar.
    """
    displacement = _real(d, 'd', (3,))
    hardness_i = _positive_scalar(ui, 'ui')
    hardness_j = _positive_scalar(uj, 'uj')
    smaller, larger = sorted((hardness_i, hardness_j))
    hardness = smaller * (2.0 / (1.0 + smaller / larger))
    a2 = math.pi / 4.0 * hardness * hardness
    argument = a2 * float(displacement @ displacement)
    moments = _gaussian_moments(argument)
    radial = hardness * (-2.0 * a2) ** np.arange(6) * moments
    powers = displacement[:, None] ** np.arange(6)[None, :]
    tensors = []
    for order, specification in enumerate(_DERIVATIVE_SPEC):
        entries = np.zeros(3 ** order, dtype=float)
        for positions, terms in specification:
            value = 0.0
            for moment_order, exponents, coefficient in terms:
                value += coefficient * radial[moment_order] * powers[0, exponents[0]] * powers[1, exponents[1]] * powers[2, exponents[2]]
            entries[positions] = value
        tensors.append(float(entries[0]) if order == 0 else entries.reshape((3,) * order))
    return tuple(tensors)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'gaussian_jet([0.7, -0.4, 0.2], 1.7, 0.8)',
      'gold_call': '_oracle_gaussian_jet([0.7, -0.4, 0.2], 1.7, 0.8)'},
     {'setup': '',
      'call': 'gaussian_jet([0, 0, 0], 1.2, 1.2)',
      'gold_call': '_oracle_gaussian_jet([0, 0, 0], 1.2, 1.2)'},
     {'setup': '',
      'call': 'gaussian_jet([1e-9, -2e-9, 3e-9], 1.8, 0.6)',
      'gold_call': '_oracle_gaussian_jet([1e-9, -2e-9, 3e-9], 1.8, 0.6)'},
     {'setup': '',
      'call': 'gaussian_jet([9, -5, 3], 0.7, 2.3)',
      'gold_call': '_oracle_gaussian_jet([9, -5, 3], 0.7, 2.3)'},
     {'setup': '',
      'call': 'gaussian_jet([0.721, 0.011, -0.001], 1.5, 1.7)',
      'gold_call': '_oracle_gaussian_jet([0.721, 0.011, -0.001], 1.5, 1.7)'},
     {'setup': '\n'
               'def _expect_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_expect_value_error(lambda: gaussian_jet([0, 0], 1, 1))',
      'gold_call': '_expect_value_error(lambda: _oracle_gaussian_jet([0, 0], 1, 1))'},
     {'setup': '\n'
               'def _expect_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_expect_value_error(lambda: gaussian_jet([0, 0, 0], 0, 1))',
      'gold_call': '_expect_value_error(lambda: _oracle_gaussian_jet([0, 0, 0], 0, 1))'},
     {'setup': '',
      'call': 'gaussian_jet([0.001, -0.002, 0.003], 1.8, 0.6)',
      'gold_call': '_oracle_gaussian_jet([0.001, -0.002, 0.003], 1.8, 0.6)'},
     {'setup': '',
      'call': 'gaussian_jet([0.0001, -0.0002, 0.0003], 1.8, 0.6)',
      'gold_call': '_oracle_gaussian_jet([0.0001, -0.0002, 0.0003], 1.8, 0.6)'},
     {'setup': '',
      'call': 'gaussian_jet([.51,.42,-.33],1.3,1.9)',
      'gold_call': '_oracle_gaussian_jet([.51,.42,-.33],1.3,1.9)'},
     {'setup': '',
      'call': 'gaussian_jet([.18,-.71,.27],2.0,.9)',
      'gold_call': '_oracle_gaussian_jet([.18,-.71,.27],2.0,.9)'}]
