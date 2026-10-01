"""
Find all distinct interior gap closures of an affine skew pencil.

Author-derived continuation of the paper’s Pfaffian homotopy invariant. A gap closure is a real c where S0+c*S1 is singular. Generalized determinant roots occur with even multiplicity for skew pencils. Count each distinct real location once, including tangencies; subsequent interval parity is evaluated independently. The input family is regular (determinant not identically zero).

Returns
-------
return result  # real ndarray (n/2+1,), first entry number K of distinct strictly interior roots, next K entries sorted roots, remaining entries zero. Root matching tolerance is 1e-6; exact endpoint roots are excluded.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def pencil_crossings(pencil: ArrayLike, lower: float, upper: float) -> np.ndarray:
    'Find all distinct interior gap closures of an affine skew pencil.\n\nParameters\n----------\npencil : real (2,n,n), ordered S0,S1, skew-symmetric of positive even size.\nlower, upper : finite floats with lower<upper. Supported instances have real crossings separated by >1e-5, endpoints separated from other roots by >1e-5, and complex roots separated from the real axis by >1e-5. Exact repeated roots and exact endpoint roots are allowed. Numerical equality to the real axis, to another root, or to an endpoint uses absolute tolerance 1e-10; this is distinct from the 1e-6 output root-matching tolerance.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Both pencil matrices satisfy skew symmetry within relative max-entry tolerance 1e-12. The pencil is regular and obeys the root-separation domain stated above.\n\nReturns\n-------\nreal ndarray (n/2+1,), first entry number K of distinct strictly interior roots, next K entries sorted roots, remaining entries zero. Root matching tolerance is 1e-6; exact endpoint roots are excluded.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_pencil_crossings(pencil: ArrayLike, lower: float, upper: float) -> np.ndarray:
    def _checked_interval(lower, upper):
        lo = _checked_scalar(lower, 'lower')
        hi = _checked_scalar(upper, 'upper')
        if lo >= hi:
            raise ValueError('bounds must be strictly increasing')
        return lo, hi

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_pencil(pencil):
        p = _checked_numeric(pencil, 'pencil')
        if p.ndim != 3 or p.shape[0] != 2:
            raise ValueError('pencil must have shape (2,n,n)')
        for a in p:
            _checked_skew(a)
        return p

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    def _exact_fallback(a, b, lower, upper):
        from decimal import Decimal, localcontext
        from fractions import Fraction
        from math import cos, pi, sin

        def _trim(poly):
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            return poly

        def _divide(numerator, denominator):
            remainder = numerator.copy()
            quotient = [Fraction(0)] * max(1, len(numerator) - len(denominator) + 1)
            while any(remainder) and len(remainder) >= len(denominator):
                shift = len(remainder) - len(denominator)
                coefficient = remainder[-1] / denominator[-1]
                quotient[shift] = coefficient
                for index, value in enumerate(denominator):
                    remainder[index + shift] -= coefficient * value
                _trim(remainder)
            return _trim(quotient), _trim(remainder)

        def _pfaffian(matrix):
            matrix = [row.copy() for row in matrix]
            result = Fraction(1)
            for k in range(0, len(matrix), 2):
                partner = next((j for j in range(k + 1, len(matrix)) if matrix[k][j]), None)
                if partner is None:
                    return Fraction(0)
                if partner != k + 1:
                    matrix[k + 1], matrix[partner] = matrix[partner], matrix[k + 1]
                    for row in matrix:
                        row[k + 1], row[partner] = row[partner], row[k + 1]
                    result = -result
                pivot = matrix[k][k + 1]
                result *= pivot
                for i in range(k + 2, len(matrix)):
                    for j in range(i + 1, len(matrix)):
                        matrix[i][j] -= (
                            matrix[k][i] * matrix[k + 1][j]
                            - matrix[k][j] * matrix[k + 1][i]
                        ) / pivot
                        matrix[j][i] = -matrix[i][j]
            return result

        def _add(z, w):
            return z[0] + w[0], z[1] + w[1]

        def _subtract(z, w):
            return z[0] - w[0], z[1] - w[1]

        def _multiply(z, w):
            return z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]

        def _quotient(z, w):
            norm = w[0] * w[0] + w[1] * w[1]
            return (z[0] * w[0] + z[1] * w[1]) / norm, (z[1] * w[0] - z[0] * w[1]) / norm

        def _norm(z):
            return abs(z[0]) + abs(z[1])

        def _decimal(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        def _simple_roots(poly, precision):
            coefficients = [_decimal(value / poly[-1]) for value in poly]
            degree = len(coefficients) - 1
            zero, one = Decimal(0), Decimal(1)
            if degree == 1:
                return [(-coefficients[0], zero)]
            if degree == 2:
                # Determine the discriminant sign exactly, including tiny pairs.
                discriminant = poly[1] ** 2 - 4 * poly[0] * poly[2]
                middle = -_decimal(poly[1] / (2 * poly[2]))
                offset = _decimal(abs(discriminant)).sqrt() / (2 * abs(_decimal(poly[2])))
                if discriminant < 0:
                    return [(middle, offset), (middle, -offset)]
                return [(middle - offset, zero), (middle + offset, zero)]

            magnitude = max(abs(value) for value in poly)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                approximate = np.roots([float(value / magnitude) for value in poly[::-1]])
            if len(approximate) == degree and np.isfinite(approximate).all():
                roots = [(Decimal(float(value.real)), Decimal(float(value.imag))) for value in approximate]
            else:
                radius = one + max(abs(value) for value in coefficients[:-1])
                roots = [
                    (radius * Decimal(cos(2 * pi * (i + .25) / degree)),
                     radius * Decimal(sin(2 * pi * (i + .25) / degree)))
                    for i in range(degree)
                ]
            for i in range(degree):
                if roots[i] in roots[:i]:
                    roots[i] = _add(roots[i], (Decimal('1e-8') * (i + 1), Decimal('1e-8')))

            tolerance = Decimal(10) ** (-(precision // 2))
            # Aberth's simultaneous correction keeps nearby simple roots distinct.
            for _ in range(400 + 20 * degree):
                largest = zero
                for i, root in enumerate(roots):
                    value, derivative = (one, zero), (zero, zero)
                    for coefficient in coefficients[-2::-1]:
                        derivative = _add(_multiply(derivative, root), value)
                        value = _add(_multiply(value, root), (coefficient, zero))
                    if _norm(value) == 0:
                        continue
                    if _norm(derivative) == 0:
                        return None
                    newton = _quotient(value, derivative)
                    repulsion = (zero, zero)
                    for j, other in enumerate(roots):
                        if i != j:
                            difference = _subtract(root, other)
                            if _norm(difference) == 0:
                                return None
                            repulsion = _add(repulsion, _quotient((one, zero), difference))
                    denominator = _subtract((one, zero), _multiply(newton, repulsion))
                    if _norm(denominator) == 0:
                        return None
                    correction = _quotient(newton, denominator)
                    roots[i] = _subtract(root, correction)
                    largest = max(largest, _norm(correction))
                if largest <= tolerance:
                    # Weierstrass corrections also detect missing/duplicated roots.
                    residual = zero
                    for i, root in enumerate(roots):
                        value = (one, zero)
                        denominator = (one, zero)
                        for coefficient in coefficients[-2::-1]:
                            value = _add(_multiply(value, root), (coefficient, zero))
                        for j, other in enumerate(roots):
                            if i != j:
                                denominator = _multiply(denominator, _subtract(root, other))
                        if _norm(denominator) == 0:
                            return None
                        residual = max(residual, _norm(_quotient(value, denominator)))
                    if residual <= tolerance * degree:
                        return roots
            return None

        def _classify(roots, center, radius):
            equality, separation = Decimal('1e-10'), Decimal('1e-5')
            lo, hi = Decimal(lower), Decimal(upper)
            interior = []
            for real, imaginary in roots:
                real, imaginary = center + radius * real, radius * imaginary
                if abs(imaginary) > equality:
                    if abs(imaginary) <= separation:
                        return 'complex roots must be separated from the real axis by more than 1e-5', []
                    continue
                if any(equality < abs(real - endpoint) <= separation for endpoint in (lo, hi)):
                    return 'non-endpoint roots must be separated from endpoints by more than 1e-5', []
                if lo + equality < real < hi - equality:
                    interior.append(real)
            interior.sort()
            groups = []
            for root in interior:
                if groups and root - groups[-1][0] <= equality:
                    groups[-1].append(root)
                else:
                    groups.append([root])
            unique = [sum(group) / len(group) for group in groups]
            if any(right - left <= separation for left, right in zip(unique, unique[1:])):
                return 'distinct real crossings must be separated by more than 1e-5', []
            return None, [float(value) for value in unique]

        n = len(a)
        center = (Fraction(lower) + Fraction(upper)) / 2
        radius = (Fraction(upper) - Fraction(lower)) / 2
        # Remove only the roundoff-scale symmetric components permitted by input validation.
        exact_a = [[(Fraction(float(a[i, j])) - Fraction(float(a[j, i]))) / 2 for j in range(n)] for i in range(n)]
        exact_b = [[(Fraction(float(b[i, j])) - Fraction(float(b[j, i]))) / 2 for j in range(n)] for i in range(n)]
        differences = [
            _pfaffian([[exact_a[i][j] + (center + radius * x) * exact_b[i][j] for j in range(n)] for i in range(n)])
            for x in range(n // 2 + 1)
        ]
        polynomial = [Fraction(0)] * (n // 2 + 1)
        basis = [Fraction(1)]
        for order in range(n // 2 + 1):
            for index, coefficient in enumerate(basis):
                polynomial[index] += differences[0] * coefficient
            differences = [right - left for left, right in zip(differences, differences[1:])]
            next_basis = [Fraction(0)] * (len(basis) + 1)
            for index, coefficient in enumerate(basis):
                next_basis[index] -= order * coefficient / (order + 1)
                next_basis[index + 1] += coefficient / (order + 1)
            basis = next_basis
        _trim(polynomial)
        if not any(polynomial):
            raise ValueError('pencil must be regular, with determinant not identically zero')
        if len(polynomial) == 1:
            return []
        derivative = [index * value for index, value in enumerate(polynomial)][1:]
        first, second = polynomial.copy(), derivative
        while any(second):
            _, remainder = _divide(first, second)
            first, second = second, remainder
        square_free, _ = _divide(polynomial, [value / first[-1] for value in first])

        previous = None
        precision = 96
        while True:
            with localcontext() as context:
                context.prec = precision
                roots = _simple_roots(square_free, precision)
                if roots is not None:
                    classified = _classify(roots, _decimal(center), _decimal(radius))
                    if previous is not None and classified[0] == previous[0] and len(classified[1]) == len(previous[1]):
                        if all(abs(x - y) <= 1e-12 for x, y in zip(classified[1], previous[1])):
                            if classified[0] is not None:
                                raise ValueError(classified[0])
                            return classified[1]
                    previous = classified
            precision = precision * 2

    p = _checked_pencil(pencil)
    lower, upper = _checked_interval(lower, upper)
    n = p.shape[1]
    a, b = p
    values = eigvals(-a, b, homogeneous_eigvals=True)
    ambiguous = bool(np.any((values[0] == 0) & (values[1] == 0)))
    equality = 1e-10
    scale = max(float(np.max(abs(a))), float(np.max(abs(b))), 1e-300)
    roots = []
    determinant_groups = []
    for alpha, beta in zip(*values):
        if abs(beta) == 0:
            continue
        root = alpha / beta
        if not np.isfinite(root):
            continue
        for group in determinant_groups:
            if abs(root - group[0]) <= equality:
                group.append(root)
                break
        else:
            determinant_groups.append([root])
        if abs(root.imag) > equality:
            if abs(root.imag) <= 1e-5:
                ambiguous = True
            continue
        root = float(root.real)
        if any(equality < abs(root - endpoint) <= 1e-5 for endpoint in (lower, upper)):
            ambiguous = True
        if lower + equality < root < upper - equality:
            residual = np.min(abs(eigvalsh(1j * (a + root * b) / scale)))
            if residual < 1e-7 * max(1., abs(root)):
                roots.append(root)
            else:
                ambiguous = True
    # Every finite determinant root of a regular skew pencil has even
    # algebraic multiplicity. Lost partners expose larger QZ root splitting,
    # including tangencies whose apparent imaginary parts exceed 1e-5.
    if any(len(group) % 2 for group in determinant_groups):
        ambiguous = True
    roots.sort()
    groups = []
    for root in roots:
        if groups and root - groups[-1][0] <= equality:
            groups[-1].append(root)
        else:
            groups.append([root])
    unique = [sum(group) / len(group) for group in groups]
    if np.any(np.diff(unique) <= 1e-5) or len(unique) > n // 2:
        ambiguous = True
    if ambiguous:
        unique = _exact_fallback(a, b, lower, upper)
    output = np.zeros(n // 2 + 1)
    output[0] = len(unique)
    output[1:len(unique) + 1] = unique
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, '
               '-2.0, 0.0]], [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '-0.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.3, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, '
               '-1.0, 0.0]], [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '-0.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.3])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.3])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.2, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.7], [0.0, 0.0, '
               '0.7, 0.0]], [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, '
               '-1.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.2, 0.7])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.2, 0.7])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.4, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.4], [0.0, 0.0, '
               '0.4, 0.0]], [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, '
               '-1.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.4])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.4])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -0.4], '
               '[0.0, 0.0, 0.0, 0.0, 0.4, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, '
               '0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, '
               '0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.4])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.4])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.6, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, 2.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.0], [0.0, '
               '0.0, 0.0, 0.0, -3.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, '
               '0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, '
               '0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.6])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.6])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.1, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, -0.10002, 0.0, 0.0], [0.0, 0.0, 0.10002, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, '
               '1.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, '
               '0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.0, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,0.5), p.shape[-1], 0.0, 0.5, [0.1, 0.10002])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,0.5), p.shape[-1], 0.0, 0.5, '
                   '[0.1, 0.10002])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.3, 0.0, 0.12], [0.3, 0.0, 0.21, 0.0], [0.0, -0.21, 0.0, 1.084], '
               '[-0.12, 0.0, -1.084, 0.0]], [[0.0, 1.0, 0.0, -0.4], [-1.0, 0.0, -0.7, 0.0], [0.0, 0.7, 0.0, '
               '-0.27999999999999997], [0.4, 0.0, 0.27999999999999997, 0.0]]],dtype=float)\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,-0.5,0.5), p.shape[-1], -0.5, 0.5, [0.3])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,-0.5,0.5), p.shape[-1], -0.5, 0.5, '
                   '[0.3])'},
     {'setup': 'import numpy as np\n'
               'z=np.zeros((2,2));i=np.eye(2)\n'
               'm=np.array([[-10000.,-0.0005],[0.0005,-10000.]])\n'
               'p=np.array([np.block([[z,m],[-m.T,z]]),np.block([[z,i],[-i,z]])])\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,9999.,10001.), p.shape[-1], 9999.0, 10001.0, [])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,9999.,10001.), p.shape[-1], 9999.0, '
                   '10001.0, [])'},
     {'setup': 'import numpy as np\n'
               'j=np.array([[0.,1.],[-1.,0.]])\n'
               'z=np.zeros((2,2))\n'
               'p=np.array([np.block([[-.2*j,z],[z,-.2000005*j]]),np.block([[j,z],[z,j]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: pencil_crossings(p,0.,1.))',
      'gold_call': '_raises_value_error(lambda: _oracle_pencil_crossings(p,0.,1.))'},
     {'setup': 'import numpy as np\n'
               'j=np.array([[0.,1.],[-1.,0.]])\n'
               'z=np.zeros((2,2))\n'
               'p=np.array([np.block([[-5e-9*j,z],[z,-.4*j]]),np.block([[j,z],[z,j]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: pencil_crossings(p,0.,1.))',
      'gold_call': '_raises_value_error(lambda: _oracle_pencil_crossings(p,0.,1.))'},
     {'setup': 'import numpy as np\n'
               'z=np.zeros((2,2));i=np.eye(2)\n'
               'm=np.array([[-.3,-0.000005],[0.000005,-.3]])\n'
               'p=np.array([np.block([[z,m],[-m.T,z]]),np.block([[z,i],[-i,z]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: pencil_crossings(p,0.,1.))',
      'gold_call': '_raises_value_error(lambda: _oracle_pencil_crossings(p,0.,1.))'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.5])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.5])'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               'b=np.array([[1.,.5,0.,-.25],[0.,1.,.25,0.],[0.,0.,1.,.5],[0.,0.,0.,1.]])\n'
               'p=np.stack([b@s@b.T for s in p])\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.5])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.5])'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.5,1.5), p.shape[-1], 0.5, 1.5, [])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.5,1.5), p.shape[-1], 0.5, 1.5, '
                   '[])'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,-0.5,0.5), p.shape[-1], -0.5, 0.5, [])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,-0.5,0.5), p.shape[-1], -0.5, 0.5, '
                   '[])'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               'j=np.array([[0.,1.],[-1.,0.]])\n'
               'q=np.zeros((2,6,6))\n'
               'q[:,:4,:4]=p\n'
               'q[0,4:,4:]=-.75*j\n'
               'q[1,4:,4:]=j\n'
               'p=q\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, [0.5, 0.75])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.0,1.0), p.shape[-1], 0.0, 1.0, '
                   '[0.5, 0.75])'},
     {'setup': 'import numpy as np\n'
               'm=4\n'
               'z=np.zeros((m,m));i=np.eye(m)\n'
               'j=-.5*i+np.diag(np.ones(m-1),1)\n'
               'c=np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [-1, -1, 1, 0, 0, 0, 0, 0], '
               '[-2, -1, 0, 1, 0, 0, 0, 0], [2, -1, 1, -2, 1, 0, 0, 0], [-1, -1, -2, -2, 1, 1, 0, 0], [-2, '
               '0, 2, -2, 2, 1, 1, 0], [-2, 2, 2, 0, 0, 2, -2, 1]],dtype=float)\n'
               'p=np.array([c@np.block([[z,j],[-j.T,z]])@c.T,c@np.block([[z,i],[-i,z]])@c.T])\n'
               '\n'
               'def _checked_crossing_record(actual, dimension, lower, upper, reference_roots):\n'
               '    # The driver has one combined relative/absolute tolerance. Check the public\n'
               '    # absolute root tolerance here, then compare the checked packed records.\n'
               '    size = dimension // 2 + 1\n'
               '    invalid = np.full(size, np.nan)\n'
               '    if not isinstance(actual, np.ndarray) or actual.shape != (size,):\n'
               '        return invalid\n'
               '    if actual.dtype.kind not in "iuf" or not np.isfinite(actual).all():\n'
               '        return invalid\n'
               '    count_value = actual[0]\n'
               '    if not 0 <= count_value <= dimension // 2 or count_value != int(count_value):\n'
               '        return invalid\n'
               '    count = int(count_value)\n'
               '    roots = actual[1:count + 1]\n'
               '    expected = np.asarray(reference_roots, dtype=float)\n'
               '    if count != expected.size or np.any(actual[count + 1:] != 0):\n'
               '        return invalid\n'
               '    if np.any(np.diff(roots) <= 0) or np.any(roots <= lower) or np.any(roots >= upper):\n'
               '        return invalid\n'
               '    if not np.allclose(roots, expected, rtol=0.0, atol=1e-6):\n'
               '        return invalid\n'
               '    checked = np.array(actual, dtype=float, copy=True)\n'
               '    checked[1:count + 1] = expected\n'
               '    return checked\n',
      'call': '_checked_crossing_record(pencil_crossings(p,0.,1.), p.shape[-1], 0.0, 1.0, [0.5])',
      'gold_call': '_checked_crossing_record(_oracle_pencil_crossings(p,0.,1.), p.shape[-1], 0.0, 1.0, '
                   '[0.5])'}]
