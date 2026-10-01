"""
Combine accepted constraints through LCM generators and membership rechecks.

Equations (4.31)–(4.37) define the simplified intersection operation. Componentwise maximum of denominator-factor exponent vectors implements the LCM. Duplicate and divisible generators are removed. Numerator membership must then be rechecked because the generated ideal is only a sub-ideal of the true intersection.

Returns
-------
Return an integer NumPy array of shape (g, f) containing the ordered minimal denominator-factor exponent vectors retained by the iterative simplified-intersection procedure. Raise ValueError for malformed polynomial or factor arrays, an invalid prime, incompatible accepted-ideal dimensions, or negative exponents.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def simplified_intersection_generators(
    numerator_coefficients: np.ndarray,
    factors: np.ndarray,
    accepted_ideals: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Construct simplified intersection generators.

    Parameters
    ----------
    numerator_coefficients : np.ndarray
        Integer matrix C where C[i,j] multiplies u**i * v**j.
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b]
        representing c + a*u + b*v.
    accepted_ideals : np.ndarray
        Nonnegative integer exponent array of shape (a,2,f).
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Ordered minimal generator-exponent matrix of shape (g,f).

    Raises
    ------
    ValueError
        If polynomial, factor, exponent, shape, or modulus data are
        invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from itertools import product

import numpy as np


def _s05_is_prime(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        return False

    value = int(value)
    if value < 3:
        return False

    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1

    return True


def _s05_minimal_generators(generators):
    unique = sorted(
        {
            tuple(int(value) for value in generator)
            for generator in generators
        },
        key=lambda row: (sum(row), row),
    )

    return [
        row
        for row in unique
        if not any(
            other != row
            and all(left <= right for left, right in zip(other, row))
            for other in unique
        )
    ]


def _s05_multiply_polynomials(left, right, prime):
    result = np.zeros(
        (
            left.shape[0] + right.shape[0] - 1,
            left.shape[1] + right.shape[1] - 1,
        ),
        dtype=np.int64,
    )

    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            coefficient = int(left[i, j]) % prime

            if coefficient:
                result[
                    i:i + right.shape[0],
                    j:j + right.shape[1],
                ] += coefficient * right
                result %= prime

    return result


def _s05_generator_polynomial(exponents, factors, prime):
    result = np.array([[1]], dtype=np.int64)

    for exponent, factor in zip(exponents, factors):
        factor_polynomial = np.zeros((2, 2), dtype=np.int64)
        factor_polynomial[0, 0] = int(factor[0]) % prime
        factor_polynomial[1, 0] = int(factor[1]) % prime
        factor_polynomial[0, 1] = int(factor[2]) % prime

        for _ in range(int(exponent)):
            result = _s05_multiply_polynomials(
                result,
                factor_polynomial,
                prime,
            )

    return result


def _s05_system_is_consistent(matrix, rhs, prime):
    augmented = [
        [int(value) % prime for value in row]
        + [int(target) % prime]
        for row, target in zip(matrix, rhs)
    ]

    row_count = len(augmented)
    column_count = len(augmented[0]) - 1
    pivot_row = 0

    for column in range(column_count):
        pivot = next(
            (
                row
                for row in range(pivot_row, row_count)
                if augmented[row][column] % prime
            ),
            None,
        )

        if pivot is None:
            continue

        augmented[pivot_row], augmented[pivot] = (
            augmented[pivot],
            augmented[pivot_row],
        )

        inverse = pow(augmented[pivot_row][column], -1, prime)
        augmented[pivot_row] = [
            value * inverse % prime
            for value in augmented[pivot_row]
        ]

        for row in range(row_count):
            if row == pivot_row:
                continue

            factor = augmented[row][column] % prime
            if factor:
                augmented[row] = [
                    (left - factor * right) % prime
                    for left, right in zip(
                        augmented[row],
                        augmented[pivot_row],
                    )
                ]

        pivot_row += 1

        if pivot_row == row_count:
            break

    for row in range(pivot_row, row_count):
        if (
            all(
                augmented[row][column] == 0
                for column in range(column_count)
            )
            and augmented[row][-1] != 0
        ):
            return False

    return True


def _s05_numerator_in_generated_ideal(numerator, generator_polynomials, prime):
    """Exact ideal membership by Buchberger reduction over the prime field."""
    def _clean(poly):
        return {monomial: int(value) % prime
                for monomial, value in poly.items() if int(value) % prime}

    def _sparse(tensor):
        return _clean({(i, j): int(tensor[i, j])
                       for i in range(tensor.shape[0])
                       for j in range(tensor.shape[1])})

    def _subtract_shift(poly, divisor, shift, coefficient):
        result = dict(poly)
        for (i, j), value in divisor.items():
            monomial = (i + shift[0], j + shift[1])
            result[monomial] = result.get(monomial, 0) - coefficient * value
        return _clean(result)

    def _remainder(poly, basis):
        poly, remainder = dict(poly), {}
        while poly:
            leading = max(poly)
            for divisor in basis:
                degree = max(divisor)
                if degree[0] <= leading[0] and degree[1] <= leading[1]:
                    coefficient = poly[leading] * pow(divisor[degree], -1, prime) % prime
                    poly = _subtract_shift(poly, divisor,
                                           (leading[0] - degree[0], leading[1] - degree[1]),
                                           coefficient)
                    break
            else:
                remainder[leading] = poly.pop(leading)
        return remainder

    def _monic(poly):
        inverse = pow(poly[max(poly)], -1, prime)
        return {degree: value * inverse % prime for degree, value in poly.items()}

    basis = []
    for tensor in generator_polynomials:
        remainder = _remainder(_sparse(tensor), basis)
        if remainder:
            basis.append(_monic(remainder))
    pairs = [(i, j) for j in range(len(basis)) for i in range(j)]
    index = 0
    while index < len(pairs):
        i, j = pairs[index]
        index += 1
        left, right = basis[i], basis[j]
        dl, dr = max(left), max(right)
        lcm = (max(dl[0], dr[0]), max(dl[1], dr[1]))
        shifted_left = {(u + lcm[0] - dl[0], v + lcm[1] - dl[1]): value
                        for (u, v), value in left.items()}
        s_poly = _subtract_shift(shifted_left, right,
                                 (lcm[0] - dr[0], lcm[1] - dr[1]), 1)
        remainder = _remainder(s_poly, basis)
        if remainder:
            pairs.extend((k, len(basis)) for k in range(len(basis)))
            basis.append(_monic(remainder))
    return not _remainder(_sparse(numerator), basis)


def _oracle_simplified_intersection_generators(
    numerator_coefficients,
    factors,
    accepted_ideals,
    prime,
):
    numerator = np.asarray(numerator_coefficients)
    factors = np.asarray(factors)
    ideals = np.asarray(accepted_ideals)

    if (
        numerator.ndim != 2
        or numerator.size == 0
        or not np.issubdtype(numerator.dtype, np.integer)
    ):
        raise ValueError(
            "numerator_coefficients must be a nonempty integer matrix"
        )

    if (
        factors.ndim != 2
        or factors.shape[1] != 3
        or factors.size == 0
        or not np.issubdtype(factors.dtype, np.integer)
    ):
        raise ValueError("factors must be an integer (f,3) array")

    if (
        ideals.ndim != 3
        or ideals.size == 0
        or not np.issubdtype(ideals.dtype, np.integer)
    ):
        raise ValueError(
            "accepted_ideals must be a nonempty integer array"
        )

    if ideals.shape[1] != 2 or ideals.shape[2] != len(factors):
        raise ValueError(
            "accepted_ideals must have shape (a,2,f)"
        )

    if np.any(ideals < 0):
        raise ValueError("factor exponents must be nonnegative")

    if not _s05_is_prime(prime):
        raise ValueError("prime must be an odd prime")

    prime = int(prime)
    numerator = numerator.astype(np.int64, copy=False) % prime

    current = _s05_minimal_generators(ideals[0])

    for ideal in ideals[1:]:
        trial = _s05_minimal_generators(
            tuple(
                max(left_exponent, right_exponent)
                for left_exponent, right_exponent in zip(
                    left,
                    right,
                )
            )
            for left, right in product(current, ideal)
        )

        generator_polynomials = [
            _s05_generator_polynomial(
                generator,
                factors,
                prime,
            )
            for generator in trial
        ]

        if _s05_numerator_in_generated_ideal(
            numerator,
            generator_polynomials,
            prime,
        ):
            current = trial

    return np.asarray(current, dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Compare results from independent equivalent input objects."""
    return [{'setup': '\n'
               'import numpy as np\n'
               'from copy import deepcopy\n'
               'coefficients=np.array([[42,25,10,3],[52,15,99,0],[17,5,0,0],[1,0,0,0]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'ideals=np.array([[[1,0,0,0],[0,1,0,0]],[[0,1,0,0],[0,0,1,0]],[[0,0,1,0],[0,0,0,1]]],dtype=np.int64)\n'
               'p=101\n',
      'call': 'simplified_intersection_generators(*deepcopy((coefficients, factors, ideals, p)))',
      'gold_call': '_oracle_simplified_intersection_generators(*deepcopy((coefficients, factors, '
                   'ideals, p)))'},
     {'setup': '\n'
               'import numpy as np\n'
               'from copy import deepcopy\n'
               'coefficients=np.array([[42,25,10,3],[52,15,99,0],[17,5,0,0],[1,0,0,0]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'ideals=np.array([[[1,0,0,0],[0,1,0,0]]],dtype=np.int64)\n'
               'p=101\n',
      'call': 'simplified_intersection_generators(*deepcopy((coefficients, factors, ideals, p)))',
      'gold_call': '_oracle_simplified_intersection_generators(*deepcopy((coefficients, factors, '
                   'ideals, p)))'},
     {'setup': '\n'
               'import numpy as np\n'
               'from copy import deepcopy\n'
               'def _expect_value_error(function, arguments):\n'
               '    try:\n'
               '        function(*arguments)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'coefficients=np.array([[1]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'ideals=np.zeros((1,2,3),dtype=np.int64)\n'
               'p=101\n'
               'arguments=(coefficients,factors,ideals,p)\n',
      'call': '_expect_value_error(simplified_intersection_generators, deepcopy(arguments))',
      'gold_call': '_expect_value_error(_oracle_simplified_intersection_generators, '
                   'deepcopy(arguments))'},
     {'setup': '\n'
               'import numpy as np\n'
               'from copy import deepcopy\n'
               'coefficients=np.array([[1]],dtype=int)\n'
               'factors=np.array([[0,1,0],[1,1,0],[2,1,0]],dtype=int)\n'
               'ideals=np.array([[[1,0,0],[0,1,0]],[[0,1,0],[0,0,1]]],dtype=int)\n'
               'p=101\n',
      'call': 'simplified_intersection_generators(*deepcopy((coefficients, factors, ideals, p)))',
      'gold_call': '_oracle_simplified_intersection_generators(*deepcopy((coefficients, factors, '
                   'ideals, p)))'},
     {'setup': '\n'
               'import numpy as np\n'
               'from copy import deepcopy\n'
               'coefficients=np.array([[0],[1]],dtype=int)\n'
               'factors=np.array([[0,1,0],[0,0,1],[0,1,1],[0,1,100]],dtype=int)\n'
               'ideals=np.array([[[1,0,0,0],[0,1,0,0]],[[0,0,1,0],[0,0,0,1]]],dtype=int)\n'
               'p=101\n',
      'call': 'simplified_intersection_generators(*deepcopy((coefficients, factors, ideals, p)))',
      'gold_call': '_oracle_simplified_intersection_generators(*deepcopy((coefficients, factors, '
                   'ideals, p)))'}]
