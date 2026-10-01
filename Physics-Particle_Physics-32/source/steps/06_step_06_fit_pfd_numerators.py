"""
Fit a deterministic set of lower-degree partial-fraction numerators.

Equation (4.38) expresses the reconstructed numerator as a sum of simplified denominator generators multiplied by lower-degree quotient polynomials. Syzygies can make this representation nonunique, so the task uses modular reduced row-echelon form and sets all free variables to zero.

Returns
-------
Return an integer NumPy array of shape (g, b), where g is the number of generators and b=(quotient_degree+1)(quotient_degree+2)/2. Columns use increasing total degree and descending u exponent within each degree; for quotient_degree=1 the basis is (1,u,v). Use modular RREF and set free variables to zero. Raise ValueError for invalid or inconsistent inputs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_canonical_pfd_numerators(
    numerator_coefficients: np.ndarray,
    factors: np.ndarray,
    generators: np.ndarray,
    quotient_degree: int,
    prime: int,
) -> np.ndarray:
    """Fit canonical quotient polynomials for the PFD generators.

    Parameters
    ----------
    numerator_coefficients : np.ndarray
        Integer matrix C where C[i,j] multiplies u**i * v**j.
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b].
    generators : np.ndarray
        Nonnegative exponent matrix of shape (g,f).
    quotient_degree : int
        Maximum total degree of each quotient polynomial.
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Generator-major quotient coefficients. Columns follow increasing
        total degree and, within each degree, descending u exponent.

    Raises
    ------
    ValueError
        If the inputs are malformed, the degree or prime is invalid, or
        the modular coefficient system is inconsistent.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _s06_is_prime(value):
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


def _s06_multiply_polynomials(left, right, prime):
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


def _s06_generator_polynomial(exponents, factors, prime):
    result = np.array([[1]], dtype=np.int64)

    for exponent, factor in zip(exponents, factors):
        factor_polynomial = np.zeros((2, 2), dtype=np.int64)
        factor_polynomial[0, 0] = int(factor[0]) % prime
        factor_polynomial[1, 0] = int(factor[1]) % prime
        factor_polynomial[0, 1] = int(factor[2]) % prime

        for _ in range(int(exponent)):
            result = _s06_multiply_polynomials(
                result,
                factor_polynomial,
                prime,
            )

    return result


def _s06_canonical_modular_solution(matrix, rhs, prime):
    augmented = [
        [int(value) % prime for value in row]
        + [int(target) % prime]
        for row, target in zip(matrix, rhs)
    ]

    if not augmented:
        raise ValueError("linear system must be nonempty")

    row_count = len(augmented)
    column_count = len(augmented[0]) - 1
    pivot_row = 0
    pivot_columns = []

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

        pivot_columns.append(column)
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
            raise ValueError("inconsistent modular system")

    # The canonical convention sets every free variable to zero.
    solution = np.zeros(column_count, dtype=np.int64)

    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row][-1]

    return solution


def _oracle_fit_canonical_pfd_numerators(
    numerator_coefficients,
    factors,
    generators,
    quotient_degree,
    prime,
):
    numerator = np.asarray(numerator_coefficients)
    factors = np.asarray(factors)
    generators = np.asarray(generators)

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
        generators.ndim != 2
        or generators.size == 0
        or not np.issubdtype(generators.dtype, np.integer)
    ):
        raise ValueError(
            "generators must be a nonempty integer matrix"
        )

    if generators.shape[1] != len(factors):
        raise ValueError(
            "generator width must equal the number of factors"
        )

    if np.any(generators < 0):
        raise ValueError("generator exponents must be nonnegative")

    if (
        isinstance(quotient_degree, (bool, np.bool_))
        or not isinstance(quotient_degree, (int, np.integer))
    ):
        raise ValueError("quotient_degree must be an integer")

    quotient_degree = int(quotient_degree)

    if quotient_degree < 0:
        raise ValueError("quotient_degree must be nonnegative")

    if not _s06_is_prime(prime):
        raise ValueError("prime must be an odd prime")

    prime = int(prime)
    numerator = numerator.astype(np.int64, copy=False) % prime

    # Increasing total degree; descending u exponent within each degree.
    quotient_basis = [
        (u_degree, total_degree - u_degree)
        for total_degree in range(quotient_degree + 1)
        for u_degree in range(total_degree, -1, -1)
    ]

    columns = []

    for exponent_row in generators:
        generator = _s06_generator_polynomial(
            exponent_row,
            factors,
            prime,
        )

        for u_shift, v_shift in quotient_basis:
            shifted = np.zeros(
                (
                    generator.shape[0] + u_shift,
                    generator.shape[1] + v_shift,
                ),
                dtype=np.int64,
            )

            shifted[
                u_shift:u_shift + generator.shape[0],
                v_shift:v_shift + generator.shape[1],
            ] = generator

            columns.append(shifted)

    row_u = max(
        [numerator.shape[0]]
        + [column.shape[0] for column in columns]
    )
    row_v = max(
        [numerator.shape[1]]
        + [column.shape[1] for column in columns]
    )

    monomials = [
        (i, j)
        for i in range(row_u)
        for j in range(row_v)
    ]

    matrix = [
        [
            (
                int(column[i, j])
                if i < column.shape[0] and j < column.shape[1]
                else 0
            )
            for column in columns
        ]
        for i, j in monomials
    ]

    rhs = [
        (
            int(numerator[i, j])
            if i < numerator.shape[0] and j < numerator.shape[1]
            else 0
        )
        for i, j in monomials
    ]

    solution = _s06_canonical_modular_solution(
        matrix,
        rhs,
        prime,
    )

    return solution.reshape(
        len(generators),
        len(quotient_basis),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Compare results from independent equivalent input objects."""
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'coefficients=np.array([[42,25,10,3],[52,15,99,0],[17,5,0,0],[1,0,0,0]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'generators=np.array([[0,1,0,1],[0,1,1,0],[1,0,1,0]],dtype=np.int64)\n'
               'p=101',
      'call': 'fit_canonical_pfd_numerators(*deepcopy((coefficients, factors, generators, 1, p)))',
      'gold_call': '_oracle_fit_canonical_pfd_numerators(*deepcopy((coefficients, factors, '
                   'generators, 1, p)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'coefficients=np.array([[7]],dtype=np.int64)\n'
               'factors=np.array([[1,0,0]],dtype=np.int64)\n'
               'generators=np.array([[0]],dtype=np.int64)\n'
               'p=101',
      'call': 'fit_canonical_pfd_numerators(*deepcopy((coefficients, factors, generators, 0, p)))',
      'gold_call': '_oracle_fit_canonical_pfd_numerators(*deepcopy((coefficients, factors, '
                   'generators, 0, p)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'def _expect_value_error(function, arguments):\n'
               '    try:\n'
               '        function(*arguments)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'coefficients=np.array([[7]],dtype=np.int64)\n'
               'factors=np.array([[1,0,0]],dtype=np.int64)\n'
               'generators=np.array([[0]],dtype=np.int64)\n'
               'p=101\n'
               'arguments=(coefficients,factors,generators,-1,p)',
      'call': '_expect_value_error(fit_canonical_pfd_numerators, deepcopy(arguments))',
      'gold_call': '_expect_value_error(_oracle_fit_canonical_pfd_numerators, deepcopy(arguments))'}]
