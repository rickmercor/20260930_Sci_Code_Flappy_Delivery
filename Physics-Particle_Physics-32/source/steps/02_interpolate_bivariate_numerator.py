"""
Recover the bivariate numerator coefficient tensor by modular interpolation.

The source uses bivariate Newton interpolation. Solving the fixed modular tensor-product Vandermonde system is algebraically equivalent and makes the coefficient ordering deterministic and testable. The modular system must use the specified pivot convention, and rearranging the input sample rows must not change the result.

Returns
-------
Return an integer NumPy array of shape (max_u+1,max_v+1), where entry [i,j] is the coefficient of u^i v^j modulo prime. Raise ValueError for invalid degree bounds, malformed samples, an incorrect sample count, an invalid prime, or a singular modular system.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def interpolate_bivariate(
    numerator_samples: np.ndarray,
    max_u: int,
    max_v: int,
    prime: int,
) -> np.ndarray:
    """Interpolate a tensor-product bivariate polynomial over a prime field.

    Parameters
    ----------
    numerator_samples : np.ndarray
        Integer array of shape (n, 3) containing u, v, and N(u,v).
    max_u : int
        Maximum exponent of u.
    max_v : int
        Maximum exponent of v.
    prime : int
        Prime modulus.

    Returns
    -------
    np.ndarray
        Coefficient matrix of shape (max_u+1, max_v+1), where entry
        [i, j] multiplies u**i * v**j.

    Raises
    ------
    ValueError
        If the inputs, degrees, prime, or sample count are invalid, or
        if the modular interpolation system is singular.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _is_prime_integer(value):
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


def _solve_unique_modular_system(matrix, rhs, prime):
    augmented = [
        [int(value) % prime for value in row]
        + [int(target) % prime]
        for row, target in zip(matrix, rhs)
    ]

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

    for row in range(pivot_row, row_count):
        if (
            all(augmented[row][column] == 0
                for column in range(column_count))
            and augmented[row][-1] != 0
        ):
            raise ValueError("inconsistent modular system")

    if len(pivot_columns) != column_count:
        raise ValueError(
            "modular system does not have a unique solution"
        )

    solution = np.zeros(column_count, dtype=np.int64)
    for row, column in enumerate(pivot_columns):
        solution[column] = augmented[row][-1]

    return solution


def _oracle_interpolate_bivariate(
    numerator_samples,
    max_u,
    max_v,
    prime,
):
    samples = np.asarray(numerator_samples)

    if (
        samples.ndim != 2
        or samples.shape[1] != 3
        or samples.size == 0
        or not np.issubdtype(samples.dtype, np.integer)
    ):
        raise ValueError(
            "numerator_samples must be a nonempty integer (n,3) array"
        )

    for degree in (max_u, max_v):
        if isinstance(degree, (bool, np.bool_)) or not isinstance(
            degree, (int, np.integer)
        ):
            raise ValueError("degree bounds must be integers")

    max_u = int(max_u)
    max_v = int(max_v)

    if max_u < 0 or max_v < 0:
        raise ValueError("degree bounds must be nonnegative")

    if not _is_prime_integer(prime):
        raise ValueError("prime must be an odd prime")

    prime = int(prime)
    coefficient_count = (max_u + 1) * (max_v + 1)

    if len(samples) != coefficient_count:
        raise ValueError(
            "sample count must match the tensor-product basis"
        )

    matrix = []
    rhs = []

    for u, v, value in samples:
        u = int(u) % prime
        v = int(v) % prime

        matrix.append([
            pow(u, i, prime) * pow(v, j, prime) % prime
            for i in range(max_u + 1)
            for j in range(max_v + 1)
        ])
        rhs.append(int(value) % prime)

    solution = _solve_unique_modular_system(
        matrix,
        rhs,
        prime,
    )

    return solution.reshape(max_u + 1, max_v + 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Compare results from independent equivalent input objects."""
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'cleared=np.array([[0,0,42],[0,1,80],[0,2,55],[0,3,86],[1,0,11],[1,1,67],[1,2,56],[1,3,97],[2,0,20],[2,1,3],[2,2,16],[2,3,77],[3,0,75],[3,1,96],[3,2,42],[3,3,32]],dtype=np.int64)\n'
               'p=101',
      'call': 'interpolate_bivariate(*deepcopy((cleared, 3, 3, p)))',
      'gold_call': '_oracle_interpolate_bivariate(*deepcopy((cleared, 3, 3, p)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'cleared=np.array([[8,9,13]],dtype=np.int64)\n'
               'p=101',
      'call': 'interpolate_bivariate(*deepcopy((cleared, 0, 0, p)))',
      'gold_call': '_oracle_interpolate_bivariate(*deepcopy((cleared, 0, 0, p)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'def _expect_value_error(function, arguments):\n'
               '    try:\n'
               '        function(*arguments)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'cleared=np.array([[0,0,1],[1,0,2]],dtype=np.int64)\n'
               'p=101\n'
               'arguments=(cleared,1,1,p)',
      'call': '_expect_value_error(interpolate_bivariate, deepcopy(arguments))',
      'gold_call': '_expect_value_error(_oracle_interpolate_bivariate, deepcopy(arguments))'}]
