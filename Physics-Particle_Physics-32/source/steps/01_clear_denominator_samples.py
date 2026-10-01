"""
Clear the known common denominator from rational finite-field samples.

Equation (4.27) obtains numerator samples by multiplying each sampled rational coefficient by its known denominator on the bivariate slice. All arithmetic is performed modulo the supplied prime. A sample lying on any denominator factor with positive power is singular and cannot be used.

Returns
-------
Return an integer NumPy array of shape (n,3) with columns u, v, and N(u,v). Raise ValueError for malformed integer arrays, an invalid prime, invalid powers, incompatible shapes, or a sample whose denominator is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def clear_denominator_samples(
    samples: np.ndarray,
    factors: np.ndarray,
    powers: np.ndarray,
    prime: int,
) -> np.ndarray:
    """Clear a known denominator from modular rational samples.

    Parameters
    ----------
    samples : np.ndarray
        Integer array of shape (n, 3) containing u, v, and r(u,v).
    factors : np.ndarray
        Integer array of shape (f, 3), with rows [c, a, b].
    powers : np.ndarray
        Nonnegative integer vector of shape (f,).
    prime : int
        Odd prime modulus.

    Returns
    -------
    np.ndarray
        Integer array of shape (n, 3) containing u, v, and N(u,v).

    Raises
    ------
    ValueError
        If the inputs are malformed, the modulus is invalid, or a sample
        lies on the supplied denominator.
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
    if value < 2:
        return False

    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1

    return True


def _oracle_clear_denominator_samples(samples, factors, powers, prime):
    samples = np.asarray(samples)
    factors = np.asarray(factors)
    powers = np.asarray(powers)

    if (
        samples.ndim != 2
        or samples.shape[1] != 3
        or samples.size == 0
        or not np.issubdtype(samples.dtype, np.integer)
    ):
        raise ValueError("samples must be a nonempty integer (n,3) array")

    if (
        factors.ndim != 2
        or factors.shape[1] != 3
        or factors.size == 0
        or not np.issubdtype(factors.dtype, np.integer)
    ):
        raise ValueError("factors must be a nonempty integer (f,3) array")

    if (
        powers.ndim != 1
        or powers.shape != (len(factors),)
        or not np.issubdtype(powers.dtype, np.integer)
        or np.any(powers < 0)
    ):
        raise ValueError(
            "powers must be nonnegative integers matching the factors"
        )

    if not _is_prime_integer(prime):
        raise ValueError("prime must be prime")

    prime = int(prime)
    output = np.empty((len(samples), 3), dtype=np.int64)

    for row, (u, v, rational_value) in enumerate(samples):
        u = int(u) % prime
        v = int(v) % prime
        denominator = 1

        for (constant, u_coefficient, v_coefficient), exponent in zip(
            factors, powers
        ):
            factor_value = (
                int(constant)
                + int(u_coefficient) * u
                + int(v_coefficient) * v
            ) % prime

            denominator = (
                denominator
                * pow(factor_value, int(exponent), prime)
            ) % prime

        if denominator == 0:
            raise ValueError("a sample lies on the supplied denominator")

        output[row] = (
            u,
            v,
            int(rational_value) * denominator % prime,
        )

    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Compare results from independent equivalent input objects."""
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'samples=np.array([[0,0,97],[0,1,35],[0,2,19],[0,3,7],[1,0,21],[1,1,12],[1,2,81],[1,3,77],[2,0,25],[2,1,98],[2,2,89],[2,3,89],[3,0,58],[3,1,91],[3,2,51],[3,3,99]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'powers=np.ones(4,dtype=np.int64)\n'
               'p=101',
      'call': 'clear_denominator_samples(*deepcopy((samples, factors, powers, p)))',
      'gold_call': '_oracle_clear_denominator_samples(*deepcopy((samples, factors, powers, p)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'samples=np.array([[0,0,7]],dtype=np.int64)\n'
               'factors=np.array([[1,0,0]],dtype=np.int64)\n'
               'powers=np.array([0],dtype=np.int64)\n'
               'p=101',
      'call': 'clear_denominator_samples(*deepcopy((samples, factors, powers, p)))',
      'gold_call': '_oracle_clear_denominator_samples(*deepcopy((samples, factors, powers, p)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'def _expect_value_error(function, arguments):\n'
               '    try:\n'
               '        function(*arguments)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'samples=np.array([[100,0,2]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0]],dtype=np.int64)\n'
               'powers=np.array([1],dtype=np.int64)\n'
               'p=101\n'
               'arguments=(samples,factors,powers,p)',
      'call': '_expect_value_error(clear_denominator_samples, deepcopy(arguments))',
      'gold_call': '_expect_value_error(_oracle_clear_denominator_samples, deepcopy(arguments))'}]
