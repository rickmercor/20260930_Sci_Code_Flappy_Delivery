#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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


def clear_denominator_samples(samples, factors, powers, prime):
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


def interpolate_bivariate(
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

from math import comb

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


def pair_ideal_membership(
    numerator_coefficients,
    factors,
    pairs,
    prime,
):
    coefficients = np.asarray(numerator_coefficients)
    factors = np.asarray(factors)
    pairs = np.asarray(pairs)

    if (
        coefficients.ndim != 2
        or coefficients.size == 0
        or not np.issubdtype(coefficients.dtype, np.integer)
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
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or pairs.size == 0
        or not np.issubdtype(pairs.dtype, np.integer)
    ):
        raise ValueError("pairs must be an integer (m,2) array")

    if not _is_prime_integer(prime):
        raise ValueError("prime must be an odd prime")

    prime = int(prime)

    def evaluate_tensor(point_u, point_v):
        return sum(
            int(coefficients[i, j])
            * pow(point_u, i, prime)
            * pow(point_v, j, prime)
            for i in range(coefficients.shape[0])
            for j in range(coefficients.shape[1])
        ) % prime

    def vanishes_on_line(constant, u_coefficient, v_coefficient):
        result_degree = (
            coefficients.shape[0] + coefficients.shape[1] - 2
        )
        result = np.zeros(result_degree + 1, dtype=np.int64)

        if v_coefficient % prime:
            inverse = pow(v_coefficient % prime, -1, prime)
            alpha = -constant * inverse % prime
            beta = -u_coefficient * inverse % prime

            # Substitute v = alpha + beta*u.
            for i in range(coefficients.shape[0]):
                for j in range(coefficients.shape[1]):
                    coefficient = int(coefficients[i, j]) % prime

                    for k in range(j + 1):
                        result[i + k] += (
                            coefficient
                            * comb(j, k)
                            * pow(alpha, j - k, prime)
                            * pow(beta, k, prime)
                        )
                        result[i + k] %= prime
        else:
            inverse = pow(u_coefficient % prime, -1, prime)
            alpha = -constant * inverse % prime
            beta = -v_coefficient * inverse % prime

            # Substitute u = alpha + beta*v.
            for i in range(coefficients.shape[0]):
                for j in range(coefficients.shape[1]):
                    coefficient = int(coefficients[i, j]) % prime

                    for k in range(i + 1):
                        result[j + k] += (
                            coefficient
                            * comb(i, k)
                            * pow(alpha, i - k, prime)
                            * pow(beta, k, prime)
                        )
                        result[j + k] %= prime

        return bool(np.all(result % prime == 0))

    flags = []

    for left, right in pairs:
        left = int(left)
        right = int(right)

        if (
            left == right
            or left < 0
            or right < 0
            or left >= len(factors)
            or right >= len(factors)
        ):
            raise ValueError(
                "pair indices must be distinct and in range"
            )

        constant_1, u_1, v_1 = (
            int(value) % prime for value in factors[left]
        )
        constant_2, u_2, v_2 = (
            int(value) % prime for value in factors[right]
        )

        determinant = (u_1 * v_2 - u_2 * v_1) % prime

        if determinant:
            inverse = pow(determinant, -1, prime)

            point_u = (
                (v_1 * constant_2 - v_2 * constant_1)
                * inverse
            ) % prime

            point_v = (
                (u_2 * constant_1 - u_1 * constant_2)
                * inverse
            ) % prime

            flags.append(
                int(evaluate_tensor(point_u, point_v) == 0)
            )
            continue

        if u_1 == v_1 == u_2 == v_2 == 0:
            if constant_1 or constant_2:
                flags.append(1)
            else:
                flags.append(
                    int(np.all(coefficients % prime == 0))
                )
            continue

        if u_1 or v_1:
            constant = constant_1
            u_coefficient = u_1
            v_coefficient = v_1
            other_constant = constant_2
            other_u = u_2
            other_v = v_2
        else:
            constant = constant_2
            u_coefficient = u_2
            v_coefficient = v_2
            other_constant = constant_1
            other_u = u_1
            other_v = v_1

        inconsistent = (
            (
                u_coefficient * other_constant
                - other_u * constant
            ) % prime != 0
            or (
                v_coefficient * other_constant
                - other_v * constant
            ) % prime != 0
        )

        if inconsistent:
            # Parallel inconsistent affine equations generate 1.
            flags.append(1)
        else:
            flags.append(
                int(
                    vanishes_on_line(
                        constant,
                        u_coefficient,
                        v_coefficient,
                    )
                )
            )

    return np.asarray(flags, dtype=np.int64)

import numpy as np


def encode_accepted_pair_ideals(
    pair_flags,
    pairs,
    factor_count,
):
    flags = np.asarray(pair_flags)
    pairs = np.asarray(pairs)

    if (
        flags.ndim != 1
        or flags.size == 0
        or not np.issubdtype(flags.dtype, np.integer)
    ):
        raise ValueError(
            "pair_flags must be a nonempty integer vector"
        )

    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or pairs.size == 0
        or not np.issubdtype(pairs.dtype, np.integer)
    ):
        raise ValueError("pairs must be an integer (m,2) array")

    if len(flags) != len(pairs):
        raise ValueError("pair_flags and pairs must have equal length")

    if np.any((flags != 0) & (flags != 1)):
        raise ValueError("pair_flags must contain only 0 or 1")

    if (
        isinstance(factor_count, (bool, np.bool_))
        or not isinstance(factor_count, (int, np.integer))
    ):
        raise ValueError("factor_count must be an integer")

    factor_count = int(factor_count)

    if factor_count < 1:
        raise ValueError("factor_count must be positive")

    accepted = []

    for flag, pair in zip(flags, pairs):
        left = int(pair[0])
        right = int(pair[1])

        if (
            left == right
            or left < 0
            or right < 0
            or left >= factor_count
            or right >= factor_count
        ):
            raise ValueError(
                "pair indices must be distinct and in range"
            )

        if int(flag) == 0:
            continue

        exponent_tensor = np.zeros(
            (2, factor_count),
            dtype=np.int64,
        )
        exponent_tensor[0, left] = 1
        exponent_tensor[1, right] = 1
        accepted.append(exponent_tensor)

    if not accepted:
        raise ValueError("at least one pair must be accepted")

    return np.stack(accepted, axis=0)

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


def simplified_intersection_generators(
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


def fit_canonical_pfd_numerators(
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

import numpy as np


def run_pfd_pipeline(
    samples,
    factors,
    powers,
    pairs,
    prime,
    heldout_points,
):
    # Explicitly chain the six preceding oracle functions.
    cleared_samples = clear_denominator_samples(
        samples,
        factors,
        powers,
        prime,
    )

    numerator_coefficients = interpolate_bivariate(
        cleared_samples,
        3,
        3,
        prime,
    )

    pair_flags = pair_ideal_membership(
        numerator_coefficients,
        factors,
        pairs,
        prime,
    )

    accepted_ideals = encode_accepted_pair_ideals(
        pair_flags,
        pairs,
        len(factors),
    )

    generators = simplified_intersection_generators(
        numerator_coefficients,
        factors,
        accepted_ideals,
        prime,
    )

    quotient_coefficients = (
        fit_canonical_pfd_numerators(
            numerator_coefficients,
            factors,
            generators,
            1,
            prime,
        )
    )

    heldout_points = np.asarray(heldout_points)

    if (
        heldout_points.ndim != 2
        or heldout_points.shape[1] != 2
        or heldout_points.size == 0
        or not np.issubdtype(
            heldout_points.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "heldout_points must be a nonempty integer (h,2) array"
        )

    factors = np.asarray(factors)
    powers = np.asarray(powers)
    prime = int(prime)

    heldout_values = []

    for point_u, point_v in heldout_points:
        point_u = int(point_u) % prime
        point_v = int(point_v) % prime

        denominator = 1

        for factor, exponent in zip(factors, powers):
            factor_value = (
                int(factor[0])
                + int(factor[1]) * point_u
                + int(factor[2]) * point_v
            ) % prime

            denominator = (
                denominator
                * pow(factor_value, int(exponent), prime)
            ) % prime

        if denominator == 0:
            raise ValueError(
                "a held-out point lies on the denominator"
            )

        numerator_value = sum(
            int(numerator_coefficients[i, j])
            * pow(point_u, i, prime)
            * pow(point_v, j, prime)
            for i in range(numerator_coefficients.shape[0])
            for j in range(numerator_coefficients.shape[1])
        ) % prime

        rational_value = (
            numerator_value
            * pow(denominator, -1, prime)
        ) % prime

        heldout_values.append(rational_value)

    checksum = sum(
        (index + 1) * int(value)
        for index, value in enumerate(
            quotient_coefficients.ravel()
        )
    )

    checksum += sum(
        (index + 17) * int(value)
        for index, value in enumerate(generators.ravel())
    )

    checksum += sum(
        (index + 31) * int(value)
        for index, value in enumerate(heldout_values)
    )

    checksum %= prime

    return np.concatenate(
        (
            np.asarray([checksum], dtype=np.int64),
            np.asarray(pair_flags, dtype=np.int64),
            np.asarray([len(generators)], dtype=np.int64),
            np.asarray(heldout_values, dtype=np.int64),
        )
    )
SCICODE_GOLD_EOF
