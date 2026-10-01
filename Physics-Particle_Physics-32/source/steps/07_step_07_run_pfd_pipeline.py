"""
Run the complete reconstruction workflow and return its checksum and diagnostics.

The orchestrator follows equations (4.27)–(4.38): clear the known denominator, interpolate the numerator, test pair-ideal membership, encode accepted pair ideals, construct the simplified intersection, and fit canonical partial-fraction numerators. It then evaluates the reconstructed rational function at nonsingular held-out points.

Returns
-------
Return a one-dimensional integer NumPy array containing [checksum, all pair-membership flags in input order, generator_count, all held-out rational values in input order]. For the declared instance, the result is [38,1,1,0,1,3,45,30,73]. Raise ValueError if a preceding contract fails or a held-out point lies on the denominator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_pfd_pipeline(
    samples: np.ndarray,
    factors: np.ndarray,
    powers: np.ndarray,
    pairs: np.ndarray,
    prime: int,
    heldout_points: np.ndarray,
) -> np.ndarray:
    """Run the complete finite-field PFD reconstruction pipeline.

    Parameters
    ----------
    samples : np.ndarray
        Integer array of shape (n,3) containing u, v, and r(u,v).
    factors : np.ndarray
        Integer array of shape (f,3), with rows [c,a,b].
    powers : np.ndarray
        Nonnegative denominator-factor powers of shape (f,).
    pairs : np.ndarray
        Integer array of shape (m,2) containing candidate factor pairs.
    prime : int
        Odd prime modulus.
    heldout_points : np.ndarray
        Integer array of shape (h,2) containing evaluation points.

    Returns
    -------
    np.ndarray
        Vector containing the checksum, pair flags, generator count,
        and held-out rational values.

    Raises
    ------
    ValueError
        If any preceding step contract fails or a held-out point lies
        on the denominator.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_pfd_pipeline(
    samples,
    factors,
    powers,
    pairs,
    prime,
    heldout_points,
):
    # Explicitly chain the six preceding oracle functions.
    cleared_samples = _oracle_clear_denominator_samples(
        samples,
        factors,
        powers,
        prime,
    )

    numerator_coefficients = _oracle_interpolate_bivariate(
        cleared_samples,
        3,
        3,
        prime,
    )

    pair_flags = _oracle_pair_ideal_membership(
        numerator_coefficients,
        factors,
        pairs,
        prime,
    )

    accepted_ideals = _oracle_encode_accepted_pair_ideals(
        pair_flags,
        pairs,
        len(factors),
    )

    generators = _oracle_simplified_intersection_generators(
        numerator_coefficients,
        factors,
        accepted_ideals,
        prime,
    )

    quotient_coefficients = (
        _oracle_fit_canonical_pfd_numerators(
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
               'pairs=np.array([[0,1],[1,2],[0,3],[2,3]],dtype=np.int64)\n'
               'p=101\n'
               'heldout=np.array([[7,11],[13,17],[19,23]],dtype=np.int64)',
      'call': 'run_pfd_pipeline(*deepcopy((samples, factors, powers, pairs, p, heldout)))',
      'gold_call': '_oracle_run_pfd_pipeline(*deepcopy((samples, factors, powers, pairs, p, '
                   'heldout)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'samples=np.array([[0,0,97],[0,1,35],[0,2,19],[0,3,7],[1,0,21],[1,1,12],[1,2,81],[1,3,77],[2,0,25],[2,1,98],[2,2,89],[2,3,89],[3,0,58],[3,1,91],[3,2,51],[3,3,99]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'powers=np.ones(4,dtype=np.int64)\n'
               'pairs=np.array([[0,1],[1,2],[0,3],[2,3]],dtype=np.int64)\n'
               'p=101\n'
               'heldout=np.array([[7,11]],dtype=np.int64)',
      'call': 'run_pfd_pipeline(*deepcopy((samples, factors, powers, pairs, p, heldout)))',
      'gold_call': '_oracle_run_pfd_pipeline(*deepcopy((samples, factors, powers, pairs, p, '
                   'heldout)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'samples=np.array([[0,0,97],[0,1,35],[0,2,19],[0,3,7],[1,0,21],[1,1,12],[1,2,81],[1,3,77],[2,0,25],[2,1,98],[2,2,89],[2,3,89],[3,0,58],[3,1,91],[3,2,51],[3,3,99]],dtype=np.int64)\n'
               'factors=np.array([[1,1,0],[2,0,1],[4,1,1],[5,2,100]],dtype=np.int64)\n'
               'powers=np.ones(4,dtype=np.int64)\n'
               'pairs=np.array([[0,1],[1,2],[0,3],[2,3]],dtype=np.int64)\n'
               'p=101\n'
               'def _expect_value_error(function, arguments):\n'
               '    try:\n'
               '        function(*arguments)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'heldout=np.array([[100,0]],dtype=np.int64)\n'
               'arguments=(samples,factors,powers,pairs,p,heldout)',
      'call': '_expect_value_error(run_pfd_pipeline, deepcopy(arguments))',
      'gold_call': '_expect_value_error(_oracle_run_pfd_pipeline, deepcopy(arguments))'}]
