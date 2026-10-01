"""
Given a finite, real, nonnegative, irreducible square matrix, return its Perron root, the positive right eigenvector belonging to it normalised to sum to one, and the positive left eigenvector belonging to it normalised so that its inner product with the right eigenvector is one. Select the Perron root as the eigenvalue of largest real part, never by largest modulus, so that the result is correct for imprimitive matrices.

Also report the number of eigenvalues whose modulus lies within a relative distance of 1e-8 of the Perron root. This count is one for a primitive matrix and equals the index of imprimitivity otherwise.

Test irreducibility directly from the pattern of nonzero entries, as reachability of every index from every other along directed paths, and reject a reducible matrix.

Every quantity in a matrix population model that survives a change of time scale is built from three objects: the asymptotic growth rate, which is the dominant eigenvalue of the projection matrix; the stable age distribution, which is the right eigenvector belonging to it; and the reproductive values, which form the left eigenvector belonging to it. For a nonnegative irreducible matrix the Perron-Frobenius theorem guarantees that the spectral radius is itself a simple, real, positive eigenvalue and that both eigenvectors belonging to it can be taken strictly positive. A Leslie matrix is irreducible exactly when every survival probability is positive and the oldest class has positive fertility.

Irreducible is weaker than primitive, and the difference matters here. A primitive matrix has a single eigenvalue of largest modulus. An imprimitive one, which is what arises whenever reproduction can occur only at ages sharing a common divisor greater than one, has several eigenvalues of the same largest modulus spaced evenly around a circle, and only one of them is real and positive. Selecting the dominant eigenvalue by modulus is then ambiguous and can return a complex member of that circle, and power iteration does not converge at all but cycles. The Perron root is nevertheless unambiguous in every case: no other eigenvalue has a real part as large as the spectral radius, because an eigenvalue with that real part and modulus no larger than the radius is the radius itself.

The two eigenvectors are fixed up to scale only, and two scales are conventional in demography: the stable distribution sums to one, and the reproductive values are then scaled so that their inner product with the stable distribution is one. Both are used downstream, and nothing computed later depends on the individual scales, only on this pairing.

Returns
-------
dict holding the float growth_rate, the Perron root; the np.ndarray stable_distribution of shape (N,), positive and summing to one; the np.ndarray reproductive_values of shape (N,), positive, with inner product one against the stable distribution; and the integer peripheral_count, the number of eigenvalues on the spectral circle.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def perron_triplet(matrix: np.ndarray) -> dict:
    """Perron root, stable distribution and reproductive values of an irreducible matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Finite real nonnegative irreducible square array, shape (N, N).

    Returns
    -------
    dict
        Under the keys growth_rate, stable_distribution, reproductive_values and
        peripheral_count.

    Raises
    ------
    ValueError
        When the matrix is not a finite real square array, has a negative entry, or is
        reducible.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_nonnegative_square(matrix, name):
    """Validate a finite real nonnegative square matrix and return it as a float array."""
    a = np.asarray(matrix)
    if np.iscomplexobj(a):
        raise ValueError(name + " must be real")
    a = a.astype(float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        raise ValueError(name + " must be a square array of size at least one")
    if not np.all(np.isfinite(a)):
        raise ValueError(name + " must be finite")
    if np.any(a < 0.0):
        raise ValueError(name + " must be nonnegative")
    return a


def _is_irreducible(a):
    """Reachability of every index from every other along the nonzero pattern."""
    size = a.shape[0]
    reach = (a > 0.0) | np.eye(size, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(size, 2)))) + 1):
        reach = (reach.astype(int) @ reach.astype(int)) > 0
    return bool(np.all(reach))


def _positive_eigenvector(vector):
    """Remove the arbitrary complex phase of a Perron eigenvector and make it positive."""
    x = np.asarray(vector)
    x = x / x[int(np.argmax(np.abs(x)))]
    x = x.real
    if np.min(x) <= 0.0:
        raise ValueError("the Perron eigenvector is not strictly positive; the matrix is ill conditioned")
    return x


def _oracle_perron_triplet(matrix: np.ndarray) -> dict:
    """Reference implementation."""
    a = _check_nonnegative_square(matrix, "matrix")
    if not _is_irreducible(a):
        raise ValueError("matrix must be irreducible")

    values, vectors = np.linalg.eig(a)
    index = int(np.argmax(values.real))
    root = float(values[index].real)
    right = _positive_eigenvector(vectors[:, index])

    left_values, left_vectors = np.linalg.eig(a.T)
    left_index = int(np.argmax(left_values.real))
    left = _positive_eigenvector(left_vectors[:, left_index])

    stable = right / right.sum()
    reproductive = left / float(left @ stable)
    peripheral = int(np.sum(np.abs(values) >= root * (1.0 - 1e-8)))
    return {"growth_rate": root,
            "stable_distribution": stable,
            "reproductive_values": reproductive,
            "peripheral_count": peripheral}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n\ndef resolved(fertility, survival, subdivisions):\n    classes = len(fertility)\n    matrix = np.zeros((classes * subdivisions, classes * subdivisions))\n    for age_class in range(1, classes + 1):\n        matrix[0, age_class * subdivisions - 1] = fertility[age_class - 1]\n    for index in range(1, classes * subdivisions):\n        matrix[index, index - 1] = survival[index // subdivisions - 1] if index % subdivisions == 0 else 1.0\n    return matrix\n"
    return [
        {
            "setup": setup + "",
            "call": "project(perron_triplet(leslie(F, P)))",
            "gold_call": "project(_oracle_perron_triplet(leslie(F, P)))",
        },
        {
            "setup": setup + "",
            "call": "project(perron_triplet(resolved(F, P, 3)))",
            "gold_call": "project(_oracle_perron_triplet(resolved(F, P, 3)))",
        },
        {
            "setup": setup + "SEMEL = leslie([0.0, 0.0, 0.0, 8.0], [0.5, 0.5, 0.5])\n",
            "call": "project(perron_triplet(SEMEL))",
            "gold_call": "project(_oracle_perron_triplet(SEMEL))",
        },
        {
            "setup": setup + "BAD = leslie([0.0, 1.0, 0.0], [0.5, 0.5])\n",
            "call": "expect_value_error(perron_triplet, BAD)",
            "gold_call": "expect_value_error(_oracle_perron_triplet, BAD)",
        },
    ]
