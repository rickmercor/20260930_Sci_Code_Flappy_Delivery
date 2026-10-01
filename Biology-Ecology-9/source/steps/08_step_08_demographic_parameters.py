"""
Given a Leslie matrix and its projection interval, return its Perron root, its net reproductive rate, its generation time in the units of the interval, and its Demetrius entropy, as defined above. Take the reproductive values and stable distribution from the Perron triplet. Treat as zero any entry outside the first row and the subdiagonal whose magnitude does not exceed 1e-10 times the largest entry, and reject the matrix if any such entry is larger.

Comparisons between models of different resolution are made through demographic parameters that summarise the fertility schedule. For a Leslie matrix with fertilities F_1 to F_L and survival probabilities P_1 to P_(L-1), write l_1 = 1 and l_i for the product of P_1 to P_(i-1), the probability of surviving to class i, and phi_i = F_i l_i for the net maternity of class i. The characteristic equation of the matrix is the discrete Euler-Lotka equation, the sum of phi_i lambda^(-i) equal to one, so the terms p_i = phi_i lambda^(-i) are nonnegative and sum to one. They are the distribution of the age of parents of newborns in the stable population, measured in projection intervals.

Three parameters follow. The net reproductive rate R_0 is the sum of the phi_i, the expected lifetime number of offspring of a newborn; it equals one exactly when lambda does, but otherwise it is not fixed by the growth rate, because it counts offspring without discounting them by the time at which they are born. The generation time defined through the reproductive values is the projection interval multiplied by lambda and divided by the bilinear form of the reproductive values, the fertility matrix (the first row of the Leslie matrix with zeros below) and the stable distribution, the vectors scaled to unit inner product; for a Leslie matrix it equals the interval times the mean of i under the distribution p, the mean age of parents. Demetrius' entropy is the Shannon entropy of the same distribution, minus the sum of p_i log p_i in natural logarithms over the classes with p_i above zero; it is zero when all reproduction falls in one class and largest when it is spread evenly.

All three are unchanged by a positive diagonal similarity of the Leslie matrix, which rescales the fertilities and survival probabilities class by class while leaving every product phi_i intact. That invariance is why these parameters, unlike the individual entries of a reduced matrix, cannot distinguish reduced models that differ only by a change of units in their classes.

Returns
-------
dict holding the floats growth_rate, net_reproductive_rate, generation_time and entropy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def demographic_parameters(leslie_matrix: np.ndarray, interval: float) -> dict:
    """Growth rate, net reproductive rate, generation time and Demetrius entropy of a Leslie matrix.

    Parameters
    ----------
    leslie_matrix : np.ndarray
        Nonnegative irreducible Leslie matrix, shape (L, L).
    interval : float
        Projection interval, finite and above zero.

    Returns
    -------
    dict
        Under the keys growth_rate, net_reproductive_rate, generation_time and entropy.

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet or does not have the Leslie
        pattern, or when the interval is not finite and above zero.
        A bool or a non-numeric value is not accepted as a real number.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numbers

import numpy as np


def _oracle_demographic_parameters(leslie_matrix: np.ndarray, interval: float) -> dict:
    """Reference implementation."""
    triplet = _oracle_perron_triplet(leslie_matrix)  # noqa: F821
    if isinstance(interval, bool) or not isinstance(interval, numbers.Real):
        raise ValueError("interval must be a real number")
    dt = float(interval)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("interval must be finite and above zero")

    a = np.asarray(leslie_matrix, dtype=float)
    size = a.shape[0]
    pattern = np.zeros((size, size), dtype=bool)
    pattern[0, :] = True
    pattern[np.arange(1, size), np.arange(size - 1)] = True
    if np.any(np.abs(a[~pattern]) > 1e-10 * np.max(np.abs(a))):
        raise ValueError("leslie_matrix must be zero outside its first row and subdiagonal")

    lam = triplet["growth_rate"]
    fertility = a[0, :]
    survival = np.array([a[i + 1, i] for i in range(size - 1)])
    survivorship = np.concatenate([[1.0], np.cumprod(survival)])
    maternity = fertility * survivorship
    parents = maternity * lam ** (-np.arange(1.0, size + 1.0))

    v = triplet["reproductive_values"]
    w = triplet["stable_distribution"]
    fertility_matrix = np.zeros((size, size))
    fertility_matrix[0, :] = fertility
    generation = dt * lam * float(v @ w) / float(v @ fertility_matrix @ w)

    positive = parents[parents > 0.0]
    return {"growth_rate": lam,
            "net_reproductive_rate": float(maternity.sum()),
            "generation_time": generation,
            "entropy": float(-np.sum(positive * np.log(positive))) + 0.0}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n"
    return [
        {
            "setup": setup + "",
            "call": "project(demographic_parameters(leslie(F, P), 1.0))",
            "gold_call": "project(_oracle_demographic_parameters(leslie(F, P), 1.0))",
        },
        {
            "setup": setup + "",
            "call": "project(demographic_parameters(leslie([0.5, 2.4, 1.9], [0.45, 0.47]), 7.0 / 3.0))",
            "gold_call": "project(_oracle_demographic_parameters(leslie([0.5, 2.4, 1.9], [0.45, 0.47]), 7.0 / 3.0))",
        },
        {
            "setup": setup + "",
            "call": "project(demographic_parameters(leslie([0.0, 0.0, 0.0, 9.0], [0.5, 0.6, 0.7]), 2.0))",
            "gold_call": "project(_oracle_demographic_parameters(leslie([0.0, 0.0, 0.0, 9.0], [0.5, 0.6, 0.7]), 2.0))",
        },
        {
            "setup": setup + "",
            "call": "expect_value_error(demographic_parameters, leslie(F, P), 0.0)",
            "gold_call": "expect_value_error(_oracle_demographic_parameters, leslie(F, P), 0.0)",
        },
    ]
