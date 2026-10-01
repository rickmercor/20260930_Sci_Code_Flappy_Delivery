"""
Return the four information quantities of the source's Eqs. (10)-(13) for a resource matrix, as the array [H(X), H(Y), H(XY), H_Y(X)]: the entropy of the resource states, the entropy of the species, their joint entropy, and the conditional entropy of the resource states given the species, all in nats, computed from the pooled-community probabilities of the source's Eqs. (5), (7) and (9).

The source's niche framework is information-theoretic: the resource matrix defines a joint distribution of species and resource states whose marginal, joint and conditional entropies measure how strongly resource use is structured by species identity.



Returns

-------

numpy.ndarray of float64 with shape (4,): [H(X), H(Y), H(XY), H_Y(X)] in nats.

Returns
-------
return entropies
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resource_entropies(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """Return the four information quantities of the source's Eqs. (10)-(13) for a resource matrix, as the array [H(X), H(Y), H(XY), H_Y(X)]: the entropy of the resource states, the entropy of the species, their joint entropy, and the conditional entropy of the resource states given the species, all in nats, computed from the pooled-community probabilities of the source's Eqs. (5), (7) and (9).

    Parameters
    ----------
    resource_matrix : numpy.ndarray
        Array of shape (s, r) of finite, nonnegative abundances: s species by r resource states.

    Returns
    -------
    entropies : numpy.ndarray
        Array of shape (4,): [H(X), H(Y), H(XY), H_Y(X)] in nats (float64).

    Raises
    ------
    ValueError
        If resource_matrix is not two-dimensional, contains a negative or non-finite entry, or has no positive entry.
    """
    return entropies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _xlogx(a):
    """x ln x with the convention 0 ln 0 = 0 (vectorised)."""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, a * np.log(np.where(a > 0.0, a, 1.0)), 0.0)


def _oracle_resource_entropies(resource_matrix: "numpy.ndarray") -> "numpy.ndarray":
    """[H(X), H(Y), H(XY), H_Y(X)] of Eqs 10-13 in nats, from the pooled probabilities of Eqs 5, 7 and 9."""
    N = np.asarray(resource_matrix, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("resource_matrix must be a 2-D array with at least one species and one state")
    if not np.all(np.isfinite(N)) or np.any(N < 0.0):
        raise ValueError("resource_matrix entries must be finite and nonnegative")
    if N.sum() <= 0.0:
        raise ValueError("resource_matrix must contain at least one positive entry")
    Z = N.sum()
    P = N.sum(axis=0) / Z                                   # marginal of the resource states
    Q = N.sum(axis=1) / Z                                   # marginal of the species
    pi = N / Z
    HX = -_xlogx(P).sum()
    HY = -_xlogx(Q).sum()
    HXY = -_xlogx(pi).sum()
    return np.array([HX, HY, HXY, HXY - HY])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[7, 39, 11, 9, 0, 6, 0], [39, 0, 25, 0, 31, 4, 0], [0, 0, 0, 21, 37, 10, 0],\n                            [0, 0, 0, 0, 14, 0, 0], [0, 0, 0, 37, 0, 0, 0], [0, 38, 0, 0, 0, 0, 0]], dtype=float)\n",
            "call": "resource_entropies(resource_matrix)",
            "gold_call": "_oracle_resource_entropies(resource_matrix)",
        },
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[50, 30, 10], [10, 30, 50], [5, 30, 55]], dtype=float)\n",
            "call": "resource_entropies(resource_matrix)",
            "gold_call": "_oracle_resource_entropies(resource_matrix)",
        },
        {
            "setup": "import numpy as np\nresource_matrix = np.array([[30, 30, 30, 0], [10, 30, 50, 0], [5, 0, 55, 0], [30, 30, 30, 0]], dtype=float)\n",
            "call": "resource_entropies(resource_matrix)",
            "gold_call": "_oracle_resource_entropies(resource_matrix)",
        },
        {
            "setup": "import numpy as np\nresource_matrix = np.zeros((3, 4))\ndef run_model():\n    try:\n        resource_entropies(resource_matrix)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_resource_entropies(resource_matrix)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
