"""
Construct the reaction expression panel.

Reaction-level expression weights describe independent candidate and perturbation axes. Each scenario is applied to the original candidate state.

Returns
-------
A positive finite ndarray of shape (C,P,N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def perturbed_weights(weights: "np.ndarray", factors: "np.ndarray") -> "np.ndarray":
    """
    weights is a positive finite (C,N) array and factors is a positive finite (P,N)
    array. C, P and N are positive. Return the aligned (C,P,N) reaction-weight panel.
    Each factor is relative to the unperturbed row. Invalid shapes, nonpositive values,
    nonfinite values or product overflow raise ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_perturbed_weights(
    weights: "np.ndarray", factors: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    g = np.asarray(weights, dtype=float)
    f = np.asarray(factors, dtype=float)
    if (
        g.ndim != 2
        or f.ndim != 2
        or min(g.shape + f.shape) == 0
        or (g.shape[1] != f.shape[1])
        or (not np.isfinite(g).all())
        or (not np.isfinite(f).all())
        or np.any(g <= 0)
        or np.any(f <= 0)
    ):
        raise ValueError("Positive finite aligned matrices required")
    result = g[:, None, :] * f[None, :, :]
    if not np.isfinite(result).all():
        raise ValueError("Product overflow")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "g = np.array([[2.0, 3.0], [5.0, 7.0]])\n"
            "f = np.array([[1.0, 1.0], [0.2, 2.0]])",
            "call": "perturbed_weights(g,f)",
            "gold_call": "_oracle_perturbed_weights(g,f)",
        },
        {
            "setup": "import numpy as np\ng = np.array([[4.0]])\nf = np.array([[1.0]])",
            "call": "perturbed_weights(g,f)",
            "gold_call": "_oracle_perturbed_weights(g,f)",
        },
        {
            "setup": "import numpy as np\n"
            "g = np.array([[1e-08, 100000000.0]])\n"
            "f = np.array([[100000000.0, 1e-08], [0.5, 0.25]])",
            "call": "perturbed_weights(g,f)",
            "gold_call": "_oracle_perturbed_weights(g,f)",
        },
        {
            "setup": "import numpy as np\n"
            "g = np.ones((2, 3))\n"
            "f = np.ones((1, 2))\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(perturbed_weights, (g,f,))",
            "gold_call": "_error_check(_oracle_perturbed_weights, (g,f,))",
            "tol": 0.0,
        },
    ]
