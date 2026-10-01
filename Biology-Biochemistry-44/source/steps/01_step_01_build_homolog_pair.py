"""
Generate a coded reference sequence and a homolog derived from it by seeded random deletions and substitutions.

Homologous nucleic acids descend from a common ancestor through point substitutions and insertion-deletion events, so a reproducible synthetic homolog supplies an alignment ensemble that contains both mismatch and gap structure.

Returns
-------
tuple[np.ndarray, np.ndarray]: coded reference x of the given length and its homolog y.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_homolog_pair(
    seed: int,
    length: int,
    delete_probability: float,
    substitute_probability: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Return a coded reference sequence ``x`` and its synthetic homolog ``y``.

    Bases are coded 0, 1, 2, 3 for A, C, G, U. A generator
    ``numpy.random.default_rng(seed)`` makes exactly three draws in this
    order: ``x = integers(0, 4, length)``, then ``u = random(length)``, then
    ``v = integers(0, 4, length)``. Visiting positions ``i = 0, ...,
    length - 1`` in order, ``y`` omits base ``i`` when ``u[i] <
    delete_probability``, writes ``v[i]`` when ``delete_probability <= u[i] <
    delete_probability + substitute_probability`` (which may equal ``x[i]``)
    and writes ``x[i]`` otherwise.

    Parameters
    ----------
    seed : int
        Nonnegative integer seed.
    length : int
        Positive number of bases in ``x``.
    delete_probability : float
        Probability in ``[0, 1]`` that a base is omitted from ``y``.
    substitute_probability : float
        Probability in ``[0, 1]`` that a base is redrawn; the two
        probabilities sum to at most 1.

    Returns
    -------
    tuple of np.ndarray
        ``(x, y)`` as one-dimensional ``int64`` arrays of codes.

    Raises
    ------
    ValueError
        If ``seed`` is not a nonnegative integer, ``length`` is not a
        positive integer (booleans are rejected for both), a probability is
        not a finite number in ``[0, 1]``, the probabilities sum to more
        than 1, or ``y`` would be empty.
    """
    return x, y

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_homolog_pair(
    seed: int,
    length: int,
    delete_probability: float,
    substitute_probability: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference implementation: three ordered draws, then one masked copy."""
    import numpy as np

    for name, value, low in (("seed", seed, 0), ("length", length, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}")
    probabilities = []
    for name, value in (("delete_probability", delete_probability),
                        ("substitute_probability", substitute_probability)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a number")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
        probabilities.append(value)
    delete, substitute = probabilities
    if delete + substitute > 1.0:
        raise ValueError("the deletion and substitution probabilities sum to more than 1")

    rng = np.random.default_rng(int(seed))
    x = rng.integers(0, 4, int(length)).astype(np.int64)
    u = rng.random(int(length))
    v = rng.integers(0, 4, int(length)).astype(np.int64)
    redrawn = np.where(u < delete + substitute, v, x)
    y = redrawn[u >= delete].astype(np.int64)
    if y.size == 0:
        raise ValueError("every base was deleted, so the homolog is empty")
    return x, y

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    pin = (
        "import numpy as np\n"
        "def _pin(pair):\n"
        "    x, y = (np.asarray(s) for s in pair)\n"
        "    if x.ndim != 1 or y.ndim != 1 or not np.issubdtype(x.dtype, np.integer) \\\n"
        "            or not np.issubdtype(y.dtype, np.integer):\n"
        "        return -1.0\n"
        "    rx = np.sqrt(np.arange(1.0, x.size + 1.0))\n"
        "    ry = np.cos(0.7 * np.arange(1.0, y.size + 1.0))\n"
        "    return float(1000.0 * x.size + y.size + np.sum((x + 1.0) * rx) / 100.0\n"
        "                 + np.sum((y + 1.0) * ry) / 10.0)\n"
    )

    def _status(args):
        return (
            "def _candidate():\n"
            "    try:\n"
            f"        build_homolog_pair({args})\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "def _reference():\n"
            "    try:\n"
            f"        _oracle_build_homolog_pair({args})\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
        )

    cases = [
        {  # Normal stochastic mutation/deletion regime.
            "setup": pin,
            "call": "_pin(build_homolog_pair(3, 40, 0.06, 0.16))",
            "gold_call": "_pin(_oracle_build_homolog_pair(3, 40, 0.06, 0.16))",
        },
        {  # Longer sequence with larger event probabilities.
            "setup": pin,
            "call": "_pin(build_homolog_pair(20260914, 64, 0.2, 0.5))",
            "gold_call": "_pin(_oracle_build_homolog_pair(20260914, 64, 0.2, 0.5))",
        },
        {  # Boundary: no substitutions or deletions.
            "setup": pin,
            "call": "_pin(build_homolog_pair(7, 25, 0.0, 0.0))",
            "gold_call": "_pin(_oracle_build_homolog_pair(7, 25, 0.0, 0.0))",
        },
        {  # Edge: one residue and certain substitution.
            "setup": pin,
            "call": "_pin(build_homolog_pair(11, 1, 0.0, 1.0))",
            "gold_call": "_pin(_oracle_build_homolog_pair(11, 1, 0.0, 1.0))",
        },
        {  # Edge: deletion and substitution thresholds coincide.
            "setup": pin,
            "call": "_pin(build_homolog_pair(5, 30, 0.5, 0.5))",
            "gold_call": "_pin(_oracle_build_homolog_pair(5, 30, 0.5, 0.5))",
        },
    ]
    for args in ("4, 10, 0.7, 0.4", "4, 10, 1.0, 0.0"):
        cases.append({"setup": _status(args), "call": "_candidate()", "gold_call": "_reference()"})
    return cases
