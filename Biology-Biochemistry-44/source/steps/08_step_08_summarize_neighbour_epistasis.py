"""
Convert single and neighbouring double substitution log weights into log-scale epistasis values and summarize them over all genuine double substitutions.

Substitutions that act on independent parts of an ensemble multiply its total weight, so departures from multiplicativity measured on the logarithmic scale isolate the coupling that two neighbouring substitutions create through the alignments that use both bases.

Returns
-------
np.ndarray: [mean epistasis, extreme epistasis, 1-based position, code at i, code at i + 1, positive fraction, count].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def summarize_neighbour_epistasis(
    x: "np.ndarray",
    single_log_weights: "np.ndarray",
    double_log_weights: "np.ndarray",
) -> "np.ndarray":
    """Return summary statistics of log-scale epistasis between neighbouring substitutions.

    ``single_log_weights[i, c]`` is ``ln Z`` after base ``i`` of ``x`` is set
    to code ``c`` and ``double_log_weights[i, c, d]`` is ``ln Z`` after bases
    ``i`` and ``i + 1`` are set to ``c`` and ``d``, as returned by
    ``evaluate_single_substitution_log_weights`` and
    ``compute_neighbour_double_log_weights``; the reference value
    ``ln Z(x)`` is ``single_log_weights[0, x[0]]``. For every ``i`` from 0 to
    ``L - 2`` and every ``c != x[i]`` and ``d != x[i + 1]``, the epistasis is
    ``double[i, c, d] + ln Z(x) - single[i, c] - single[i + 1, d]``. Only
    these ``9 (L - 1)`` genuine double substitutions enter the summary. The
    largest absolute epistasis is resolved in favour of the earliest entry in
    ``(i, c, d)`` order when values are exactly equal.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional integer array of codes 0 to 3 with ``L >= 2``.
    single_log_weights : np.ndarray
        Shape ``(L, 4)``, finite.
    double_log_weights : np.ndarray
        Shape ``(L - 1, 4, 4)``, finite.

    Returns
    -------
    np.ndarray
        Float array ``[mean, extreme, position, code_i, code_next,
        positive_fraction, count]``: the mean epistasis; the epistasis with
        the largest absolute value, its 1-based position ``i + 1`` and the
        codes ``c`` and ``d``; the fraction of strictly positive values; and
        the number of genuine double substitutions.

    Raises
    ------
    ValueError
        If ``x`` is not a one-dimensional integer array of codes 0 to 3 with
        at least two bases, if an array has the wrong shape or a non-finite
        entry, or if the unsubstituted entries ``single[i, x[i]]`` and
        ``double[i, x[i], x[i + 1]]`` differ from ``ln Z(x)`` by more than
        ``1e-8 * max(1, |ln Z(x)|)``.
    """
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_summarize_neighbour_epistasis(
    x: "np.ndarray",
    single_log_weights: "np.ndarray",
    double_log_weights: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation on the natural-log scale."""
    import numpy as np

    codes = np.asarray(x)
    if (codes.ndim != 1 or codes.size < 2 or not np.issubdtype(codes.dtype, np.integer)
            or codes.min() < 0 or codes.max() > 3):
        raise ValueError("x must be a 1-D integer array of codes 0 to 3 with at least two bases")
    n = codes.size
    single = np.asarray(single_log_weights, dtype=float)
    double = np.asarray(double_log_weights, dtype=float)
    if single.shape != (n, 4) or double.shape != (n - 1, 4, 4):
        raise ValueError("log weights must have shapes (L, 4) and (L - 1, 4, 4)")
    if not (np.all(np.isfinite(single)) and np.all(np.isfinite(double))):
        raise ValueError("log weights must be finite")
    positions = np.arange(n)
    log_z = single[0, codes[0]]
    reference = np.concatenate([single[positions, codes],
                                double[positions[:-1], codes[:-1], codes[1:]]])
    if np.max(np.abs(reference - log_z)) > 1e-8 * max(1.0, abs(log_z)):
        raise ValueError("unsubstituted entries disagree with the reference total weight")

    epistasis = double + log_z - single[:-1, :, None] - single[1:, None, :]
    genuine = np.ones(epistasis.shape, dtype=bool)
    genuine[positions[:-1], codes[:-1], :] = False
    genuine[positions[:-1], :, codes[1:]] = False
    values = epistasis[genuine]
    flat = np.argmax(np.where(genuine, np.abs(epistasis), -1.0))
    i, c, d = np.unravel_index(flat, epistasis.shape)
    return np.array([values.mean(), epistasis[i, c, d], i + 1, c, d,
                     np.mean(values > 0.0), values.size], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    model = (
        "import math\n"
        "import numpy as np\n"
        "M = np.array([[0.16, 0.03, 0.05, 0.03], [0.03, 0.16, 0.03, 0.05],\n"
        "              [0.05, 0.03, 0.16, 0.03], [0.03, 0.05, 0.03, 0.16]])\n"
        "EX = np.array([0.12, 0.38, 0.38, 0.12])\n"
        "EY = np.array([0.16, 0.34, 0.34, 0.16])\n"
        "def _lae(*v):\n"
        "    top = max(v)\n"
        "    return top if top == -math.inf else top + math.log(sum(math.exp(t - top) for t in v))\n"
        "def _log_z(x, y, em, ex, ey, go, ge):\n"
        "    n, m = len(x), len(y)\n"
        "    lo, le = math.log(go), math.log(ge)\n"
        "    F = np.full((3, n + 1, m + 1), -math.inf)\n"
        "    F[0, 0, 0] = 0.0\n"
        "    for i in range(n + 1):\n"
        "        for j in range(m + 1):\n"
        "            if i and j:\n"
        "                F[0, i, j] = math.log(em[x[i-1], y[j-1]]) + _lae(*F[:, i-1, j-1])\n"
        "            if i:\n"
        "                F[1, i, j] = math.log(ex[x[i-1]]) + _lae(lo + F[0, i-1, j], le + F[1, i-1, j])\n"
        "            if j:\n"
        "                F[2, i, j] = math.log(ey[y[j-1]]) + _lae(lo + F[0, i, j-1], le + F[2, i, j-1])\n"
        "    return _lae(*F[:, n, m])\n"
        "def _rerun(x, y, *model):\n"
        "    n = len(x)\n"
        "    S = np.empty((n, 4))\n"
        "    D = np.empty((n - 1, 4, 4))\n"
        "    for i in range(n):\n"
        "        for c in range(4):\n"
        "            z = x.copy(); z[i] = c; S[i, c] = _log_z(z, y, *model)\n"
        "    for i in range(n - 1):\n"
        "        for c in range(4):\n"
        "            for d in range(4):\n"
        "                z = x.copy(); z[i] = c; z[i + 1] = d; D[i, c, d] = _log_z(z, y, *model)\n"
        "    return S, D\n"
        "def _pin(summary):\n"
        "    s = np.asarray(summary, dtype=float)\n"
        "    if s.shape != (7,):\n"
        "        return -1.0\n"
        "    w = np.array([1.0e6, 1.0e4, 1.0, 1.0, 1.0, 1.0e3, 1.0e-2])\n"
        "    return float(np.sum(w * s))\n"
    )
    instances = [
        ("x = np.array([0, 2, 1, 3, 3, 0]); y = np.array([0, 2, 2, 3, 0, 1])\n"
         "S, D = _rerun(x, y, M, EX, EY, 0.04, 0.4)\n"),
        ("x = np.array([2, 2, 1, 0]); y = np.array([2, 1, 1, 0, 3])\n"
         "S, D = _rerun(x, y, M * 1e-150, EX * 1e-150, EY * 1e-150, 1e-120, 0.5)\n"),
        ("x = np.array([3, 1]); y = np.array([1, 3, 3])\n"
         "S, D = _rerun(x, y, M, np.array([0.05, 0.9, 0.4, 0.02]), EY, 0.2, 0.9)\n"),
        ("x = np.array([0, 0, 0])\n"
         "S = np.zeros((3, 4))\n"
         "D = np.zeros((2, 4, 4)); D[0, 1, 2] = 0.5; D[1, 3, 1] = -0.5; D[0, 2, 3] = 0.25\n"),
    ]
    cases = [
        {  # Normal epistasis scan from pair-HMM weights.
            "setup": model + instances[0],
            "call": "_pin(summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
            "gold_call": "_pin(_oracle_summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
        },
        {  # Numerically extreme but finite weights.
            "setup": model + instances[1],
            "call": "_pin(summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
            "gold_call": "_pin(_oracle_summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
        },
        {  # Boundary: the shortest sequence with one neighbouring pair.
            "setup": model + instances[2],
            "call": "_pin(summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
            "gold_call": "_pin(_oracle_summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
        },
        {  # Edge: synthetic positive, negative, and zero interactions.
            "setup": model + instances[3],
            "call": "_pin(summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
            "gold_call": "_pin(_oracle_summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy()))",
        },
    ]
    cases.append({
        "setup": model + instances[0] + "S[2, x[2]] += 1.0e-3\n" + (
            "def _candidate():\n"
            "    try:\n"
            "        summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy())\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "def _reference():\n"
            "    try:\n"
            "        _oracle_summarize_neighbour_epistasis(x.copy(), S.copy(), D.copy())\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"),
        "call": "_candidate()",
        "gold_call": "_reference()",
    })
    return cases
