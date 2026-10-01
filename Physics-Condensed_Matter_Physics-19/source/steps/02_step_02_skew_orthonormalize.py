"""
Reduce one new vector against a partially built symplectic basis so that it becomes skew-orthogonal to every complete pair, either opening a new pair or completing the unpaired last vector.

Skew-orthogonal polynomial bases come in pairs, and the gauge of each pair is fixed by a normalisation choice: here the vector opening a pair keeps its scale and the vector closing a pair keeps its component along its partner.

Returns
-------
tuple[np.ndarray, np.ndarray]: the new vector v and the coefficients h.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def skew_orthonormalize(
    basis: np.ndarray,
    vector: np.ndarray,
    gram: np.ndarray,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Skew-orthonormalize one vector against a partially built symplectic basis.

    The skew product is ``<f, g> = f @ gram @ g``. The ``n`` columns
    ``s_1, ..., s_n`` of ``basis`` (in this 1-based labelling ``s_i`` is
    ``basis[:, i - 1]``) are grouped into consecutive complete pairs
    ``(s_1, s_2), (s_3, s_4), ...`` with ``<s_{2k-1}, s_{2k}> = 1`` and zero
    skew product between columns of different pairs; when ``n`` is odd the
    last column ``s_n`` is unpaired and has zero skew product with every
    earlier column. Return ``(v, h)`` with ``h`` of length ``n + 1`` such that
    ``vector = sum_{i=1}^{n} h_i s_i + h_{n+1} v`` and ``v`` has zero skew
    product with every column of every complete pair, where

    * if ``n`` is even, ``h_{n+1} = 1`` (``v`` opens a new pair and is not
      rescaled);
    * if ``n`` is odd, ``h_n = 0`` and ``<s_n, v> = 1`` (``v`` completes the
      pair of ``s_n``).

    The reduction against the complete pairs is applied in passes, and a new
    pass is made while the previous pass shrank the Euclidean norm of the
    working vector below ``eta`` times its value before that pass; at least
    one pass is always made. The inputs are not modified.

    Parameters
    ----------
    basis : np.ndarray
        Array of shape ``(L, n)`` with ``n >= 0`` columns as described above.
    vector : np.ndarray
        Array of shape ``(L,)`` to be reduced.
    gram : np.ndarray
        Real skew-symmetric array of shape ``(L, L)`` defining the skew product.
    eta : float
        Reorthogonalization threshold, ``0 < eta <= 1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(v, h)``: the new basis vector of shape ``(L,)`` and the coefficient
        vector of shape ``(n + 1,)``, ``h[i - 1]`` holding ``h_i``.

    Raises
    ------
    ValueError
        If ``basis`` is not two-dimensional, ``vector`` is not one-dimensional
        with the same length ``L`` as the columns of ``basis``, ``gram`` does
        not have shape ``(L, L)``, any entry is non-finite, ``eta`` is not a
        real number with ``0 < eta <= 1`` (booleans are rejected), or ``n`` is
        odd and ``<s_n, vector>`` is zero, so that the pair cannot be
        completed.
    """
    return v, h

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_skew_orthonormalize(
    basis: np.ndarray,
    vector: np.ndarray,
    gram: np.ndarray,
    eta: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation (modified symplectic Gram-Schmidt, iterated reorthogonalization, ESR3m)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    columns = np.asarray(basis, dtype=float)
    work = np.array(vector, dtype=float)
    form = np.asarray(gram, dtype=float)
    if columns.ndim != 2 or work.ndim != 1 or columns.shape[0] != work.shape[0]:
        raise ValueError("basis must be (L, n) and vector (L,)")
    size, count = columns.shape
    if form.shape != (size, size):
        raise ValueError("gram must have shape (L, L)")
    if not (np.all(np.isfinite(columns)) and np.all(np.isfinite(work))
            and np.all(np.isfinite(form))):
        raise ValueError("inputs must be finite")
    if not (_is_number(eta) and 0.0 < eta <= 1.0):
        raise ValueError("eta must satisfy 0 < eta <= 1")
    coefficients = np.zeros(count + 1)
    previous = np.inf
    passes = 0
    while np.linalg.norm(work) < eta * previous and passes < 100:
        previous = np.linalg.norm(work)
        passes += 1
        for k in range(count // 2):
            first, second = columns[:, 2 * k], columns[:, 2 * k + 1]
            along_first = -(second @ form @ work)
            along_second = first @ form @ work
            coefficients[2 * k] += along_first
            coefficients[2 * k + 1] += along_second
            work = work - along_first * first - along_second * second
    # ESR3m: r11 = 1 opens a pair unscaled; r12 = 0 keeps the partner component.
    if count % 2 == 1:
        scale = columns[:, count - 1] @ form @ work
        if scale == 0.0 or not np.isfinite(scale):
            raise ValueError("the unpaired column cannot be completed by this vector")
    else:
        scale = 1.0
    coefficients[count] = scale
    return work / scale, coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    build = (
        "import numpy as np\n"
        "def _form(q, n):\n"
        "    x = np.arange(n, dtype=float)\n"
        "    w = q ** (0.5 * x)\n"
        "    return 0.5 * np.sign(x[None, :] - x[:, None]) * np.outer(w, w)\n"
        "def _generic(n, seed):\n"
        "    a = np.random.default_rng(seed).normal(size=(n, n))\n"
        "    return a - a.T\n"
        "def _pairs(G, raw):\n"
        "    cols = []\n"
        "    for r in raw.T:\n"
        "        v = np.array(r, dtype=float)\n"
        "        for k in range(len(cols) // 2):\n"
        "            a, b = cols[2 * k], cols[2 * k + 1]\n"
        "            v = v + (b @ G @ v) * a - (a @ G @ v) * b\n"
        "        if len(cols) % 2:\n"
        "            v = v / (cols[-1] @ G @ v)\n"
        "        cols.append(v)\n"
        "    return np.array(cols).T.reshape(G.shape[0], len(cols))\n"
        "def _red(out):\n"
        "    v, h = out\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    h = np.asarray(h, dtype=float)\n"
        "    k = np.arange(v.size)\n"
        "    j = np.arange(h.size)\n"
        "    return float(np.sum(v * np.cos(0.7 * k)) + 3.0 * np.sum(h * np.sin(1.3 * j + 0.2))\n"
        "                 + 10.0 * v.size + 100.0 * h.size)\n"
        "rng = np.random.default_rng(5)\n"
        "raw = rng.normal(size=(10, 6))\n"
        "G = _form(0.4, 10)\n"
    )
    preserve = (
        'import copy\n'
        'def _preserved(fn, *args, **kwargs):\n'
        '    saved_args = copy.deepcopy(args)\n'
        '    saved_kwargs = copy.deepcopy(kwargs)\n'
        '    result = fn(*args, **kwargs)\n'
        '    for actual, before in list(zip(args, saved_args)) + [(kwargs[k], saved_kwargs[k]) for k in kwargs]:\n'
        '        if isinstance(actual, np.ndarray):\n'
        "            assert np.array_equal(actual, before, equal_nan=True), 'Array input was modified'\n"
        '        else:\n'
        "            assert actual == before, 'Input was modified'\n"
        '    return result\n'
    )
    return [
        {
            "setup": (build + "B = _pairs(G, raw[:, :2])\nv0 = raw[:, 2].copy()\n") + preserve,
            "call": "_red(_preserved(skew_orthonormalize, B.copy(), v0.copy(), G.copy()))",
            "gold_call": "_red(_preserved(_oracle_skew_orthonormalize, B.copy(), v0.copy(), G.copy()))",
        },
        {
            "setup": (build + "B = _pairs(G, raw[:, :3])\nv0 = raw[:, 3].copy()\n") + preserve,
            "call": "_red(_preserved(skew_orthonormalize, B.copy(), v0.copy(), G.copy()))",
            "gold_call": "_red(_preserved(_oracle_skew_orthonormalize, B.copy(), v0.copy(), G.copy()))",
        },
        {
            "setup": (build + "B = np.zeros((10, 0))\nv0 = raw[:, 0].copy()\n") + preserve,
            "call": "_red(_preserved(skew_orthonormalize, B.copy(), v0.copy(), G.copy()))",
            "gold_call": "_red(_preserved(_oracle_skew_orthonormalize, B.copy(), v0.copy(), G.copy()))",
        },
        {
            "setup": (build + "H = _generic(10, 11)\nB = _pairs(H, raw[:, :5])\nv0 = raw[:, 5].copy()\n") + preserve,
            "call": "_red(_preserved(skew_orthonormalize, B.copy(), v0.copy(), H.copy(), eta=0.5))",
            "gold_call": "_red(_preserved(_oracle_skew_orthonormalize, B.copy(), v0.copy(), H.copy(), eta=0.5))",
        },
        {
            "setup": (build + "H = _generic(10, 3)\nB = _pairs(H, raw[:, :4])\nv0 = B @ np.array([0.3, -1.2, 2.0, 0.7]) + 1e-3 * raw[:, 4]\n") + preserve,
            "call": "_red(_preserved(skew_orthonormalize, B.copy(), v0.copy(), H.copy()))",
            "gold_call": "_red(_preserved(_oracle_skew_orthonormalize, B.copy(), v0.copy(), H.copy()))",
        },
        {
            "setup": (status + "e = np.zeros(6)\ne[0] = 1.0\nx = np.arange(6.0)\nG = 0.5 * np.sign(x[None, :] - x[:, None])\n") + preserve,
            "call": "_status(lambda: _preserved(skew_orthonormalize, e.reshape(6, 1), e.copy(), G))",
            "gold_call": "_status(lambda: _preserved(_oracle_skew_orthonormalize, e.reshape(6, 1), e.copy(), G))",
        },
        {
            "setup": (status + "G = np.zeros((4, 4))\n") + preserve,
            "call": "_status(lambda: _preserved(skew_orthonormalize, np.zeros((4, 0)), np.ones(4), G, eta=0.0))",
            "gold_call": "_status(lambda: _preserved(_oracle_skew_orthonormalize, np.zeros((4, 0)), np.ones(4), G, eta=0.0))",
        },
    ]
