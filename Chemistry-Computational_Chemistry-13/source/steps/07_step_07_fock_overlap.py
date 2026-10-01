"""
Extract normalized overlap coefficients for repeated vibrational occupations.

Higher vibrational occupations require repeated-index Gaussian contractions. Pair contractions, displacement terms, and occupation factorials all contribute.

Returns
-------
overlaps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fock_overlap(B, b, vacuum_overlap, bra_labels, ket_labels):
    r"""For F(z)=G*exp(z.T B z/2+b.T z), return the matrix
    O[k,l]=G*[partial_z**(k,l) exp(z.T B z/2+b.T z)] at z=0 / sqrt(k!*l!).
    Here z consists of D bra variables followed by D ket variables. Multi-index
    factorials mean products of ordinary factorials. Do not take absolute values.
    The zero-occupation derivative is one. Label order and duplicates are preserved.

    Parameters
    ----------
    B : finite complex symmetric array, shape (2D,2D), D>=1
    b : finite complex array, shape (2D,)
    vacuum_overlap : finite complex scalar G
    bra_labels, ket_labels : nonnegative integer-valued arrays, shapes (L,D),(R,D)
        Each row has total occupation <=8. Empty arrays of shape (0,D) are allowed.

    Returns
    -------
    overlaps : complex array, shape (L,R)

    Raises
    ------
    ValueError
        For incompatible or nonfinite data; B symmetry error above 1e-10;
        negative/noninteger occupations; or a row total above eight.
    """
    return overlaps

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fock_overlap(B, b, vacuum_overlap, bra_labels, ket_labels):
    import math
    import numpy as np
    B = np.asarray(B, dtype=complex)
    b = np.asarray(b, dtype=complex)
    left = np.asarray(bra_labels)
    right = np.asarray(ket_labels)
    if (B.ndim != 2 or B.shape[0] == 0 or B.shape[0] != B.shape[1] or B.shape[0] % 2
            or b.shape != (B.shape[0],)
            or left.ndim != 2 or right.ndim != 2
            or left.shape[1] != B.shape[0] // 2 or right.shape[1] != left.shape[1]
            or not all(np.all(np.isfinite(a)) for a in (B, b, vacuum_overlap, left, right))
            or not np.allclose(B, B.T, atol=1e-10, rtol=0)
            or np.any(left < 0) or np.any(right < 0)
            or np.any(left != np.floor(left)) or np.any(right != np.floor(right))
            or (left.size and np.max(left.sum(axis=1)) > 8)
            or (right.size and np.max(right.sum(axis=1)) > 8)):
        raise ValueError("Invalid generating function or occupation labels")
    left, right = left.astype(int), right.astype(int)

    zero = (0,) * len(b)
    required = {zero}
    stack = [tuple(k) + tuple(ell) for k in left for ell in right]
    while stack:
        alpha = stack.pop()
        if alpha in required:
            continue
        required.add(alpha)
        i = next(j for j, count in enumerate(alpha) if count)
        beta = list(alpha)
        beta[i] -= 1
        stack.append(tuple(beta))
        for j, count in enumerate(beta):
            if count:
                gamma = beta.copy()
                gamma[j] -= 1
                stack.append(tuple(gamma))
    moments = {zero: 1.0 + 0j}
    for alpha in sorted(required, key=lambda k: (sum(k), k)):
        if alpha == zero:
            continue
        i = next(j for j, count in enumerate(alpha) if count)
        beta = list(alpha)
        beta[i] -= 1
        value = b[i] * moments[tuple(beta)]
        for j, count in enumerate(beta):
            if count:
                gamma = beta.copy()
                gamma[j] -= 1
                value += B[i, j] * count * moments[tuple(gamma)]
        moments[alpha] = value
    output = np.empty((len(left), len(right)), dtype=complex)
    for row, k in enumerate(left):
        for column, ell in enumerate(right):
            alpha = tuple(k) + tuple(ell)
            denominator = math.sqrt(math.prod(math.factorial(int(n)) for n in alpha))
            output[row, column] = vacuum_overlap * moments[alpha] / denominator
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
B=np.array([[.2+.1j,.1,-.4,.3j],[.1,-.2,.2j,.5],[-.4,.2j,.15,-.1],[.3j,.5,-.1,.3j]]);b=np.array([.2,-.3j,.1+.1j,-.2]);G=.7-.2j;left=np.array([[0,0],[2,1],[1,3],[4,2]]);right=np.array([[1,0],[0,0],[2,2]])
""",
            "call": 'fock_overlap(B, b, G, left, right)',
            "gold_call": '_oracle_fock_overlap(B, b, G, left, right)',
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
B=np.zeros((2,2),complex);b=np.zeros(2);G=.8+.2j;left=np.array([[0]]);right=left.copy()
""",
            "call": 'fock_overlap(B, b, G, left, right)',
            "gold_call": '_oracle_fock_overlap(B, b, G, left, right)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
B=np.eye(2);b=np.ones(2);G=1.;left=np.empty((0,1),int);right=np.array([[1],[0]])
""",
            "call": 'fock_overlap(B, b, G, left, right)',
            "gold_call": '_oracle_fock_overlap(B, b, G, left, right)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
B=np.array([[0.,1.],[1.,0.]]);b=np.zeros(2);G=1.;left=np.array([[8],[2],[8],[0],[3]]);right=np.array([[3],[8],[0],[2]])
""",
            "call": 'fock_overlap(B, b, G, left, right)',
            "gold_call": '_oracle_fock_overlap(B, b, G, left, right)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
B=np.array([[0.,1.],[.3,0.]]);b=np.zeros(2);G=1.;left=np.array([[0]]);right=left.copy()

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(fock_overlap, B, b, G, left, right)',
            "gold_call": '_status(_oracle_fock_overlap, B, b, G, left, right)',
        },
    ]
