"""
Construct spin-completed, size-consistent fragment and dimer determinant spaces.

Use the retrieved sampled-configuration construction with the supplied samples and fragment electron counts. No further sampling or determinant selection is performed.

Returns
-------
tuple (space_a, space_b, space_ab) of ascending unique np.int64 packed-determinant arrays using local orbital counts n_a, n_b and n_a+n_b.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def selected_spaces(n_a: int, n_b: int, n_e_a: int, samples: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Construct the size-consistent selected configuration spaces.

    Parameters
    ----------
    n_a, n_b : int
        Positive counts of active spatial orbitals on A and B, sum<=10.
        Dimer spatial labels put A first, then B.
    n_e_a : int
        Neutral active electron count on A; the neutral B count is
        total sampled electron count minus n_e_a.
    samples : np.ndarray
        Raw alpha/beta masks in a nonempty (k,2) integer array, using
        bit p for spatial orbital p and common Nalpha,Nbeta.

    Apply the retrieved spin-completion and size-consistent
    fragment/dimer-space prescription. Retain the fragment spin sectors
    induced by the samples. The dimer retains the sampled total alpha
    and beta counts. Packed determinants use alpha_mask | (beta_mask
    << local_orbital_count), with alpha spin orbitals before beta.
    Sort each output by packed integer.

    Returns
    -------
    tuple (space_a, space_b, space_ab) of ascending unique np.int64 packed-determinant arrays using local orbital counts n_a, n_b and n_a+n_b.

    Raises
    ------
    ValueError
    If n_a or n_b is not a positive integer (booleans invalid),
    or n_a+n_b>10; if samples is not a nonempty integer (k,2) array
    of masks in [0,2**(n_a+n_b)) with common alpha and beta counts;
    if n_e_a is not an integer in [0,2*n_a], or the implied n_e_b is
    outside [0,2*n_b]; or if no spin-completed sample has n_e_a
    electrons on A. Duplicate sample rows are valid.
    """
    return space_a, space_b, space_ab

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_selected_spaces(n_a: int, n_b: int, n_e_a: int, samples: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np
    for n in (n_a, n_b):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError('Fragment orbital counts must be positive integers.')
    n_a, n_b = int(n_a), int(n_b)
    completed = _qc38_complete_spins(n_a+n_b, samples)
    if isinstance(n_e_a, (bool, np.bool_)) or not isinstance(n_e_a, (int, np.integer)):
        raise ValueError('Fragment electron count must be an integer.')
    n = n_a+n_b
    a0, b0 = np.asarray(samples)[0]
    na, nb = int(a0).bit_count(), int(b0).bit_count()
    n_e_b = na+nb-int(n_e_a)
    if not 0 <= n_e_a <= 2*n_a or not 0 <= n_e_b <= 2*n_b:
        raise ValueError('Infeasible fragment electron counts.')
    fa, fb, ct = set(), set(), set()
    for packed in completed:
        a, b = int(packed) & ((1 << n)-1), int(packed) >> n
        aa, ba = a & ((1 << n_a)-1), b & ((1 << n_a)-1)
        ab, bb = a >> n_a, b >> n_a
        if aa.bit_count()+ba.bit_count() == n_e_a:
            fa.add(aa | (ba << n_a)); fb.add(ab | (bb << n_b))
        else:
            ct.add(int(packed))
    if not fa or not fb:
        raise ValueError('No neutral fragment configurations.')
    dimer = set(ct)
    for da in fa:
        aa, ba = da & ((1 << n_a)-1), da >> n_a
        for db in fb:
            ab, bb = db & ((1 << n_b)-1), db >> n_b
            a, b = aa | (ab << n_a), ba | (bb << n_a)
            if a.bit_count() == na and b.bit_count() == nb:
                dimer.add(a | (b << n))
    return tuple(np.array(sorted(s), dtype=np.int64) for s in (fa, fb, dimer))

def _qc38_complete_spins(norb: int, samples: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from itertools import combinations
    if isinstance(norb, (bool, np.bool_)) or not isinstance(norb, (int, np.integer)) or not 1 <= norb <= 10:
        raise ValueError('norb must be an integer in [1,10].')
    try:
        s = np.asarray(samples)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid samples.') from exc
    if s.ndim != 2 or s.shape[1] != 2 or s.shape[0] == 0 or s.dtype.kind not in 'iu':
        raise ValueError('Samples must be a nonempty integer (k,2) array.')
    if np.any(s < 0) or np.any(s >= 1 << int(norb)):
        raise ValueError('Mask out of range.')
    counts = {(int(a).bit_count(), int(b).bit_count()) for a, b in s}
    if len(counts) != 1:
        raise ValueError('Inconsistent spin-resolved particle counts.')
    na, nb = next(iter(counts))
    out = set()
    for a, b in s:
        a, b = int(a), int(b)
        double = a & b
        singles = a ^ b
        positions = [p for p in range(norb) if (singles >> p) & 1]
        for selected in combinations(positions, na-double.bit_count()):
            alpha_single = sum(1 << p for p in selected)
            aa = double | alpha_single
            bb = double | (singles ^ alpha_single)
            out.add(aa | (bb << int(norb)))
    return np.array(sorted(out), dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases = []
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[9, 9], [9, 10], [9, 17], [9, 18], [12, 12], [33, 33], [9, 3], [9, 24]], dtype=int)
""",
        'call': 'selected_spaces(3, 3, 2, samples.copy())',
        'gold_call': '_oracle_selected_spaces(3, 3, 2, samples.copy())',
    })
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[1, 2]])
""",
        'call': 'selected_spaces(1, 1, 1, samples.copy())',
        'gold_call': '_oracle_selected_spaces(1, 1, 1, samples.copy())',
    })
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[5, 5], [5, 3], [5, 12]])
""",
        'call': 'selected_spaces(2, 2, 2, samples.copy())',
        'gold_call': '_oracle_selected_spaces(2, 2, 2, samples.copy())',
    })
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[3, 4], [5, 2]])
""",
        'call': 'selected_spaces(2, 2, 2, samples.copy())',
        'gold_call': '_oracle_selected_spaces(2, 2, 2, samples.copy())',
    })
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[4, 4], [8, 8]])
""",
        'call': 'selected_spaces(2, 2, 0, samples.copy())',
        'gold_call': '_oracle_selected_spaces(2, 2, 0, samples.copy())',
    })
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[1, 1]])

def run_model():
    try:
        selected_spaces(1, 1, 1, samples.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_selected_spaces(1, 1, 1, samples.copy())
        return 0
    except ValueError:
        return 1
""",
        'call': 'run_model()',
        'gold_call': 'run_gold()',
    })
    cases.append({
        "setup": """
import numpy as np
samples = np.array([[1, 2]])

def run_model():
    try:
        selected_spaces(1, 1, 3, samples.copy())
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_selected_spaces(1, 1, 3, samples.copy())
        return 0
    except ValueError:
        return 1
""",
        'call': 'run_model()',
        'gold_call': 'run_gold()',
    })
    return cases
