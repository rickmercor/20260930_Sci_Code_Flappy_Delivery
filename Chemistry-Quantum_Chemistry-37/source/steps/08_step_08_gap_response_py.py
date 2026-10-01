"""
Implement gap_response. The requested observable is the mixed control derivative of the gap between the second and third sorted corrected energies, not an exact-diagonalization gap.

The requested observable is the mixed control derivative of the gap between the second and third sorted corrected energies, not an exact-diagonalization gap. For the benchmark at a=b=0, after the initial reference-space rotation, the smaller direct coupling ratio determining inclusion of both remaining perturbers is approximately 0.4010913817, exceeding the primary threshold 0.4 by approximately 0.0010913817. Thus this deciding comparison is strict, rather than an equality requiring a tie convention. Previously optimized references remain frozen according to sequential optimization order and continue to act as external perturbers.

Returns
-------
float, the unrounded mixed derivative of epsilon_2-epsilon_1 as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gap_response(a: float = 0.0, b: float = 0.0) -> float:
    """Return the mixed response of the specified corrected excitation gap.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls with |a|<=0.001 and |b|<=0.001. This
        neighborhood retains the benchmark's selection branch. Build the
        published four-state model with model_jet and use ssrsbw_jet with
        targets=3, rho=0.4, enrich=0.6, max_updates=50.

    Returns
    -------
    result : float
        Native Python float d_a d_b (epsilon_2-epsilon_1), in |t| units.
        Return the unrounded value. Use analytic derivative propagation.

    Raises
    ------
    ValueError
        If controls are not finite real scalars convertible to float or
        exceed the stated neighborhood; a numerical eigensolver fails;
        computations become nonfinite; a required reference denominator,
        effective spectral gap, or corrected spectral gap has magnitude
        <=1e-10; a rotation jet fails orthogonality at atol=1e-9; a BW
        reference is within 1e-10 of a pole or its interval lacks exactly
        one root >1e-10 from finite bounds; or a target needs more than
        50 transformations.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_gap_response(a: float = 0.0, b: float = 0.0) -> float:
    """Return the mixed response of the specified corrected excitation gap.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls with |a|<=0.001 and |b|<=0.001. This
        neighborhood retains the benchmark's selection branch. Build the
        published four-state model with model_jet and use ssrsbw_jet with
        targets=3, rho=0.4, enrich=0.6, max_updates=50.

    Returns
    -------
    result : float
        Native Python float d_a d_b (epsilon_2-epsilon_1), in |t| units.
        Return the unrounded value. Use analytic derivative propagation.

    Raises
    ------
    ValueError
        If controls are not finite real scalars convertible to float or
        exceed the stated neighborhood; a numerical eigensolver fails;
        computations become nonfinite; a required reference denominator,
        effective spectral gap, or corrected spectral gap has magnitude
        <=1e-10; a rotation jet fails orthogonality at atol=1e-9; a BW
        reference is within 1e-10 of a pole or its interval lacks exactly
        one root >1e-10 from finite bounds; or a target needs more than
        50 transformations.
    """
    H = _oracle_model_jet(a,b)
    if abs(float(a))>0.001 or abs(float(b))>0.001:
        raise ValueError('controls outside benchmark neighborhood')
    energies = _oracle_ssrsbw_jet(H,targets=3,rho=0.4,enrich=0.6,max_updates=50)
    return float(energies[3,2]-energies[3,1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(0.0, 0.0)',
            "gold_call": '_oracle_gap_response(0.0, 0.0)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(0.0005, 0.0)',
            "gold_call": '_oracle_gap_response(0.0005, 0.0)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(0.0, -0.0005)',
            "gold_call": '_oracle_gap_response(0.0, -0.0005)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(-0.0005, 0.0005)',
            "gold_call": '_oracle_gap_response(-0.0005, 0.0005)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(0.001, 0.001)',
            "gold_call": '_oracle_gap_response(0.001, 0.001)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(0.001, -0.001)',
            "gold_call": '_oracle_gap_response(0.001, -0.001)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(-0.001, 0.001)',
            "gold_call": '_oracle_gap_response(-0.001, 0.001)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(-0.001, -0.001)',
            "gold_call": '_oracle_gap_response(-0.001, -0.001)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(0.00037, -0.00061)',
            "gold_call": '_oracle_gap_response(0.00037, -0.00061)',
        },
        {
            "setup": """import numpy as np""",
            "call": 'gap_response(-0.00073, 0.00029)',
            "gold_call": '_oracle_gap_response(-0.00073, 0.00029)',
        },
        {
            "setup": """import numpy as np
a, b = (np.nextafter(0.001, np.inf), 0.0)

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (-np.nextafter(0.001, np.inf), 0.0)

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (0.0, np.nextafter(0.001, np.inf))

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (0.0, -np.nextafter(0.001, np.inf))

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (np.nan, 0.0)

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (0.0, np.inf)

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (1j, 0.0)

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
        {
            "setup": """import numpy as np
a, b = (np.array([0.0, 0.0]), 0.0)

def rejects(fn):
    aa = a.copy() if isinstance(a, np.ndarray) else a
    bb = b.copy() if isinstance(b, np.ndarray) else b
    try:
        fn(aa, bb)
    except ValueError:
        return 1
    return 0
""",
            "call": 'rejects(gap_response)',
            "gold_call": 'rejects(_oracle_gap_response)',
        },
    ]
