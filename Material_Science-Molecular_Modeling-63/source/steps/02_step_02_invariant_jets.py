"""
Construct the five scalar moment invariants and propagate their Cartesian gradients and Hessians.

Tensor contractions produce rotationally invariant features for the charge model. Their second derivatives require the product rule for every factor, including both gradient cross terms and all three factors of the vector–matrix–vector contraction.

Returns
-------
return f, df, ddf
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invariant_jets(moments, first, second):
    """Contract moment jets into five scalar invariants and their first two derivatives.
    
    Parameters
    ----------
    moments : (N,13) real array
        M0, M1[3], M2[3,3] flattened row-major; N>=1.
    first : (N,13,D) real array
        Derivatives with respect to D>=1 variables.
    second : (N,13,D,D) real array
        Unnormalized second derivatives.
    
    Return invariants in this order: M0, M0^2, M1 dot M1,
    sum_ab M2_ab^2, sum_ab M1_a*M2_ab*M1_b.
    Differentiate every factor, including both occurrences of M1 and M2.
    The inputs are arbitrary finite jets; do not impose symmetry on M2.
    
    Returns
    -------
    f : (N,5) float array
        Invariant values.
    df : (N,5,D) float array
        First derivatives.
    ddf : (N,5,D,D) float array
        Second derivatives.
    
    Raises
    ------
    ValueError
        If shapes do not match the stated contract or any entry is nonfinite.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (f, df, ddf)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _ij_mul(a, b):
    import numpy as np
    av, ag, ah = a
    bv, bg, bh = b
    return (av * bv, ag * bv + av * bg, ah * bv + av * bh + np.outer(ag, bg) + np.outer(bg, ag))

def _ij_sum(items):
    return tuple((sum((x[k] for x in items)) for k in range(3)))

def _oracle_invariant_jets(moments, first, second):
    """Reference implementation."""
    import numpy as np
    m = np.asarray(moments, float)
    g = np.asarray(first, float)
    h = np.asarray(second, float)
    if m.ndim != 2 or m.shape[1] != 13 or len(m) < 1 or (g.ndim != 3) or (g.shape[:2] != m.shape) or (g.shape[2] < 1) or (h.shape != g.shape + (g.shape[-1],)):
        raise ValueError('jet shape')
    if not all((np.all(np.isfinite(a)) for a in (m, g, h))):
        raise ValueError('nonfinite jet')
    n, D = (g.shape[0], g.shape[2])
    f = np.zeros((n, 5))
    df = np.zeros((n, 5, D))
    ddf = np.zeros((n, 5, D, D))
    for i in range(n):
        jets = [(m[i, k], g[i, k], h[i, k]) for k in range(13)]
        out = [jets[0], _ij_mul(jets[0], jets[0])]
        out.append(_ij_sum([_ij_mul(jets[k], jets[k]) for k in range(1, 4)]))
        out.append(_ij_sum([_ij_mul(jets[k], jets[k]) for k in range(4, 13)]))
        out.append(_ij_sum([_ij_mul(_ij_mul(jets[1 + a], jets[4 + 3 * a + b]), jets[1 + b]) for a in range(3) for b in range(3)]))
        for k, (v, dv, hv) in enumerate(out):
            f[i, k] = v
            df[i, k] = dv
            ddf[i, k] = hv
    return (f, df, ddf)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(1729)\n'
                'm=rng.normal(size=(2,13)); g=rng.normal(size=(2,13,4)); h=rng.normal(size=(2,13,4,4)); h=(h+h.swapaxes(-1,-2))/2\n'
                'm_g = m.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'invariant_jets(m,g,h)',
            'gold_call': '_oracle_invariant_jets(m_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(1729)\n'
                'm=rng.normal(size=(2,13)); g=rng.normal(size=(2,13,4)); h=rng.normal(size=(2,13,4,4)); h=(h+h.swapaxes(-1,-2))/2\n'
                'm[:]=0\n'
                'm_g = m.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'invariant_jets(m,g,h)',
            'gold_call': '_oracle_invariant_jets(m_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'm=np.arange(13,dtype=float).reshape(1,13)/10\n'
                'g=np.ones((1,13,1)); h=-np.ones((1,13,1,1))\n'
                'm_g = m.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': 'invariant_jets(m,g,h)',
            'gold_call': '_oracle_invariant_jets(m_g,g_g,h_g)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'm=np.zeros((1,12));g=np.zeros((1,12,2));h=np.zeros((1,12,2,2))\n'
                'def _raise_code(f):\n'
                '    try:\n'
                '        f()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        raise\n'
                '    raise AssertionError("Expected ValueError")\n'
                'm_g = m.copy(); g_g = g.copy(); h_g = h.copy()\n'
            ),
            'call': '_raise_code(lambda: invariant_jets(m,g,h))',
            'gold_call': '_raise_code(lambda: _oracle_invariant_jets(m_g,g_g,h_g))',
        },
    ]
