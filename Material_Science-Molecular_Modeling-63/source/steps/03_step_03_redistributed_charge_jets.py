"""
Evaluate the local raw charges, impose the prescribed total charge and propagate first and second coordinate derivatives through the redistribution map.

Fixed type softness weights distribute the difference between the prescribed total charge and the raw-charge sum. The normalization counts atoms of each type. Applying the same linear redistribution map to both derivative orders preserves total-charge conservation under atomic displacement.

Returns
-------
return q, jq, hq
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def redistributed_charge_jets(features, first, second, species, bias, coefficients, softness, total_charge):
    """Evaluate environment charges and redistribute their residual with type softness.
    
    Parameters
    ----------
    features : (N,5) real array
    first : (N,5,D) real array
    second : (N,5,D,D) real array
        Feature values and unnormalized derivatives; N,D>=1.
    species : (N,) integer array
        Indices in [0,S).
    bias : (S,) real array
    coefficients : (S,5) real array
    softness : (S,) positive real array
        Geometry-independent redistribution weights.
    total_charge : finite float
        Prescribed charge, held constant during differentiation.
    
    For y_i=bias[z_i]+sum_k coefficients[z_i,k]*features[i,k], define
    s_i=softness[z_i], w_i=s_i/sum_j s_j and
    q_i=y_i+w_i*(total_charge-sum_j y_j).
    Return q and its full first and second derivatives. The sums run over
    all atoms, not over species. Do not minimize a QEq energy.
    
    Returns
    -------
    q : (N,) float array
    jq : (N,D) float array
    hq : (N,D,D) float array
    
    Raises
    ------
    ValueError
        If shapes do not match, entries are nonfinite, softness is nonpositive,
        or species is not integer-valued or is out of range.

    Runtime
    -------
    Import required libraries inside the submitted function and inside any
    helper that uses them. Do not assume that np, itertools or erfc exists
    in the function's global namespace. Imports in test setup are separate.
    """
    return (q, jq, hq)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_redistributed_charge_jets(features, first, second, species, bias, coefficients, softness, total_charge):
    """Reference implementation."""
    import numpy as np
    f = np.asarray(features, float)
    g = np.asarray(first, float)
    h = np.asarray(second, float)
    z0 = np.asarray(species)
    b = np.asarray(bias, float)
    c = np.asarray(coefficients, float)
    s = np.asarray(softness, float)
    if f.ndim != 2 or f.shape[1] != 5 or len(f) < 1 or (g.ndim != 3) or (g.shape[:2] != f.shape) or (g.shape[-1] < 1) or (h.shape != g.shape + (g.shape[-1],)):
        raise ValueError('feature shape')
    if b.ndim != 1 or len(b) < 1 or c.shape != (len(b), 5) or (s.shape != b.shape) or (z0.shape != (len(f),)):
        raise ValueError('parameter shape')
    if not all((np.all(np.isfinite(a)) for a in (f, g, h, z0, b, c, s))) or not np.isfinite(total_charge):
        raise ValueError('nonfinite input')
    if np.any(s <= 0) or np.any(z0 != np.floor(z0)) or np.any(z0 < 0) or np.any(z0 >= len(b)):
        raise ValueError('parameter domain')
    z = z0.astype(int)
    w = s[z] / s[z].sum()
    y = b[z] + np.einsum('nk,nk->n', c[z], f)
    dy = np.einsum('nk,nkd->nd', c[z], g)
    hy = np.einsum('nk,nkde->nde', c[z], h)
    return (y + w * (total_charge - y.sum()), dy - w[:, None] * dy.sum(axis=0), hy - w[:, None, None] * hy.sum(axis=0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Normal, boundary, edge and declared-invalid fixtures."""
    return [
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(431)\n'
                'f=rng.normal(size=(3,5));g=rng.normal(size=(3,5,2));h=rng.normal(size=(3,5,2,2));h=(h+h.swapaxes(-1,-2))/2\n'
                'b=np.array([.4,-.7]);c=rng.normal(size=(2,5));s=np.array([1.,3.])\n'
                'f_g = f.copy(); g_g = g.copy(); h_g = h.copy(); b_g = b.copy(); c_g = c.copy(); s_g = s.copy()\n'
            ),
            'call': 'redistributed_charge_jets(f,g,h,[0,1,1],b,c,s,1.25)',
            'gold_call': '_oracle_redistributed_charge_jets(f_g,g_g,h_g,[0,1,1],b_g,c_g,s_g,1.25)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(431)\n'
                'f=rng.normal(size=(3,5));g=rng.normal(size=(3,5,2));h=rng.normal(size=(3,5,2,2));h=(h+h.swapaxes(-1,-2))/2\n'
                'b=np.array([.4,-.7]);c=rng.normal(size=(2,5));s=np.array([1.,3.])\n'
                'f_g = f.copy(); g_g = g.copy(); h_g = h.copy(); b_g = b.copy(); c_g = c.copy(); s_g = s.copy()\n'
            ),
            'call': 'redistributed_charge_jets(f[:1],g[:1],h[:1],[1],b,c,s,-2.)',
            'gold_call': '_oracle_redistributed_charge_jets(f_g[:1],g_g[:1],h_g[:1],[1],b_g,c_g,s_g,-2.)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(431)\n'
                'f=rng.normal(size=(3,5));g=rng.normal(size=(3,5,2));h=rng.normal(size=(3,5,2,2));h=(h+h.swapaxes(-1,-2))/2\n'
                'b=np.array([.4,-.7]);c=rng.normal(size=(2,5));s=np.array([1.,3.])\n'
                'c[:]=0\n'
                'f_g = f.copy(); g_g = g.copy(); h_g = h.copy(); b_g = b.copy(); c_g = c.copy(); s_g = s.copy()\n'
            ),
            'call': 'redistributed_charge_jets(f,g,h,[0,1,1],b,c,s,0.)',
            'gold_call': '_oracle_redistributed_charge_jets(f_g,g_g,h_g,[0,1,1],b_g,c_g,s_g,0.)',
        },
        {
            'setup': (
                'import numpy as np\n'
                'rng=np.random.default_rng(431)\n'
                'f=rng.normal(size=(3,5));g=rng.normal(size=(3,5,2));h=rng.normal(size=(3,5,2,2));h=(h+h.swapaxes(-1,-2))/2\n'
                'b=np.array([.4,-.7]);c=rng.normal(size=(2,5));s=np.array([1.,3.])\n'
                's[0]=0\n'
                'def _raise_code(f):\n'
                '    try:\n'
                '        f()\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        raise\n'
                '    raise AssertionError("Expected ValueError")\n'
                'f_g = f.copy(); g_g = g.copy(); h_g = h.copy(); b_g = b.copy(); c_g = c.copy(); s_g = s.copy()\n'
            ),
            'call': '_raise_code(lambda: redistributed_charge_jets(f,g,h,[0,1,1],b,c,s,0.))',
            'gold_call': '_raise_code(lambda: _oracle_redistributed_charge_jets(f_g,g_g,h_g,[0,1,1],b_g,c_g,s_g,0.))',
        },
    ]
