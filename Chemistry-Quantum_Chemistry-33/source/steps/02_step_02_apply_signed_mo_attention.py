"""
Validate embedded shape (P,A,K,M), integer fragment_ids shape (P,), three weight tensors shape (H,K,K), head_weights shape (H,), and positive epsilon. Project query, key, and value channels with the supplied head matrices. Recover the source normalization and aggregation rule, mask scores for unequal fragment labels, combine heads with head_weights, and return mixed states, the H by P by P score tensor, and the pre-renormalization H by P output norms.

The normalization and non-softmax aggregation are source-derived and load-bearing. The explicit fragment mask is a deterministic proxy for the source separated-fragment limit.

Returns
-------
tuple : mixed states, attention scores, and output norms.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_signed_mo_attention(embedded: "np.ndarray", fragment_ids: "np.ndarray", q_weights: "np.ndarray", k_weights: "np.ndarray", v_weights: "np.ndarray", head_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    """Return mixed states, masked attention scores, and output norms.

    The implementation must recover from the source paper the normalization,
    attention aggregation, and fragment-coupling choices that preserve the stated
    orbital sign, rotation, and separated-fragment behavior.

    Parameters
    ----------
    embedded : np.ndarray
        Orbital states with shape (P,A,K,M).
    fragment_ids : np.ndarray
        Integer fragment label for each orbital, shape (P,).
    q_weights, k_weights, v_weights : np.ndarray
        Head-specific channel maps with shape (H,K,K).
    head_weights : np.ndarray
        Finite head-combination coefficients with shape (H,).
    epsilon : float
        Positive normalization stabilizer.

    Returns
    -------
    mixed : np.ndarray
        Attention-mixed states with shape (P,A,K,M).
    scores : np.ndarray
        Masked signed scores with shape (H,P,P).
    output_norms : np.ndarray
        Pre-renormalization output norms with shape (H,P).

    Raises
    ------
    ValueError
        If shapes, fragment labels, numeric values, or epsilon are invalid.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_apply_signed_mo_attention(embedded: "np.ndarray", fragment_ids: "np.ndarray", q_weights: "np.ndarray", k_weights: "np.ndarray", v_weights: "np.ndarray", head_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    x = np.asarray(embedded, dtype=float)
    f = np.asarray(fragment_ids)
    qw = np.asarray(q_weights, dtype=float)
    kw = np.asarray(k_weights, dtype=float)
    vw = np.asarray(v_weights, dtype=float)
    hw = np.asarray(head_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("embedded must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if f.shape != (p,) or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("fragment_ids must be an integer vector of length P")
    if qw.ndim != 3 or qw.shape[1:] != (k, k) or kw.shape != qw.shape or vw.shape != qw.shape:
        raise ValueError("q, k, and v weights must all have shape (H,K,K)")
    h = qw.shape[0]
    if h < 1 or hw.shape != (h,):
        raise ValueError("head_weights must have shape (H,)")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, qw, kw, vw, hw)):
        raise ValueError("all numeric inputs must be finite")
    q = np.einsum("hjk,pakm->hpajm", qw, x, optimize=True)
    key = np.einsum("hjk,pakm->hpajm", kw, x, optimize=True)
    value = np.einsum("hjk,pakm->hpajm", vw, x, optimize=True)
    qn = np.sqrt(np.sum(q * q, axis=(2, 3, 4))) + abs(float(epsilon))
    kn = np.sqrt(np.sum(key * key, axis=(2, 3, 4))) + abs(float(epsilon))
    q_unit = q / qn[:, :, None, None, None]
    k_unit = key / kn[:, :, None, None, None]
    scores = np.einsum("hpajm,hqajm->hpq", q_unit, k_unit, optimize=True)
    same_fragment = f[:, None] == f[None, :]
    scores = scores * same_fragment[None, :, :]
    raw = np.einsum("hpq,hqajm->hpajm", scores, value, optimize=True)
    output_norms = np.sqrt(np.sum(raw * raw, axis=(2, 3, 4)))
    heads = raw / (output_norms[:, :, None, None, None] + abs(float(epsilon)))
    mixed = np.einsum("h,hpajm->pajm", hw, heads, optimize=True)
    return mixed, scores, output_norms

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = lambda call: "(lambda z: [len(z),*z[0].shape,*z[1].shape,*z[2].shape,*np.concatenate([z[0].ravel(),z[1].ravel(),z[2].ravel()]).tolist()])(" + call + ")"
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    setup = lambda seed,p,a,k,m,h: f'import numpy as np\nr=np.random.default_rng({seed});x=r.normal(size=({p},{a},{k},{m}));f=np.arange({p})%2;qw=r.normal(size=({h},{k},{k}));kw=r.normal(size=({h},{k},{k}));vw=r.normal(size=({h},{k},{k}));hw=r.normal(size={h})'
    return [
        {"tol": 1e-10, "setup": setup(13,6,4,3,3,2), "call": pack("apply_signed_mo_attention(x,f,qw,kw,vw,hw,1e-7)"), "gold_call": pack("_oracle_apply_signed_mo_attention(x,f,qw,kw,vw,hw,1e-7)")},
        {"tol": 1e-10, "setup": setup(17,2,1,1,1,1) + ";f=np.zeros(2,dtype=int)", "call": pack("apply_signed_mo_attention(x,f,qw,kw,vw,hw)"), "gold_call": pack("_oracle_apply_signed_mo_attention(x,f,qw,kw,vw,hw)")},
        {"tol": 1e-10, "setup": setup(19,5,3,4,2,3) + ";x[0]=0", "call": pack("apply_signed_mo_attention(x,f,qw,kw,vw,hw,1e-6)"), "gold_call": pack("_oracle_apply_signed_mo_attention(x,f,qw,kw,vw,hw,1e-6)")},
        {"tol": 0.0, "setup": setup(23,4,2,3,3,2) + "\n" + err + "f=np.array([0,1,2])", "call": "et(apply_signed_mo_attention,x,f,qw,kw,vw,hw)", "gold_call": "et(_oracle_apply_signed_mo_attention,x,f,qw,kw,vw,hw)"},
    ]
