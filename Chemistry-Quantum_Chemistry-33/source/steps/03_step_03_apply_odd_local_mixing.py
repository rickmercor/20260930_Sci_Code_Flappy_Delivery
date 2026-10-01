"""
Validate features shape (P,A,K,M), positions shape (A,3), two channel maps shape (K,K), and positive cutoff. For each ordered atom pair inside the strict cutoff, add exp(-distance/cutoff)*features[:,neighbor] to the center aggregate and require at least one edge. Form the rotationally equivariant odd cubic as aggregate*sum(aggregate*aggregate,axis=-1,keepdims=True). Apply linear_weights to aggregate and cubic_weights to that cubic, add both to the residual features, and return updated, aggregate, and the directed-edge count.

The odd-only local nonlinearity and finite support are source-derived. The exponential radial factor, cubic truncation, and residual connection are disclosed benchmark conventions.

Returns
-------
tuple : updated states, local aggregate, and directed-edge count.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_odd_local_mixing(features: "np.ndarray", positions: "np.ndarray", cutoff: float, linear_weights: "np.ndarray", cubic_weights: "np.ndarray") -> tuple:
    """Return updated orbital states, the local aggregate, and edge count.

    Recover from the source paper which polynomial orders are admissible and why
    the finite message-passing cutoff is required. This benchmark uses the
    documented radial factor exp(-d/cutoff) and an explicit residual connection.

    Parameters
    ----------
    features : np.ndarray
        Orbital states with shape (P,A,K,M).
    positions : np.ndarray
        Atomic positions with shape (A,3).
    cutoff : float
        Positive finite neighbor cutoff.
    linear_weights, cubic_weights : np.ndarray
        Channel maps with shape (K,K).

    Returns
    -------
    updated : np.ndarray
        Residual plus odd local update, shape (P,A,K,M).
    aggregate : np.ndarray
        Distance-weighted neighbor aggregate, shape (P,A,K,M).
    directed_edge_count : int
        Number of directed atom pairs inside the cutoff.

    Raises
    ------
    ValueError
        If shapes, values, or cutoff are invalid, or if no directed edge exists.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_apply_odd_local_mixing(features: "np.ndarray", positions: "np.ndarray", cutoff: float, linear_weights: "np.ndarray", cubic_weights: "np.ndarray") -> tuple:
    x = np.asarray(features, dtype=float)
    pos = np.asarray(positions, dtype=float)
    w1 = np.asarray(linear_weights, dtype=float)
    w3 = np.asarray(cubic_weights, dtype=float)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError("features must have shape (P,A,K,M)")
    p, a, k, m = x.shape
    if pos.shape != (a, 3) or w1.shape != (k, k) or w3.shape != (k, k):
        raise ValueError("positions or channel maps have invalid shape")
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be positive and finite")
    if not all(np.all(np.isfinite(z)) for z in (x, pos, w1, w3)):
        raise ValueError("numeric inputs must be finite")
    aggregate = np.zeros_like(x)
    count = 0
    for center in range(a):
        for neighbor in range(a):
            if center == neighbor:
                continue
            distance = float(np.linalg.norm(pos[center] - pos[neighbor]))
            if distance < cutoff:
                aggregate[:, center] += np.exp(-distance / cutoff) * x[:, neighbor]
                count += 1
    if count == 0:
        raise ValueError("the cutoff graph must contain at least one directed edge")
    linear = np.einsum("jk,pakm->pajm", w1, aggregate, optimize=True)
    cubic_input = aggregate * np.sum(aggregate * aggregate, axis=-1, keepdims=True)
    cubic = np.einsum("jk,pakm->pajm", w3, cubic_input, optimize=True)
    updated = x + linear + cubic
    return updated, aggregate, int(count)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = lambda call: "(lambda z: [len(z),*z[0].shape,*z[1].shape,*np.concatenate([z[0].ravel(),z[1].ravel(),[z[2]]]).tolist()])(" + call + ")"
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    setup = lambda seed,p,a,k,m: f'import numpy as np\nr=np.random.default_rng({seed});x=r.normal(scale=.3,size=({p},{a},{k},{m}));pos=r.normal(size=({a},3));w1=r.normal(scale=.1,size=({k},{k}));w3=r.normal(scale=.03,size=({k},{k}))'
    return [
        {"tol": 1e-10, "setup": setup(29,6,5,4,3) + ";cut=2.4", "call": pack("apply_odd_local_mixing(x,pos,cut,w1,w3)"), "gold_call": pack("_oracle_apply_odd_local_mixing(x,pos,cut,w1,w3)")},
        {"tol": 1e-10, "setup": setup(31,2,2,1,1) + ";pos=np.array([[0.,0,0],[.5,0,0]]);cut=.5000001", "call": pack("apply_odd_local_mixing(x,pos,cut,w1,w3)"), "gold_call": pack("_oracle_apply_odd_local_mixing(x,pos,cut,w1,w3)")},
        {"tol": 1e-10, "setup": setup(37,4,3,3,2) + ";pos=np.array([[0.,0,0],[1.,0,0],[2.,0,0]]);cut=1.5", "call": pack("apply_odd_local_mixing(x,pos,cut,w1,w3)"), "gold_call": pack("_oracle_apply_odd_local_mixing(x,pos,cut,w1,w3)")},
        {"tol": 0.0, "setup": setup(41,3,3,2,3) + "\n" + err + "pos=np.eye(3)*10\ncut=.1", "call": "et(apply_odd_local_mixing,x,pos,cut,w1,w3)", "gold_call": "et(_oracle_apply_odd_local_mixing,x,pos,cut,w1,w3)"},
        {
            "tol": 1e-10,
            "setup": "import numpy as np\nr=np.random.default_rng(101);x=r.normal(size=(3,4,2,3));p=r.normal(size=(4,3));q,_=np.linalg.qr(r.normal(size=(3,3)));w1=r.normal(size=(2,2));w3=r.normal(size=(2,2))\ndef rotate_result(z,q):\n return (np.einsum('pakm,mn->pakn',z[0],q),np.einsum('pakm,mn->pakn',z[1],q),z[2])",
            "call": "apply_odd_local_mixing(np.einsum('pakm,mn->pakn',x,q),p@q,10.0,w1,w3)",
            "gold_call": "rotate_result(_oracle_apply_odd_local_mixing(x,p,10.0,w1,w3),q)",
        },
    ]
