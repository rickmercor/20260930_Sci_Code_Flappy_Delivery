"""
Construct the full static-cRPA active interaction tensor.

The effective interaction acts on every pair of active orbital indices. Use the environmental response returned by environmental_response.

Returns
-------
real array w of shape (a,a,a,a), in active input order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screened_interaction(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> "np.ndarray":
    """Construct the full static-cRPA active interaction tensor.
    
    Parameters and conventions
    --------------------------
    v is a real finite chemists' tensor of shape (n,n,n,n), n>=1,
    with p/q, r/s and pair-exchange symmetry (atol=1e-12, rtol=0).
    eps is a real finite vector of length n. occupied is a vector of
    distinct integer indices in [0,n), each doubly occupied; empty is
    allowed. Every occupied-to-virtual gap must be positive. active is
    a nonempty vector of distinct integer indices in [0,n); its order
    defines the local active orbital order. Booleans are not indices.
    The reference orbitals, energies and occupations remain fixed.
    Complex dtype numerical inputs are rejected even if their imaginary
    parts are zero. Raise ValueError for violations, failed numerical
    solves, or nonfinite computed results.
    
    Use environmental_response and its spatial, spin-summed convention.
    Return the static screened chemists' interaction on all active pair
    indices. The constrained response must be strictly stable. Empty
    constrained transition spaces and a fully active system are valid.
    
    Returns
    -------
    real array w of shape (a,a,a,a), in active input order.
    """
    return w

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_screened_interaction(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray") -> "np.ndarray":
    import numpy as np
    pairs, response = _oracle_environmental_response(v, eps, occupied, active)
    act = np.asarray(active, dtype=int)
    v = np.asarray(v, dtype=float)
    m = len(act)
    bare = v[np.ix_(act, act, act, act)]
    c = np.array([v[:,:,i,a][np.ix_(act,act)].reshape(-1)
                  for i,a in pairs], dtype=float).reshape(len(pairs), m*m)
    w = (bare.reshape(m*m,m*m) + c.T @ response @ c).reshape(bare.shape)
    return _qc38_screened_input(w, m).copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = "import numpy as np\nfactors=np.array([[[.50,.16,.30],[.16,.36,.09],[.30,.09,.30]],\n                  [[.20,-.08,.22],[-.08,.28,.07],[.22,.07,.24]]])\nv=np.einsum('lpq,lrs->pqrs',factors,factors)\neps=np.array([-.8,-.35,1.1])\noccupied=np.array([0],dtype=int)\nactive=np.array([0,1],dtype=int)\n"
    variants = [
        "",
        "active=np.arange(3,dtype=int)\n",
        "occupied=np.array([0,1],dtype=int)\nactive=np.array([1,2],dtype=int)\n",
        "active=active[::-1].copy()\n",
        "occupied=np.array([],dtype=int)\n",
    ]
    model = 'screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy())'
    gold = '_oracle_screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy())'
    cases = [
        {"setup": base + variants[0], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[1], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[2], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[3], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[4], "call": model, "gold_call": gold, "tol": 1e-10},
    ]
    cases.append({"setup":base+'v=v.astype(complex)\n\ndef run_model():\n    try:\n        screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy())\n    except ValueError:\n        return True\n    return False\n\ndef run_gold():\n    try:\n        _oracle_screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy())\n    except ValueError:\n        return True\n    return False\n', "call":"run_model()", "gold_call":"run_gold()"})
    return cases
