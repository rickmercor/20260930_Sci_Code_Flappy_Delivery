"""
Construct the corrected one-electron part of the molecular effective Hamiltonian.

Use the molecular one-body correction with the supplied reference and screened interaction. Occupied environment orbitals are supported.

Returns
-------
real array t_eff of shape (a,a), in active input order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def corrected_one_body(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> "np.ndarray":
    """Construct the corrected one-electron part of the molecular effective Hamiltonian.
    
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
    
    w is the real finite screened chemists' tensor of shape (a,a,a,a),
    a=len(active), with the same three Coulomb symmetries and tolerance
    as v. It is already in active input order. Reject complex dtype w.
    Compute the one-body term of the retrieved molecular prescription.
    The bare reference h is defined by the problem statement. Occupied
    environment orbitals are supported. Do not recompute w or test that
    it equals a particular screening calculation. Nuclear repulsion is zero.
    
    Returns
    -------
    real array t_eff of shape (a,a), in active input order.
    """
    return t_eff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_corrected_one_body(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> "np.ndarray":
    import numpy as np
    h, _, _, _, act, ia, ie = _qc38_active_reference(v, eps, occupied, active)
    v = np.asarray(v, dtype=float)
    w = _qc38_screened_input(w, len(act))
    delta = w - v[np.ix_(act,act,act,act)]
    t = h[np.ix_(act,act)].copy()
    for i in ie:
        t += (2*v[:,:,i,i] - v[:,i,i,:])[np.ix_(act,act)]
    for i in ia:
        t -= 2*delta[:,:,i,i] - delta[:,i,i,:]
    if not np.all(np.isfinite(t)):
        raise ValueError('Nonfinite one-electron term.')
    return t

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
    model = 'corrected_one_body(v.copy(), eps.copy(), occupied.copy(), active.copy(), screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy()))'
    gold = '_oracle_corrected_one_body(v.copy(), eps.copy(), occupied.copy(), active.copy(), _oracle_screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy()))'
    cases = [
        {"setup": base + variants[0], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[1], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[2], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[3], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[4], "call": model, "gold_call": gold, "tol": 1e-10},
    ]
    cases.append({"setup":base+'w=v[np.ix_(active,active,active,active)].copy()\nv=v.astype(complex)\n\ndef run_model():\n    try:\n        corrected_one_body(v.copy(), eps.copy(), occupied.copy(), active.copy(), w.copy())\n    except ValueError:\n        return True\n    return False\n\ndef run_gold():\n    try:\n        _oracle_corrected_one_body(v.copy(), eps.copy(), occupied.copy(), active.copy(), w.copy())\n    except ValueError:\n        return True\n    return False\n', "call":"run_model()", "gold_call":"run_gold()"})
    return cases
