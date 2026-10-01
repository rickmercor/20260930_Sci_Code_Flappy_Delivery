"""
Evaluate the reference-energy contribution to the molecular scalar offset.

This scalar is the non-correlation portion of the molecular offset. Occupied environment orbitals and a screening-dependent active reference contribution are supported.

Returns
-------
native Python float: the reference-energy part of the scalar offset, in hartree.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reference_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    """Evaluate the reference-energy contribution to the molecular scalar offset.
    
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
    Return the reference-energy part of the molecular scalar offset,
    excluding its RPA correlation part. Occupied environment orbitals
    are supported. The bare reference h is defined by the problem.
    Do not recompute w or test that it equals a particular screening
    calculation. Nuclear repulsion is zero.
    
    Returns
    -------
    native Python float: the reference-energy part of the scalar offset, in hartree.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reference_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    import numpy as np
    h, _, _, _, act, ia, ie = _qc38_active_reference(v, eps, occupied, active)
    v = np.asarray(v, dtype=float)
    w = _qc38_screened_input(w, len(act))
    delta = w - v[np.ix_(act,act,act,act)]
    value = 2*sum(h[i,i] for i in ie)
    value += sum(2*v[i,i,j,j]-v[i,j,j,i] for i in ie for j in ie)
    value += sum(2*delta[i,i,j,j]-delta[i,j,j,i] for i in ia for j in ia)
    if not np.isfinite(value):
        raise ValueError('Nonfinite reference offset.')
    return float(value)

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
    model = 'reference_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy()))'
    gold = '_oracle_reference_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), _oracle_screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy()))'
    cases = [
        {"setup": base + variants[0], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[1], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[2], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[3], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[4], "call": model, "gold_call": gold, "tol": 1e-10},
    ]
    cases.append({"setup":base+'w=v[np.ix_(active,active,active,active)].copy()\nv=v.astype(complex)\n\ndef run_model():\n    try:\n        reference_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), w.copy())\n    except ValueError:\n        return True\n    return False\n\ndef run_gold():\n    try:\n        _oracle_reference_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), w.copy())\n    except ValueError:\n        return True\n    return False\n', "call":"run_model()", "gold_call":"run_gold()"})
    return cases
