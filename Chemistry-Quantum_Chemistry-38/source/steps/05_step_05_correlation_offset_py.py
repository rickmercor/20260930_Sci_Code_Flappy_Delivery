"""
Evaluate the correlation-energy contribution to the molecular scalar offset.

This scalar is the correlation portion of the molecular offset. All reference energies and occupations remain fixed.

Returns
-------
native Python float: the correlation-energy part of the scalar offset, in hartree.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    """Evaluate the correlation-energy contribution to the molecular scalar offset.
    
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
    Return the correlation part of the molecular scalar offset under
    the supplied direct-Coulomb RPA convention. Its required full and
    active RPA problems must be strictly stable: the corresponding
    symmetric charge-channel squared-frequency matrices are positive
    definite. Empty transition spaces are valid. Do not recompute w or
    replace the supplied orbital energies by eigenvalues of t_eff.
    
    Returns
    -------
    native Python float: the correlation-energy part of the scalar offset, in hartree.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_correlation_offset(v: "np.ndarray", eps: "np.ndarray", occupied: "np.ndarray", active: "np.ndarray", w: "np.ndarray") -> float:
    import numpy as np
    _, _, gaps, k, act, ia, _ = _qc38_active_reference(v, eps, occupied, active)
    w = _qc38_screened_input(w, len(act))
    ec_full = _qc38_rpa_response(gaps, k)[2]
    _, _, ga, ka = _qc38_reference_data(w, np.asarray(eps)[act], np.array(ia, dtype=int))
    ec_active = _qc38_rpa_response(ga, ka)[2]
    value = float(ec_full - ec_active)
    if not np.isfinite(value):
        raise ValueError('Nonfinite correlation offset.')
    return value

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
    model = 'correlation_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy()))'
    gold = '_oracle_correlation_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), _oracle_screened_interaction(v.copy(), eps.copy(), occupied.copy(), active.copy()))'
    cases = [
        {"setup": base + variants[0], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[1], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[2], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[3], "call": model, "gold_call": gold, "tol": 1e-10},
        {"setup": base + variants[4], "call": model, "gold_call": gold, "tol": 1e-10},
    ]
    cases.append({"setup":base+'w=v[np.ix_(active,active,active,active)].copy()\nv=v.astype(complex)\n\ndef run_model():\n    try:\n        correlation_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), w.copy())\n    except ValueError:\n        return True\n    return False\n\ndef run_gold():\n    try:\n        _oracle_correlation_offset(v.copy(), eps.copy(), occupied.copy(), active.copy(), w.copy())\n    except ValueError:\n        return True\n    return False\n', "call":"run_model()", "gold_call":"run_gold()"})
    return cases
