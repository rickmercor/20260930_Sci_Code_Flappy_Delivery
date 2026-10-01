"""
Contract the basis overlaps with coherent dipole coefficients at each time.

For an isotropic ensemble, Cartesian dipole autocorrelations are averaged after forming each coherent wavepacket overlap. Cross-occupation terms remain within each polarization.

Returns
-------
correlation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_path(qs, ps, actions, Qs, Ps, logdets, labels, coefficients, hbar):
    r"""Evaluate C(t)=sum_{a=x,y,z} sum_{k,l} conj(c[k,a])*O[k,l](0,t)*c[l,a]/3.
    Coefficients are constant in the moving Hagedorn basis. Do not normalize C
    and do not delete off-diagonal occupation pairs. Initial state is path entry 0.
    The wavefunction/basis and determinant lifts have the overlap_generator contract.
    Use earlier overlap_generator and fock_overlap to construct O at each time.

    Parameters
    ----------
    qs, ps : finite real arrays, shape (N,D), N,D>=1
    actions : finite real array, shape (N,)
    Qs, Ps : finite canonical complex frames, shape (N,D,D)
    logdets : finite lifted log determinants, shape (N,)
    labels : nonnegative integer-valued array, shape (L,D), row totals<=8
    coefficients : finite complex array, shape (L,3)
    hbar : finite positive float

    Returns
    -------
    correlation : complex array, shape (N,)
        C(0)=sum(abs(coefficients)**2)/3. A zero coefficient matrix returns zeros.

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes/data or nonpositive hbar; invalid canonical
        frames/determinant lifts under overlap_generator; or invalid labels under
        fock_overlap.
    """
    return correlation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_correlation_path(qs, ps, actions, Qs, Ps, logdets, labels, coefficients, hbar):
    import numpy as np
    qs, ps, actions = [np.asarray(a, dtype=float) for a in (qs, ps, actions)]
    Qs, Ps, logdets, coefficients = [np.asarray(a, dtype=complex) for a in (Qs, Ps, logdets, coefficients)]
    labels = np.asarray(labels)
    if (qs.ndim != 2 or qs.shape[0] == 0 or qs.shape[1] == 0 or ps.shape != qs.shape
            or actions.shape != (qs.shape[0],) or logdets.shape != actions.shape
            or Qs.shape != (qs.shape[0], qs.shape[1], qs.shape[1]) or Ps.shape != Qs.shape
            or labels.ndim != 2 or labels.shape[1] != qs.shape[1]
            or coefficients.shape != (labels.shape[0], 3)
            or not all(np.all(np.isfinite(a)) for a in (qs, ps, actions, Qs, Ps, logdets, labels, coefficients))
            or not np.isfinite(hbar) or hbar <= 0):
        raise ValueError("Invalid correlation inputs")
    answer = np.empty(len(qs), dtype=complex)
    initial = (qs[0], ps[0], Qs[0], Ps[0], actions[0], logdets[0])
    for t in range(len(qs)):
        current = (qs[t], ps[t], Qs[t], Ps[t], actions[t], logdets[t])
        B, b, overlap = _oracle_overlap_generator(*initial, *current, hbar)
        matrix = _oracle_fock_overlap(B, b, overlap, labels, labels)
        answer[t] = np.einsum("ia,ij,ja->", coefficients.conj(), matrix, coefficients) / 3.0
    return answer

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
Q=np.array([[0.9,0.12],[0.12,1.15]]);P=1j*np.linalg.inv(Q)
times=np.linspace(0,1.2,7)
M=np.array([[1.2,0.13],[0.13,0.85]]);K=np.array([[1.7,-0.21],[-0.21,0.8]])
Qs,Ps,logs=_oracle_width_path(M,K,Q,P,times)
qs=np.column_stack((0.2*np.sin(times),0.1*np.cos(times)))
ps=np.column_stack((0.3*np.cos(times),-0.15*np.sin(times)))
S=-0.45*times
labels=np.array([[0,0],[1,0],[0,2],[2,1]])
coef=np.array([[0.4,0.2,-0.1],[0.2+0.1j,-0.3,0.1],[0.15,0.1,-0.2j],[-0.05,0.02,0.1]])
""",
            "call": 'correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
            "gold_call": '_oracle_correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
Q=np.array([[0.9,0.12],[0.12,1.15]]);P=1j*np.linalg.inv(Q)
times=np.linspace(0,1.2,7)[:1]
M=np.array([[1.2,0.13],[0.13,0.85]]);K=np.array([[1.7,-0.21],[-0.21,0.8]])
Qs,Ps,logs=_oracle_width_path(M,K,Q,P,times)
qs=np.column_stack((0.2*np.sin(times),0.1*np.cos(times)))
ps=np.column_stack((0.3*np.cos(times),-0.15*np.sin(times)))
S=-0.45*times
labels=np.array([[0,0],[1,0],[0,2],[2,1]])
coef=np.array([[0.4,0.2,-0.1],[0.2+0.1j,-0.3,0.1],[0.15,0.1,-0.2j],[-0.05,0.02,0.1]])
""",
            "call": 'correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
            "gold_call": '_oracle_correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
Q=np.array([[0.9,0.12],[0.12,1.15]]);P=1j*np.linalg.inv(Q)
times=np.linspace(0,1.2,7)
M=np.array([[1.2,0.13],[0.13,0.85]]);K=np.array([[1.7,-0.21],[-0.21,0.8]])
Qs,Ps,logs=_oracle_width_path(M,K,Q,P,times)
qs=np.column_stack((0.2*np.sin(times),0.1*np.cos(times)))
ps=np.column_stack((0.3*np.cos(times),-0.15*np.sin(times)))
S=-0.45*times
labels=np.array([[0,0],[1,0],[0,2],[2,1]])
coef=np.array([[0.4,0.2,-0.1],[0.2+0.1j,-0.3,0.1],[0.15,0.1,-0.2j],[-0.05,0.02,0.1]])

coef[:]=0
""",
            "call": 'correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
            "gold_call": '_oracle_correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
Q=np.array([[0.9,0.12],[0.12,1.15]]);P=1j*np.linalg.inv(Q)
times=np.linspace(0,1.2,7)
M=np.array([[1.2,0.13],[0.13,0.85]]);K=np.array([[1.7,-0.21],[-0.21,0.8]])
Qs,Ps,logs=_oracle_width_path(M,K,Q,P,times)
qs=np.column_stack((0.2*np.sin(times),0.1*np.cos(times)))
ps=np.column_stack((0.3*np.cos(times),-0.15*np.sin(times)))
S=-0.45*times
labels=np.array([[0,0],[1,0],[0,2],[2,1]])
coef=np.array([[0.4,0.2,-0.1],[0.2+0.1j,-0.3,0.1],[0.15,0.1,-0.2j],[-0.05,0.02,0.1]])

coef[:,1]*=1j;coef[1,0]*=np.exp(.7j);coef[2,0]*=-1
""",
            "call": 'correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
            "gold_call": '_oracle_correlation_path(qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
Q=np.array([[0.9,0.12],[0.12,1.15]]);P=1j*np.linalg.inv(Q)
times=np.linspace(0,1.2,7)
M=np.array([[1.2,0.13],[0.13,0.85]]);K=np.array([[1.7,-0.21],[-0.21,0.8]])
Qs,Ps,logs=_oracle_width_path(M,K,Q,P,times)
qs=np.column_stack((0.2*np.sin(times),0.1*np.cos(times)))
ps=np.column_stack((0.3*np.cos(times),-0.15*np.sin(times)))
S=-0.45*times
labels=np.array([[0,0],[1,0],[0,2],[2,1]])
coef=np.array([[0.4,0.2,-0.1],[0.2+0.1j,-0.3,0.1],[0.15,0.1,-0.2j],[-0.05,0.02,0.1]])

coef=coef[:,:2]

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(correlation_path, qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
            "gold_call": '_status(_oracle_correlation_path, qs, ps, S, Qs, Ps, logs, labels, coef, 0.7)',
        },
    ]
