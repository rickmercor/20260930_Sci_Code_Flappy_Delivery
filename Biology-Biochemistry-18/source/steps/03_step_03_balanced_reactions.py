"""
Construct organic, buffered-proton and membrane-charge coefficients for each condition.

Reactant pooling gives fractional bound-proton counts. Chemical re-equilibration and the selected transported species together determine the proton coefficients; charge transfer follows from the interior species and proton balance.

Returns
-------
A finite ndarray of shape (P,2M+3,N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def balanced_reactions(stoich: "np.ndarray", reactants: "np.ndarray",
                       transports: "np.ndarray", transported: "np.ndarray") -> "np.ndarray":
    """
    stoich is the (2M,N) organic matrix, exterior reactants followed by
    interior reactants. reactants and transports follow steps 1 and 2.
    transported has shape (P,T,4) from step 2. Return (P,2M+3,N), rows
    organic, buffered exterior H+, buffered interior H+, membrane charge.
    The charge coefficient is positive for exterior-to-interior transfer.
    Chemical reactions without transport records remain compartment-local.
    The proton rows encode re-equilibration plus explicitly transported
    protons; they are buffered and are not additional steady-state rows.
    Inputs obey chemical skeleton conservation and aligned species counts.
    Shape mismatches, nonfinite stoichiometry/transport data and reaction
    indices outside the matrix raise ValueError. Numerical zeros below
    1e-13 may be returned as exact zero.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_balanced_reactions(stoich: "np.ndarray", reactants: "np.ndarray",
                       transports: "np.ndarray", transported: "np.ndarray") -> "np.ndarray":
    import numpy as np
    s, a = np.asarray(stoich, dtype=float), np.asarray(reactants, dtype=float)
    t, u = np.asarray(transports, dtype=int), np.asarray(transported, dtype=float)
    if (s.ndim != 2 or a.ndim != 4 or s.shape[0] != 2 * a.shape[2]
            or a.shape[1] != 2 or t.ndim != 2 or t.shape[1] != 6
            or u.shape != (a.shape[0], t.shape[0], 4)
            or np.any((t[:, 0] < 0) | (t[:, 0] >= s.shape[1]))
            or not np.isfinite(s).all() or not np.isfinite(u).all()):
        raise ValueError('Aligned reaction, reactant and transport dimensions required')
    p_count, m, r_count = a.shape[0], a.shape[2], s.shape[1]
    out = np.zeros((p_count, 2*m + 3, r_count))
    out[:, :2*m] = s
    for p in range(p_count):
        protons = np.stack([-a[p, c, :, 1] @ s[c*m:(c+1)*m] for c in range(2)])
        for row, (j, compound, src, dst, k, override) in enumerate(t):
            carried = u[p, row, 0] + u[p, row, 2]
            protons[src, j] -= carried
            protons[dst, j] += carried
        out[p, 2*m:2*m+2] = protons
        out[p, -1] = a[p, 1, :, 2] @ s[m:] + protons[1]
    out[np.abs(out) < 1e-13] = 0.
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               's=np.array([[-1.,0.],[1.,0.]])\n'
               't=np.array([[0,0,0,1,0,-1],[1,-1,0,1,1,-1]])\n'
               'u=_oracle_transport_state(a,t,h,z)\n',
      'call': 'balanced_reactions(s,a,t,u)',
      'gold_call': '_oracle_balanced_reactions(s,a,t,u)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               's=np.array([[-1.,0.],[1.,0.]])\n'
               't=np.array([[0,0,0,1,0,-1],[1,-1,0,1,1,-1]])\n'
               'u=_oracle_transport_state(a,t,h,z)\n'
               't[0,2:4]=[1,0]; s[:,0]*=-1.\n'
               'u=_oracle_transport_state(a,t,h,z)',
      'call': 'balanced_reactions(s,a,t,u)',
      'gold_call': '_oracle_balanced_reactions(s,a,t,u)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               's=np.array([[-1.],[1.],[0.],[0.]])\n'
               'a=np.zeros((1,2,2,5)); a[:,:,:,1]=[[[.8,.3],[.2,.4]]]; '
               'a[:,:,:,2]=a[:,:,:,1]-1.\n'
               't=np.empty((0,6),dtype=int); u=np.empty((1,0,4))',
      'call': 'balanced_reactions(s,a,t,u)',
      'gold_call': '_oracle_balanced_reactions(s,a,t,u)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               's=np.array([[-1.,0.],[1.,0.]])\n'
               't=np.array([[0,0,0,1,0,-1],[1,-1,0,1,1,-1]])\n'
               'u=_oracle_transport_state(a,t,h,z)\n'
               's=np.zeros((3,2))\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(balanced_reactions, (s,a,t,u,))',
      'gold_call': '_raises_value_error(_oracle_balanced_reactions, (s,a,t,u,))',
      'tol': 0.0}]
