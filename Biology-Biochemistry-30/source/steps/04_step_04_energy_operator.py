"""
Form each reaction energy as an affine function of concentrations and intrinsic formation errors.

Reactant-level energies carry the pooling transformation. Membrane transport contributes pH and electrical work; shared formation uncertainties propagate through organic stoichiometry into coupled reaction-energy errors.

Returns
-------
A finite ndarray of shape (P,N,1+2M+K).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def energy_operator(stoich: "np.ndarray", reactants: "np.ndarray",
                    balanced: "np.ndarray", transports: "np.ndarray",
                    conditions: "np.ndarray", loadings: "np.ndarray",
                    rt: float, faraday: float) -> "np.ndarray":
    """
    stoich, reactants, balanced and transports follow preceding steps.
    conditions has shape (P,2,2), trailing columns pH and electrical
    potential in volts. loadings has shape (2M,K), kJ/mol per intrinsic
    factor; its rows follow organic reactant order. rt and faraday are
    positive finite constants in kJ/mol and kJ/mol/V. Return (P,N,1+2M+K):
    constant energy, the 2M concentration coefficients, then K intrinsic
    factor coefficients. Thus one packed row represents a+B*x+Q*m.
    Use the stated electrochemical signs and organic concentration
    coordinates. K may be zero. Invalid aligned energy dimensions,
    nonfinite condition/loading data or nonpositive constants raise
    ValueError; other inputs satisfy preceding contracts.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_energy_operator(stoich: "np.ndarray", reactants: "np.ndarray",
                    balanced: "np.ndarray", transports: "np.ndarray",
                    conditions: "np.ndarray", loadings: "np.ndarray",
                    rt: float, faraday: float) -> "np.ndarray":
    import numpy as np
    s, a, bal = map(lambda x: np.asarray(x, dtype=float), (stoich, reactants, balanced))
    t = np.asarray(transports, dtype=int)
    cond, load = np.asarray(conditions, dtype=float), np.asarray(loadings, dtype=float)
    p_count, m = a.shape[0], s.shape[0]
    if (cond.shape != (p_count, 2, 2) or load.ndim != 2 or load.shape[0] != m
            or bal.shape != (p_count, m+3, s.shape[1])
            or not np.isfinite(cond).all() or not np.isfinite(load).all()
            or not np.isfinite(rt) or rt <= 0 or not np.isfinite(faraday) or faraday <= 0):
        raise ValueError('Aligned finite energy data and positive constants required')
    offset = a[..., 0].reshape(p_count, m) @ s
    dph = cond[:, 1, 0] - cond[:, 0, 0]
    dpsi = cond[:, 1, 1] - cond[:, 0, 1]
    offset += faraday * dpsi[:, None] * bal[:, -1, :]
    for j, compound, src, dst, k, override in t:
        offset[:, j] -= k * (dst-src) * rt * np.log(10.) * dph
    out = np.empty((p_count, s.shape[1], 1 + m + load.shape[1]))
    out[:, :, 0] = offset
    out[:, :, 1:1+m] = rt * s.T
    out[:, :, 1+m:] = s.T @ load
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
               'u=_oracle_transport_state(a,t,h,z)\n'
               'bal=_oracle_balanced_reactions(s,a,t,u)\n'
               'cond=np.array([[[6.,0.],[7.,.04]]])\n'
               'load=np.array([[.4,-.2],[.4,-.2]])\n',
      'call': 'energy_operator(s,a,bal,t,cond,load,2.5,96.)',
      'gold_call': '_oracle_energy_operator(s,a,bal,t,cond,load,2.5,96.)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               's=np.array([[-1.,0.],[1.,0.]])\n'
               't=np.array([[0,0,0,1,0,-1],[1,-1,0,1,1,-1]])\n'
               'u=_oracle_transport_state(a,t,h,z)\n'
               'bal=_oracle_balanced_reactions(s,a,t,u)\n'
               'cond=np.array([[[6.,0.],[7.,.04]]])\n'
               'load=np.array([[.4,-.2],[.4,-.2]])\n'
               'cond[0,1]=cond[0,0]; load=np.empty((2,0))',
      'call': 'energy_operator(s,a,bal,t,cond,load,2.5,96.)',
      'gold_call': '_oracle_energy_operator(s,a,bal,t,cond,load,2.5,96.)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               's=np.array([[-1.,0.],[1.,0.]])\n'
               't=np.array([[0,0,0,1,0,-1],[1,-1,0,1,1,-1]])\n'
               'u=_oracle_transport_state(a,t,h,z)\n'
               'bal=_oracle_balanced_reactions(s,a,t,u)\n'
               'cond=np.array([[[6.,0.],[7.,.04]]])\n'
               'load=np.array([[.4,-.2],[.4,-.2]])\n'
               't[0,4]=1; u=_oracle_transport_state(a,t,h,z); '
               'bal=_oracle_balanced_reactions(s,a,t,u)\n'
               'load=np.array([[.4,-.2],[-.3,.6]])',
      'call': 'energy_operator(s,a,bal,t,cond,load,2.5,96.)',
      'gold_call': '_oracle_energy_operator(s,a,bal,t,cond,load,2.5,96.)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'a=np.array([[[[0.,.9,-.1,.1,.9]],[[0.,.2,-.8,.8,.2]]]])\n'
               'h=np.array([0.,1.]); z=h-1.\n'
               's=np.array([[-1.,0.],[1.,0.]])\n'
               't=np.array([[0,0,0,1,0,-1],[1,-1,0,1,1,-1]])\n'
               'u=_oracle_transport_state(a,t,h,z)\n'
               'bal=_oracle_balanced_reactions(s,a,t,u)\n'
               'cond=np.array([[[6.,0.],[7.,.04]]])\n'
               'load=np.array([[.4,-.2],[.4,-.2]])\n'
               'load=np.ones((3,2))\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(energy_operator, (s,a,bal,t,cond,load,2.5,96.,))',
      'gold_call': '_raises_value_error(_oracle_energy_operator, '
                   '(s,a,bal,t,cond,load,2.5,96.,))',
      'tol': 0.0}]
