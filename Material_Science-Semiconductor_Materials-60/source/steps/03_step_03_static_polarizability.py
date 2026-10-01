"""
Compute the static independent-particle polarizability, including both band directions.

$$\chi_{GG\prime}(q)=\frac{g_s}{N_k}\sum_{k,n\ne m}\frac{f_n-f_m}{E_n(k)-E_m(k+q)}I_G I_{G\prime}^*.$$

Returns
-------
Complex Hermitian (ng,ng) polarizability in inverse energy units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_polarizability(momentum, transfer, reciprocal, positions, parameters, spin=2.0):
    """Compute the static independent-particle polarizability, including both band directions.

    momentum/parameters follow band_structure; transfer, reciprocal and
    positions follow density_vertices, with positions shape (2,2).
    spin is a finite positive real scalar. At zero temperature f_v=1,f_c=0.
    Evaluate bands at k and k+q without folding q or shifting reciprocal indices.
    chi_GG'=spin/nk * sum_{k,n!=m} [(f_n-f_m)/(E_nk-E_m,k+q)]
    * I_G(nk,mk+q)*conj(I_G'(nk,mk+q)). Include (n,m)=(0,1),(1,0).
    Cell area=1. Do not replace this matrix by its head or use a factor-two
    shortcut for the two transition directions.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) polarizability in inverse energy units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, nonpositive/non-real/nonfinite spin,
        or an interband energy denominator with absolute value <=1e-10.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_static_polarizability(momentum, transfer, reciprocal, positions, parameters, spin=2.0):
    """Compute the static independent-particle polarizability, including both band directions.

    momentum/parameters follow band_structure; transfer, reciprocal and
    positions follow density_vertices, with positions shape (2,2).
    spin is a finite positive real scalar. At zero temperature f_v=1,f_c=0.
    Evaluate bands at k and k+q without folding q or shifting reciprocal indices.
    chi_GG'=spin/nk * sum_{k,n!=m} [(f_n-f_m)/(E_nk-E_m,k+q)]
    * I_G(nk,mk+q)*conj(I_G'(nk,mk+q)). Include (n,m)=(0,1),(1,0).
    Cell area=1. Do not replace this matrix by its head or use a factor-two
    shortcut for the two transition directions.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) polarizability in inverse energy units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, nonpositive/non-real/nonfinite spin,
        or an interband energy denominator with absolute value <=1e-10.
    """
    import numpy as np
    try:
        if np.iscomplexobj(spin):
            raise ValueError('real spin')
        s = np.asarray(spin, dtype=float)
        q = np.asarray(transfer, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if s.ndim != 0 or not np.isfinite(s) or s <= 0 or (q.shape != (2,)) or np.iscomplexobj(transfer) or (not np.all(np.isfinite(q))):
        raise ValueError('spin or transfer')
    e, u, _, _ = _oracle_band_structure(momentum, parameters)
    ep, up, _, _ = _oracle_band_structure(np.asarray(momentum, dtype=float) + q, parameters)
    chi = None
    for n, m, sign in [(0, 1, 1.0), (1, 0, -1.0)]:
        vertex = _oracle_density_vertices(u[:, :, n], up[:, :, m], q, reciprocal, positions)
        den = e[:, n] - ep[:, m]
        if np.any(abs(den) <= 1e-10):
            raise ValueError('singular transition')
        term = np.einsum('k,kg,kh->gh', sign / den, vertex, vertex.conj())
        chi = term if chi is None else chi + term
    return float(s) * chi / len(e)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'def make_inputs():\n'
               '    """Synthetic two-orbital square-lattice semiconductor, a=hbar=cell area=1."""\n'
               '    import numpy as np\n'
               '    n=7;axis=2*np.pi*(np.arange(n)-(n-1)/2)/n\n'
               '    momentum=np.array([(x,y) for x in axis for y in axis])\n'
               '    return dict(momentum=momentum,\n'
               '                reciprocal=2*np.pi*np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),\n'
               '                positions=np.array([[0.,0.],[.23,.31]]),\n'
               '                parameters=np.array([.92,.31,.24,.12,.43,.54,.29]),\n'
               '                coupling=.48,spin=2.,radial_order=12,angular_order=20,\n'
               '                polarization=np.array([1.,.7j]),window=np.array([1.10,2.20]),broadening=.065)\n'
               '\n'
               'd=make_inputs()\n'
               'q=np.array([.4,-.2])\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(static_polarizability(d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin']))",
      'gold_call': "numeric(_oracle_static_polarizability(d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin']))"},
     {'setup': 'import numpy as np\n'
               'def make_inputs():\n'
               '    """Synthetic two-orbital square-lattice semiconductor, a=hbar=cell area=1."""\n'
               '    import numpy as np\n'
               '    n=7;axis=2*np.pi*(np.arange(n)-(n-1)/2)/n\n'
               '    momentum=np.array([(x,y) for x in axis for y in axis])\n'
               '    return dict(momentum=momentum,\n'
               '                reciprocal=2*np.pi*np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),\n'
               '                positions=np.array([[0.,0.],[.23,.31]]),\n'
               '                parameters=np.array([.92,.31,.24,.12,.43,.54,.29]),\n'
               '                coupling=.48,spin=2.,radial_order=12,angular_order=20,\n'
               '                polarization=np.array([1.,.7j]),window=np.array([1.10,2.20]),broadening=.065)\n'
               '\n'
               'd=make_inputs()\n'
               'q=np.zeros(2)\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(static_polarizability(d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin']))",
      'gold_call': "numeric(_oracle_static_polarizability(d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin']))"},
     {'setup': 'import numpy as np\n'
               'def make_inputs():\n'
               '    """Synthetic two-orbital square-lattice semiconductor, a=hbar=cell area=1."""\n'
               '    import numpy as np\n'
               '    n=7;axis=2*np.pi*(np.arange(n)-(n-1)/2)/n\n'
               '    momentum=np.array([(x,y) for x in axis for y in axis])\n'
               '    return dict(momentum=momentum,\n'
               '                reciprocal=2*np.pi*np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),\n'
               '                positions=np.array([[0.,0.],[.23,.31]]),\n'
               '                parameters=np.array([.92,.31,.24,.12,.43,.54,.29]),\n'
               '                coupling=.48,spin=2.,radial_order=12,angular_order=20,\n'
               '                polarization=np.array([1.,.7j]),window=np.array([1.10,2.20]),broadening=.065)\n'
               '\n'
               'd=make_inputs()\n'
               "q=np.array([-.6,.3]);d['spin']=1.\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(static_polarizability(d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin']))",
      'gold_call': "numeric(_oracle_static_polarizability(d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin']))"},
     {'setup': 'import numpy as np\n'
               'def make_inputs():\n'
               '    """Synthetic two-orbital square-lattice semiconductor, a=hbar=cell area=1."""\n'
               '    import numpy as np\n'
               '    n=7;axis=2*np.pi*(np.arange(n)-(n-1)/2)/n\n'
               '    momentum=np.array([(x,y) for x in axis for y in axis])\n'
               '    return dict(momentum=momentum,\n'
               '                reciprocal=2*np.pi*np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),\n'
               '                positions=np.array([[0.,0.],[.23,.31]]),\n'
               '                parameters=np.array([.92,.31,.24,.12,.43,.54,.29]),\n'
               '                coupling=.48,spin=2.,radial_order=12,angular_order=20,\n'
               '                polarization=np.array([1.,.7j]),window=np.array([1.10,2.20]),broadening=.065)\n'
               '\n'
               'd=make_inputs()\n'
               "q=np.array([.4,-.2]);d['spin']=-1.\n"
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': 'raises_value_error(static_polarizability, '
              "d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin'])",
      'gold_call': 'raises_value_error(_oracle_static_polarizability, '
                   "d['momentum'],q,d['reciprocal'],d['positions'],d['parameters'],d['spin'])"}]
