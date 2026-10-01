"""
Evaluate point-orbital density vertices in the cell-periodic orbital gauge.

$$I_G=\sum_l C_l^*C_l\prime\,e^{-i(\mathbf q+\mathbf G)\cdot\boldsymbol\tau_l}.$$

Returns
-------
Complex (n,ng) array of density vertices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def density_vertices(left, right, transfer, reciprocal, positions):
    """Evaluate point-orbital density vertices in the cell-periodic orbital gauge.

    left and right are finite complex (n,orb) coefficient arrays with n,orb>=1.
    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1,
    positions finite real (orb,2). All coefficients use the same orbital basis.
    I_G=sum_l conj(left_l)*right_l*exp[-i*(transfer+G) dot positions_l].
    Coefficients need not be normalized. Lattice constant=1.

    Returns
    -------
    result
        Complex (n,ng) array of density vertices.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, or non-real geometric arrays.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_density_vertices(left, right, transfer, reciprocal, positions):
    """Evaluate point-orbital density vertices in the cell-periodic orbital gauge.

    left and right are finite complex (n,orb) coefficient arrays with n,orb>=1.
    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1,
    positions finite real (orb,2). All coefficients use the same orbital basis.
    I_G=sum_l conj(left_l)*right_l*exp[-i*(transfer+G) dot positions_l].
    Coefficients need not be normalized. Lattice constant=1.

    Returns
    -------
    result
        Complex (n,ng) array of density vertices.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, or non-real geometric arrays.
    """
    import numpy as np
    try:
        l = np.asarray(left, dtype=complex)
        r = np.asarray(right, dtype=complex)
        if any((np.iscomplexobj(v) for v in (transfer, reciprocal, positions))):
            raise ValueError('real geometry')
        q, g, t = [np.asarray(v, dtype=float) for v in (transfer, reciprocal, positions)]
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if l.ndim != 2 or 0 in l.shape or r.shape != l.shape or (q.shape != (2,)) or (g.ndim != 2) or (g.shape[1] != 2) or (len(g) == 0) or (t.shape != (l.shape[1], 2)) or any((not np.all(np.isfinite(v)) for v in (l, r, q, g, t))):
        raise ValueError('shapes or finite inputs')
    phase = np.exp(-1j * (q[None, :] + g) @ t.T)
    return l.conj() * r @ phase.T

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
               'rng=np.random.default_rng(19);l=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));r=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));q=np.array([.23,-.41])\n'
               '\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(density_vertices(l,r,q,d['reciprocal'],d['positions']))",
      'gold_call': "numeric(_oracle_density_vertices(l,r,q,d['reciprocal'],d['positions']))"},
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
               'rng=np.random.default_rng(19);l=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));r=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));q=np.array([.23,-.41])\n'
               "d['positions'][:]=0.\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(density_vertices(l,r,q,d['reciprocal'],d['positions']))",
      'gold_call': "numeric(_oracle_density_vertices(l,r,q,d['reciprocal'],d['positions']))"},
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
               'rng=np.random.default_rng(19);l=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));r=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));q=np.array([.23,-.41])\n'
               'r=l;q[:]=0.\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(density_vertices(l,r,q,d['reciprocal'],d['positions']))",
      'gold_call': "numeric(_oracle_density_vertices(l,r,q,d['reciprocal'],d['positions']))"},
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
               'rng=np.random.default_rng(19);l=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));r=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2));q=np.array([.23,-.41])\n'
               "d['positions']=np.ones((3,2))\n"
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': "raises_value_error(density_vertices, l,r,q,d['reciprocal'],d['positions'])",
      'gold_call': "raises_value_error(_oracle_density_vertices, l,r,q,d['reciprocal'],d['positions'])"}]
