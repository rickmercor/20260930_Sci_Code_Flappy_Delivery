"""
Compute interband dipoles with the intracell position contribution to velocity.

$$v_\alpha=\partial_{k_\alpha}H-i[\tau_\alpha,H],\quad d_k=C_c^\dagger(\mathbf e\cdot\mathbf v)C_v/(E_c-E_v).$$

Returns
-------
Complex (nk,) dipole vector in lattice-length units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optical_dipole(momentum, positions, parameters, polarization):
    """Compute interband dipoles with the intracell position contribution to velocity.

    momentum and parameters follow band_structure. positions is finite real
    (2,2); polarization is a finite nonzero complex (2,) vector, normalized
    internally by its Euclidean norm. In the point-orbital, cell-periodic gauge,
    v_alpha=dH/dk_alpha-i*[diag(tau_alpha),H], with hbar=1.
    d_k=C_c,k^dagger*(sum_alpha polarization_alpha*v_alpha)*C_v,k/(Ec_k-Ev_k).
    The omitted common phase i does not affect oscillator strengths.

    Returns
    -------
    result
        Complex (nk,) dipole vector in lattice-length units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, non-real/nonfinite/wrong-shaped positions,
        or nonfinite, zero or wrong-shaped polarization.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_optical_dipole(momentum, positions, parameters, polarization):
    """Compute interband dipoles with the intracell position contribution to velocity.

    momentum and parameters follow band_structure. positions is finite real
    (2,2); polarization is a finite nonzero complex (2,) vector, normalized
    internally by its Euclidean norm. In the point-orbital, cell-periodic gauge,
    v_alpha=dH/dk_alpha-i*[diag(tau_alpha),H], with hbar=1.
    d_k=C_c,k^dagger*(sum_alpha polarization_alpha*v_alpha)*C_v,k/(Ec_k-Ev_k).
    The omitted common phase i does not affect oscillator strengths.

    Returns
    -------
    result
        Complex (nk,) dipole vector in lattice-length units.

    Raises
    ------
    ValueError
        For invalid delegated inputs, non-real/nonfinite/wrong-shaped positions,
        or nonfinite, zero or wrong-shaped polarization.
    """
    import numpy as np
    e, u, h, dh = _oracle_band_structure(momentum, parameters)
    try:
        if np.iscomplexobj(positions):
            raise ValueError('real positions')
        pos = np.asarray(positions, dtype=float)
        pol = np.asarray(polarization, dtype=complex)
    except (TypeError, ValueError) as ex:
        raise ValueError('numeric inputs') from ex
    if pos.shape != (2, 2) or pol.shape != (2,) or (not np.all(np.isfinite(pos))) or (not np.all(np.isfinite(pol))) or (np.linalg.norm(pol) == 0):
        raise ValueError('positions or polarization')
    pol = pol / np.linalg.norm(pol)
    v = np.zeros_like(h)
    for axis in range(2):
        comm = (pos[:, axis, None] - pos[None, :, axis]) * h
        v += pol[axis] * (dh[:, axis] - 1j * comm)
    return np.einsum('ki,kij,kj->k', u[:, :, 1].conj(), v, u[:, :, 0]) / (e[:, 1] - e[:, 0])

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
               '\n'
               'def dipole_summary(fn,*args):\n'
               '    return abs(fn(*args))**2\n'
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
      'call': "dipole_summary(optical_dipole, d['momentum'],d['positions'],d['parameters'],d['polarization'])",
      'gold_call': 'dipole_summary(_oracle_optical_dipole, '
                   "d['momentum'],d['positions'],d['parameters'],d['polarization'])"},
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
               "d['polarization']=np.array([0.,1.])\n"
               '\n'
               'def dipole_summary(fn,*args):\n'
               '    return abs(fn(*args))**2\n'
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
      'call': "dipole_summary(optical_dipole, d['momentum'],d['positions'],d['parameters'],d['polarization'])",
      'gold_call': 'dipole_summary(_oracle_optical_dipole, '
                   "d['momentum'],d['positions'],d['parameters'],d['polarization'])"},
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
               "d['positions'][:]=0.\n"
               '\n'
               'def dipole_summary(fn,*args):\n'
               '    return abs(fn(*args))**2\n'
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
      'call': "dipole_summary(optical_dipole, d['momentum'],d['positions'],d['parameters'],d['polarization'])",
      'gold_call': 'dipole_summary(_oracle_optical_dipole, '
                   "d['momentum'],d['positions'],d['parameters'],d['polarization'])"},
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
               "d['polarization'][:]=0.\n"
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': 'raises_value_error(optical_dipole, '
              "d['momentum'],d['positions'],d['parameters'],d['polarization'])",
      'gold_call': 'raises_value_error(_oracle_optical_dipole, '
                   "d['momentum'],d['positions'],d['parameters'],d['polarization'])"}]
