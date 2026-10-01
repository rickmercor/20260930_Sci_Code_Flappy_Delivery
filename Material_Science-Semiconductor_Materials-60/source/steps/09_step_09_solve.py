"""
Return the exciton oscillator fraction for the supplied semiconductor model.

$$\text{microscopic screening}\to H^{\rm exc},\quad\text{cell-gauge velocity}\to d,\quad(H^{\rm exc},d)\to R.$$

Returns
-------
Float normalized oscillator fraction in the prescribed energy window.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(data):
    """Return the exciton oscillator fraction for the supplied semiconductor model.

    data is the make_inputs() dictionary with keys momentum, reciprocal,
    positions, parameters, coupling, spin, radial_order, angular_order,
    polarization, window, broadening. Every value follows the preceding
    contracts. Assemble the direct microscopic-screened Tamm-Dancoff matrix,
    compute the position-corrected dipole, and integrate the oscillator measure.
    The finite k/G grids and circle quadrature are fixed benchmark definitions.

    Returns
    -------
    result
        Float normalized oscillator fraction in the prescribed energy window.

    Raises
    ------
    ValueError
        For a non-dictionary input, absent required key, or any invalid delegated input.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve(data):
    """Return the exciton oscillator fraction for the supplied semiconductor model.

    data is the make_inputs() dictionary with keys momentum, reciprocal,
    positions, parameters, coupling, spin, radial_order, angular_order,
    polarization, window, broadening. Every value follows the preceding
    contracts. Assemble the direct microscopic-screened Tamm-Dancoff matrix,
    compute the position-corrected dipole, and integrate the oscillator measure.
    The finite k/G grids and circle quadrature are fixed benchmark definitions.

    Returns
    -------
    result
        Float normalized oscillator fraction in the prescribed energy window.

    Raises
    ------
    ValueError
        For a non-dictionary input, absent required key, or any invalid delegated input.
    """
    keys = 'momentum reciprocal positions parameters coupling spin radial_order angular_order polarization window broadening'.split()
    if not isinstance(data, dict) or any((k not in data for k in keys)):
        raise ValueError('missing data')
    h = _oracle_exciton_hamiltonian(*[data[k] for k in keys[:8]])
    d = _oracle_optical_dipole(data['momentum'], data['positions'], data['parameters'], data['polarization'])
    return _oracle_spectral_fraction(h, d, data['window'], data['broadening'])

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
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': 'numeric(solve(d))',
      'gold_call': 'numeric(_oracle_solve(d))'},
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
               "axis=2*np.pi*np.arange(-1,2)/3;d['momentum']=np.array([(x,y) for x in axis for y in axis])\n"
               "d['radial_order']=5;d['angular_order']=8;d['polarization']=np.array([1.,0.])\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': 'numeric(solve(d))',
      'gold_call': 'numeric(_oracle_solve(d))'},
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
               "axis=2*np.pi*np.arange(-1,2)/3;d['momentum']=np.array([(x,y) for x in axis for y in axis])\n"
               "d['reciprocal']=d['reciprocal'][:1];d['radial_order']=5;d['angular_order']=8\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': 'numeric(solve(d))',
      'gold_call': 'numeric(_oracle_solve(d))'},
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
               "d['broadening']=-.1\n"
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': 'raises_value_error(solve, d)',
      'gold_call': 'raises_value_error(_oracle_solve, d)'}]
