"""
Diagonalize the periodic two-orbital Bloch Hamiltonian and its derivatives.

$$H=\mathbf d\cdot\boldsymbol\sigma,\quad E_v=-|\mathbf d|,\quad E_c=|\mathbf d|.$$

Returns
-------
Tuple (energies real (nk,2), eigenvectors complex (nk,2,2) in columns,     H complex (nk,2,2), dH complex (nk,2,2,2), derivative axis second).     Eigenvector phases are arbitrary; only physical invariants are specified.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def band_structure(momentum, parameters):
    """Diagonalize the periodic two-orbital Bloch Hamiltonian and its derivatives.

    momentum is finite real (nk,2), nk>=1, in inverse lattice units.
    parameters is finite real (7,)=(m,bx,by,a,tx,ty,tz).
    H=dx*sigma_x+dy*sigma_y+dz*sigma_z, where
    dx=a+tx*cos(kx)+0.19*cos(ky), dy=ty*sin(kx)+tz*sin(ky),
    dz=m+bx*(1-cos(kx))+by*(1-cos(ky)). Energy unit is fixed by parameters;
    lattice constant and hbar are 1. Use ascending energies (valence,conduction).
    dH/dkx uses (-tx*sin(kx),ty*cos(kx),bx*sin(kx));
    dH/dky uses (-0.19*sin(ky),tz*cos(ky),by*sin(ky)).

    Returns
    -------
    result
        Tuple (energies real (nk,2), eigenvectors complex (nk,2,2) in columns,
        H complex (nk,2,2), dH complex (nk,2,2,2), derivative axis second).
        Eigenvector phases are arbitrary; only physical invariants are specified.

    Raises
    ------
    ValueError
        For non-real/nonfinite inputs, wrong shapes, or a sampled gap <=1e-10.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_band_structure(momentum, parameters):
    """Diagonalize the periodic two-orbital Bloch Hamiltonian and its derivatives.

    momentum is finite real (nk,2), nk>=1, in inverse lattice units.
    parameters is finite real (7,)=(m,bx,by,a,tx,ty,tz).
    H=dx*sigma_x+dy*sigma_y+dz*sigma_z, where
    dx=a+tx*cos(kx)+0.19*cos(ky), dy=ty*sin(kx)+tz*sin(ky),
    dz=m+bx*(1-cos(kx))+by*(1-cos(ky)). Energy unit is fixed by parameters;
    lattice constant and hbar are 1. Use ascending energies (valence,conduction).
    dH/dkx uses (-tx*sin(kx),ty*cos(kx),bx*sin(kx));
    dH/dky uses (-0.19*sin(ky),tz*cos(ky),by*sin(ky)).

    Returns
    -------
    result
        Tuple (energies real (nk,2), eigenvectors complex (nk,2,2) in columns,
        H complex (nk,2,2), dH complex (nk,2,2,2), derivative axis second).
        Eigenvector phases are arbitrary; only physical invariants are specified.

    Raises
    ------
    ValueError
        For non-real/nonfinite inputs, wrong shapes, or a sampled gap <=1e-10.
    """
    import numpy as np
    try:
        if np.iscomplexobj(momentum) or np.iscomplexobj(parameters):
            raise ValueError('real inputs')
        k = np.asarray(momentum, dtype=float)
        p = np.asarray(parameters, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if k.ndim != 2 or k.shape[1] != 2 or len(k) == 0 or (p.shape != (7,)) or (not np.all(np.isfinite(k))) or (not np.all(np.isfinite(p))):
        raise ValueError('shape or finite inputs')
    x, y = k.T
    m, bx, by, a, tx, ty, tz = p

    def matrix(dx, dy, dz):
        h = np.empty((len(k), 2, 2), dtype=complex)
        h[:, 0, 0] = dz
        h[:, 1, 1] = -dz
        h[:, 0, 1] = dx - 1j * dy
        h[:, 1, 0] = dx + 1j * dy
        return h
    h = matrix(a + tx * np.cos(x) + 0.19 * np.cos(y), ty * np.sin(x) + tz * np.sin(y), m + bx * (1 - np.cos(x)) + by * (1 - np.cos(y)))
    dh = np.stack([matrix(-tx * np.sin(x), ty * np.cos(x), bx * np.sin(x)), matrix(-0.19 * np.sin(y), tz * np.cos(y), by * np.sin(y))], axis=1)
    e, u = np.linalg.eigh(h)
    if np.any(e[:, 1] - e[:, 0] <= 1e-10):
        raise ValueError('closed gap')
    return (e, u, h, dh)

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
               'def band_summary(fn,*args):\n'
               '    e,u,h,dh=fn(*args)\n'
               "    proj=np.einsum('kia,kja->kaij',u,u.conj())\n"
               '    return numeric((e,proj,h,dh))\n'
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
      'call': "band_summary(band_structure, d['momentum'],d['parameters'])",
      'gold_call': "band_summary(_oracle_band_structure, d['momentum'],d['parameters'])"},
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
               '\n'
               'def band_summary(fn,*args):\n'
               '    e,u,h,dh=fn(*args)\n'
               "    proj=np.einsum('kia,kja->kaij',u,u.conj())\n"
               '    return numeric((e,proj,h,dh))\n'
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
      'call': "band_summary(band_structure, np.array([[0.,0.],[.3,-.9]]),d['parameters'])",
      'gold_call': "band_summary(_oracle_band_structure, np.array([[0.,0.],[.3,-.9]]),d['parameters'])"},
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
               "d['parameters'][0]=1.7\n"
               '\n'
               'def band_summary(fn,*args):\n'
               '    e,u,h,dh=fn(*args)\n'
               "    proj=np.einsum('kia,kja->kaij',u,u.conj())\n"
               '    return numeric((e,proj,h,dh))\n'
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
      'call': "band_summary(band_structure, d['momentum'][:5],d['parameters'])",
      'gold_call': "band_summary(_oracle_band_structure, d['momentum'][:5],d['parameters'])"},
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
               '\n'
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': "raises_value_error(band_structure, np.array([[1j,0.]],dtype=object),d['parameters'])",
      'gold_call': 'raises_value_error(_oracle_band_structure, '
                   "np.array([[1j,0.]],dtype=object),d['parameters'])"}]
