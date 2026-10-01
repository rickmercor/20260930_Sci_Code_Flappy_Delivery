"""
Regularize the zero-transfer cell by a circular microscopic head average.

$$R=\sqrt{4\pi/N_k},\quad\overline W_{00}=\frac1{\pi R^2}\int_0^Rr\,dr\int_0^{2\pi}W_{00}(q)\,d\theta,\quad W_{0G}=W_{G0}=0\ (G\ne0).$$

Returns
-------
Complex Hermitian (ng,ng) regularized zero-transfer interaction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gamma_interaction(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Regularize the zero-transfer cell by a circular microscopic head average.

    Inputs follow static_polarizability and screened_interaction. reciprocal[0]
    must be zero to atol=1e-14 and all other vectors nonzero (>1e-12).
    radial_order>=2 and angular_order>=4 are integers. For nk k-points,
    replace their reciprocal cell of area (2*pi)^2/nk by a circle of radius
    R=sqrt(4*pi/nk). Set W00_bar=(1/pi/R^2)*int_0^R r dr int_0^{2pi} W00(q)dtheta.
    Use radial_order Gauss-Legendre nodes on [0,R] and equally spaced angles
    theta_j=2*pi*j/angular_order. At each node recompute the full chi and W.
    At q=0 use chi's nonzero-G body to obtain the screened body. Set wings
    W0G=WG0=0 for G!=0. This benchmark uses a direct circular average, not a
    fitted linear small-q dielectric approximation.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) regularized zero-transfer interaction.

    Raises
    ------
    ValueError
        For invalid delegated inputs, invalid integer quadrature orders,
        a nonzero first reciprocal vector, or an additional zero reciprocal vector.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_gamma_interaction(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Regularize the zero-transfer cell by a circular microscopic head average.

    Inputs follow static_polarizability and screened_interaction. reciprocal[0]
    must be zero to atol=1e-14 and all other vectors nonzero (>1e-12).
    radial_order>=2 and angular_order>=4 are integers. For nk k-points,
    replace their reciprocal cell of area (2*pi)^2/nk by a circle of radius
    R=sqrt(4*pi/nk). Set W00_bar=(1/pi/R^2)*int_0^R r dr int_0^{2pi} W00(q)dtheta.
    Use radial_order Gauss-Legendre nodes on [0,R] and equally spaced angles
    theta_j=2*pi*j/angular_order. At each node recompute the full chi and W.
    At q=0 use chi's nonzero-G body to obtain the screened body. Set wings
    W0G=WG0=0 for G!=0. This benchmark uses a direct circular average, not a
    fitted linear small-q dielectric approximation.

    Returns
    -------
    result
        Complex Hermitian (ng,ng) regularized zero-transfer interaction.

    Raises
    ------
    ValueError
        For invalid delegated inputs, invalid integer quadrature orders,
        a nonzero first reciprocal vector, or an additional zero reciprocal vector.
    """
    import numpy as np
    if any((isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in (radial_order, angular_order))) or radial_order < 2 or angular_order < 4:
        raise ValueError('quadrature orders')
    chi = _oracle_static_polarizability(momentum, [0.0, 0.0], reciprocal, positions, parameters, spin)
    g = np.asarray(reciprocal, dtype=float)
    if np.linalg.norm(g[0]) > 1e-14 or np.any(np.linalg.norm(g[1:], axis=1) <= 1e-12):
        raise ValueError('head index')
    radius = np.sqrt(4 * np.pi / len(momentum))
    x, w = np.polynomial.legendre.leggauss(int(radial_order))
    r = radius * (x + 1) / 2
    wr = radius * w / 2
    head = 0.0
    for ri, wi in zip(r, wr):
        for j in range(angular_order):
            theta = 2 * np.pi * j / angular_order
            q = ri * np.array([np.cos(theta), np.sin(theta)])
            response = _oracle_static_polarizability(momentum, q, g, positions, parameters, spin)
            screened, _ = _oracle_screened_interaction(q, g, response, coupling)
            head += 2 * ri * wi * float(screened[0, 0].real) / (radius ** 2 * angular_order)
    out = np.zeros_like(chi)
    out[0, 0] = head
    if len(g) > 1:
        out[1:, 1:] = _oracle_screened_interaction([0.0, 0.0], g[1:], chi[1:, 1:], coupling)[0]
    return out

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
      'call': "numeric(gamma_interaction(d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8))",
      'gold_call': "numeric(_oracle_gamma_interaction(d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8))"},
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
               "d['reciprocal']=d['reciprocal'][:1]\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(gamma_interaction(d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8))",
      'gold_call': "numeric(_oracle_gamma_interaction(d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8))"},
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
               "d['coupling']=.19\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(gamma_interaction(d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8))",
      'gold_call': "numeric(_oracle_gamma_interaction(d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8))"},
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
               "d['reciprocal'][0]=[.1,0.]\n"
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': 'raises_value_error(gamma_interaction, '
              "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)",
      'gold_call': 'raises_value_error(_oracle_gamma_interaction, '
                   "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)"}]
