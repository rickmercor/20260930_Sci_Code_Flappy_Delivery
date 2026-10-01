"""
Assemble the direct Tamm-Dancoff exciton Hamiltonian on the supplied k-grid.

$$H^{\rm exc}_{ij}=(E_{c,i}-E_{v,i})\delta_{ij}-a_{ij}^TW(k_i-k_j)b_{ij}^*/N_k.$$

Returns
-------
Complex Hermitian (nk,nk) direct exciton Hamiltonian, in energy units.     Band phases may change entries; compare gauge-invariant quantities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exciton_hamiltonian(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Assemble the direct Tamm-Dancoff exciton Hamiltonian on the supplied k-grid.

    Inputs follow gamma_interaction. Require distinct k-points, no coincident
    nonzero-transfer q+G singularities, and an inversion-symmetric reciprocal
    set containing zero first. k differences are unwrapped; do not fold them.
    Band order is v=0,c=1. For q=k_i-k_j define
    a_G=sum_l conj(C_c,i,l)*C_c,j,l*exp[+i*(q+G) dot tau_l],
    b_G=sum_l conj(C_v,i,l)*C_v,j,l*exp[+i*(q+G) dot tau_l].
    D_ij=(a @ W(q) @ conj(b))/nk; H_ij=(Ec_i-Ev_i)*delta_ij-D_ij.
    At i=j use gamma_interaction; elsewhere recompute full static RPA and W.
    Cache equal transfers using component rounding to 13 decimal places;
    evaluate a cache entry at its first unrounded transfer. Omit exchange.
    Check Hermiticity (atol=2e-10,rtol=0); then symmetrize roundoff only.

    Returns
    -------
    result
        Complex Hermitian (nk,nk) direct exciton Hamiltonian, in energy units.
        Band phases may change entries; compare gauge-invariant quantities.

    Raises
    ------
    ValueError
        For invalid delegated inputs, duplicate k-points at 13-decimal precision,
        an inversion-asymmetric/duplicate reciprocal set, or failed Hermiticity.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_exciton_hamiltonian(momentum, reciprocal, positions, parameters, coupling, spin, radial_order, angular_order):
    """Assemble the direct Tamm-Dancoff exciton Hamiltonian on the supplied k-grid.

    Inputs follow gamma_interaction. Require distinct k-points, no coincident
    nonzero-transfer q+G singularities, and an inversion-symmetric reciprocal
    set containing zero first. k differences are unwrapped; do not fold them.
    Band order is v=0,c=1. For q=k_i-k_j define
    a_G=sum_l conj(C_c,i,l)*C_c,j,l*exp[+i*(q+G) dot tau_l],
    b_G=sum_l conj(C_v,i,l)*C_v,j,l*exp[+i*(q+G) dot tau_l].
    D_ij=(a @ W(q) @ conj(b))/nk; H_ij=(Ec_i-Ev_i)*delta_ij-D_ij.
    At i=j use gamma_interaction; elsewhere recompute full static RPA and W.
    Cache equal transfers using component rounding to 13 decimal places;
    evaluate a cache entry at its first unrounded transfer. Omit exchange.
    Check Hermiticity (atol=2e-10,rtol=0); then symmetrize roundoff only.

    Returns
    -------
    result
        Complex Hermitian (nk,nk) direct exciton Hamiltonian, in energy units.
        Band phases may change entries; compare gauge-invariant quantities.

    Raises
    ------
    ValueError
        For invalid delegated inputs, duplicate k-points at 13-decimal precision,
        an inversion-asymmetric/duplicate reciprocal set, or failed Hermiticity.
    """
    import numpy as np
    e, u, _, _ = _oracle_band_structure(momentum, parameters)
    k = np.asarray(momentum, dtype=float)
    g = np.asarray(reciprocal, dtype=float)
    zero = _oracle_gamma_interaction(k, g, positions, parameters, coupling, spin, radial_order, angular_order)
    gs = {tuple(v) for v in np.round(g, 13)}
    if len(gs) != len(g) or any((tuple(-v) not in gs for v in np.round(g, 13))):
        raise ValueError('reciprocal inversion')
    if len({tuple(v) for v in np.round(k, 13)}) != len(k):
        raise ValueError('duplicate momentum')
    h = np.diag(e[:, 1] - e[:, 0]).astype(complex)
    cache = {}
    n = len(k)
    for i in range(n):
        for j in range(n):
            q = k[i] - k[j]
            if i == j:
                w = zero
            else:
                key = tuple(np.round(q, 13))
                if key not in cache:
                    chi = _oracle_static_polarizability(k, q, g, positions, parameters, spin)
                    cache[key] = _oracle_screened_interaction(q, g, chi, coupling)[0]
                w = cache[key]
            a = _oracle_density_vertices(u[i:i + 1, :, 1], u[j:j + 1, :, 1], -q, -g, positions)[0]
            b = _oracle_density_vertices(u[i:i + 1, :, 0], u[j:j + 1, :, 0], -q, -g, positions)[0]
            h[i, j] -= a @ w @ b.conj() / n
    if not np.allclose(h, h.conj().T, atol=2e-10, rtol=0):
        raise ValueError('BSE Hermiticity')
    return (h + h.conj().T) / 2

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
               "axis=2*np.pi*np.arange(-1,2)/3;d['momentum']=np.array([(x,y) for x in axis for y in axis])\n"
               '\n'
               'def exciton_summary(fn,*args):\n'
               '    h=fn(*args)\n'
               '    cycles=h[:,:,None]*h[None,:,:]*h.T[:,None,:]\n'
               '    return numeric((np.linalg.eigvalsh(h),abs(h),cycles))\n'
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
      'call': 'exciton_summary(exciton_hamiltonian, '
              "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)",
      'gold_call': 'exciton_summary(_oracle_exciton_hamiltonian, '
                   "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)"},
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
               "d['reciprocal']=d['reciprocal'][:1]\n"
               '\n'
               'def exciton_summary(fn,*args):\n'
               '    h=fn(*args)\n'
               '    cycles=h[:,:,None]*h[None,:,:]*h.T[:,None,:]\n'
               '    return numeric((np.linalg.eigvalsh(h),abs(h),cycles))\n'
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
      'call': 'exciton_summary(exciton_hamiltonian, '
              "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)",
      'gold_call': 'exciton_summary(_oracle_exciton_hamiltonian, '
                   "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)"},
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
               "d['positions'][1]=[.1,.4];d['coupling']=.25\n"
               '\n'
               'def exciton_summary(fn,*args):\n'
               '    h=fn(*args)\n'
               '    cycles=h[:,:,None]*h[None,:,:]*h.T[:,None,:]\n'
               '    return numeric((np.linalg.eigvalsh(h),abs(h),cycles))\n'
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
      'call': 'exciton_summary(exciton_hamiltonian, '
              "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)",
      'gold_call': 'exciton_summary(_oracle_exciton_hamiltonian, '
                   "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)"},
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
               "d['momentum'][1]=d['momentum'][0]\n"
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': 'raises_value_error(exciton_hamiltonian, '
              "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)",
      'gold_call': 'raises_value_error(_oracle_exciton_hamiltonian, '
                   "d['momentum'],d['reciprocal'],d['positions'],d['parameters'],d['coupling'],d['spin'],4,8)"}]
