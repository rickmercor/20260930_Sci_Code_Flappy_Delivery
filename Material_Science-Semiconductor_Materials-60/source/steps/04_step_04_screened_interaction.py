"""
Invert the symmetrized microscopic dielectric matrix at nonsingular momentum.

$$v_G=2\pi g/|q+G|,\quad\mathcal E=I-\sqrt v\chi\sqrt v,\quad W=\sqrt v\mathcal E^{-1}\sqrt v.$$

Returns
-------
Tuple (W complex (ng,ng), inverse_dielectric complex (ng,ng)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screened_interaction(transfer, reciprocal, polarizability, coupling):
    """Invert the symmetrized microscopic dielectric matrix at nonsingular momentum.

    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1.
    polarizability is finite Hermitian complex (ng,ng), atol=1e-11,rtol=0.
    coupling>0 is finite real. Cell area=1; v_G=2*pi*coupling/|q+G|.
    E=I-sqrt(v)*chi*sqrt(v), W=sqrt(v)*E^-1*sqrt(v).
    Require every |q+G|>1e-12 and E positive definite. The first reciprocal
    vector is the head for callers using a macroscopic dielectric function.
    Return E^-1 as well as W; epsilon_M=1/(E^-1)[0,0], not E[0,0].

    Returns
    -------
    result
        Tuple (W complex (ng,ng), inverse_dielectric complex (ng,ng)).

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-real geometry/coupling,
        nonpositive coupling, non-Hermitian chi, singular q+G, or nonpositive E.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_screened_interaction(transfer, reciprocal, polarizability, coupling):
    """Invert the symmetrized microscopic dielectric matrix at nonsingular momentum.

    transfer is finite real (2,), reciprocal finite real (ng,2), ng>=1.
    polarizability is finite Hermitian complex (ng,ng), atol=1e-11,rtol=0.
    coupling>0 is finite real. Cell area=1; v_G=2*pi*coupling/|q+G|.
    E=I-sqrt(v)*chi*sqrt(v), W=sqrt(v)*E^-1*sqrt(v).
    Require every |q+G|>1e-12 and E positive definite. The first reciprocal
    vector is the head for callers using a macroscopic dielectric function.
    Return E^-1 as well as W; epsilon_M=1/(E^-1)[0,0], not E[0,0].

    Returns
    -------
    result
        Tuple (W complex (ng,ng), inverse_dielectric complex (ng,ng)).

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-real geometry/coupling,
        nonpositive coupling, non-Hermitian chi, singular q+G, or nonpositive E.
    """
    import numpy as np
    try:
        if any((np.iscomplexobj(v) for v in (transfer, reciprocal, coupling))):
            raise ValueError('real inputs')
        q, g, c = [np.asarray(v, dtype=float) for v in (transfer, reciprocal, coupling)]
        chi = np.asarray(polarizability, dtype=complex)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if q.shape != (2,) or g.ndim != 2 or g.shape[1] != 2 or (len(g) == 0) or (c.ndim != 0) or (chi.shape != (len(g), len(g))) or any((not np.all(np.isfinite(v)) for v in (q, g, c, chi))) or (c <= 0) or (not np.allclose(chi, chi.conj().T, atol=1e-11, rtol=0)):
        raise ValueError('shape or bounds')
    norm = np.linalg.norm(q + g, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError('Coulomb singularity')
    root = np.sqrt(2 * np.pi * float(c) / norm)
    eps = np.eye(len(g)) - root[:, None] * chi * root[None, :]
    try:
        l = np.linalg.cholesky(eps)
        inv = np.linalg.solve(l.conj().T, np.linalg.solve(l, np.eye(len(g))))
    except np.linalg.LinAlgError as e:
        raise ValueError('dielectric not positive') from e
    return (root[:, None] * inv * root[None, :], inv)

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
               'rng=np.random.default_rng(2);z=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3));chi=-z@z.conj().T*.015;q=np.array([.32,.19])\n'
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
      'call': "numeric(screened_interaction(q,d['reciprocal'],chi,d['coupling']))",
      'gold_call': "numeric(_oracle_screened_interaction(q,d['reciprocal'],chi,d['coupling']))"},
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
               'rng=np.random.default_rng(2);z=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3));chi=-z@z.conj().T*.015;q=np.array([.32,.19])\n'
               'chi[:]=0.\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(screened_interaction(q,d['reciprocal'],chi,d['coupling']))",
      'gold_call': "numeric(_oracle_screened_interaction(q,d['reciprocal'],chi,d['coupling']))"},
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
               'rng=np.random.default_rng(2);z=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3));chi=-z@z.conj().T*.015;q=np.array([.32,.19])\n'
               "d['coupling']=.91\n"
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': "numeric(screened_interaction(q,d['reciprocal'],chi,d['coupling']))",
      'gold_call': "numeric(_oracle_screened_interaction(q,d['reciprocal'],chi,d['coupling']))"},
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
               'rng=np.random.default_rng(2);z=rng.normal(size=(5,3))+1j*rng.normal(size=(5,3));chi=-z@z.conj().T*.015;q=np.array([.32,.19])\n'
               'q[:]=0.\n'
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': "raises_value_error(screened_interaction, q,d['reciprocal'],chi,d['coupling'])",
      'gold_call': "raises_value_error(_oracle_screened_interaction, q,d['reciprocal'],chi,d['coupling'])"}]
