"""
Construct the field derivatives of the Gaussian-QEq energy coefficients.

Field-dependent Gaussian QEq energy at one snapshot. Atomic positions and widths are held fixed when differentiating the external field. The supplied scalar and vector readouts replace network training.

Returns
-------
Tuple (A,A1,A2,b,b1,b2,c2) of real arrays with shapes (n,n), (3,n,n), (3,3,n,n), (n,), (3,n), (3,3,n), (3,3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def field_matrices(z: float, field: 'np.ndarray', positions: 'np.ndarray', widths: 'np.ndarray', parameters: 'np.ndarray', curvature: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    """Construct the field jet of U=c+b.T@q+q.T@A@q/2, in atomic units.
    
    Parameters
    ----------
    z : float
        Finite environmental descriptor.
    field, positions, widths, parameters, curvature : numpy.ndarray
        Real finite molecular inputs with the shapes and constraints below.

    z is a finite scalar; z is a finite scalar; field has shape (3,); positions (n,3), n>=1;
    widths (n,) are positive; parameters (n,9) have columns
    [chi,h,s,u,t,w,eta,l,k]; curvature (2,3,3) contains symmetric K0,K1.
    All arrays are real and finite. Put v_i=R_i-mean_j(R_j), d_i=v_i.field.
    The Gaussian matrix G has G_ii=1/(sqrt(pi)*width_i), and for i!=j
    G_ij=erf(r_ij/sqrt(2*(width_i**2+width_j**2)))/r_ij.
    At r_ij=0 use sqrt(2/pi)/sqrt(width_i**2+width_j**2).
    A=G+diag(h+eta*z+l*d+k*d**2/2).
    b=chi+s*z+(u+w*z)*d+t*d**2/2; c=-field.T@(K0+z*K1)@field/2.
    Return (A,A1,A2,b,b1,b2,c2), the values and field derivatives at fixed z,R.
    A1[a,i,j]=partial_a A_ij; A2[a,b,i,j]=partial_a partial_b A_ij;
    b1[a,i]=partial_a b_i; b2[a,b,i]=partial_a partial_b b_i;
    c2[a,b]=partial_a partial_b c. No factorials in returned derivatives.
    Shapes: (n,n),(3,n,n),(3,3,n,n),(n,),(3,n),(3,3,n),(3,3).
    No explicit -q_i R_i.field term is to be added: b already defines all field coupling.
    
    Returns
    -------
    result
        Tuple (A,A1,A2,b,b1,b2,c2) of real arrays with shapes (n,n), (3,n,n), (3,3,n,n), (n,), (3,n), (3,3,n), (3,3).
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If any shape or finiteness condition above fails, n<1, a width<=0,
        or either curvature matrix is nonsymmetric beyond absolute 1e-12.
    
    Notes
    --------------------------
    Import required packages inside the function. NumPy and SciPy are installed.
    An import in test setup does not define a global in this function.
    Implement the public function only; the reference implementation is separate.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_field_matrices(z: float, field: 'np.ndarray', positions: 'np.ndarray', widths: 'np.ndarray', parameters: 'np.ndarray', curvature: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    import numpy as np
    from math import erf as _erf_scalar
    erf = np.vectorize(_erf_scalar, otypes=[float])

    for _name, _value in (('z', z), ('field', field), ('positions', positions), ('widths', widths), ('parameters', parameters), ('curvature', curvature),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)
    z = _scalar(z, 'z')
    field = _array(field, (3,), 'field')
    try:
        positions = np.asarray(positions, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('positions') from exc
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions)<1 or not np.isfinite(positions).all():
        raise ValueError('positions')
    n = len(positions)
    widths = _array(widths, (n,), 'widths')
    p = _array(parameters, (n,9), 'parameters')
    curvature = _array(curvature, (2,3,3), 'curvature')
    if np.any(widths<=0) or not np.allclose(curvature, curvature.swapaxes(1,2), atol=1e-12, rtol=0):
        raise ValueError('widths or curvature')
    v = positions-positions.mean(axis=0)
    d = v@field
    chi,h,s,u,t,w,eta,l,k = p.T
    distances = np.linalg.norm(positions[:,None,:]-positions[None,:,:], axis=2)
    spread = np.sqrt(2*(widths[:,None]**2+widths[None,:]**2))
    G = np.empty((n,n))
    np.divide(erf(distances/spread), distances, out=G, where=distances!=0)
    G[distances==0] = (2/np.sqrt(np.pi)/spread)[distances==0]
    A = G+np.diag(h+eta*z+l*d+0.5*k*d*d)
    A1 = np.zeros((3,n,n)); A2 = np.zeros((3,3,n,n))
    ids = np.arange(n)
    A1[:,ids,ids] = ((l+k*d)[:,None]*v).T
    vv = np.einsum('ia,ib->abi',v,v)
    A2[:,:,ids,ids] = vv*k
    b = chi+s*z+(u+w*z)*d+0.5*t*d*d
    b1 = (((u+w*z)+t*d)[:,None]*v).T
    b2 = vv*t
    return A,A1,A2,b,b1,b2,-curvature[0]-z*curvature[1]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Numerical cases and explicitly declared invalid inputs."""
    return [{'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n',
      'call': "field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])",
      'gold_call': "_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               "d['field']=np.zeros(3)\n",
      'call': "field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])",
      'gold_call': "_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               "d['positions'][1]=d['positions'][0]\n",
      'call': "field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])",
      'gold_call': "_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               "d['positions']=d['positions'][:1];d['widths']=d['widths'][:1];d['parameters']=d['parameters'][:1]\n",
      'call': "field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])",
      'gold_call': "_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               "d['positions']+=np.array([7.,-4.,2.])\n",
      'call': "field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])",
      'gold_call': "_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n',
      'call': "field_matrices(-0.8,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])",
      'gold_call': "_oracle_field_matrices(-0.8,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature'])"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               "d['widths'][0]=0\n",
      'call': "_error_code(lambda:field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature']))",
      'gold_call': "_error_code(lambda:_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature']))"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               '\n'
               'def _error_code(thunk):\n'
               '    try:thunk()\n'
               '    except ValueError:return 1\n'
               '    except Exception:return 2\n'
               '    return 0\n'
               "d['curvature'][0,0,1]+=0.1\n",
      'call': "_error_code(lambda:field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature']))",
      'gold_call': "_error_code(lambda:_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature']))"},
     {'setup': '"""Supplied deterministic benchmark inputs; no fitted outputs are stored here."""\n'
               'import numpy as np\n'
               '\n'
               'def make_inputs(seed=271828, replicas=24, length=256):\n'
               '    rng=np.random.default_rng(seed)\n'
               '    noise=rng.standard_normal((replicas,length+129,2))\n'
               '    slow=noise[:,0,0].copy(); fast=noise[:,0,1].copy()\n'
               '    states=np.empty((replicas,length))\n'
               '    for t in range(length+128):\n'
               '        slow=0.94*slow+np.sqrt(1-0.94**2)*noise[:,t+1,0]\n'
               '        fast=0.20*fast+np.sqrt(1-0.20**2)*noise[:,t+1,1]\n'
               '        if t>=128:states[:,t-128]=0.22*slow+0.055*fast\n'
               '    positions=np.array([[0.0,0.0,0.0],[1.4,0.2,-0.1],[-0.3,1.2,0.4],[0.2,-0.4,1.3]])\n'
               '    widths=np.array([0.70,0.85,0.75,0.90])\n'
               '    parameters=np.array([\n'
               '        [-0.6,2.1,0.30,0.80,0.17,0.21,0.08,0.12,0.25],\n'
               '        [ 0.4,2.4,-0.20,-0.40,-0.11,0.13,-0.05,-0.07,0.18],\n'
               '        [-0.2,1.9,0.25,0.65,0.09,-0.17,0.06,0.08,0.22],\n'
               '        [ 0.7,2.3,-0.15,-0.55,0.13,0.19,-0.04,-0.10,0.16]])\n'
               '    curvature=np.array([[[1.9,0.2,-0.1],[0.2,1.6,0.15],[-0.1,0.15,1.8]],\n'
               '                        [[0.4,0.35,-0.2],[0.35,-0.3,0.25],[-0.2,0.25,0.1]]])\n'
               '    field=np.array([0.23,-0.17,0.11])\n'
               '    ein=np.array([2.,-1.,1.])/np.sqrt(6)\n'
               '    eout=np.array([1.,2.,-1.])/np.sqrt(6)\n'
               '    return dict(states=states,field=field,positions=positions,widths=widths,\n'
               '                parameters=parameters,curvature=curvature,total_charge=0.0,\n'
               '                projector=np.outer(eout,ein),time_step=0.2,\n'
               '                cutoffs=np.array([0.30,0.37,0.46,0.57,0.71,0.88]),\n'
               '                initial=np.array([0.05,0.04,400.0]))\n'
               '\n'
               'd=make_inputs(replicas=1,length=8)\n'
               '\n'
               "d['widths']=d['widths'].astype(complex)+1j\n"
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: '
              "field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature']))",
      'gold_call': '_error_code(lambda: '
                   "_oracle_field_matrices(0.17,d['field'],d['positions'],d['widths'],d['parameters'],d['curvature']))"}]
