"""
Run the molecular response and spectral estimation pipeline.

ORCHESTRATOR. Compute the relaxed field response for every supplied state, form its polarized fluctuation spectrum, fit every specified cutoff and marginalize the physical spectral integral.

Returns
-------
Native Python float containing the cutoff-marginalized integral.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def polarizability_integral(states: 'np.ndarray', field: 'np.ndarray', positions: 'np.ndarray', widths: 'np.ndarray', parameters: 'np.ndarray', curvature: 'np.ndarray', total_charge: float, projector: 'np.ndarray', time_step: float, cutoffs: 'np.ndarray', initial: 'np.ndarray') -> float:
    """Return the cutoff-marginalized polarizability autocorrelation integral as float.
    
    Parameters
    ----------
    states, field, positions, widths, parameters, curvature, projector, cutoffs, initial : numpy.ndarray
        Real finite trajectory, molecular, projection, cutoff, and initialization data.
    total_charge, time_step : float
        Finite total charge and positive finite sampling interval.

    states is a finite real (M,N) array, M>=1,N>=8. Each element is the
    scalar z supplied to field_matrices. field has shape (3,), positions (n,3),
    widths (n,), parameters (n,9), curvature (2,3,3), n>=1; total_charge is a
    finite scalar. Widths are positive and both curvature matrices are symmetric
    within absolute 1e-12. All input arrays are finite and real. The molecular
    inputs obey the earlier field_matrices and constrained_response contracts.
    The per-state hardness matrix must be symmetric positive definite.
    For every state independently compute its field matrices, constrained
    q,dq at the supplied total_charge, and relaxed alpha. Use projected_spectrum
    with projector (3,3) and positive time_step. cutoffs is a finite positive
    strictly increasing 1D array, length >=1; initial is a finite (3,) vector.
    For EACH cutoff independently call fit_cutoff starting from the SAME
    supplied initial, then cv2l_score on that fit and the full spectrum.
    Do not reuse a fit for another cutoff or reuse its fitted parameters as
    the next initial value. Feed all parameter/covariance/criterion rows to
    lorentz_marginal; return mean[0]. All earlier public functions are available.
    This is the integral of the projected polarizability fluctuation ACF;
    no temperature, volume, refractive-index or photon-frequency prefactor.
    Both endpoint conventions and all thresholds are specified in prior steps.
    
    Returns
    -------
    result
        Native Python float containing the cutoff-marginalized integral.
    
    Raises
    ------
    ValueError
        If any input contains complex values, including a complex dtype with
        zero imaginary part.
        If states/cutoffs violate the stated conditions, or any earlier
        function encounters one of its declared invalid conditions.
    
    Notes
    --------------------------
    Import required packages inside the function. NumPy and SciPy are installed.
    An import in test setup does not define a global in this function.
    Implement the public function only; the reference implementation is separate.
    Pipeline acceptance uses rtol=2e-5 and atol=1e-10.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_polarizability_integral(states: 'np.ndarray', field: 'np.ndarray', positions: 'np.ndarray', widths: 'np.ndarray', parameters: 'np.ndarray', curvature: 'np.ndarray', total_charge: float, projector: 'np.ndarray', time_step: float, cutoffs: 'np.ndarray', initial: 'np.ndarray') -> float:
    import numpy as np

    for _name, _value in (('states', states), ('field', field), ('positions', positions), ('widths', widths), ('parameters', parameters), ('curvature', curvature), ('total_charge', total_charge), ('projector', projector), ('time_step', time_step), ('cutoffs', cutoffs), ('initial', initial),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")


    try:
        states=np.asarray(states,dtype=float); cutoffs=np.asarray(cutoffs,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('states or cutoffs') from exc
    if states.ndim!=2 or states.shape[0]<1 or states.shape[1]<8 or not np.isfinite(states).all():raise ValueError('states')
    if cutoffs.ndim!=1 or len(cutoffs)<1 or not np.isfinite(cutoffs).all() or np.any(cutoffs<=0) or np.any(np.diff(cutoffs)<=0):raise ValueError('cutoffs')
    alpha=np.empty(states.shape+(3,3))
    for m in range(states.shape[0]):
        for t in range(states.shape[1]):
            A,A1,A2,b,b1,b2,c2=_oracle_field_matrices(states[m,t],field,positions,widths,parameters,curvature)
            q,dq=_oracle_constrained_response(A,A1,b,b1,total_charge)
            alpha[m,t]=_oracle_relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)
    f,y,nu=_oracle_projected_spectrum(alpha,projector,time_step)
    rows=[]; cov=[]; scores=[]
    for cutoff in cutoffs:
        p,C=_oracle_fit_cutoff(f,y,nu,cutoff,initial)
        score,_,_=_oracle_cv2l_score(p,f,y,nu,cutoff)
        rows.append(p);cov.append(C);scores.append(score)
    mean,_,_,_,_=_oracle_lorentz_marginal(rows,cov,scores)
    return float(mean[0])

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
               'd=make_inputs()\n'
               '\n',
      'call': 'polarizability_integral(**d)',
      'gold_call': '_oracle_polarizability_integral(**d)',
      'tol': 2e-05},
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
               'd=make_inputs()\n'
               "d['projector']*=2.;d['initial'][:2]*=4.\n"
               '\n',
      'call': 'polarizability_integral(**d)',
      'gold_call': '_oracle_polarizability_integral(**d)',
      'tol': 2e-05},
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
               'd=make_inputs()\n'
               "d['time_step']*=3.;d['cutoffs']/=3.;d['initial']*=np.array([3.,27.,9.])\n"
               '\n',
      'call': 'polarizability_integral(**d)',
      'gold_call': '_oracle_polarizability_integral(**d)',
      'tol': 2e-05},
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
               'd=make_inputs()\n'
               'd=make_inputs(seed=17,replicas=32,length=384)\n'
               '\n',
      'call': 'polarizability_integral(**d)',
      'gold_call': '_oracle_polarizability_integral(**d)',
      'tol': 2e-05},
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
               'd=make_inputs()\n'
               'd=make_inputs(seed=42,replicas=32,length=384)\n'
               '\n',
      'call': 'polarizability_integral(**d)',
      'gold_call': '_oracle_polarizability_integral(**d)',
      'tol': 2e-05},
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
               'd=make_inputs()\n'
               'expected=0.04507739052731866\n'
               '\n'
               'def _answer_check(value):\n'
               '    return int(np.asarray(value).shape==() and np.isfinite(value) and '
               'np.isclose(value,expected,rtol=2e-5,atol=1e-10))\n'
               '\n'
               "d['states']=d['states'].astype(complex)+1j\n"
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: polarizability_integral(**d))',
      'gold_call': '_error_code(lambda: _oracle_polarizability_integral(**d))'},
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
               'd=make_inputs()\n'
               'expected=0.04507739052731866\n'
               '\n'
               'def _answer_check(value):\n'
               '    return int(np.asarray(value).shape==() and np.isfinite(value) and '
               'np.isclose(value,expected,rtol=2e-5,atol=1e-10))\n'
               '\n'
               "d['states']=d['states'][:,:7]\n"
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: polarizability_integral(**d))',
      'gold_call': '_error_code(lambda: _oracle_polarizability_integral(**d))'},
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
               'd=make_inputs()\n'
               'expected=0.04507739052731866\n'
               '\n'
               'def _answer_check(value):\n'
               '    return int(np.asarray(value).shape==() and np.isfinite(value) and '
               'np.isclose(value,expected,rtol=2e-5,atol=1e-10))\n'
               '\n'
               "d['cutoffs']=np.array([.57,.57])\n"
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: polarizability_integral(**d))',
      'gold_call': '_error_code(lambda: _oracle_polarizability_integral(**d))'},
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
               'd=make_inputs()\n'
               'expected=0.04507739052731866\n'
               '\n'
               'def _answer_check(value):\n'
               '    return int(np.asarray(value).shape==() and np.isfinite(value) and '
               'np.isclose(value,expected,rtol=2e-5,atol=1e-10))\n'
               '\n'
               "d['projector'][:]=0\n"
               '\n'
               'def _error_code(thunk):\n'
               '    try:\n'
               '        thunk()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_error_code(lambda: polarizability_integral(**d))',
      'gold_call': '_error_code(lambda: _oracle_polarizability_integral(**d))'}]
