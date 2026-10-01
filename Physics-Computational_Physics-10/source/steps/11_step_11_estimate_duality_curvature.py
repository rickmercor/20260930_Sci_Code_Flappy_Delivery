"""
Compute the second logarithmic coupled material-and-shape sensitivity of the conversion gain, composing mesh, conduction, compensation, field-derived duality, laminate realization, angular maxima, implicit compensation derivatives and duality-Jacobian and converted-response derivatives from the previous steps.

The curvature of the logarithmic suppression ratio measures how its fractional sensitivity changes as the insulating material is varied and the physical design is recalibrated. It combines movement of the local converted axes with movement of each device's worst applied direction.

Returns
-------
float: dimensionless second ordinary $\eta$ derivative of the natural logarithm of the converted/original worst-direction disturbance ratio along the prescribed material-and-shape path.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_duality_curvature(
    n_nodes: int = 121,
    core_boundary: tuple = ((0.20, 0.0), (0.0, 0.0), (0.04, 0.0)),
    inner_boundary: tuple = ((0.30, 0.0), (0.0, 0.0), (0.06, 0.0), (0.0, 0.0), (0.015, 0.0)),
    outer_boundary: tuple = ((0.45, 0.0), (0.0, 0.0), (0.09, 0.0), (0.02, 0.0)),
    inner_conductivity: float = 0.1,
    background_conductivity: float = 1.0,
    bracket: tuple[float, float] = (1.0, 20.0),
    tolerance: float = 1e-10,
    far_threshold: float = 0.61,
    high_fraction: "float | np.ndarray" = 0.5,
    shape_strength: float = 1.0,
) -> float:
    r"""Return $\mathrm{d}^2\ln(D^*_{\mathrm{converted}}/D^*_{\mathrm{original}})/\mathrm{d}\eta^2$ at $\eta=0$.

    Set the inner conductivity to ``inner_conductivity`` $\cdot\,e^{\eta}$, keep core
    and background at background_conductivity, and retune the outer
    conductivity to keep the exterior x-drive dipole moment fixed at its
    calibrated value. Material labels and connectivity are transported
    without reclassification. For baseline coordinates $(u,v)$, set
    $b=(1-u^2)(1-v^2)$, $V=b\,(0.18u+0.11v,\ -0.13u+0.16v)$,
    $W=b\,(0.09\sin(2u+v),\ 0.07\cos(u-2v))$, and
    $X(\eta)=(u,v)+s\,(\eta V+\eta^2W/2)$ with $s$ = ``shape_strength``.
    The boundary is fixed. The exterior moment uses current $x$ coordinates;
    the far-node indices are selected at zero and tracked thereafter. Build the
    mesh and calibrate the baseline outer shell using the preceding steps.
    Differentiate that regular implicit constraint using
    differentiate_compensation_design; treat the finite-tolerance baseline
    root as the center of its constant-moment branch, rather than trying
    to differentiate the bisection algorithm.

    At every nearby $\eta$, recompute the design field and duality conversion.
    The baseline converted laminate is realized from the duality Jacobian;
    evaluate both baseline $D^*$ values with compute_worst_direction_disturbance.
    Use differentiate_duality_response for their ordinary first and second
    derivatives, using differentiate_duality_jacobian followed by
    differentiate_duality_tensor for the constitutive jet
    and including changes of local axes and maximizing directions.
    Assemble the scalar second derivative of the log ratio. high_fraction
    is fixed along the perturbation and used for both laminate realization
    and forward evaluation; it does not change the ideal converted tensor.

    Parameters
    ----------
    n_nodes : int
        Nodes per side of the square grid.
    core_boundary : tuple
        Fourier $(A_k,B_k)$ coefficients for the core boundary.
    inner_boundary : tuple
        Coefficients for the outer edge of the inner shell.
    outer_boundary : tuple
        Coefficients for the outer edge of the compensation shell.
    inner_conductivity : float
        Positive baseline inner-shell conductivity.
    background_conductivity : float
        Positive, fixed core and background conductivity.
    bracket : tuple[float, float]
        Baseline compensation bisection bracket.
    tolerance : float
        Bisection stopping width; use the final midpoint.
    far_threshold : float
        Max-norm radius selecting interior far-field nodes.
    high_fraction : float or np.ndarray
        Fixed high-material share, scalar or (e,) array, strictly in $(0,1)$.
    shape_strength : float
        Finite real multiplier of both $V$ and $W$ in the coordinate curve.
        Zero recovers purely material sensitivity on the fixed mesh.

    Returns
    -------
    float
        Dimensionless curvature of the natural logarithm of suppression
        along the specified coupled material-and-shape perturbation.

    Raises
    ------
    ValueError
        If shape_strength is not finite, any preceding stage rejects its
        inputs, or either device has
        a nonregular worst-direction response as defined in step 10.
    """
    return logarithmic_curvature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_estimate_duality_curvature(
    n_nodes: int = 121,
    core_boundary: tuple = ((0.20, 0.0), (0.0, 0.0), (0.04, 0.0)),
    inner_boundary: tuple = ((0.30, 0.0), (0.0, 0.0), (0.06, 0.0), (0.0, 0.0), (0.015, 0.0)),
    outer_boundary: tuple = ((0.45, 0.0), (0.0, 0.0), (0.09, 0.0), (0.02, 0.0)),
    inner_conductivity: float = 0.1,
    background_conductivity: float = 1.0,
    bracket: tuple[float, float] = (1.0, 20.0),
    tolerance: float = 1e-10,
    far_threshold: float = 0.61,
    high_fraction: "float | np.ndarray" = 0.5,
    shape_strength: float = 1.0,
) -> float:
    """Compose the ten reference stages and differentiate the log ratio."""
    x,tri,lab=_oracle_build_cloak_mesh(n_nodes,core_boundary,inner_boundary,outer_boundary)
    q=_oracle_calibrate_compensation_shell(x,tri,lab,inner_conductivity,
                                          background_conductivity,bracket,tolerance)
    if not np.isscalar(shape_strength) or not np.isfinite(shape_strength):
        raise ValueError('shape strength must be finite')
    u,v=x.T
    envelope=(1-u*u)*(1-v*v)
    motion=shape_strength*np.stack([
        envelope[:,None]*np.column_stack([.18*u+.11*v,-.13*u+.16*v]),
        envelope[:,None]*np.column_stack([.09*np.sin(2*u+v),.07*np.cos(u-2*v)])])
    qjet,tjet=_oracle_differentiate_compensation_design(x,tri,lab,inner_conductivity,
                                                       q,background_conductivity,motion)
    kjet=np.empty((3,len(tri)))
    kjet[0]=np.array([background_conductivity,q,inner_conductivity,background_conductivity])[lab]
    for order in [1,2]:
        kjet[order]=np.array([0.,qjet[order],inner_conductivity,0.])[lab]
    jac=_oracle_compute_duality_jacobian(x,tri,tjet[0],kjet[0],background_conductivity,
                                        np.array([1.,0.]))
    layers=_oracle_realize_duality_laminate(jac,background_conductivity,high_fraction)
    original=np.column_stack([kjet[0],kjet[0],np.zeros(len(tri))])
    d=np.array([_oracle_compute_worst_direction_disturbance(x,tri,original,far_threshold),
                _oracle_compute_worst_direction_disturbance(x,tri,layers,far_threshold,high_fraction)])
    derivatives=_oracle_differentiate_duality_response(x,tri,kjet,tjet,
                                                       background_conductivity,far_threshold,motion)
    curvature=derivatives[:,1]/d-(derivatives[:,0]/d)**2
    return float(curvature[1]-curvature[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Explicit normal, boundary and shape-derivative cases."""
    return [{'setup': 'import numpy as np\n'
               'N=121\n'
               'KI=0.1\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.61\n'
               'STRENGTH=1.0\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=31\n'
               'KI=0.1\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.61\n'
               'STRENGTH=1.0\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=45\n'
               'KI=0.23\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.55\n'
               'STRENGTH=1.0\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=33\n'
               'KI=0.1\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.61\n'
               'STRENGTH=1.0\n'
               'FRACTION=np.linspace(.05,.95,2*(N-1)**2)\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION.copy(),shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION.copy(),shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=43\n'
               'KI=0.2\n'
               'K0=2.0\n'
               'BRACKET=(2.0, 40.0)\n'
               'FAR=0.61\n'
               'STRENGTH=1.0\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=35\n'
               'KI=0.1\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.61\n'
               'STRENGTH=0.0\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=37\n'
               'KI=0.17\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.55\n'
               'STRENGTH=-0.8\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'N=39\n'
               'KI=0.08\n'
               'K0=1.0\n'
               'BRACKET=(1.0, 20.0)\n'
               'FAR=0.61\n'
               'STRENGTH=1.7\n'
               'FRACTION=.5\n',
      'call': 'estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'gold_call': '_oracle_estimate_duality_curvature(n_nodes=N,inner_conductivity=KI,background_conductivity=K0,bracket=BRACKET,far_threshold=FAR,high_fraction=FRACTION,shape_strength=STRENGTH)',
      'tol': 2e-07},
     {'setup': 'N=21\n'
               'BRACKET=(10.,20.)\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n',
      'call': '_status(lambda: estimate_duality_curvature(n_nodes=N,bracket=BRACKET))',
      'gold_call': '_status(lambda: _oracle_estimate_duality_curvature(n_nodes=N,bracket=BRACKET))'},
     {'setup': 'N=19\n'
               'STRENGTH=float("nan")\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n',
      'call': '_status(lambda: estimate_duality_curvature(n_nodes=N,shape_strength=STRENGTH))',
      'gold_call': '_status(lambda: _oracle_estimate_duality_curvature(n_nodes=N,shape_strength=STRENGTH))'}]
