"""
Compute a high-order perturbation response of the largest retained condition number.

An affine coefficient perturbation makes the resultant polynomial depend on

both the elimination variable and the perturbation parameter. Fourier sampling

in the parameter and confluent sampling in the elimination variable recover

this bivariate polynomial. The maximizing admissible base root then defines a

local implicit branch whose logarithmic condition can be continued.

Returns
-------
One finite float, the ordinary derivative of order $P$ of the continued maximal natural-log condition at zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_condition_response(
    alpha: np.ndarray, gamma: np.ndarray, u: float, v: float,
    row_mixing: np.ndarray, alpha_direction: np.ndarray,
    gamma_direction: np.ndarray, response_order: int = 6,
    determinant_degree: int = 12, imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    r"""Return an ordinary derivative of the locally maximal log condition.
    
    Parameters
    ----------
    alpha : np.ndarray
        Finite real array of shape $(5,3)$ for the base quartic in $y$,
        with ascending $y$ degree followed by ascending $x$ degree.
    gamma : np.ndarray
        Finite real array of shape $(4,2)$ for the base cubic, with the
        same degree conventions.
    u : float
        Finite first root of the fixed row multiplier $q(x)=(x-u)(x-v)$.
    v : float
        Finite second root, with $|u-v|>10^{-12}$.
    row_mixing : np.ndarray
        Finite real matrix $L$ of shape $(7,7)$ with
        $\bigl||\det L|-1\bigr|\le10^{-9}$, fixed under perturbation.
    alpha_direction : np.ndarray
        Finite real array of shape $(5,3)$ defining
        $\alpha(t)=\alpha+t\,\alpha_{\mathrm{direction}}$.
    gamma_direction : np.ndarray
        Finite real array of shape $(4,2)$ defining
        $\gamma(t)=\gamma+t\,\gamma_{\mathrm{direction}}$.
    response_order : int, optional
        Ordinary derivative order $P$, default $6$, with $0\le P\le6$.
    determinant_degree : int, optional
        Degree bound $k$, default $12$, with $K=k+1$. It must hold for
        the determinant for all sufficiently small $t$, including its
        parameter coefficient polynomials, and accommodate the matrix rows.
    imaginary_tolerance : float, optional
        Finite positive base-root reality tolerance, default $10^{-8}$.
    residual_threshold : float, optional
        Finite positive base-pair residual threshold, default $10^{-3}$.
    
    Returns
    -------
    response : float
        The finite ordinary derivative $\ell^{(P)}(0)$, where $\ell(t)$
        is the log condition continued from the unique base root attaining
        the largest admissible condition number. Order zero returns its
        natural logarithm. Base candidates lie in $[0,3]$ and use the
        preceding cofactor recovery and original-equation residual gates.
    
    Raises
    ------
    ValueError
        If construction data, directions, order, or tolerances are invalid;
        coefficient recovery fails the realness or consistency checks below;
        no base root survives; a retained base root is numerically multiple;
        the largest and second-largest base condition numbers differ by at
        most $10^{-8}\max\{1,\kappa_{\max}\}$; or the response is nonfinite.
        Errors from the preceding stages propagate.
    
    Notes
    -----
    Compose the earlier stages, including the base maximum and the continuation
    step. The admissible set, anchor choices, and maximizing root label are
    fixed at $t=0$ for the local derivative; do not differentiate comparisons
    or independently reselect roots at perturbed parameters.
    
    Use eight parameter nodes $t_h=\exp(-2\pi i h/8)$ and
    $S=\max\{2,\lceil K/4\rceil\}$ elimination-variable nodes
    $x_j=\exp(-2\pi i j/S)$. The affine $7\times7$ matrix has determinant
    parameter degree at most seven. Obtain its elimination-variable derivatives
    of orders zero through three analytically at each parameter node, including
    singular matrices. Recover all eight parameter coefficient polynomials,
    then retain Taylor orders zero through $P$ for continuation. Every such
    polynomial has at most $K$ coefficients; preserve padding.
    
    Separate elimination-degree aliases with the same row-scaled
    falling-factorial moment systems as in the earlier confluent recovery step.
    Take an inverse transform in the parameter index to obtain Taylor
    coefficients; this transform does not give ordinary parameter derivatives.
    The recovered real bivariate coefficients must reproduce all sampled
    elimination derivatives within maximum absolute error
    $10^{-7}\max\{1,\max|Y|\}$, where $Y$ is the supplied sample array.
    Their imaginary parts must be at most $10^{-7}$ times the larger of one
    and the maximum coefficient magnitude. Also require the reconstructed
    values at $x=0.37$ and all eight parameter nodes to agree with direct
    matrix determinants within $10^{-7}(1+\max|D(t_h,0.37)|)$.
    Compute the high-order response from these polynomial coefficients and
    implicit continuation. Do not mutate inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_condition_response(
    alpha: np.ndarray, gamma: np.ndarray, u: float, v: float,
    row_mixing: np.ndarray, alpha_direction: np.ndarray,
    gamma_direction: np.ndarray, response_order: int = 6,
    determinant_degree: int = 12, imaginary_tolerance: float = 1e-8,
    residual_threshold: float = 1e-3,
) -> float:
    da, dg = np.asarray(alpha_direction), np.asarray(gamma_direction)
    if (da.shape != (5,3) or dg.shape != (4,2) or not np.isrealobj(da)
            or not np.isrealobj(dg) or not np.all(np.isfinite(da))
            or not np.all(np.isfinite(dg))):
        raise ValueError("invalid coefficient directions")
    if not isinstance(response_order,(int,np.integer)) or not 0 <= response_order <= 6:
        raise ValueError("invalid response order")
    base_maximum = _oracle_compute_hidden_root_condition(
        alpha,gamma,u,v,row_mixing,determinant_degree,
        imaginary_tolerance,residual_threshold)
    tensor = _oracle_construct_resultant_tensor(alpha,gamma,u,v,row_mixing,determinant_degree)
    direction = _oracle_construct_resultant_tensor(da,dg,u,v,row_mixing,determinant_degree)
    width = tensor.shape[0]
    count = max(2,(width+3)//4)
    points = _oracle_generate_unit_circle_samples(count-1)
    parameters = _oracle_generate_unit_circle_samples(7)
    sampled = np.asarray([
        _oracle_compute_determinant_samples(
            _oracle_evaluate_resultant_fft(tensor+t*direction,points,3))
        for t in parameters])
    moments = np.fft.ifft(sampled*points[None,None,:]**np.arange(4)[None,:,None],axis=2)
    at_parameters = np.zeros((8,width),dtype=complex)
    for residue in range(count):
        degrees = np.arange(residue,width,count)
        if not degrees.size:
            continue
        system = np.ones((4,len(degrees)))
        for order in range(1,4):
            system[order] = system[order-1]*(degrees-order+1)
        scale = np.maximum(1,np.max(np.abs(system),axis=1))
        at_parameters[:,degrees] = np.linalg.lstsq(
            system/scale[:,None],moments[:,:,residue].T/scale[:,None],rcond=None)[0].T
    jets = np.fft.ifft(at_parameters,axis=0)
    if np.max(np.abs(jets.imag)) > 1e-7*max(1,float(np.max(np.abs(jets)))):
        raise ValueError("coefficient response is not real")
    jets = jets.real
    recovered = np.zeros_like(sampled)
    for h,t in enumerate(parameters):
        coefficients = np.polynomial.polynomial.polyval(t,jets)
        for order in range(4):
            recovered[h,order] = np.polynomial.polynomial.polyval(
                points,np.polynomial.polynomial.polyder(coefficients,order))
    if np.max(np.abs(recovered-sampled)) > 1e-7*max(1,float(np.max(np.abs(sampled)))):
        raise ValueError("inconsistent confluent response")
    direct = _oracle_compute_determinant_samples(np.asarray([
        np.polynomial.polynomial.polyval(0.37,tensor+t*direction) for t in parameters]))
    predicted = np.asarray([
        np.polynomial.polynomial.polyval(0.37,np.polynomial.polynomial.polyval(t,jets))
        for t in parameters])
    if np.max(np.abs(predicted-direct)) > 1e-7*(1+float(np.max(np.abs(direct)))):
        raise ValueError("off-grid response validation failed")
    roots = _oracle_extract_hidden_candidates(jets[0],imaginary_tolerance,interval=(0,3))
    solutions = _oracle_recover_and_filter_solutions(
        tensor,roots,alpha,gamma,residual_threshold=residual_threshold)
    responses = [_oracle_continue_condition_jets(jets[:response_order+1],x) for x in solutions[:,0]]
    log_values = np.array([value[0,1] for value in responses])
    selected = int(np.argmin(np.abs(log_values-np.log(base_maximum))))
    if len(log_values)>1:
        ordered = np.sort(np.exp(log_values))
        if ordered[-1]-ordered[-2] <= 1e-8*max(1,base_maximum):
            raise ValueError("nonunique maximizing branch")
    factorial = 1
    for n in range(2,response_order+1):
        factorial *= n
    result = float(factorial*responses[selected][response_order,1])
    if not np.isfinite(result):
        raise ValueError("nonfinite condition response")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Response benchmarks and scaling/singularity checks."""
    return [{'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n'
               'da[:] = 0\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'tol': 1e-07},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n'
               'da = alpha.copy(); dg = gamma.copy()\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n'
               'da = alpha.copy()\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy(),response_order=0)',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy(),response_order=0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n'
               'da *= 0.5\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n'
               'def checked(fn):\n'
               '    try:\n'
               '        '
               'fn(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy(),response_order=7)\n'
               '    except ValueError:\n'
               '        pass\n'
               '    else:\n'
               '        return 0\n'
               '    try:\n'
               '        fn(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),np.zeros((4,3)),dg.copy())\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n',
      'call': 'checked(compute_condition_response)',
      'gold_call': 'checked(_oracle_compute_condition_response)',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'alpha = '
               'np.array([[1.0,-0.5,0.25],[-1.5,1.0,0.5],[2.0,0.5,-1.0],[-1.0,0.25,0.5],[1.0,0.0,0.0]])\n'
               'gamma = np.array([[-2.0,0.5],[1.0,-1.5],[0.5,0.75],[1.0,0.25]])\n'
               'row_mixing = np.array([[1,0,0,0,0,0,-1],[0,1,2,0,0,0,0],[0,0,1,0,0,0,0],\n'
               '                       [-2,0,0,1,-2,0,2],[0,0,0,0,1,0,1],[0,0,0,0,0,1,0],\n'
               '                       [0,0,0,0,0,0,1]], dtype=float)\n'
               'u, v = 0.6, 2.5\n'
               '\n'
               'da = np.zeros((5,3)); da[0,0] = 1.0\n'
               'dg = np.zeros((4,2))\n'
               'u,v=1.0,-1.0\n',
      'call': 'compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy(),response_order=4)',
      'gold_call': '_oracle_compute_condition_response(alpha.copy(),gamma.copy(),u,v,row_mixing.copy(),da.copy(),dg.copy(),response_order=4)',
      'tol': 1e-08}]
