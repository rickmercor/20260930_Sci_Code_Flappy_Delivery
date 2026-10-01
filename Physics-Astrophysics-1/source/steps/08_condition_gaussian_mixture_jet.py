"""
Response of the physicality-conditioned coefficient posterior

The coefficient prior is proportional to $$\sum_g w_g\mathcal N(\boldsymbol u;\boldsymbol\mu_g,\Sigma_g)$$, globally conditioned on the common finite box $$B$$. As the delay-mixture fraction $$f$$ varies, the joint formation-compatible region $$A_f\subseteq B$$ moves, while the Gaussian components, box and pre-truncation weights remain fixed. The preceding step supplies each component's unnormalized mass and raw first and second moments over $$A_f$$ together with their first two ordinary $$f$$ derivatives. Determine the probability of $$A_f$$ under the box-conditioned mixture, and the coefficient mean and centered covariance after conditioning on $$A_f$$, with their first two ordinary derivatives. The component box masses can differ, and the requested derivatives include the change in the accepted mixture's normalization. This calculation measures how the physicality constraint changes the inferred rate uncertainty, including its curvature with respect to the delay mixture.

Returns
-------
np.ndarray of shape (3,10), the conditional summary and its first two ordinary fraction derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def condition_gaussian_mixture_jet(accepted_moment_jets: np.ndarray,
                                   prior_moments: np.ndarray,
                                   weights: np.ndarray) -> np.ndarray:
    r'''Reweight a Gaussian mixture and propagate its local fraction response.

    Parameters
    ----------
    accepted_moment_jets : np.ndarray
        Finite real shape (G,3,10), G >= 1. The middle axis contains value,
        first derivative and second ordinary fraction derivative. The last
        axis orders the raw integrals of [1,u1,u2,u3,u1^2,u1*u2,u1*u3,
        u2^2,u2*u3,u3^2]. Component masses at order zero are nonnegative.
    prior_moments : np.ndarray
        Finite real shape (G,10), in the same moment order, for each
        untruncated Gaussian integrated over the common fixed box.
        These moments are constant with respect to fraction. Inputs are
        consistent with a smooth family whose accepted region is contained
        in the fixed box at each fraction; containment need not be
        numerically certified. A zero accepted mass is locally
        zero, with vanishing response.
    weights : np.ndarray
        Finite real nonnegative shape (G,), with positive total. These
        are weights before global box truncation. Their sum need not be one.

    Returns
    -------
    summary_jet : np.ndarray
        Finite shape (3,10): rows give value and first/second ordinary
        fraction derivatives. Columns contain the physicality probability
        conditional on the box, the three conditional means, and centered
        covariance entries [11,12,13,22,23,33]. Locally zero accepted mass
        returns all zeros; the mean/covariance entries are then placeholders.

    Raises
    ------
    ValueError
        If finite real shapes, nonnegative order-zero masses or weight
        contracts fail, the weighted prior mass is nonpositive, or the
        conditional response is nonfinite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_condition_gaussian_mixture_jet(accepted_moment_jets: np.ndarray,
                                           prior_moments: np.ndarray,
                                           weights: np.ndarray) -> np.ndarray:
    if any(not np.isrealobj(v) for v in (accepted_moment_jets, prior_moments, weights)):
        raise ValueError("moments and weights must be real")
    try:
        accepted, prior, weight = [np.asarray(v, dtype=float)
                                  for v in (accepted_moment_jets, prior_moments, weights)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite real moments and weights required") from exc
    if accepted.ndim != 3 or accepted.shape[1:] != (3,10) or accepted.shape[0] < 1:
        raise ValueError("accepted_moment_jets must have shape (G,3,10)")
    if prior.shape != (accepted.shape[0],10) or weight.shape != (accepted.shape[0],):
        raise ValueError("prior moments and weights must match the components")
    if not all(np.all(np.isfinite(v)) for v in (accepted, prior, weight)):
        raise ValueError("all inputs must be finite")
    if np.any(accepted[:,0,0] < 0) or np.any(prior[:,0] < 0) or np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError("nonnegative masses and nonnegative nonzero weights required")
    weight = weight / np.max(weight)
    denominator = float(weight @ prior[:,0])
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("the weighted box mass must be positive")
    raw = np.einsum('g,gij->ij', weight, accepted)
    if raw[0,0] == 0:
        return np.zeros((3,10))
    normalized = np.empty((3,9))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        normalized[0] = raw[0,1:] / raw[0,0]
        normalized[1] = (raw[1,1:] - raw[1,0]*normalized[0]) / raw[0,0]
        normalized[2] = (raw[2,1:] - raw[2,0]*normalized[0] - 2*raw[1,0]*normalized[1]) / raw[0,0]
        result = np.column_stack((raw[:,0]/denominator, normalized))
        pairs = ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))
        for j, (a,b) in enumerate(pairs):
            ma, mb = normalized[:,a], normalized[:,b]
            product = np.array([ma[0]*mb[0], ma[1]*mb[0]+ma[0]*mb[1],
                                ma[2]*mb[0]+2*ma[1]*mb[1]+ma[0]*mb[2]])
            result[:,4+j] -= product
    if not np.all(np.isfinite(result)):
        raise ValueError("the conditional response must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Explicit cases keep static Studio validation aligned with execution."""
    return [{'setup': 'import numpy as np\n'
               'components=[([0, 0, 0], [1, 1, 1])]\n'
               'w=np.array([1],dtype=float)\n'
               'lo=[-1, -1, -1];hi=[0.3, 0.7, 0.9];velocity=[0.2, -0.1, 0.15];acceleration=[0.1, 0.04, '
               '-0.08]\n'
               'import numpy as np\n'
               'from scipy.special import ndtr\n'
               'def multiply(a,b):\n'
               '    return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])\n'
               'def raw_box(mu,sd,lo,hi,velocity,acceleration):\n'
               '    zlo=(lo-mu)/sd;zhi=(hi-mu)/sd\n'
               '    plo=np.exp(-zlo*zlo/2)/np.sqrt(2*np.pi)\n'
               '    phi=np.exp(-zhi*zhi/2)/np.sqrt(2*np.pi)\n'
               '    mass=ndtr(zhi)-ndtr(zlo)\n'
               '    first=mu*mass+sd*(plo-phi)\n'
               '    second=(mu*mu+sd*sd)*mass+2*mu*sd*(plo-phi)+sd*sd*(zlo*plo-zhi*phi)\n'
               '    base=np.stack((mass,first,second))\n'
               '    marginal=np.empty((3,3,3))\n'
               '    for coordinate in range(3):\n'
               '        u=hi[coordinate];g=phi[coordinate]/sd[coordinate]\n'
               '        for power in range(3):\n'
               '            polynomial_derivative=0. if power==0 else power*u**(power-1)\n'
               '            '
               'marginal[coordinate,power]=[base[power,coordinate],u**power*g*velocity[coordinate],\n'
               '                '
               'g*((polynomial_derivative-u**power*(u-mu[coordinate])/sd[coordinate]**2)*velocity[coordinate]**2+u**power*acceleration[coordinate])]\n'
               '    '
               'exponents=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]\n'
               '    return np.array([multiply(multiply(marginal[0,a],marginal[1,b]),marginal[2,c]) for '
               'a,b,c in exponents]).T\n'
               'prior=np.array([raw_box(np.array(mu),np.array(sd),-np.ones(3),np.ones(3),np.zeros(3),np.zeros(3))[0] '
               'for mu,sd in components])\n'
               'accepted=np.array([raw_box(np.array(mu),np.array(sd),np.array(lo),np.array(hi),np.array(velocity),np.array(acceleration)) '
               'for mu,sd in components])\n',
      'call': 'condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'gold_call': '_oracle_condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'components=[([-0.7, 0.3, -0.4], [0.6, 0.8, 0.7]), ([0.5, -0.2, 0.3], [0.9, 0.5, '
               '1.1])]\n'
               'w=np.array([0.35, 0.65],dtype=float)\n'
               'lo=[-1, -1, -1];hi=[0.1, 0.6, 0.4];velocity=[-0.4, 0.2, -0.1];acceleration=[0.05, '
               '-0.1, 0.2]\n'
               'import numpy as np\n'
               'from scipy.special import ndtr\n'
               'def multiply(a,b):\n'
               '    return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])\n'
               'def raw_box(mu,sd,lo,hi,velocity,acceleration):\n'
               '    zlo=(lo-mu)/sd;zhi=(hi-mu)/sd\n'
               '    plo=np.exp(-zlo*zlo/2)/np.sqrt(2*np.pi)\n'
               '    phi=np.exp(-zhi*zhi/2)/np.sqrt(2*np.pi)\n'
               '    mass=ndtr(zhi)-ndtr(zlo)\n'
               '    first=mu*mass+sd*(plo-phi)\n'
               '    second=(mu*mu+sd*sd)*mass+2*mu*sd*(plo-phi)+sd*sd*(zlo*plo-zhi*phi)\n'
               '    base=np.stack((mass,first,second))\n'
               '    marginal=np.empty((3,3,3))\n'
               '    for coordinate in range(3):\n'
               '        u=hi[coordinate];g=phi[coordinate]/sd[coordinate]\n'
               '        for power in range(3):\n'
               '            polynomial_derivative=0. if power==0 else power*u**(power-1)\n'
               '            '
               'marginal[coordinate,power]=[base[power,coordinate],u**power*g*velocity[coordinate],\n'
               '                '
               'g*((polynomial_derivative-u**power*(u-mu[coordinate])/sd[coordinate]**2)*velocity[coordinate]**2+u**power*acceleration[coordinate])]\n'
               '    '
               'exponents=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]\n'
               '    return np.array([multiply(multiply(marginal[0,a],marginal[1,b]),marginal[2,c]) for '
               'a,b,c in exponents]).T\n'
               'prior=np.array([raw_box(np.array(mu),np.array(sd),-np.ones(3),np.ones(3),np.zeros(3),np.zeros(3))[0] '
               'for mu,sd in components])\n'
               'accepted=np.array([raw_box(np.array(mu),np.array(sd),np.array(lo),np.array(hi),np.array(velocity),np.array(acceleration)) '
               'for mu,sd in components])\n',
      'call': 'condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'gold_call': '_oracle_condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'components=[([-0.5, -0.4, 0.1], [0.4, 0.7, 0.8]), ([0.6, 0.2, -0.5], [0.8, 0.6, 0.4]), '
               '([0, 0.7, 0.6], [0.7, 0.5, 0.9])]\n'
               'w=np.array([2, 0, 3],dtype=float)\n'
               'lo=[-0.9, -0.8, -0.9];hi=[0.7, 0.9, 0.2];velocity=[0.1, -0.3, 0.2];acceleration=[0.2, '
               '0.1, -0.15]\n'
               'import numpy as np\n'
               'from scipy.special import ndtr\n'
               'def multiply(a,b):\n'
               '    return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])\n'
               'def raw_box(mu,sd,lo,hi,velocity,acceleration):\n'
               '    zlo=(lo-mu)/sd;zhi=(hi-mu)/sd\n'
               '    plo=np.exp(-zlo*zlo/2)/np.sqrt(2*np.pi)\n'
               '    phi=np.exp(-zhi*zhi/2)/np.sqrt(2*np.pi)\n'
               '    mass=ndtr(zhi)-ndtr(zlo)\n'
               '    first=mu*mass+sd*(plo-phi)\n'
               '    second=(mu*mu+sd*sd)*mass+2*mu*sd*(plo-phi)+sd*sd*(zlo*plo-zhi*phi)\n'
               '    base=np.stack((mass,first,second))\n'
               '    marginal=np.empty((3,3,3))\n'
               '    for coordinate in range(3):\n'
               '        u=hi[coordinate];g=phi[coordinate]/sd[coordinate]\n'
               '        for power in range(3):\n'
               '            polynomial_derivative=0. if power==0 else power*u**(power-1)\n'
               '            '
               'marginal[coordinate,power]=[base[power,coordinate],u**power*g*velocity[coordinate],\n'
               '                '
               'g*((polynomial_derivative-u**power*(u-mu[coordinate])/sd[coordinate]**2)*velocity[coordinate]**2+u**power*acceleration[coordinate])]\n'
               '    '
               'exponents=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]\n'
               '    return np.array([multiply(multiply(marginal[0,a],marginal[1,b]),marginal[2,c]) for '
               'a,b,c in exponents]).T\n'
               'prior=np.array([raw_box(np.array(mu),np.array(sd),-np.ones(3),np.ones(3),np.zeros(3),np.zeros(3))[0] '
               'for mu,sd in components])\n'
               'accepted=np.array([raw_box(np.array(mu),np.array(sd),np.array(lo),np.array(hi),np.array(velocity),np.array(acceleration)) '
               'for mu,sd in components])\n',
      'call': 'condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'gold_call': '_oracle_condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'components=[([-0.5, 0.4, 0], [0.7, 0.7, 0.7]), ([0.6, -0.3, 0.4], [1, 0.6, 0.8])]\n'
               'w=np.array([1, 4],dtype=float)\n'
               'lo=[-1, -1, -1];hi=[1, 1, 1];velocity=[0, 0, 0];acceleration=[0, 0, 0]\n'
               'import numpy as np\n'
               'from scipy.special import ndtr\n'
               'def multiply(a,b):\n'
               '    return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])\n'
               'def raw_box(mu,sd,lo,hi,velocity,acceleration):\n'
               '    zlo=(lo-mu)/sd;zhi=(hi-mu)/sd\n'
               '    plo=np.exp(-zlo*zlo/2)/np.sqrt(2*np.pi)\n'
               '    phi=np.exp(-zhi*zhi/2)/np.sqrt(2*np.pi)\n'
               '    mass=ndtr(zhi)-ndtr(zlo)\n'
               '    first=mu*mass+sd*(plo-phi)\n'
               '    second=(mu*mu+sd*sd)*mass+2*mu*sd*(plo-phi)+sd*sd*(zlo*plo-zhi*phi)\n'
               '    base=np.stack((mass,first,second))\n'
               '    marginal=np.empty((3,3,3))\n'
               '    for coordinate in range(3):\n'
               '        u=hi[coordinate];g=phi[coordinate]/sd[coordinate]\n'
               '        for power in range(3):\n'
               '            polynomial_derivative=0. if power==0 else power*u**(power-1)\n'
               '            '
               'marginal[coordinate,power]=[base[power,coordinate],u**power*g*velocity[coordinate],\n'
               '                '
               'g*((polynomial_derivative-u**power*(u-mu[coordinate])/sd[coordinate]**2)*velocity[coordinate]**2+u**power*acceleration[coordinate])]\n'
               '    '
               'exponents=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]\n'
               '    return np.array([multiply(multiply(marginal[0,a],marginal[1,b]),marginal[2,c]) for '
               'a,b,c in exponents]).T\n'
               'prior=np.array([raw_box(np.array(mu),np.array(sd),-np.ones(3),np.ones(3),np.zeros(3),np.zeros(3))[0] '
               'for mu,sd in components])\n'
               'accepted=np.array([raw_box(np.array(mu),np.array(sd),np.array(lo),np.array(hi),np.array(velocity),np.array(acceleration)) '
               'for mu,sd in components])\n',
      'call': 'condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'gold_call': '_oracle_condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'components=[([-0.7, 0.3, -0.4], [0.6, 0.8, 0.7]), ([0.5, -0.2, 0.3], [0.9, 0.5, '
               '1.1])]\n'
               'w=np.array([0.35, 0.65],dtype=float)\n'
               'lo=[-1, -1, -1];hi=[0.1, 0.6, 0.4];velocity=[-0.4, 0.2, -0.1];acceleration=[0.05, '
               '-0.1, 0.2]\n'
               'import numpy as np\n'
               'from scipy.special import ndtr\n'
               'def multiply(a,b):\n'
               '    return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])\n'
               'def raw_box(mu,sd,lo,hi,velocity,acceleration):\n'
               '    zlo=(lo-mu)/sd;zhi=(hi-mu)/sd\n'
               '    plo=np.exp(-zlo*zlo/2)/np.sqrt(2*np.pi)\n'
               '    phi=np.exp(-zhi*zhi/2)/np.sqrt(2*np.pi)\n'
               '    mass=ndtr(zhi)-ndtr(zlo)\n'
               '    first=mu*mass+sd*(plo-phi)\n'
               '    second=(mu*mu+sd*sd)*mass+2*mu*sd*(plo-phi)+sd*sd*(zlo*plo-zhi*phi)\n'
               '    base=np.stack((mass,first,second))\n'
               '    marginal=np.empty((3,3,3))\n'
               '    for coordinate in range(3):\n'
               '        u=hi[coordinate];g=phi[coordinate]/sd[coordinate]\n'
               '        for power in range(3):\n'
               '            polynomial_derivative=0. if power==0 else power*u**(power-1)\n'
               '            '
               'marginal[coordinate,power]=[base[power,coordinate],u**power*g*velocity[coordinate],\n'
               '                '
               'g*((polynomial_derivative-u**power*(u-mu[coordinate])/sd[coordinate]**2)*velocity[coordinate]**2+u**power*acceleration[coordinate])]\n'
               '    '
               'exponents=[(0,0,0),(1,0,0),(0,1,0),(0,0,1),(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]\n'
               '    return np.array([multiply(multiply(marginal[0,a],marginal[1,b]),marginal[2,c]) for '
               'a,b,c in exponents]).T\n'
               'prior=np.array([raw_box(np.array(mu),np.array(sd),-np.ones(3),np.ones(3),np.zeros(3),np.zeros(3))[0] '
               'for mu,sd in components])\n'
               'accepted=np.array([raw_box(np.array(mu),np.array(sd),np.array(lo),np.array(hi),np.array(velocity),np.array(acceleration)) '
               'for mu,sd in components])\n'
               'accepted[:]=0\n',
      'call': 'condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'gold_call': '_oracle_condition_gaussian_mixture_jet(accepted.copy(),prior.copy(),w.copy())',
      'tol': 1e-06}]
