"""
Conditional order-statistic construction around Eq. (15): obtain zero/one probabilities from the gap generating polynomial. Polynomial coefficients extend the nonsingular derivative expression to valid deterministic-count limits.

Conditional order-statistic construction around Eq. (15): obtain zero/one probabilities from the gap generating polynomial. Polynomial coefficients extend the nonsingular derivative expression to valid deterministic-count limits.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def zero_one_level_probabilities(kernel):
    """Return zero- and one-level probabilities on a restricted domain.

    Parameters
    ----------
    kernel : real or complex array, (2*m,2*m)
        Valid Pfaffian correlation kernel restricted to the queried nodes,
        with adjacent components. Complex symplectic gauge representations are
        allowed when their inclusion probabilities describe a real probability law.

    Returns
    -------
    float64 array, (2,)
        [P(number of nodes occupied=0), P(number of nodes occupied=1)]. These
        are the first two coefficients of Pf(J_m+(z-1)*K). This includes cases
        where J_m-K is singular and the empty domain gives [1,0]. Equivalent
        polynomial, coefficient, or nonsingular derivative calculations are valid.
        Outputs below zero by at most 2e-12 may be clipped to zero.

    Raises
    ------
    ValueError
        For the same finite even skew-matrix violations as signed_pfaffian,
        or coefficient imaginary part above 2e-10, or real coefficients outside
        [-2e-12,1+2e-12]. Validity of the entire point process is a precondition.
    """
    return np.zeros(2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def _oracle_zero_one_level_probabilities(kernel):
    """Return zero- and one-level probabilities on a restricted domain.

    Parameters
    ----------
    kernel : real or complex array, (2*m,2*m)
        Valid Pfaffian correlation kernel restricted to the queried nodes,
        with adjacent components. Complex symplectic gauge representations are
        allowed when their inclusion probabilities describe a real probability law.

    Returns
    -------
    float64 array, (2,)
        [P(number of nodes occupied=0), P(number of nodes occupied=1)]. These
        are the first two coefficients of Pf(J_m+(z-1)*K). This includes cases
        where J_m-K is singular and the empty domain gives [1,0]. Equivalent
        polynomial, coefficient, or nonsingular derivative calculations are valid.
        Outputs below zero by at most 2e-12 may be clipped to zero.

    Raises
    ------
    ValueError
        For the same finite even skew-matrix violations as signed_pfaffian,
        or coefficient imaginary part above 2e-10, or real coefficients outside
        [-2e-12,1+2e-12]. Validity of the entire point process is a precondition.
    """
    k=np.asarray(kernel,dtype=complex);_oracle_signed_pfaffian(k)
    m=len(k)//2
    if m==0:return np.array([1.,0.])
    count=m+1;j=_j(m)
    zs=np.exp(2j*np.pi*np.arange(count)/count)
    values=np.array([_oracle_signed_pfaffian(j+(z-1)*k) for z in zs])
    coeff=np.fft.fft(values)/count
    if np.max(abs(coeff[:2].imag))>2e-10 or np.any(coeff[:2].real < -2e-12) or np.any(coeff[:2].real>1+2e-12):
        raise ValueError('Invalid probability coefficients')
    return np.clip(coeff[:2].real,0,1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'empty queried domain',
      'setup': 'import numpy as np\narg0=np.array([],dtype=float).reshape((0, 0))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'one Bernoulli variable',
      'setup': 'import numpy as np\narg0=np.array([[0.0, 0.37], [-0.37, 0.0]],dtype=float).reshape((2, 2))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'correlated restricted domain',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 0.5381585157720508, -0.09254897243355609, -0.038439618060990055, '
               '0.0750000086832478, -0.03441904323469544], [-0.5381585157720508, 0.0, -0.11008454382437773, '
               '-0.36836088194834715, 0.15500533637285846, 0.13402421121405975], [0.09254897243355609, '
               '0.11008454382437773, 0.0, 0.44513146765961, -0.07525019526295518, 0.09018429557262697], '
               '[0.038439618060990055, 0.36836088194834715, -0.44513146765961, 0.0, -0.09811382450888616, '
               '-0.267663045362813], [-0.0750000086832478, -0.15500533637285846, 0.07525019526295518, '
               '0.09811382450888616, 0.0, 0.4228421133717575], [0.03441904323469544, -0.13402421121405975, '
               '-0.09018429557262697, 0.267663045362813, -0.4228421133717575, 0.0]],dtype=float).reshape((6, 6))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'singular J-K with exactly one point',
      'setup': 'import numpy as np\narg0=np.array([[0.0, 1.0], [-1.0, 0.0]],dtype=float).reshape((2, 2))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'singular J-K with exactly two points',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, -0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-0.0, 0.0, '
               '-1.0, 0.0]],dtype=float).reshape((4, 4))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'three independent unequal probabilities',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 0.2, 0.0, 0.0, 0.0, 0.0], [-0.2, 0.0, -0.0, 0.0, -0.0, 0.0], [0.0, 0.0, '
               '0.0, 0.5, 0.0, 0.0], [-0.0, 0.0, -0.5, 0.0, -0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.8], [-0.0, '
               '0.0, -0.0, 0.0, -0.8, 0.0]],dtype=float).reshape((6, 6))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'complex local symplectic gauge',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0j, (0.5381585157720508+0j), -0.18509794486711217j, -0.019219809030495028j, '
               '0.0750000086832478j, -0.03441904323469544j], [(-0.5381585157720508+0j), 0j, '
               '0.22016908764875545j, 0.18418044097417358j, -0.15500533637285846j, -0.13402421121405975j], '
               '[0.18509794486711217j, -0.22016908764875545j, 0j, (0.44513146765961+0j), '
               '(-0.15050039052591035+0j), (0.18036859114525394+0j)], [0.019219809030495028j, '
               '-0.18418044097417358j, (-0.44513146765961+0j), 0j, (-0.04905691225444308+0j), '
               '(-0.1338315226814065+0j)], [-0.0750000086832478j, 0.15500533637285846j, '
               '(0.15050039052591035+0j), (0.04905691225444308+0j), 0j, (0.4228421133717575+0j)], '
               '[0.03441904323469544j, 0.13402421121405975j, (-0.18036859114525394+0j), (0.1338315226814065+0j), '
               '(-0.4228421133717575+0j), 0j]],dtype=complex).reshape((6, 6))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'empty point process on nonempty domain',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, '
               '0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, '
               '0.0, 0.0, 0.0]],dtype=float).reshape((6, 6))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'},
     {'name': 'rare zero-count coefficient in a strongly correlated four-node tail',
      'setup': 'import numpy as np\n'
               'arg0=np.array([[0.0, 0.5045688892111452, 0.04365298925349696, 0.40812513228186087, '
               '-0.01852367299837168, -0.28377180777812383, 0.002975921056243075, 0.24077261966889485], '
               '[-0.5045688892111452, 0.0, -0.2547120579928719, -0.4997136624365468, 0.05570515177744968, '
               '-0.12112075710409624, -0.00645132569546192, 0.2454337120499196], [-0.04365298925349696, '
               '0.2547120579928719, 0.0, 0.6579819635001344, 0.005523731401643509, 0.3309705736965786, '
               '-0.0012053385695723643, -0.31751243375414634], [-0.40812513228186087, 0.4997136624365468, '
               '-0.6579819635001344, 0.0, -0.1037999605774801, -1.1083640058453084, 0.014813844947092758, '
               '0.8711363947961631], [0.01852367299837168, -0.05570515177744968, -0.005523731401643509, '
               '0.1037999605774801, 0.0, 0.8851525766192327, 0.00013568999392547947, 0.11457604339291627], '
               '[0.28377180777812383, 0.12112075710409624, -0.3309705736965786, 1.1083640058453084, '
               '-0.8851525766192327, 0.0, -0.017495902074088878, -1.7353133490902881], [-0.002975921056243075, '
               '0.00645132569546192, 0.0012053385695723643, -0.014813844947092758, -0.00013568999392547947, '
               '0.017495902074088878, 0.0, 0.9822327179380598], [-0.24077261966889485, -0.2454337120499196, '
               '0.31751243375414634, -0.8711363947961631, -0.11457604339291627, 1.7353133490902881, '
               '-0.9822327179380598, 0.0]],dtype=float).reshape((8, 8))',
      'call': 'zero_one_level_probabilities(arg0)',
      'gold_call': '_oracle_zero_one_level_probabilities(arg0)'}]
