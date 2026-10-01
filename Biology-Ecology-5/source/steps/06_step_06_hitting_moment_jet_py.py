"""
Calculate fixation probability and the fixation-weighted first passage-time moment.

Conditioning on eventual fixation changes the moment equation: the right-hand side is the fixation probability rather than a constant. The killed diffusion Green function accounts for both absorbing endpoints.

Returns
-------
Shape (4,2), columns p=P(f(T)=1) and u=E[T*1_{f(T)=1], with T the first hitting time of 0 or 1 for the same diffusion as log_scale_integrals. Rows are value, epsilon, zeta, epsilon-zeta. Evaluate the killed-generator Green integral using order-point Gauss-Legendre quadrature on each outer interval (0,f0),(f0,1), and order-point quadrature for every inner scale integral. Integrate complementary scale intervals directly rather than subtracting nearly equal cumulative integrals. Differentiate the quadrature-defined moments at fixed f0,N0,D; no factorial scaling. Use the earlier scale-integral step. The moment u is not conditional time itself and is not the unconditional absorption-time moment.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hitting_moment_jet(selection: 'np.ndarray', N0: float, D: float, f0: float, order: int = 64) -> 'np.ndarray':
    """Calculate fixation probability and the fixation-weighted first passage-time moment.

    Parameters
    ----------
    selection : np.ndarray
        Finite (4,2) jet of (s,eta), rows value, epsilon, zeta, epsilon-zeta.
    N0 : float
        Positive fixed combined focal abundance.
    D : float
        Positive fixed demographic-noise coefficient, lineage variance 2*D*N.
    f0 : float
        Fixed initial frequency strictly between zero and one.
    order : int
        Gauss-Legendre order in [16,256], integer and not boolean.

    Returns
    -------
    result : np.ndarray
        Shape (4,2), columns p=P(f(T)=1) and u=E[T*1_{f(T)=1}],
        with T the first hitting time of 0 or 1 for the same diffusion as
        log_scale_integrals. Rows are value, epsilon, zeta, epsilon-zeta.
        Evaluate the killed-generator Green integral using order-point
        Gauss-Legendre quadrature on each outer interval (0,f0),(f0,1),
        and order-point quadrature for every inner scale integral.
        Integrate complementary scale intervals directly rather than
        subtracting nearly equal cumulative integrals. Differentiate the
        quadrature-defined moments at fixed f0,N0,D; no factorial scaling.
        Use the earlier scale-integral step. The moment u is not conditional
        time itself and is not the unconditional absorption-time moment.

    Raises
    ------
    ValueError
        If selection has invalid shape or nonreal/nonfinite entries, N0 or D
        is nonpositive or nonfinite, f0 is not finite and strictly interior,
        order is not an integer in [16,256] or is boolean, or intermediate
        effective size, arithmetic or output is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_hitting_moment_jet(selection: 'np.ndarray', N0: float, D: float, f0: float, order: int = 64) -> 'np.ndarray':
    sel,n=_diffusion_inputs(selection,N0,D);f0=float(_arr(f0,0));nodes,weights=_order(order)
    if not 0<f0<1:raise ValueError('initial frequency must be interior')
    base=_oracle_log_scale_integrals(sel,N0,D,np.array([[0,f0],[f0,1],[0,1]]),order)
    logS,logT,logZ=base.T
    p=_exp(logS-logZ);terms=[];quadweights=[]
    for lo,hi,left in [(0.,f0,True),(f0,1.,False)]:
        y=lo+(hi-lo)*(nodes+1)/2
        intervals=np.array([[0.,z] for z in y]+[[z,1.] for z in y])
        logs=_oracle_log_scale_integrals(sel,N0,D,intervals,order);Sy=logs[:,:order];Ty=logs[:,order:]
        V=_potential(sel,n,y)
        log_integrand=(2*Sy+logT[:,None] if left else Sy+Ty+logS[:,None])-2*logZ[:,None]-V
        log_integrand[0]+=np.log(2*n)-np.log(y)-np.log1p(-y)
        terms.append(log_integrand);quadweights.append(weights*(hi-lo)/2)
    log_terms=np.concatenate(terms,axis=1);qw=np.concatenate(quadweights);shift=log_terms[0].max();log_terms[0]-=shift
    u=(_exp(log_terms)*qw).sum(axis=1)*np.exp(shift)
    return _arr(np.stack([p,u],axis=1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Four original cases plus overflow/underflow and cancellation coverage."""
    return [
        {
            "setup": """import numpy as np
import copy
# normal
a0=np.array([[1.2011099483335636e-06, 2.4022125203113796e-06], [0.00030386556673751256, 1.590454928850751e-07], [0.0004714218741368572, 2.3138128145951028e-07], [3.692447978065164e-05, 3.976566960660779e-09]], dtype=float)
a1=0.1
a2=2e-08
a3=0.07
""",
            "call": 'hitting_moment_jet(a0.copy(), a1, a2, a3)',
            "gold_call": '_oracle_hitting_moment_jet(a0.copy(), a1, a2, a3)',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# boundary
a0=np.array([[0.0, 0.0], [0.01, 0.02], [-0.02, 0.01], [0.003, -0.004]], dtype=float)
a1=1.0
a2=0.1
a3=0.5
""",
            "call": 'hitting_moment_jet(a0.copy(), a1, a2, a3)',
            "gold_call": '_oracle_hitting_moment_jet(a0.copy(), a1, a2, a3)',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# edge
a0=np.array([[0.0009113819757888439, 1.2142612899191465e-05], [0.001451918719749543, 8.170959985959672e-07], [-0.0038140415980013886, -2.824406990374669e-06], [-0.0004439591716340586, -9.943762137924551e-08]], dtype=float)
a1=0.08
a2=0.0002
a3=0.999
a4=96
""",
            "call": 'hitting_moment_jet(a0.copy(), a1, a2, a3, a4)',
            "gold_call": '_oracle_hitting_moment_jet(a0.copy(), a1, a2, a3, a4)',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# invalid_declared_condition
a0=np.array([[1.2011099483335636e-06, 2.4022125203113796e-06], [0.00030386556673751256, 1.590454928850751e-07], [0.0004714218741368572, 2.3138128145951028e-07], [3.692447978065164e-05, 3.976566960660779e-09]], dtype=float)
a1=0.1
a2=2e-08
a3=0.0

def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": 'expect_value_error(hitting_moment_jet, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2), copy.deepcopy(a3))',
            "gold_call": 'expect_value_error(_oracle_hitting_moment_jet, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2), copy.deepcopy(a3))',
            "tol": 0,
        },
        {
            "setup": """import numpy as np
a0 = np.array([[-0.008, 0.0], [0.0001, 0.0002], [-0.0002, 0.0001], [3e-05, -4e-05]], dtype=float)
a1 = 1.0
a2 = 1e-5
a3 = 0.25
a4 = 128
normalizers = np.array([[2.6503965529986713e-261, 2.771611533825148e-258], [4.919798601503884e-261, 5.163704301773612e-258], [5.215483466947718e-260, 5.462588467423219e-257], [1.0773365349179499e-259, 1.1321223485349278e-256]], dtype=float)
""",
            "call": 'hitting_moment_jet(a0.copy(), a1, a2, a3, a4) / normalizers',
            "gold_call": '_oracle_hitting_moment_jet(a0.copy(), a1, a2, a3, a4) / normalizers',
            "tol": 1e-07,
        },
        {
            "setup": """import numpy as np
a0 = np.array([[0.008, 0.0], [0.0001, 0.0002], [-0.0002, 0.0001], [3e-05, -4e-05]], dtype=float)
a1 = 1.0
a2 = 1e-5
a3 = 0.01
a4 = 128
normalizers = np.array([[0.9996645373691446, 1499.8604147164974], [3.3126934472765384e-05, 2.180943016995646], [6.730218965750011e-05, 44.587638392653986], [1.6792247513318273e-05, 8.921140937279473]], dtype=float)
""",
            "call": 'hitting_moment_jet(a0.copy(), a1, a2, a3, a4) / normalizers',
            "gold_call": '_oracle_hitting_moment_jet(a0.copy(), a1, a2, a3, a4) / normalizers',
            "tol": 1e-07,
        },
    ]
