"""
Calculate the two first responses and mixed stationary response of the cavity community.

A stationary branch must preserve all resident growth constraints while supply changes; curvature of the resource and growth balances generates a nonzero mixed state response even though supply is affine.

Returns
-------
Shape (4,S+M), rows value, epsilon derivative, zeta derivative, mixed epsilon-zeta derivative of concatenated (N,R) at zero. Resident growth is (1-leakage)sum_a q[i,a]-mortality[i]; resource loss is R+(I-leakageB)@q.T@N. Baseline mortality and supply are defined by the given state and are held fixed except for supply K0+epsilonv+zetaw. Preserve the resident set. No mixed supply forcing is present. The coupled Jacobian is nonsingular.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cavity_state_jet(tensors: 'np.ndarray', B: 'np.ndarray', leakage: float, N: 'np.ndarray', R: 'np.ndarray', v: 'np.ndarray', w: 'np.ndarray') -> 'np.ndarray':
    """Calculate the two first responses and mixed stationary response of the cavity community.

    Parameters
    ----------
    tensors : np.ndarray
        Finite (4,S,M) uptake derivatives from uptake_tensors for residents,
        1 <= S <= M. The derivative-order axis is not factorial-scaled.
    B : np.ndarray
        Nonnegative (M,M) conversion matrix, columns sum to one within 1e-12.
    leakage : float
        Common leakage fraction in [0,1).
    N : np.ndarray
        Positive resident abundances (S,).
    R : np.ndarray
        Positive resource abundances (M,).
    v : np.ndarray
        First signed supply direction (M,).
    w : np.ndarray
        Second signed supply direction (M,).

    Returns
    -------
    result : np.ndarray
        Shape (4,S+M), rows value, epsilon derivative, zeta derivative,
        mixed epsilon-zeta derivative of concatenated (N,R) at zero.
        Resident growth is (1-leakage)*sum_a q[i,a]-mortality[i];
        resource loss is R+(I-leakage*B)@q.T@N. Baseline mortality
        and supply are defined by the given state and are held fixed except
        for supply K0+epsilon*v+zeta*w. Preserve the resident set.
        No mixed supply forcing is present. The coupled Jacobian is nonsingular.

    Raises
    ------
    ValueError
        If inputs are nonreal or nonfinite, stated shapes or positivity fail,
        leakage is outside [0,1), B is negative or not column-stochastic
        to 1e-12, the coupled response is singular, or a result is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _oracle_cavity_state_jet(tensors: 'np.ndarray', B: 'np.ndarray', leakage: float, N: 'np.ndarray', R: 'np.ndarray', v: 'np.ndarray', w: 'np.ndarray') -> 'np.ndarray':
    t=_arr(tensors,3);s,m=t.shape[1:]
    if t.shape[0]!=4 or not 1<=s<=m:raise ValueError('invalid tensor shape')
    B=_arr(B,2,(m,m));N=_arr(N,1,(s,));R=_arr(R,1,(m,));v=_arr(v,1,(m,));w=_arr(w,1,(m,));ell=float(_arr(leakage,0))
    if np.any(B<0) or not np.allclose(B.sum(0),1,atol=1e-12,rtol=0) or not 0<=ell<1 or np.any(N<=0) or np.any(R<=0):raise ValueError('invalid ecology domain')
    L=np.eye(m)-ell*B;C=(1-ell)*t[1];E=L@t[0].T;Q=np.eye(m)+L@np.diag(N@t[1]);A=np.block([[np.zeros((s,s)),C],[E,Q]])
    a=_solve(A,np.r_[np.zeros(s),v]);b=_solve(A,np.r_[np.zeros(s),w]);Na,Ra=a[:s],a[s:];Nb,Rb=b[:s],b[s:]
    top=(1-ell)*(t[2]@(Ra*Rb))
    bottom=L@((Na@t[1])*Rb+(Nb@t[1])*Ra+(N@t[2])*Ra*Rb)
    ab=_solve(A,-np.r_[top,bottom])
    return _arr(np.stack([np.r_[N,R],a,b,ab]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Four existing cases plus one independently checked coupled-response case."""
    return [
        {
            "setup": """import numpy as np
import copy
# normal
a0=np.array([[[0.7142857142857143, 0.10666666666666669, 0.08, 0.026470588235294117], [0.0625, 0.7200000000000001, 0.14285714285714285, 0.0642857142857143], [0.11538461538461538, 0.05000000000000001, 0.6, 0.18000000000000002]], [[0.20408163265306126, 0.062222222222222213, 0.013333333333333332, 0.01384083044982699], [0.023437499999999993, 0.18000000000000002, 0.05102040816326531, 0.025510204081632657], [0.02662721893491124, 0.03125, 0.125, 0.08]], [[-0.29154518950437325, -0.08296296296296296, -0.017777777777777778, -0.016283329940972927], [-0.029296874999999993, -0.36000000000000004, -0.048590864917395525, -0.036443148688046656], [-0.040964952207555756, -0.0390625, -0.15625, -0.10666666666666666]], [[0.6247396917950856, 0.16592592592592595, 0.03555555555555556, 0.028735288131128692], [0.05493164062499999, 1.08, 0.06941552131056504, 0.0780924614743857], [0.09453450509435941, 0.0732421875, 0.29296875, 0.2133333333333333]]], dtype=float)
a1=np.array([[0.0, 0.1, 0.2, 0.3], [0.5, 0.0, 0.3, 0.2], [0.3, 0.6, 0.0, 0.5], [0.2, 0.3, 0.5, 0.0]], dtype=float)
a2=0.35
a3=np.array([1.2, 0.9, 1.1], dtype=float)
a4=np.array([1.0, 0.8, 1.2, 0.9], dtype=float)
a5=np.array([0.3, -0.2, 0.15, -0.1], dtype=float)
a6=np.array([-0.1, 0.25, 0.2, -0.15], dtype=float)
""",
            "call": 'cavity_state_jet(a0.copy(), a1.copy(), a2, a3.copy(), a4.copy(), a5.copy(), a6.copy())',
            "gold_call": '_oracle_cavity_state_jet(a0.copy(), a1.copy(), a2, a3.copy(), a4.copy(), a5.copy(), a6.copy())',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# boundary
a0=np.array([[[0.7142857142857143, 0.10666666666666669, 0.08, 0.026470588235294117], [0.0625, 0.7200000000000001, 0.14285714285714285, 0.0642857142857143], [0.11538461538461538, 0.05000000000000001, 0.6, 0.18000000000000002]], [[0.20408163265306126, 0.062222222222222213, 0.013333333333333332, 0.01384083044982699], [0.023437499999999993, 0.18000000000000002, 0.05102040816326531, 0.025510204081632657], [0.02662721893491124, 0.03125, 0.125, 0.08]], [[-0.29154518950437325, -0.08296296296296296, -0.017777777777777778, -0.016283329940972927], [-0.029296874999999993, -0.36000000000000004, -0.048590864917395525, -0.036443148688046656], [-0.040964952207555756, -0.0390625, -0.15625, -0.10666666666666666]], [[0.6247396917950856, 0.16592592592592595, 0.03555555555555556, 0.028735288131128692], [0.05493164062499999, 1.08, 0.06941552131056504, 0.0780924614743857], [0.09453450509435941, 0.0732421875, 0.29296875, 0.2133333333333333]]], dtype=float)
a1=np.array([[0.0, 0.1, 0.2, 0.3], [0.5, 0.0, 0.3, 0.2], [0.3, 0.6, 0.0, 0.5], [0.2, 0.3, 0.5, 0.0]], dtype=float)
a2=0.35
a3=np.array([1.2, 0.9, 1.1], dtype=float)
a4=np.array([1.0, 0.8, 1.2, 0.9], dtype=float)
a5=np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
a6=np.array([-0.1, 0.25, 0.2, -0.15], dtype=float)
""",
            "call": 'cavity_state_jet(a0.copy(), a1.copy(), a2, a3.copy(), a4.copy(), a5.copy(), a6.copy())',
            "gold_call": '_oracle_cavity_state_jet(a0.copy(), a1.copy(), a2, a3.copy(), a4.copy(), a5.copy(), a6.copy())',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# edge
a0=np.array([[[0.6285714285714286, 0.16875]], [[0.1224489795918367, 0.08203124999999999]], [[-0.17492711370262387, -0.10253906249999997]], [[0.3748438150770511, 0.19226074218749994]]], dtype=float)
a1=np.array([[0.2, 0.7], [0.8, 0.3]], dtype=float)
a2=0.2
a3=np.array([0.8], dtype=float)
a4=np.array([1.1, 0.9], dtype=float)
a5=np.array([0.2, -0.1], dtype=float)
a6=np.array([-0.15, 0.3], dtype=float)
""",
            "call": 'cavity_state_jet(a0.copy(), a1.copy(), a2, a3.copy(), a4.copy(), a5.copy(), a6.copy())',
            "gold_call": '_oracle_cavity_state_jet(a0.copy(), a1.copy(), a2, a3.copy(), a4.copy(), a5.copy(), a6.copy())',
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
import copy
# invalid_declared_condition
a0=np.array([[[0.7142857142857143, 0.10666666666666669, 0.08, 0.026470588235294117], [0.0625, 0.7200000000000001, 0.14285714285714285, 0.0642857142857143], [0.11538461538461538, 0.05000000000000001, 0.6, 0.18000000000000002]], [[0.20408163265306126, 0.062222222222222213, 0.013333333333333332, 0.01384083044982699], [0.023437499999999993, 0.18000000000000002, 0.05102040816326531, 0.025510204081632657], [0.02662721893491124, 0.03125, 0.125, 0.08]], [[-0.29154518950437325, -0.08296296296296296, -0.017777777777777778, -0.016283329940972927], [-0.029296874999999993, -0.36000000000000004, -0.048590864917395525, -0.036443148688046656], [-0.040964952207555756, -0.0390625, -0.15625, -0.10666666666666666]], [[0.6247396917950856, 0.16592592592592595, 0.03555555555555556, 0.028735288131128692], [0.05493164062499999, 1.08, 0.06941552131056504, 0.0780924614743857], [0.09453450509435941, 0.0732421875, 0.29296875, 0.2133333333333333]]], dtype=float)
a1=np.array([[0.0, 0.1, 0.2, 0.3], [0.5, 0.0, 0.3, 0.2], [0.3, 0.6, 0.0, 0.5], [0.2, 0.3, 0.5, 0.0]], dtype=float)
a2=1.0
a3=np.array([1.2, 0.9, 1.1], dtype=float)
a4=np.array([1.0, 0.8, 1.2, 0.9], dtype=float)
a5=np.array([0.3, -0.2, 0.15, -0.1], dtype=float)
a6=np.array([-0.1, 0.25, 0.2, -0.15], dtype=float)

def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
""",
            "call": 'expect_value_error(cavity_state_jet, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2), copy.deepcopy(a3), copy.deepcopy(a4), copy.deepcopy(a5), copy.deepcopy(a6))',
            "gold_call": 'expect_value_error(_oracle_cavity_state_jet, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2), copy.deepcopy(a3), copy.deepcopy(a4), copy.deepcopy(a5), copy.deepcopy(a6))',
            "tol": 0,
        },
        {
            "setup": """import numpy as np
a0 = np.array([[[0.765217391304348, 0.08593750000000001, 0.2648648648648649, 0.05279187817258883, 0.24747899159663866], [0.06357615894039735, 0.7572463768115941, 0.07730061349693253, 0.49337349397590363, 0.16764705882352943]], [[0.2911153119092628, 0.033365885416666664, 0.13976138300462626, 0.013811229354015818, 0.05253866252383306], [0.03736678215867725, 0.13967653854232304, 0.06300575859083896, 0.08230512411090143, 0.06689734717416379]], [[-0.506287498972631, -0.03475613064236111, -0.2518223117200474, -0.014021552643670882, -0.08830027314929927], [-0.04949242670023477, -0.20242976600336668, -0.07730767925256314, -0.09916280013361617, -0.08744751264596576]], [[1.320749997319907, 0.05430645412868924, 0.6806008424866146, 0.02135261823909271, 0.22260573062848554], [0.09832932457000285, 0.440064708702971, 0.1422840722439813, 0.17920987975954727, 0.17146571107052114]]], dtype=float)
a1 = np.array([[0.05, 0.15, 0.1, 0.3, 0.2], [0.3, 0.05, 0.2, 0.1, 0.15], [0.2, 0.35, 0.1, 0.15, 0.25], [0.25, 0.2, 0.35, 0.05, 0.1], [0.2, 0.25, 0.25, 0.4, 0.3]], dtype=float)
a2 = 0.27
a3 = np.array([1.15, 0.85], dtype=float)
a4 = np.array([0.8, 1.1, 0.7, 1.3, 0.95], dtype=float)
a5 = np.array([0.23, -0.17, 0.31, -0.11, 0.07], dtype=float)
a6 = np.array([-0.13, 0.29, 0.05, 0.19, -0.21], dtype=float)
""",
            "call": 'cavity_state_jet(a0, a1, a2, a3, a4, a5, a6)',
            "gold_call": 'np.array([[1.15, 0.85, 0.8, 1.1, 0.7, 1.3, 0.95], [0.46243649277421345, -0.15852082262836453, -0.07963990828492086, -0.04780845068725865, 0.17802557163239233, -0.022153348284833632, 0.003891094503318037], [-0.15398910044006148, 0.3467542106027842, -0.008632410770231207, 0.035100668605501366, 0.07364076161489273, 0.03362818292267163, -0.17919601901702845], [-0.01986111529596603, 0.004992719519032306, 0.009763781002874734, -0.0012391423256784904, 0.003615159203224519, -0.001815819194472214, 0.00452768459477273]], dtype=float)',
            "tol": 1e-08,
        },
    ]
