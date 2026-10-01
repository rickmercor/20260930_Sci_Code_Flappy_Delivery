"""
Task-specific inverse use of the main paper's general-weight finite ensemble. Root finding is a supporting operation; its probability map must use the preceding scientific functions.

Task-specific inverse use of the main paper's general-weight finite ensemble. Root finding is a supporting operation; its probability map must use the preceding scientific functions.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def infer_confinement(nodes, count, record, threshold, target, bracket, tilt=0.07):
    """Infer a quartic confinement parameter from a conditional tail probability.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes, supplied order fixes observation labels.
    count : even int, 2<=count<=m
        Fixed total number of levels before conditioning.
    record : integer array, (r,2)
        Original node labels and binary observations as in condition_spectral_record.
    threshold : finite float
        Queried unobserved nodes satisfy x>threshold (strict inequality).
    target : float in (0,1)
        Observed conditional probability of at most one level in that domain.
    bracket : real array, (2,)
        Finite increasing nonnegative parameter endpoints. The user-supplied
        equation is continuous with a unique root in this bracket, possibly
        at an endpoint. Endpoint probability residual <=1e-13 counts as a root.
    tilt : finite float
        Linear coefficient in log w=-x^2/4-theta*x^4/150+tilt*x.

    Returns
    -------
    float
        Dimensionless theta at that root, with parameter accuracy 1e-9 or better.
        Use the exact finite beta=1 ensemble and condition on the full record.
        Positive-mass observations and well-conditioned SOPs throughout the
        bracket are caller preconditions. This inverse problem is a task-specific
        application of the source method, not a fitted parameter from the paper.

    Raises
    ------
    ValueError
        For invalid scalar/bracket inputs, a root not bracketed, or any invalid
        contract propagated from symplectic_arnoldi/condition_spectral_record.
    """
    return 0.0

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

def _pf_curve(nodes,count,record,threshold,theta,tilt):
    x=np.asarray(nodes,float);w=np.exp(-x*x/4-theta*x**4/150+tilt*x)
    s=_oracle_symplectic_arnoldi(x,w,count,2)
    k=_oracle_orthogonal_ensemble_kernel(x,w,s)
    packed=_oracle_condition_spectral_record(k,record)
    ids=np.array([i for i in range(len(x)) if i not in set(np.asarray(record)[:,0])],int)
    kc=packed[1:].reshape(2*len(ids),2*len(ids))
    rows=np.where(x[ids]>threshold)[0];pair=np.column_stack((2*rows,2*rows+1)).ravel()
    return np.r_[packed[0],_oracle_zero_one_level_probabilities(kc[np.ix_(pair,pair)])]

def _oracle_infer_confinement(nodes, count, record, threshold, target, bracket, tilt=0.07):
    """Infer a quartic confinement parameter from a conditional tail probability.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes, supplied order fixes observation labels.
    count : even int, 2<=count<=m
        Fixed total number of levels before conditioning.
    record : integer array, (r,2)
        Original node labels and binary observations as in condition_spectral_record.
    threshold : finite float
        Queried unobserved nodes satisfy x>threshold (strict inequality).
    target : float in (0,1)
        Observed conditional probability of at most one level in that domain.
    bracket : real array, (2,)
        Finite increasing nonnegative parameter endpoints. The user-supplied
        equation is continuous with a unique root in this bracket, possibly
        at an endpoint. Endpoint probability residual <=1e-13 counts as a root.
    tilt : finite float
        Linear coefficient in log w=-x^2/4-theta*x^4/150+tilt*x.

    Returns
    -------
    float
        Dimensionless theta at that root, with parameter accuracy 1e-9 or better.
        Use the exact finite beta=1 ensemble and condition on the full record.
        Positive-mass observations and well-conditioned SOPs throughout the
        bracket are caller preconditions. This inverse problem is a task-specific
        application of the source method, not a fitted parameter from the paper.

    Raises
    ------
    ValueError
        For invalid scalar/bracket inputs, a root not bracketed, or any invalid
        contract propagated from symplectic_arnoldi/condition_spectral_record.
    """
    bounds=_real(bracket,1)
    if bounds.shape!=(2,) or bounds[0]<0 or bounds[0]>=bounds[1] or not np.isfinite([threshold,target,tilt]).all() or not 0<target<1:
        raise ValueError('Invalid inverse problem')
    def residual(theta):return np.sum(_pf_curve(nodes,count,record,threshold,theta,tilt)[1:])-target
    lo,hi=map(float,bounds);fl=residual(lo);fh=residual(hi)
    if abs(fl)<=1e-13:return lo
    if abs(fh)<=1e-13:return hi
    if fl*fh>0:raise ValueError('Root not bracketed')
    return float(brentq(residual,lo,hi,xtol=2e-10,rtol=1e-12))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'interior confinement root',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.448816700078522, -2.9379330731698166, -2.416664664281899, '
               '-1.8577726263461718, -1.3189605653061107, -0.8125977443766836, -0.27779313468693323, '
               '0.2821315793340835, 0.8071413303163508, 1.316028178479501, 1.8639846842020107, '
               '2.4179962722025685, 2.9313778746157, 3.449174294643277, '
               '4.006463050372399],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[2, 0], [5, 1]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.38470139855234564\n'
               'arg5=np.array([1.39, 1.77],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'lower endpoint root',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.448816700078522, -2.9379330731698166, -2.416664664281899, '
               '-1.8577726263461718, -1.3189605653061107, -0.8125977443766836, -0.27779313468693323, '
               '0.2821315793340835, 0.8071413303163508, 1.316028178479501, 1.8639846842020107, '
               '2.4179962722025685, 2.9313778746157, 3.449174294643277, '
               '4.006463050372399],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[2, 0], [5, 1]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.3577867816474117\n'
               'arg5=np.array([0.0, 0.27],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'upper endpoint root',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.448816700078522, -2.9379330731698166, -2.416664664281899, '
               '-1.8577726263461718, -1.3189605653061107, -0.8125977443766836, -0.27779313468693323, '
               '0.2821315793340835, 0.8071413303163508, 1.316028178479501, 1.8639846842020107, '
               '2.4179962722025685, 2.9313778746157, 3.449174294643277, '
               '4.006463050372399],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[2, 0], [5, 1]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.42519944477896526\n'
               'arg5=np.array([3.89, 4.0],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'unconditioned inference',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.36676541802724, -2.773830509067252, -2.170510818128053, '
               '-1.5295674981410434, -0.9087041550497, -0.32029005206899125, 0.29656583967204175, '
               '0.9385418357443402, 1.5456028687778893, 2.1365409989923223, 2.7665487867661143, '
               '3.402611656817953, 3.9980445412823666],dtype=float).reshape((14,))\n'
               'arg1=4\n'
               'arg2=np.array([],dtype=int).reshape((0, 2))\n'
               'arg3=1.1\n'
               'arg4=0.701209587462767\n'
               'arg5=np.array([1.0899999999999999, 1.47],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'absences without observed inclusion',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.448816700078522, -2.9379330731698166, -2.416664664281899, '
               '-1.8577726263461718, -1.3189605653061107, -0.8125977443766836, -0.27779313468693323, '
               '0.2821315793340835, 0.8071413303163508, 1.316028178479501, 1.8639846842020107, '
               '2.4179962722025685, 2.9313778746157, 3.449174294643277, '
               '4.006463050372399],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[1, 0], [12, 0]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.41707304133173\n'
               'arg5=np.array([2.39, 2.77],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'multiple occupied observations',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.5115617981177376, -3.0634232692482475, -2.6048999583995465, '
               '-2.1087530185030343, -1.632686055502189, -1.1890683326119782, -0.7170088209614432, '
               '-0.21982920497964195, 0.24243544796340993, 0.6885771980873446, 1.1737886057706388, '
               '1.6650550957319792, 2.1156916001058956, 2.570742922094257, 3.0652865797841637, '
               '3.5452386798155953, 3.989458527880463],dtype=float).reshape((18,))\n'
               'arg1=6\n'
               'arg2=np.array([[4, 1], [7, 1], [15, 0]],dtype=int).reshape((3, 2))\n'
               'arg3=0.6\n'
               'arg4=0.03554998761965078\n'
               'arg5=np.array([1.2899999999999998, 1.67],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'negative asymmetry tilt',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.448816700078522, -2.9379330731698166, -2.416664664281899, '
               '-1.8577726263461718, -1.3189605653061107, -0.8125977443766836, -0.27779313468693323, '
               '0.2821315793340835, 0.8071413303163508, 1.316028178479501, 1.8639846842020107, '
               '2.4179962722025685, 2.9313778746157, 3.449174294643277, '
               '4.006463050372399],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[2, 0], [5, 1]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.5739267130412591\n'
               'arg5=np.array([1.89, 2.27],dtype=float).reshape((2,))\n'
               'arg6=-0.2',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'zero perturbation of node spacing',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.466666666666667, -2.9333333333333336, -2.4, -1.8666666666666667, '
               '-1.3333333333333335, -0.7999999999999998, -0.2666666666666666, 0.2666666666666666, '
               '0.7999999999999998, 1.333333333333333, 1.8666666666666663, 2.4000000000000004, '
               '2.9333333333333336, 3.466666666666667, 4.0],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[2, 0], [5, 1]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.39681724558829384\n'
               'arg5=np.array([2.0900000000000003, 2.47],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'},
     {'name': 'rare-tail inverse on a high-degree finite ensemble',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-2.0, -1.9655289425765166, -1.9377744973218287, -1.9082892550706052, '
               '-1.8725334079780802, -1.8401242237014996, -1.8131232461100246, -1.7813819703918292, '
               '-1.7454540106184226, -1.7153452113514744, -1.687923562554379, -1.6539903041640567, '
               '-1.61904786539406, -1.5908770908886345, -1.5619705134474677, -1.5264818800560436, '
               '-1.4934251729410322, -1.4663238411537811, -1.4352559484452272, -1.3992548194809076, '
               '-1.3684914920991202, -1.3413116771808549, -1.307970628977729, -1.2726337502847305, '
               '-1.2439724084169268, -1.2155866847062093, -1.18045440040616, -1.1467850123237573, '
               '-1.1194852125839478, -1.089081143848747, -1.0531074227086903, -1.0216732086590359, '
               '-0.9946388660234096, -0.9619312178827697, -0.9262856466856566, -0.8970704514145378, '
               '-0.8691361421312627, -0.8344413867072852, -0.8002078378238859, -0.772615887234216, '
               '-0.7428512633570286, -0.707004915168111, -0.6748984336534906, -0.6479099539898348, '
               '-0.6158627549313, -0.5800011303062327, -0.5501811605423399, -0.5226187484143876, '
               '-0.48843293368136914, -0.45369633923074115, -0.4257250772535467, -0.3965612433114002, '
               '-0.3609393895592909, -0.3281742683652637, -0.30113101845763207, -0.269756572767279, '
               '-0.23377635725116952, -0.20331419368586404, -0.17603585872011943, -0.14241903423190252, '
               '-0.10725174009836576, -0.07882247444204472, -0.05020736268765211, -0.014902112949089618, '
               '0.018493315800726204, 0.04569074640535198, 0.07639515387204768, 0.11239385016333436, '
               '0.14352128966003344, 0.17060970956870647, 0.2036101971445047, 0.23912622957070595, '
               '0.2680819661215258, 0.29621267383092764, 0.3311162782634144, 0.36509961063091567, '
               '0.3925471902529873, 0.42259923345469497, 0.458515887816174, 0.4903168328797753, '
               '0.51731373974991, 0.5496643036611497, 0.5854393272976702, 0.6149782483300075, '
               '0.6426996846098859, 0.6771255693015888, 0.7116412762154463, 0.7394293885198995, '
               '0.7688613056527458, 0.8045972269913265, 0.8370648710336649, 0.8640707099411725, '
               '0.8957522731859989, 0.9316907588902416, 0.9618565582140882, 0.9892529927933357, '
               '1.0231357485530548, 1.0581164155127003, 1.0863278416652027, 1.1151857154349003, '
               '1.1506462475790893, 1.1837588997475483, 1.2108739163577948, 1.241882337584705, '
               '1.2778851068719836, 1.3087074830422505, 1.3358704414149758, 1.3691567845757933, '
               '1.4045246167013554, 1.4332326872527368, 1.4615754160636818, 1.4966720509923905, '
               '1.53039362042501, 1.5577156229767997, 1.5880617889416202, 1.6240282281760325, '
               '1.6555222214629677, 1.6825484415530447, 1.7151984035469516, 1.7508669623744215, '
               '1.7801339201330564, 1.8080319031629197, 1.8426842569748818, 1.8769650585402076, '
               '1.9045872342460795, 1.934296816761716, 1.9701271233872208, '
               '2.0022927799976435],dtype=float).reshape((128,))\n'
               'arg1=32\n'
               'arg2=np.array([[7, 0], [42, 1], [96, 0]],dtype=int).reshape((3, 2))\n'
               'arg3=1.9\n'
               'arg4=0.0012\n'
               'arg5=np.array([0.0, 4.0],dtype=float).reshape((2,))\n'
               'arg6=0.07',
      'call': 'infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)',
      'gold_call': '_oracle_infer_confinement(arg0,arg1,arg2,arg3,arg4,arg5,arg6)'}]
