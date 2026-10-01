"""
Final orchestrator for inferred conditional second-largest-level statistics, using every preceding public function through the composed call chain. The synthetic inference and observation protocol are explicitly task-specific.

Final orchestrator for inferred conditional second-largest-level statistics, using every preceding public function through the composed call chain. The synthetic inference and observation protocol are explicitly task-specific.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def spectral_tail_surprisal(nodes, count, record, calibration_cut, target, prediction_cut, bracket, tilt=0.07):
    """Assemble the inferred finite-ensemble joint-event surprisal.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes in supplied label order.
    count : even int, 4<=count<=m
        Fixed number of levels. The final assembly extends a count-2 SOP prefix
        by the last pair, using discrete_skew_metric and skew_project.
    record : integer array, (r,2)
        Original labels and occupancies as in condition_spectral_record.
    target : float in (0,1)
        Conditional at-most-one calibration probability.
    bracket : real array, (2,)
        Increasing nonnegative confinement bounds containing the unique root.
    tilt : finite float
        Linear coefficient of the weight, as in infer_confinement.
    calibration_cut : finite float
        Strict upper-domain boundary for inference.
    prediction_cut : finite float
        Strict upper-domain boundary for prediction, larger than calibration_cut.
        All recorded-present nodes must be <= both cuts, so the resulting event
        states that the second-largest level is <=prediction_cut.

    Returns
    -------
    float
        -ln P(record AND at most one level above prediction_cut), at the inferred
        theta; natural logarithm, dimensionless nats. Compose the preceding
        public functions on the submission path, including through their calls
        to earlier steps. This final orchestrator must use every earlier function.
        No rounding is fed back into the calculation.

    Raises
    ------
    ValueError
        For nonfinite/misordered cuts, a recorded-present node above either cut,
        a nonpositive computed joint probability, or an earlier contract violation.
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

def _oracle_spectral_tail_surprisal(nodes,count,record,calibration_cut,target,prediction_cut,bracket,tilt=0.07):
    """Assemble the inferred finite-ensemble joint-event surprisal.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes in supplied label order.
    count : even int, 4<=count<=m
        Fixed number of levels. The final assembly extends a count-2 SOP prefix
        by the last pair, using discrete_skew_metric and skew_project.
    record : integer array, (r,2)
        Original labels and occupancies as in condition_spectral_record.
    target : float in (0,1)
        Conditional at-most-one calibration probability.
    bracket : real array, (2,)
        Increasing nonnegative confinement bounds containing the unique root.
    tilt : finite float
        Linear coefficient of the weight, as in infer_confinement.
    calibration_cut : finite float
        Strict upper-domain boundary for inference.
    prediction_cut : finite float
        Strict upper-domain boundary for prediction, larger than calibration_cut.
        All recorded-present nodes must be <= both cuts, so the resulting event
        states that the second-largest level is <=prediction_cut.

    Returns
    -------
    float
        -ln P(record AND at most one level above prediction_cut), at the inferred
        theta; natural logarithm, dimensionless nats. Compose the preceding
        public functions on the submission path, including through their calls
        to earlier steps. This final orchestrator must use every earlier function.
        No rounding is fed back into the calculation.

    Raises
    ------
    ValueError
        For nonfinite/misordered cuts, a recorded-present node above either cut,
        a nonpositive computed joint probability, or an earlier contract violation.
    """
    if not np.isfinite([calibration_cut,prediction_cut]).all() or prediction_cut<=calibration_cut:
        raise ValueError('Invalid prediction interval')
    x=_real(nodes,1);r=_real(record,2)
    if not isinstance(count,(int,np.integer)) or count<4 or count%2:raise ValueError('At least two polynomial pairs required')
    theta=_oracle_infer_confinement(x,count,r,calibration_cut,target,bracket,tilt)
    if any(x[int(i)]>calibration_cut for i,b in r if b==1):raise ValueError('Recorded level exceeds calibration cut')
    weights=np.exp(-x*x/4-theta*x**4/150+tilt*x)
    metric=_oracle_discrete_skew_metric(x,weights)
    basis=_oracle_symplectic_arnoldi(x,weights,count-2,2)
    for _ in range(2):
        extension=_oracle_skew_project(basis,x*basis[:,-1],metric,2)[:len(x)]
        basis=np.column_stack((basis,extension))
    kernel=_oracle_orthogonal_ensemble_kernel(x,weights,basis)
    packed=_oracle_condition_spectral_record(kernel,r)
    labels=np.array([i for i in range(len(x)) if i not in set(r[:,0])],int)
    kc=packed[1:].reshape(2*len(labels),2*len(labels))
    rows=np.where(x[labels]>prediction_cut)[0];pair=np.column_stack((2*rows,2*rows+1)).ravel()
    restricted=kc[np.ix_(pair,pair)]
    q1=_oracle_zero_one_level_probabilities(restricted)[1]
    q0=_oracle_signed_pfaffian(_j(len(rows))-restricted).real
    p=packed[0]*(q0+q1)
    if p<=0:raise ValueError('Zero predicted joint probability')
    return float(-np.log(p))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'canonical finite ensemble',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.685853737115559, -3.4120071472438904, -3.1277757753930104, '
               '-2.8059207744943198, -2.5041457504912956, -2.234819966598906, -1.9370523939461926, '
               '-1.6141647169622129, -1.3261920030169825, -1.0543421918908693, -0.7434227232053965, '
               '-0.4264481722418766, -0.15010360686578172, 0.13065577612475837, 0.45090749481684367, '
               '0.7565676558504535, 1.026495564917499, 1.3202228395123643, 1.6435495316418218, '
               '1.935449414276086, 2.205848222129218, 2.5132144296915135, 2.8325556201930464, '
               '3.1118435897434082, 3.3894778503512537, 3.707591474495091, '
               '4.0169277213330465],dtype=float).reshape((28,))\n'
               'arg1=8\n'
               'arg2=np.array([[3, 0], [8, 1], [20, 0]],dtype=int).reshape((3, 2))\n'
               'arg3=1.8\n'
               'arg4=0.2\n'
               'arg5=2.4\n'
               'arg6=np.array([0.0, 4.0],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'unconditioned joint law',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.36676541802724, -2.773830509067252, -2.170510818128053, '
               '-1.5295674981410434, -0.9087041550497, -0.32029005206899125, 0.29656583967204175, '
               '0.9385418357443402, 1.5456028687778893, 2.1365409989923223, 2.7665487867661143, '
               '3.402611656817953, 3.9980445412823666],dtype=float).reshape((14,))\n'
               'arg1=4\n'
               'arg2=np.array([],dtype=int).reshape((0, 2))\n'
               'arg3=1.1\n'
               'arg4=0.701209587462767\n'
               'arg5=2.2\n'
               'arg6=np.array([1.0899999999999999, 1.47],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'two observed levels',
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
               'arg5=1.6\n'
               'arg6=np.array([1.2899999999999998, 1.67],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'lower endpoint inferred model',
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
               'arg5=1.5\n'
               'arg6=np.array([0.0, 0.27],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'upper endpoint inferred model',
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
               'arg5=1.5\n'
               'arg6=np.array([3.89, 4.0],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'asymmetric confinement model',
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
               'arg5=1.5\n'
               'arg6=np.array([1.89, 2.27],dtype=float).reshape((2,))\n'
               'arg7=-0.2',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'absence-only record evidence',
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
               'arg5=1.5\n'
               'arg6=np.array([2.39, 2.77],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'uniform finite grid',
      'setup': 'import numpy as np\n'
               'arg0=np.array([-4.0, -3.466666666666667, -2.9333333333333336, -2.4, -1.8666666666666667, '
               '-1.3333333333333335, -0.7999999999999998, -0.2666666666666666, 0.2666666666666666, '
               '0.7999999999999998, 1.333333333333333, 1.8666666666666663, 2.4000000000000004, '
               '2.9333333333333336, 3.466666666666667, 4.0],dtype=float).reshape((16,))\n'
               'arg1=4\n'
               'arg2=np.array([[2, 0], [5, 1]],dtype=int).reshape((2, 2))\n'
               'arg3=0.4\n'
               'arg4=0.39681724558829384\n'
               'arg5=1.5\n'
               'arg6=np.array([2.0900000000000003, 2.47],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'},
     {'name': 'large canonical instance requiring a stable polynomial representation',
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
               'arg5=1.94\n'
               'arg6=np.array([0.0, 4.0],dtype=float).reshape((2,))\n'
               'arg7=0.07',
      'call': 'spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)',
      'gold_call': '_oracle_spectral_tail_surprisal(arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7)'}]
