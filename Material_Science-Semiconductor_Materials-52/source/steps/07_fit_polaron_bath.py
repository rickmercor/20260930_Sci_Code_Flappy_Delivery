"""
Infer the two coupling strengths and phonon decay rates.

Fit the recovered Sigma samples by unweighted nonlinear least squares to sum_i weights[i]*Psi0(times;omega[i],S_i,gamma0,gamma_ph_i). Parameter order is [S1,S2,gamma_ph1,gamma_ph2]. Bounds are [0.01,0.005,0.04,0.04] and [0.18,0.12,0.35,0.35]. Inputs have an isolated best fit in this box. Round all four fitted coordinates to seven decimal places before any prediction. Equivalent converged optimization algorithms are accepted.

Returns
-------
return result  # real ndarray, shape (4,), ordered [S1,S2,gamma_ph1,gamma_ph2]; first two dimensionless, last two ps^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_polaron_bath(times: np.ndarray, channels: np.ndarray, mixing: np.ndarray, omega: np.ndarray, weights: np.ndarray, gamma0: float) -> np.ndarray:
    """Infer the two coupling strengths and phonon decay rates.

    Parameters
    ----------
    times : nonnegative real ndarray, shape (T,)
        Calibration pulse separations in ps.
    channels : real ndarray, shape (2,T)
        Measured dimensionless parallel and cross amplitudes.
    mixing : nonsingular real ndarray, shape (2,2)
        Dimensionless detector response.
    omega : positive real ndarray, shape (2,)
        Distinct known phonon angular frequencies in ps^-1, in mode order.
    weights : positive real ndarray, shape (2,)
        Known mode amplitude weights, summing to one.
    gamma0 : positive float
        Known zero-phonon optical population-decay rate in ps^-1.

    Returns
    -------
    real ndarray, shape (4,), ordered [S1,S2,gamma_ph1,gamma_ph2]; first two dimensionless, last two ps^-1.

    Raises
    ------
    ValueError
        If inputs fall outside the documented numerical domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def _oracle_fit_polaron_bath(times: np.ndarray, channels: np.ndarray, mixing: np.ndarray, omega: np.ndarray, weights: np.ndarray, gamma0: float) -> np.ndarray:
    if np.ndim(times)!=1 or np.shape(channels)!=(2,len(times)) or np.shape(omega)!=(2,) or np.shape(weights)!=(2,) or not np.all(np.isfinite(omega)) or not np.all(np.isfinite(weights)) or min(omega)<=0 or min(weights)<=0 or not np.isclose(sum(weights),1.,rtol=0,atol=1e-12) or omega[0]==omega[1] or not np.isfinite(gamma0) or gamma0<=0:raise ValueError("invalid two-mode fit data")
    target=_oracle_recover_intrinsic_echo(channels,mixing)[0]
    def residual(z):
        prediction=sum(weights[i]*_oracle_weak_polaron_echo(times,omega[i],z[i],gamma0,z[2+i]) for i in range(2))
        return prediction-target
    candidates=[]
    for start in [(.06,.035,.11,.16),(.14,.075,.23,.075)]:
        result=least_squares(residual,start,bounds=([.01,.005,.04,.04],[.18,.12,.35,.35]),
          xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=400)
        candidates.append((float(result.fun@result.fun),tuple(result.x)))
    candidates.sort()
    return np.round(np.array(candidates[0][1]),7)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ntimes=np.array([0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37])\nchannels=np.array([[2.375653279, 1.7582586482, 1.7035211663, 1.7225594908, 1.9080361027, 1.5395260639, 1.0925985855, 1.2827918215, 1.0345356415, 0.9666848089, 1.2239222733, 1.3924916511, 1.2512701821, 1.2274585368, 1.1048938415, 1.0960834178, 1.3421942552, 1.2251986651], [-0.0693478451, -0.0276750754, 0.0207849291, 0.0812233679, 0.1776811678, 0.2618057848, 0.2747511869, 0.4449630364, 0.4247748017, 0.3941245946, 0.4120523666, 0.3671592284, 0.3403742145, 0.4030103264, 0.3380416278, 0.3464142758, 0.2605912086, 0.3240218881]])\nmixing=np.array([[1.0, 0.075], [-0.035, 0.91]])\nomega=np.array([4.861655833587608, 7.748263984780249])\nweights=np.array([0.63, 0.37])\ngamma0=0.0032258064516129032\n', 'call': 'fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)', 'gold_call': '_oracle_fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)'}, {'setup': 'import numpy as np\ntimes=np.array([0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37])\nchannels=np.array([[2.3256822295, 1.7607021634, 1.7485407355, 1.7432889583, 1.8306797518, 1.5480094831, 1.1069000228, 1.2694032831, 1.0221777425, 0.981395341, 1.212440373, 1.3503121881, 1.2334996094, 1.1910050349, 1.1104620577, 1.0981549908, 1.3220428473, 1.1961622247], [-0.067889137, -0.0277135365, 0.02133422, 0.0822008187, 0.1704775479, 0.2632484419, 0.2783475094, 0.4403189433, 0.4197007144, 0.400122188, 0.4081868072, 0.3560377404, 0.3355402108, 0.3910415819, 0.3397452203, 0.347068991, 0.2566787498, 0.3163427724]])\nmixing=np.array([[1.0, 0.075], [-0.035, 0.91]])\nomega=np.array([4.861655833587608, 7.748263984780249])\nweights=np.array([0.63, 0.37])\ngamma0=0.0032258064516129032\n', 'call': 'fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)', 'gold_call': '_oracle_fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)'}, {'setup': 'import numpy as np\ntimes=np.array([0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37])\nchannels=np.array([[2.2971484499, 1.756471513, 1.7893709506, 1.7699913257, 1.7672207714, 1.5821965593, 1.1096864266, 1.2843587034, 1.0051064853, 1.000265741, 1.2357023567, 1.3121263433, 1.2082601582, 1.1434407325, 1.1674108356, 1.1240510344, 1.3369685621, 1.2055391924], [-0.0670562057, -0.0276469458, 0.0218323958, 0.083459908, 0.1645680865, 0.2690621624, 0.2790481947, 0.445506542, 0.4126913475, 0.4078157906, 0.4160183139, 0.3459692526, 0.3286745007, 0.3754248384, 0.3571686658, 0.3552533673, 0.259576624, 0.3188226501]])\nmixing=np.array([[1.0, 0.075], [-0.035, 0.91]])\nomega=np.array([4.861655833587608, 7.748263984780249])\nweights=np.array([0.63, 0.37])\ngamma0=0.0032258064516129032\n', 'call': 'fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)', 'gold_call': '_oracle_fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)'}, {'setup': 'import numpy as np\ntimes=np.array([0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37])\nchannels=np.array([[2.1551818975, 1.7578950498, 1.9167565756, 1.8266472967, 1.5805678577, 1.6044223682, 1.1428749823, 1.2574719297, 0.9800769588, 1.0261709946, 1.2082988927, 1.2648627969, 1.1948635676, 1.1263789599, 1.1428293658, 1.1079758721, 1.3080900861, 1.1804377457], [-0.062912051, -0.0276693524, 0.0233866478, 0.0861313913, 0.1471864931, 0.2728417966, 0.2873939817, 0.4361803051, 0.4024143578, 0.4183775554, 0.4067925138, 0.3335072409, 0.3250303205, 0.3698229624, 0.3496479794, 0.3501728547, 0.2539697776, 0.3121842017]])\nmixing=np.array([[1.0, 0.075], [-0.035, 0.91]])\nomega=np.array([4.861655833587608, 7.748263984780249])\nweights=np.array([0.63, 0.37])\ngamma0=0.0032258064516129032\n', 'call': 'fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)', 'gold_call': '_oracle_fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)'}, {'setup': 'import numpy as np\ntimes=np.array([0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37])\nchannels=np.array([[2.7428812356, 1.6904699314, 1.4775595731, 1.6416122335, 2.2400125976, 1.5057290401, 1.0176859776, 1.3007290457, 1.0724089797, 0.9167016242, 1.2056123089, 1.4198792288, 1.2637032148, 1.2206609768, 1.0728915721, 1.0900014954, 1.3078530892, 1.1744968735], [-0.0800676196, -0.0266080778, 0.0180279363, 0.0774064845, 0.2085956621, 0.256058395, 0.2559132274, 0.4511849359, 0.4403253918, 0.373746078, 0.4058880338, 0.3743805298, 0.3437562849, 0.4007784898, 0.3282505521, 0.3444921002, 0.253923764, 0.3106130502]])\nmixing=np.array([[1.0, 0.075], [-0.035, 0.91]])\nomega=np.array([4.861655833587608, 7.748263984780249])\nweights=np.array([0.63, 0.37])\ngamma0=0.0032258064516129032\n', 'call': 'fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)', 'gold_call': '_oracle_fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)'}, {'setup': 'import numpy as np\ntimes=np.array([0.17, 0.39, 0.68, 0.94, 1.21, 1.58, 1.93, 2.37, 2.86, 3.41, 4.13, 5.07, 6.24, 7.81, 9.68, 12.03, 15.29, 19.37])\nchannels=np.array([[2.5397012845, 1.7200076294, 1.595554716, 1.6793507418, 2.1447203595, 1.5270920453, 1.0384207172, 1.329119359, 1.0756518465, 0.906435122, 1.2424211529, 1.5922832732, 1.3510866405, 1.4534048607, 0.9941041813, 1.0354818764, 1.4201684179, 1.4103859864], [-0.0741365808, -0.0270730026, 0.0194676135, 0.0791859578, 0.1997218069, 0.2596913041, 0.2611273054, 0.4610327068, 0.4416568956, 0.3695603486, 0.4182803006, 0.4198384224, 0.3675265827, 0.477195074, 0.3041455958, 0.3272613183, 0.275730136, 0.3729974112]])\nmixing=np.array([[1.0, 0.075], [-0.035, 0.91]])\nomega=np.array([4.861655833587608, 7.748263984780249])\nweights=np.array([0.63, 0.37])\ngamma0=0.0032258064516129032\n', 'call': 'fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)', 'gold_call': '_oracle_fit_polaron_bath(times,channels,mixing,omega,weights,gamma0)'}]
