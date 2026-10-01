"""
Infer one steady flux state with the flux-sum/logit capacity model.

The fixed topology has eight irreversible reactions. At a stated uptake, three independent branch variables determine the complete steady flux vector.

Returns
-------
eight fluxes, eight saturation fractions, the minimized objective, the RMS residual, and the smallest remaining capacity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kineflux_state(
    uptake: float,
    enzyme_abundance: "np.ndarray",
    apparent_rate: "np.ndarray",
    intercept: "np.ndarray",
    coefficients: "np.ndarray",
    stoichiometry: "np.ndarray",
    weight: float,
) -> "np.ndarray":
    """Infer one condition-specific steady flux state.

    Parameters
    ----------
    uptake
        Positive finite flux fixed on R01.
    enzyme_abundance, apparent_rate, intercept
        Finite reaction vectors of length eight. Enzyme abundances and
        apparent rates must be positive.
    coefficients
        Finite 8-by-4 flux-sum coefficient matrix.
    stoichiometry
        The finite 4-by-8 matrix for the stated branched network.
    weight
        Finite nonnegative quadratic flux penalty.

    Returns
    -------
    np.ndarray
        R01--R08 fluxes, R01--R08 saturation fractions, minimized objective,
        RMS capacity residual, and minimum capacity margin.

    Raises
    ------
    ValueError
        If an input violates the stated shape, range, topology or finite-value
        requirements, or if no feasible converged state is found.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize


def _ks5_unpack(uptake, independent):
    r02, r04, r05 = independent
    return np.array([
        uptake, r02, uptake-r02, r04, r05, r04+r05,
        r02-r04, uptake-r02-r05,
    ], dtype=np.float64)


def _oracle_kineflux_state(
    uptake: float,
    enzyme_abundance: "np.ndarray",
    apparent_rate: "np.ndarray",
    intercept: "np.ndarray",
    coefficients: "np.ndarray",
    stoichiometry: "np.ndarray",
    weight: float,
) -> "np.ndarray":
    enzyme_abundance = np.asarray(enzyme_abundance, dtype=np.float64)
    apparent_rate = np.asarray(apparent_rate, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    stoichiometry = np.asarray(stoichiometry, dtype=np.float64)
    expected = np.array([
        [1,-1,-1,0,0,0,0,0], [0,1,0,-1,0,0,-1,0],
        [0,0,1,0,-1,0,0,-1], [0,0,0,1,1,-1,0,0],
    ], dtype=np.float64)
    if (
        not np.isfinite(uptake) or uptake <= 0.0
        or enzyme_abundance.shape != (8,) or apparent_rate.shape != (8,)
        or intercept.shape != (8,) or coefficients.shape != (8,4)
        or stoichiometry.shape != (4,8)
        or not np.isfinite(enzyme_abundance).all()
        or not np.isfinite(apparent_rate).all()
        or not np.isfinite(intercept).all()
        or not np.isfinite(coefficients).all()
        or not np.isfinite(stoichiometry).all()
        or np.any(enzyme_abundance <= 0.0) or np.any(apparent_rate <= 0.0)
        or not np.array_equal(stoichiometry, expected)
        or not np.isfinite(weight) or weight < 0.0
    ):
        raise ValueError("KineFlux inputs violate the stated contract")
    capacity = enzyme_abundance * apparent_rate

    def _ks5_objective(independent):
        flux = _ks5_unpack(uptake, independent)
        # SLSQP samples infinitesimally outside inequality faces while taking
        # numerical derivatives.  Project only those trial values for the
        # flux-sum evaluation; accepted states remain constrained.
        flux_sum = _oracle_flux_sums(stoichiometry, np.maximum(flux,0.0))
        linear = np.clip(intercept + coefficients @ flux_sum, -700.0, 700.0)
        eta = 1.0 / (1.0 + np.exp(-linear))
        mismatch = flux - capacity * eta
        return float(np.sum(mismatch*mismatch) + weight*np.sum(flux*flux))

    constraints = []
    for reaction in range(8):
        constraints.append({"type":"ineq", "fun":lambda x, reaction=reaction:
            capacity[reaction]-_ks5_unpack(uptake,x)[reaction]})
        constraints.append({"type":"ineq", "fun":lambda x, reaction=reaction:
            _ks5_unpack(uptake,x)[reaction]})
    starts = (
        np.array([.50*uptake,.25*uptake,.25*uptake]),
        np.array([.70*uptake,.50*uptake,.15*uptake]),
        np.array([.30*uptake,.12*uptake,.50*uptake]),
    )
    fits=[]
    for start in starts:
        fit = minimize(_ks5_objective, start, method="SLSQP", constraints=constraints,
            options={"ftol":1e-13,"maxiter":2000,"disp":False})
        feasible = min(float(rule["fun"](fit.x)) for rule in constraints)
        if fit.success and feasible >= -2e-8:
            fits.append((_ks5_objective(fit.x),fit.x.copy()))
    if not fits:
        raise ValueError("no feasible converged flux state")
    lowest=min(value for value,_ in fits)
    independent=min((x for value,x in fits if value<=lowest+1e-12),
        key=lambda x:tuple(float(v) for v in x))
    flux = np.maximum(_ks5_unpack(uptake, independent),0.0)
    flux_sum = _oracle_flux_sums(stoichiometry, flux)
    linear = np.clip(intercept + coefficients @ flux_sum, -700.0, 700.0)
    eta = 1.0 / (1.0 + np.exp(-linear))
    mismatch = flux - capacity*eta
    residual = float(np.sqrt(np.mean(mismatch*mismatch)))
    margin = float(np.min(capacity-flux))
    return np.concatenate((flux,eta,[_ks5_objective(independent),residual,margin])).astype(np.float64)


def _ks5_fixture():
    stoich=np.array([[1,-1,-1,0,0,0,0,0],[0,1,0,-1,0,0,-1,0],
        [0,0,1,0,-1,0,0,-1],[0,0,0,1,1,-1,0,0]],float)
    kmax=np.array([11.,12.5,12.,13.5,13.,10.5,8.5,8.])
    beta=np.array([-.30,-.10,-.22,.05,-.08,-.18,.12,.09])
    alpha=np.array([[.35,-.12,.08,.04],[.16,.25,-.18,.09],
        [.18,-.14,.27,.07],[-.05,.22,.03,.28],[.02,-.03,.24,.31],
        [.08,.04,.06,.38],[-.09,.21,-.02,.05],[.03,-.05,.19,.02]])
    return stoich,kmax,beta,alpha

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup":"""import numpy as np
s,k,b,a=_ks5_fixture(); e=np.array([.16,.13,.12,.11,.105,.15,.115,.11])
cargs=(.82,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),.012)
gargs=(.82,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),.012)""",
            "call":"kineflux_state(*cargs)",
            "gold_call":"_oracle_kineflux_state(*gargs)","tol":1e-8,
        },
        {
            "setup":"""import numpy as np
s,k,b,a=_ks5_fixture(); e=np.full(8,.30)
cargs=(.40,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),0.)
gargs=(.40,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),0.)""",
            "call":"kineflux_state(*cargs)",
            "gold_call":"_oracle_kineflux_state(*gargs)","tol":1e-8,
        },
        {
            "setup":"""import numpy as np
s,k,b,a=_ks5_fixture(); e=np.full(8,.20)
cargs=(.10,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),.05)
gargs=(.10,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),.05)""",
            "call":"kineflux_state(*cargs)",
            "gold_call":"_oracle_kineflux_state(*gargs)","tol":1e-8,
        },
        {
            "setup":"""import numpy as np
s,k,b,a=_ks5_fixture(); e=np.full(8,.001)
cargs=(1.,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),.01)
gargs=(1.,e.copy(),k.copy(),b.copy(),a.copy(),s.copy(),.01)
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call":"case_raises(kineflux_state,*cargs)",
            "gold_call":"case_raises(_oracle_kineflux_state,*gargs)","tol":0.0,
        },
    ]
