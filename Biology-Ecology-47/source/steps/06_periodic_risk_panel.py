"""
Calculate the periodic risk panel for one management plan.

Preserve the ordered dry-season, contraction, and wet-season map, carry correlated innovations through that order, and solve the stationary discrete Lyapunov equation. Return one row per climate condition with columns [minimum equilibrium density, Floquet radius, normalized covariance trace, security radius, zero-based limiting species, usable flag]. If a condition is unusable, retain its computed minimum density and Floquet radius but return 1e12, -1e12, -1, and 0 in the last four columns.

Returns
-------
return a float64 matrix of shape (conditions, 6): minimum equilibrium, Floquet radius, normalized covariance trace, security radius, limiting species, and usable flag.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def periodic_risk_panel(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", effort: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    """Calculate equilibrium, Floquet, covariance, and security diagnostics for one plan.
 
    Returns
    -------
    The numerical object stated in the step contract.
 
    Raises
    ------
    ValueError
        If an input violates the stated contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov
 
 
def _oracle_periodic_risk_panel(competition: "np.ndarray", slope: "np.ndarray", loadings: "np.ndarray", reserves: "np.ndarray", pulse_decay: "np.ndarray", climate_covariance: "np.ndarray", idiosyncratic_noise: "np.ndarray", durations: "np.ndarray", exposure: "np.ndarray", effort: float, contraction_gain: float, stability_limit: float, chi_radius: float) -> "np.ndarray":
    B = np.asarray(competition, dtype=np.float64)
    d = np.asarray(slope, dtype=np.float64)
    U = np.asarray(loadings, dtype=np.float64)
    reserve = np.asarray(reserves, dtype=np.float64)
    pulse = np.asarray(pulse_decay, dtype=np.float64)
    climate = np.asarray(climate_covariance, dtype=np.float64)
    idio = np.asarray(idiosyncratic_noise, dtype=np.float64)
    duration = np.asarray(durations, dtype=np.float64)
    e = np.asarray(exposure, dtype=np.float64)
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError("competition must contain square condition matrices")
    s, q = B.shape[:2]
    if d.shape != (q,) or U.shape != (s, q, 2) or reserve.shape != (q,) or pulse.shape != (q,):
        raise ValueError("unaligned plan arrays")
    if climate.shape != (s, 2, 2) or idio.shape != (s, q) or duration.shape != (s, 2) or e.shape != (q,):
        raise ValueError("unaligned climate arrays")
    arrays = (B, d, U, reserve, pulse, climate, idio, duration, e)
    if not all(np.isfinite(x).all() for x in arrays) or np.any(d <= 0.0) or np.any(pulse <= 0.0) or np.any(idio <= 0.0) or np.any(duration <= 0.0):
        raise ValueError("periodic-risk inputs must be finite with positive scales")
    if not np.isfinite(effort) or effort <= 0.0 or contraction_gain < 0.0 or not (0.0 < stability_limit < 1.0) or chi_radius <= 0.0:
        raise ValueError("invalid periodic-risk scalars")
    rows = []
    for c in range(s):
        M = float(effort) * np.diag(d) + B[c]
        x = np.linalg.solve(M, np.ones(q, dtype=np.float64))
        scale0 = 0.78 + 0.55 * np.abs(e) + 0.08 * c
        scale1 = 1.22 - 0.35 * np.abs(e) - 0.05 * c
        J0 = -np.diag(x * scale0) @ M
        J1 = -np.diag(x * scale1) @ M
        E0 = expm(J0 * duration[c, 0])
        E1 = expm(J1 * duration[c, 1])
        P = np.diag(np.exp(-(pulse + contraction_gain * float(effort))))
        F = E1 @ P @ E0
        rho = float(np.max(np.abs(np.linalg.eigvals(F))))
        G0 = np.diag(x) @ U[c]
        rotation = np.array([[0.82 + 0.03 * c, -0.21], [0.17, 0.91 - 0.02 * c]], dtype=np.float64)
        G1 = np.diag(x) @ (U[c] @ rotation)
        Q0 = G0 @ climate[c] @ G0.T + np.diag(idio[c] * x * x)
        Q1 = G1 @ climate[c] @ G1.T + np.diag((1.15 - 0.08 * c) * idio[c] * x * x)
        Qcycle = E1 @ P @ Q0 @ P.T @ E1.T + Q1
        usable = bool(np.all(x > reserve) and rho < stability_limit)
        if usable:
            C = solve_discrete_lyapunov(F, Qcycle)
            C = (C + C.T) / 2.0
            diagonal = np.diag(C)
            if np.any(diagonal <= 0.0) or not np.isfinite(C).all():
                raise ValueError("stationary covariance is not positive on the diagonal")
            standardized = (x - reserve) / (float(chi_radius) * np.sqrt(diagonal))
            limiting = int(np.argmin(standardized))
            trace_ratio = float(np.trace(C) / np.dot(x, x))
            security = float(standardized[limiting])
        else:
            trace_ratio = 1.0e12
            security = -1.0e12
            limiting = -1
        rows.append([float(np.min(x)), rho, trace_ratio, security, float(limiting), float(usable)])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = 'import numpy as np\nB = np.array([[[0.0,0.08,0.05],[0.06,0.0,0.07],[0.05,0.06,0.0]],[[0.0,0.09,0.04],[0.05,0.0,0.08],[0.06,0.05,0.0]],[[0.0,0.07,0.06],[0.07,0.0,0.05],[0.04,0.08,0.0]]])\nd = np.array([0.9,1.1,1.0])\nU = np.array([[[0.05,0.03],[-0.04,0.05],[0.03,-0.02]],[[0.04,-0.03],[0.02,0.05],[-0.05,0.03]],[[0.03,0.04],[-0.03,0.02],[0.04,-0.05]]])\nreserve = np.array([0.015,0.016,0.014])\npulse = np.array([0.35,0.4,0.32])\nclimate = np.array([[[1.0,0.25],[0.25,1.1]],[[0.9,-0.2],[-0.2,1.2]],[[1.2,0.3],[0.3,0.95]]])\nidio = np.full((3,3),0.001)\nduration = np.array([[0.45,0.55],[0.5,0.5],[0.6,0.4]])\nexposure = np.array([0.01,-0.02,0.015])'
    return [
        {"setup": common, "call": "periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,1.4,0.04,0.92,2.447746830680816)", "gold_call": "_oracle_periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,1.4,0.04,0.92,2.447746830680816)", "tol": 1e-9},
        {"setup": common, "call": "periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,0.45,0.04,0.92,2.447746830680816)", "gold_call": "_oracle_periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,0.45,0.04,0.92,2.447746830680816)", "tol": 1e-9},
        {"setup": common + "\nclimate[:,0,1]=0.0; climate[:,1,0]=0.0", "call": "periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,1.4,0.04,0.92,2.447746830680816)", "gold_call": "_oracle_periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,1.4,0.04,0.92,2.447746830680816)", "tol": 1e-9},
        {"setup": common + "\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,-1.0,0.04,0.92,2.447746830680816))", "gold_call": "caught(lambda: _oracle_periodic_risk_panel(B,d,U,reserve,pulse,climate,idio,duration,exposure,-1.0,0.04,0.92,2.447746830680816))", "tol": 0.0},
    ]
