"""
Assemble the trajectory-conditioned periodic community system.

Apply the fixed coupling equations in the main problem exactly. Read breadth from the first q niche entries and overlap from the following q*q entries; read the four resilience values followed by q signed exposures from the PETRA vector. Return one float64 vector containing B, D, U, reserve, pulse, climate covariance, and nu in that stated C-order packing.

Returns
-------
return one float64 vector packing competition, self-regulation, loadings, reserves, pulse decay, climate covariance, and idiosyncratic noise in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_periodic_system(base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", niche_profile: "np.ndarray", petra_signal: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    """Assemble the trajectory-conditioned periodic community arrays in the stated packed order.
 
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
 
 
def _oracle_assemble_periodic_system(base_competition: "np.ndarray", base_designs: "np.ndarray", base_loadings: "np.ndarray", base_reserves: "np.ndarray", pulse_decay_base: "np.ndarray", climate_covariance_base: "np.ndarray", idiosyncratic_noise_base: "np.ndarray", niche_profile: "np.ndarray", petra_signal: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    b = np.asarray(base_competition, dtype=np.float64)
    d = np.asarray(base_designs, dtype=np.float64)
    u = np.asarray(base_loadings, dtype=np.float64)
    reserve0 = np.asarray(base_reserves, dtype=np.float64)
    pulse0 = np.asarray(pulse_decay_base, dtype=np.float64)
    climate0 = np.asarray(climate_covariance_base, dtype=np.float64)
    idio0 = np.asarray(idiosyncratic_noise_base, dtype=np.float64)
    n = np.asarray(niche_profile, dtype=np.float64)
    p = np.asarray(petra_signal, dtype=np.float64)
    coef = np.asarray(coefficients, dtype=np.float64)
    if b.ndim != 3 or b.shape[1] != b.shape[2] or d.ndim != 2 or d.shape[1] != b.shape[1]:
        raise ValueError("unaligned competition and design arrays")
    s, q = b.shape[:2]
    plans = d.shape[0]
    if u.shape != (s, q, 2) or reserve0.shape != (q,) or pulse0.shape != (plans, q):
        raise ValueError("unaligned loading, reserve, or pulse arrays")
    if climate0.shape != (s, 2, 2) or idio0.shape != (s, q):
        raise ValueError("unaligned climate arrays")
    if n.size != q + q * q or p.size != 4 + q or coef.shape != (11,):
        raise ValueError("unaligned niche, PETRA, or coefficient arrays")
    arrays = (b, d, u, reserve0, pulse0, climate0, idio0, n, p, coef)
    if not all(np.isfinite(x).all() for x in arrays) or np.any(d <= 0.0) or np.any(reserve0 < 0.0) or np.any(pulse0 <= 0.0) or np.any(idio0 <= 0.0):
        raise ValueError("periodic-system inputs must be finite with positive scales")
    breadth = n[:q]
    overlap = n[q:].reshape(q, q)
    resistance, amplitude, recovery, net = p[:4]
    exposure = p[4:]
    tau0, tau1, chi, db, de, du, dl, dr, dp, dc, di = coef
    tau = tau0 + tau1 * np.array([net, amplitude, recovery], dtype=np.float64)
    if s != 3:
        raise ValueError("the periodic task requires three climate conditions")
    direction = exposure[:, None] - exposure[None, :]
    B = b * np.clip(1.0 + tau[:, None, None] * overlap[None, :, :] + chi * direction[None, :, :], 0.08, None)
    for matrix in B:
        np.fill_diagonal(matrix, 0.0)
    D = d * (1.0 + db * (1.0 - breadth)[None, :]) * (1.0 + de * np.abs(exposure)[None, :])
    U = u.copy()
    U[:, :, 0] += du * exposure[None, :]
    U[:, :, 1] -= 0.7 * du * exposure[None, :]
    reserve = reserve0 * (1.0 + dl * np.abs(exposure)) * (1.0 + dr * (1.0 - resistance))
    pulse = pulse0 * (1.0 + dp * (1.0 - breadth)[None, :]) * (1.0 + 0.5 * dp * np.abs(exposure)[None, :])
    climate = climate0.copy()
    climate[:, 0, 1] += dc * np.array([net, -amplitude, recovery], dtype=np.float64)
    climate[:, 1, 0] = climate[:, 0, 1]
    if np.any(np.linalg.eigvalsh(climate) <= 0.0):
        raise ValueError("adjusted climate covariance must be positive definite")
    idio = idio0 * (1.0 + di * np.abs(exposure)[None, :])
    return np.concatenate([B.ravel(), D.ravel(), U.ravel(), reserve.ravel(), pulse.ravel(), climate.ravel(), idio.ravel()]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
B0 = np.array([[[0.0,0.2,0.1],[0.15,0.0,0.18],[0.12,0.16,0.0]],[[0.0,0.22,0.12],[0.14,0.0,0.2],[0.11,0.17,0.0]],[[0.0,0.18,0.13],[0.16,0.0,0.19],[0.1,0.15,0.0]]])
D0 = np.array([[0.8,1.0,1.2],[1.1,0.9,1.0]])
U0 = np.full((3,3,2),0.04)
r0 = np.array([0.01,0.012,0.011])
p0 = np.full((2,3),0.35)
c0 = np.array([[[1.0,0.2],[0.2,1.1]],[[1.1,-0.1],[-0.1,0.9]],[[0.9,0.15],[0.15,1.2]]])
i0 = np.full((3,3),0.001)
n = np.array([0.8,0.85,0.9,1.0,0.2,0.1,0.2,1.0,0.3,0.1,0.3,1.0])
sig = np.array([0.7,0.12,0.08,0.03,0.01,-0.02,0.015])
coef = np.array([1.1,1.6,0.8,0.4,0.6,0.7,0.5,0.3,0.2,0.4,0.25])"""
    return [
        {"setup": common, "call": "assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef)", "gold_call": "_oracle_assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef)", "tol": 1e-10},
        {"setup": common + "\nn[0] = 0.999", "call": "assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef)", "gold_call": "_oracle_assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef)", "tol": 1e-10},
        {"setup": common + "\nsig[4:] = 0.0", "call": "assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef)", "gold_call": "_oracle_assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef)", "tol": 1e-10},
        {"setup": common + "\nD0[0,0] = 0.0\ndef caught(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    except Exception:\n        return np.array([-1.0])\n    return np.array([0.0])", "call": "caught(lambda: assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef))", "gold_call": "caught(lambda: _oracle_assemble_periodic_system(B0,D0,U0,r0,p0,c0,i0,n,sig,coef))", "tol": 0.0},
    ]
