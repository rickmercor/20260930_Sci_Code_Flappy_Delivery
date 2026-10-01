"""
Compute the converged steady state of the cell at the applied voltage V under the uniform free-carrier generation rate G = s G_ex, where s is the generation scale (s = 0 is the dark state) and par the 14-entry parameter vector of the first step. The steady state solves Poisson's equation eps eps0 psi'' = -q (p - n) with the continuity equations dJ_n/dx = q (R - G) and dJ_p/dx = -q (R - G), R = gamma_L (n p - n_i^2), J_n = -q mu_n n psi' + q D_n n' and J_p = -q mu_p p psi' - q D_p p' with D = mu kT/q, on the uniform grid x_i = i d/N (spacing h = d/N): the three-point second difference for Poisson at every interior node, R and G evaluated at the interior nodes (the contact nodes hold their fixed densities and generate nothing), the currents on every interval by the Scharfetter-Gummel fluxes J_n = (q D_n/h) [n_{i+1} B(delta_i) - n_i B(-delta_i)] and J_p = (q D_p/h) [p_i B(delta_i) - p_{i+1} B(-delta_i)] with delta_i = (psi_{i+1} - psi_i) q/kT and B(x) = x/(exp(x) - 1), and the divergence at a node equal to the difference of its two interval fluxes divided by h. Solve the discrete system to machine precision (every residual, scaled as in the reference: Poisson in units of kT/q, each continuity equation multiplied by h^2 over q D times the local density, below 1e-12 or at the roundoff floor) by damped Newton iteration in the unknowns psi q/kT, ln n and ln p, started from the dark zero-bias Poisson-Boltzmann equilibrium with the generation switched on at zero bias and continued in the bias in steps of at most 0.05 V. If start = (state, voltage0) is given, a converged (N + 1, 3) state of the same cell at the bias voltage0 (any generation), start the continuation from it instead of from the dark equilibrium, in bias steps of at most 0.05 V. Return the converged state at the N + 1 nodes as the (N + 1, 3) array with columns psi q/kT, ln n and ln p (densities in 1/m^3). Raise ValueError if par does not have 14 entries, if N is not an even integer of at least 4, if the generation scale is negative, if V is not below V_bi + 0.5 V, or if start is not an (N + 1, 3) state satisfying the contact potentials at its bias.

The exact steady state of the drift-diffusion model is the reference against which every closed-form or procedural statement about charge collection in a thin-film cell must be checked; the Scharfetter-Gummel discretisation keeps the discrete currents exact for exponentially varying densities so that the converged discrete state is unique and reproducible to machine precision.

Returns
-------
An (N + 1, 3) float64 array [psi q/kT, ln n, ln p] at the nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def steady_state(voltage: float, generation_scale: float, intervals: int, par: "np.ndarray", start: "tuple[np.ndarray, float] | None" = None) -> "np.ndarray":
    """Compute the converged steady state of the cell at the applied voltage V under the
        uniform free-carrier generation rate G = s G_ex, where s is the generation scale (s = 0
        is the dark state) and par the 14-entry parameter vector of the first step. An (N + 1,
        3) float64 array [psi q/kT, ln n, ln p] at the nodes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dbernoulli(x):
    """dB/dx with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-5
    out[s] = -0.5 + x[s] / 6.0
    xb = x[~s]; ex = np.expm1(xb); out[~s] = (ex - xb * (ex + 1.0)) / ex ** 2
    return out

def _block_tridiagonal_solve(A, B, C, r):
    """Solve the block tridiagonal system with diagonal blocks A[i] (k x k), super-diagonal B[i] (row i to i+1) and
    sub-diagonal C[i] (row i to i-1) for the right-hand side r (M, k), through a LAPACK banded solve."""
    M, k, _ = A.shape; n = k * M; u = 2 * k - 1
    ab = np.zeros((2 * u + 1, n)); i = np.arange(M)
    for rr in range(k):
        for cc in range(k):
            rows = k * i + rr; cols = k * i + cc; ab[u + rows - cols, cols] = A[:, rr, cc]
            rows = k * i[:-1] + rr; cols = k * (i[:-1] + 1) + cc; ab[u + rows - cols, cols] = B[:-1, rr, cc]
            rows = k * i[1:] + rr; cols = k * (i[1:] - 1) + cc; ab[u + rows - cols, cols] = C[1:, rr, cc]
    return solve_banded((u, u), ab, r.reshape(-1)).reshape(M, k)

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT = prm["VT"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [prm["Vbi"] - V]])
    n = np.concatenate([[prm["n_an"]], np.exp(state[:, 1]), [prm["n_cat"]]])
    p = np.concatenate([[prm["p_an"]], np.exp(state[:, 2]), [prm["p_cat"]]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h, G):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each interior node; G in 1/(m^3 s)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    U = prm["gamma"] * (n * p - prm["ni2"]) - G
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q() / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h, G):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p]."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q() * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    gn = h ** 2 * G / prm["Dn"]; gp = h ** 2 * G / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3))
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"]) + gn
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"]) - gp
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    return A, B, C

def _dd_equilibrium(prm, N):
    """Dark zero-bias state: Poisson-Boltzmann solution (damped Newton) in the state layout."""
    VT = prm["VT"]; d = prm["d"]; h = d / N; M = N - 1; c = _Q() * h ** 2 / (prm["ee"] * VT)
    psi = np.linspace(0.0, prm["Vbi"], N + 1)
    def _res(ps):
        n = prm["n_an"] * np.exp(ps / VT); p = prm["p_an"] * np.exp(-ps / VT)
        return (ps[2:] - 2.0 * ps[1:-1] + ps[:-2]) / VT + c * (p[1:-1] - n[1:-1]), n, p
    for it in range(300):
        F, n, p = _res(psi); nrm = np.max(np.abs(F))
        if nrm < 1e-13: break
        A = ((-2.0 - c * (p[1:-1] + n[1:-1])) / VT).reshape(M, 1, 1)
        Bq = np.full((M, 1, 1), 1.0 / VT); Cq = np.full((M, 1, 1), 1.0 / VT)
        dpsi = _block_tridiagonal_solve(A, Bq, Cq, -F.reshape(M, 1))[:, 0]
        lam = 1.0
        for _ in range(60):
            pt = psi.copy(); pt[1:-1] += lam * dpsi
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _res(pt)[0]
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        psi = pt
    n = prm["n_an"] * np.exp(psi / VT); p = prm["p_an"] * np.exp(-psi / VT)
    return np.stack([psi[1:-1] / VT, np.log(n[1:-1]), np.log(p[1:-1])], axis=1)

def _dd_newton(state, prm, V, h, G, tol=1e-12, maxit=100):
    """Damped Newton on the scaled residual; stops below tol or when a full line search no longer reduces it."""
    U = state.copy()
    for it in range(maxit):
        F = _dd_residual(U, prm, V, h, G); nrm = np.max(np.abs(F))
        if nrm < tol: return U, nrm
        A, B, C = _dd_jacobian(U, prm, V, h, G)
        dU = _block_tridiagonal_solve(A, B, C, -F); lam = 1.0; ok = False
        for _ in range(40):
            Ut = U + lam * dU
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _dd_residual(Ut, prm, V, h, G)
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: ok = True; break
            lam *= 0.5
        if not ok: return U, nrm
        U = Ut
    return U, nrm

def _dd_solve(prm, N, V, G, state0=None, V0=0.0, dvmax=0.05):
    """Steady state at bias V under the generation rate G: from the dark equilibrium with the generation ramped in at
    zero bias and continuation in steps of at most dvmax, or from a given state at bias V0."""
    h = prm["d"] / N
    if state0 is None:
        U = _dd_equilibrium(prm, N); V0 = 0.0
        if G > 0.0:
            for f in (1e-3, 1e-2, 1e-1, 0.3, 1.0):
                U, nrm = _dd_newton(U, prm, 0.0, h, G * f)
                if not nrm < 1e-10:
                    raise ValueError("generation ramp did not converge")
    else:
        U = state0.copy()
    nsteps = max(1, int(np.ceil(abs(V - V0) / dvmax - 1e-12)))
    for k in range(1, nsteps + 1):
        Vk = V0 + (V - V0) * k / nsteps
        U, nrm = _dd_newton(U, prm, Vk, h, G)
        if not nrm < 1e-10:
            raise ValueError("drift-diffusion solve did not converge at V = %g" % Vk)
    return U

def _oracle_steady_state(voltage: float, generation_scale: float, intervals: int, par: "np.ndarray", start: "tuple[np.ndarray, float] | None" = None) -> "np.ndarray":
    """Converged steady state at bias V with the generation rate scaled by s (0 = dark), as [psi/VT, ln n, ln p];
    optionally continued from a converged state at another bias, start = (state, voltage0)."""
    par = np.asarray(par, dtype=float).ravel(); N = int(intervals); V = float(voltage); s = float(generation_scale)
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    if s < 0.0:
        raise ValueError("the generation scale must be non-negative")
    prm = _dd_params(par)
    if not (V < prm["Vbi"] + 0.5):
        raise ValueError("bias must lie below the built-in voltage plus 0.5 V")
    G = prm["G"] * s
    if start is None:
        U = _dd_solve(prm, N, V, G)
    else:
        st = np.asarray(start[0], dtype=float); V0 = float(start[1])
        if st.shape != (N + 1, 3) or abs(st[0, 0]) > 1e-9 or abs(st[-1, 0] * prm["VT"] - (prm["Vbi"] - V0)) > 1e-9:
            raise ValueError("start must be an (N + 1, 3) state satisfying the contact potentials at its bias")
        U = _dd_solve(prm, N, V, G, state0=st[1:-1].copy(), V0=V0)
    psi, n, p = _dd_unpack(U, prm, V)
    return np.stack([psi / prm["VT"], np.log(n), np.log(p)], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltage = 0.0\ngeneration_scale = 1.0\nintervals = 200\n',
         'call': 'steady_state(voltage, generation_scale, intervals, par)',
         'gold_call': '_oracle_steady_state(voltage, generation_scale, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltage = -1.0\ngeneration_scale = 1.0\nintervals = 400\n',
         'call': 'steady_state(voltage, generation_scale, intervals, par)',
         'gold_call': '_oracle_steady_state(voltage, generation_scale, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltage = 0.6\ngeneration_scale = 0.0\nintervals = 200\n',
         'call': 'steady_state(voltage, generation_scale, intervals, par)',
         'gold_call': '_oracle_steady_state(voltage, generation_scale, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)\nvoltage = 0.3\ngeneration_scale = 0.5\nintervals = 200\n',
         'call': 'steady_state(voltage, generation_scale, intervals, par)',
         'gold_call': '_oracle_steady_state(voltage, generation_scale, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltage = -0.4\ngeneration_scale = 1.0\nintervals = 200\nstart = (np.array([[0.0, 15.383226285236546, 50.19678064988678], [0.1823410869661751, 44.72351355398066, 50.15225509239725], [0.36438165144118634, 45.41666754536563, 50.11309535616184], [0.5461346099428938, 45.822136045558544, 50.078723123884586], [0.7276110943966431, 46.1098167297321, 50.04858328491456], [0.9088207560513959, 46.33295308276157, 50.02215458644672], [1.0897720171787733, 46.51526086574136, 49.99895652074242], [1.270472279643485, 46.66939063999577, 49.978552972813844], [1.4509280978227341, 46.802893604118935, 49.96055327067258], [1.6311453220344714, 46.920640426608266, 49.94461129019935], [1.811129217554342, 47.02595678349219, 49.93042320787801], [1.9908845634145114, 47.12121477580694, 49.91772440048949], [2.170415734448495, 47.208165912950896, 49.90628588592753], [2.349726769446009, 47.28814034961242, 49.89591059918347], [2.5288214277872014, 47.362172073983515, 49.88642971056377], [2.707703236517618, 47.431080799329486, 49.87769912249465], [2.886375529488347, 47.495527371170326, 49.86959622676542], [3.064841479907424, 47.556052346881344, 49.86201696397667], [3.2431041274184014, 47.61310353092164, 49.85487319881093], [3.421166400631565, 47.667056055219085, 49.848090405962225], [3.5990311358756433, 47.71822730283883, 49.84160564982674], [3.776701092807278, 47.766888186776924, 49.835365834421765], [3.9541789674073864, 47.8132718026091, 49.829326196922636], [4.1314674028038105, 47.85758015626932, 49.82344901751047], [4.308568998285355, 47.89998945903096, 49.81770251904331], [4.485486316810556, 47.940654340979115, 49.81205993178136], [4.6622218912633855, 47.979711237709125, 49.80649870058644], [4.838778229665541, 48.0172811376201, 49.80099981438975], [5.015157819519757, 48.05347182942737, 49.795547240097115], [5.191363131429152, 48.0883797551903, 49.79012744536386], [5.367396622113396, 48.122091549143924, 49.78472899676045], [5.54326073692212, 48.1546853241804, 49.779342221735824], [5.718957911929253, 48.18623175407228, 49.773958924459144], [5.894490575677924, 48.2167949891642, 49.76857214708873], [6.06986115063395, 48.24643343537119, 49.763175969292476], [6.245072054396236, 48.275200420264504, 49.75776533994291], [6.420125700704381, 48.303144765334004, 49.75233593585312], [6.5950245002770185, 48.33031127985371, 49.74688404322324], [6.769770861508923, 48.35674118889629, 49.74140645815081], [6.944367191050171, 48.3824725057604, 49.73590040313729], [7.118815894286836, 48.40754035725537, 49.730363457012494], [7.293119375739458, 48.43197726882774, 49.72479349611169], [7.4672800393928, 48.455813415335854, 49.71918864488806], [7.641300288968225, 48.479076842322286, 49.713547234435545], [7.815182528148111, 48.50179366185327, 49.70786776764341], [7.988929160760173, 48.52398822635411, 49.70214888991007], [8.162542590928284, 48.54568328334222, 49.69638936451723], [8.336025223195275, 48.56690011352228, 49.690588051910545], [8.509379462622297, 48.587658654345134, 49.68474389225511], [8.682607714868597, 48.607977610828485, 49.67885589073598], [8.855712386254888, 48.62787455518337, 49.67292310515983], [9.028695883813011, 48.64736601657621, 49.666944635485315], [9.201560615324095, 48.666467562175704, 49.66091961497003], [9.374308989347133, 48.68519387048046, 49.6548472026721], [9.546943415239493, 48.70355879779293, 49.64872657708693], [9.719466303170698, 48.72157543859432, 49.642556930734564], [9.891880064130568, 48.739256180479565, 49.63633746554319], [10.06418710993262, 48.756612754230225, 49.630067388898766], [10.236389853213531, 48.77365627953251, 49.6237459102518], [10.408490707429253, 48.79039730678708, 49.617372238189525], [10.58049208684839, 48.80684585540478, 49.61094557789655], [10.752396406543188, 48.823011448936796, 49.60446512893911], [10.924206082378607, 48.83890314734792, 49.597930083318374], [11.095923530999764, 48.854529576707485, 49.59133962374699], [11.267551169817954, 48.869898956541626, 49.584692922109994], [11.439091416995565, 48.88501912506474, 49.577989138077484], [11.610546691429995, 48.899897562484185, 49.57122741784149], [11.78191941273676, 48.91454141255221, 49.564406892953514], [11.953212001231956, 48.92895750252087, 49.557526679243054], [12.12442687791407, 48.94315236163986, 49.55058587580023], [12.29556646444541, 48.957132238323, 49.54358356400811], [12.466633183133032, 48.97090311609674, 49.53651880661244], [12.637629456909412, 48.98447072843297, 49.529390646818364], [12.808557709312767, 48.99784057255827, 49.52219810740479], [12.97942036446719, 49.011017922323404, 49.5149401898488], [13.150219847062527, 49.0240078402086, 49.50761587345293], [13.320958582334141, 49.03681518853336, 49.50022411446938], [13.491638996042491, 49.049444639933206, 49.49276384521581], [13.662263514452622, 49.06190068716005, 49.48523397317768], [13.83283456431352, 49.07418765225803, 49.477633380092925], [14.003354572837415, 49.08630969516183, 49.469960921014845], [14.17382596767899, 49.098270821760565, 49.46221542334941], [14.344251176914543, 49.110074891466624, 49.45439568586348], [14.514632629021092, 49.12172562432551, 49.44650047766061], [14.68497275285544, 49.13322660769948, 49.43852853712094], [14.85527397763319, 49.14458130255555, 49.430478570802286], [15.025538732907764, 49.155793049385366, 49.422349252298886], [15.195769448549353, 49.16686507378262, 49.41413922105485], [15.365968554723874, 49.17780049170153, 49.40584708112895], [15.536138481871914, 49.1886023144179, 49.397471399907474], [15.706281660687623, 49.199273453212804, 49.389010706761745], [15.87640052209763, 49.20981672379728, 49.380463491646694], [16.046497497239947, 49.220234850494975, 49.37182820363695], [16.21657501744284, 49.23053047019853, 49.36310324939644], [16.386635514203693, 49.240706136114206, 49.35428699157754], [16.556681419167905, 49.25076432130816, 49.34537774714555], [16.726715164107745, 49.26070742206697, 49.33637378562384], [16.8967391809012, 49.270537761083894, 49.327273327255085], [17.06675590151085, 49.28025759048161, 49.31807454107328], [17.236767757962717, 49.28986909468149, 49.30877554288139], [17.406777182325122, 49.29937439312864, 49.29937439312864], [17.576786606687524, 49.30877554288139, 49.28986909468149], [17.746798463139392, 49.31807454107328, 49.28025759048161], [17.91681518374904, 49.327273327255085, 49.270537761083894], [18.086839200542496, 49.33637378562384, 49.26070742206697], [18.256872945482332, 49.34537774714555, 49.25076432130816], [18.426918850446548, 49.35428699157754, 49.240706136114206], [18.5969793472074, 49.36310324939644, 49.23053047019853], [18.767056867410293, 49.37182820363695, 49.220234850494975], [18.93715384255261, 49.380463491646694, 49.20981672379728], [19.10727270396262, 49.389010706761745, 49.199273453212804], [19.277415882778328, 49.397471399907474, 49.1886023144179], [19.447585809926363, 49.40584708112895, 49.17780049170153], [19.617784916100884, 49.414139221054846, 49.16686507378262], [19.788015631742475, 49.422349252298886, 49.155793049385366], [19.958280387017048, 49.430478570802286, 49.14458130255555], [20.128581611794804, 49.43852853712094, 49.13322660769948], [20.298921735629147, 49.44650047766061, 49.12172562432551], [20.4693031877357, 49.45439568586348, 49.110074891466624], [20.63972839697125, 49.462215423349406, 49.098270821760565], [20.810199791812828, 49.469960921014845, 49.08630969516183], [20.98071980033672, 49.477633380092925, 49.07418765225803], [21.151290850197615, 49.48523397317768, 49.06190068716005], [21.32191536860775, 49.49276384521581, 49.049444639933206], [21.492595782316098, 49.50022411446938, 49.03681518853336], [21.66333451758771, 49.50761587345293, 49.0240078402086], [21.83413400018305, 49.5149401898488, 49.011017922323404], [22.004996655337465, 49.52219810740479, 48.99784057255827], [22.17592490774083, 49.529390646818364, 48.984470728432974], [22.346921181517207, 49.53651880661244, 48.97090311609674], [22.517987900204833, 49.54358356400811, 48.957132238323], [22.689127486736172, 49.55058587580023, 48.94315236163986], [22.860342363418287, 49.557526679243054, 48.92895750252087], [23.031634951913478, 49.564406892953514, 48.91454141255221], [23.20300767322025, 49.57122741784149, 48.899897562484185], [23.374462947654674, 49.577989138077484, 48.88501912506474], [23.546003194832284, 49.584692922109994, 48.869898956541626], [23.717630833650475, 49.59133962374699, 48.854529576707485], [23.88934828227163, 49.597930083318374, 48.83890314734792], [24.061157958107053, 49.60446512893911, 48.823011448936796], [24.233062277801853, 49.61094557789655, 48.80684585540478], [24.405063657220985, 49.617372238189525, 48.79039730678708], [24.577164511436713, 49.6237459102518, 48.77365627953251], [24.74936725471762, 49.630067388898766, 48.756612754230225], [24.92167430051967, 49.63633746554319, 48.739256180479565], [25.094088061479543, 49.642556930734564, 48.721575438594314], [25.266610949410747, 49.64872657708693, 48.70355879779293], [25.439245375303106, 49.6548472026721, 48.68519387048046], [25.611993749326142, 49.66091961497003, 48.666467562175704], [25.78485848083723, 49.666944635485315, 48.64736601657621], [25.95784197839535, 49.67292310515983, 48.62787455518337], [26.13094664978164, 49.67885589073598, 48.607977610828485], [26.304174902027945, 49.68474389225511, 48.587658654345134], [26.477529141454962, 49.690588051910545, 48.56690011352228], [26.651011773721955, 49.69638936451723, 48.545683283342214], [26.824625203890072, 49.70214888991007, 48.52398822635411], [26.998371836502134, 49.70786776764341, 48.50179366185327], [27.172254075682016, 49.713547234435545, 48.479076842322286], [27.346274325257443, 49.719188644888064, 48.455813415335854], [27.520434988910782, 49.72479349611169, 48.43197726882774], [27.694738470363404, 49.730363457012494, 48.40754035725537], [27.869187173600068, 49.7359004031373, 48.3824725057604], [28.043783503141317, 49.74140645815082, 48.35674118889629], [28.218529864373224, 49.74688404322324, 48.33031127985371], [28.39342866394586, 49.75233593585312, 48.303144765334004], [28.568482310254005, 49.75776533994291, 48.275200420264504], [28.74369321401629, 49.763175969292476, 48.24643343537119], [28.919063788972316, 49.76857214708873, 48.2167949891642], [29.094596452720985, 49.773958924459144, 48.18623175407228], [29.27029362772812, 49.779342221735824, 48.1546853241804], [29.446157742536844, 49.78472899676045, 48.122091549143924], [29.622191233221088, 49.79012744536386, 48.0883797551903], [29.798396545130483, 49.795547240097115, 48.05347182942737], [29.9747761349847, 49.80099981438975, 48.01728113762011], [30.151332473386855, 49.80649870058644, 47.979711237709125], [30.328068047839682, 49.81205993178136, 47.940654340979115], [30.504985366364885, 49.81770251904331, 47.89998945903096], [30.68208696184643, 49.82344901751048, 47.85758015626932], [30.859375397242857, 49.829326196922636, 47.8132718026091], [31.03685327184296, 49.835365834421765, 47.766888186776924], [31.214523228774592, 49.84160564982674, 47.71822730283883], [31.392387964018678, 49.848090405962225, 47.667056055219085], [31.57045023723184, 49.85487319881093, 47.61310353092164], [31.748712884742815, 49.86201696397667, 47.556052346881344], [31.927178835161897, 49.86959622676542, 47.495527371170326], [32.10585112813262, 49.87769912249465, 47.431080799329486], [32.28473293686304, 49.88642971056377, 47.362172073983515], [32.46382759520423, 49.89591059918347, 47.28814034961242], [32.64313863020175, 49.90628588592753, 47.208165912950896], [32.822669801235726, 49.91772440048949, 47.12121477580694], [33.0024251470959, 49.93042320787801, 47.02595678349219], [33.18240904261577, 49.94461129019935, 46.920640426608266], [33.362626266827505, 49.96055327067258, 46.802893604118935], [33.54308208500676, 49.978552972813844, 46.66939063999577], [33.72378234747147, 49.99895652074242, 46.51526086574136], [33.904733608598846, 50.02215458644672, 46.33295308276157], [34.085943270253594, 50.04858328491456, 46.1098167297321], [34.26741975470735, 50.078723123884586, 45.822136045558544], [34.44917271320905, 50.11309535616184, 45.41666754536563], [34.63121327768406, 50.15225509239725, 44.723513553980666], [34.81355436465024, 50.19678064988678, 15.383226285236546]], dtype=float), 0.0)\n',
         'call': 'steady_state(voltage, generation_scale, intervals, par, start)',
         'gold_call': '_oracle_steady_state(voltage, generation_scale, intervals, par, start)'},
    ]
