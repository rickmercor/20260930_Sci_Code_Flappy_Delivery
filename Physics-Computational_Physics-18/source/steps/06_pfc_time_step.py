"""
Advance the whole system by ****one**** step of the fully discrete Fourier-IEQ-ZEC scheme. The state is the three most recent layers, each holding the nine fields $[\\phi_1,\\ \\phi_2,\\ M_x,\\ M_y,\\ \\mu_1,\\ \\mu_2,\\ \\mu_{3x},\\ \\mu_{3y},\\ U]$ in that order, so `state[0]` is layer $n$, `state[1]` is layer $n-1$ and `state[2]` is layer $n-2$; the nonlocal scalar $Q^n$ is passed separately. Build the second-order extrapolations, form the three coefficient functions at the extrapolated state, assemble the six explicit right-hand sides, solve the four density subsystems of step 5 and the two magnetization subsystems, evaluate the two nonlocal scalars, close the step for the nonlocal scalar and recombine. Return the ten planes $[\\phi_1,\\ \\phi_2,\\ M_x,\\ M_y,\\ \\mu_1,\\ \\mu_2,\\ \\mu_{3x},\\ \\mu_{3y},\\ U,\\ Q^{n+1}]$ of the new layer, the last plane being constant and equal to the new nonlocal scalar, so that the shape is `(10, Nx, Ny)`.

Every unknown of the step is written as a part independent of the nonlocal scalar plus that scalar times a second part, and the scheme separates accordingly into a subsystem driven by the explicit data and a subsystem driven by the quadratized term. The two density subsystems of each pair are coupled only through the interspecies operator, so summing and differencing them decouples them completely, which is what step 5 solves; the magnetization subsystems are already decoupled and are ordinary constant-coefficient elliptic solves. Two structural points govern the assembly, and both are places where a plausible reconstruction differs from the source. First, the scheme is posed in the Crank-Nicolson ****average**** of every unknown while the splitting is a splitting of the ****new layer****, so the conversion between the two leaves old-layer quantities inside the explicit right-hand sides, with coefficients that the conversion itself fixes - which quantities, and with which coefficients, is exactly what the problem statement tells you to take from the source. Second, the two nonlocal scalars are assembled from quantities the step has already computed, the first from the parts independent of the scalar and the second from the parts multiplying it, and their ratio closes the step; the second is non-positive by the source's own sum-of-squares identity, so the closing equation never degenerates.

****--- Formulas ---****

There is no formula sheet for this step. The six explicit right-hand sides, the four density solves and the two magnetization solves, the two explicit updates of the auxiliary variable, the two scalars whose ratio gives the nonlocal scalar at the half step, the relation between that half-step value and the new one, and the recombination of the two parts of every unknown are the source's fully discrete scheme, and the problem statement lists each of them as something to fetch from it.

Fixed here by the task rather than by the paper, and graded: the field extrapolation is $\\psi^{*}=\\tfrac32\\psi^{n}-\\tfrac12\\psi^{n-1}$ and the derivative extrapolation to the half step is $\\psi^{*}_t=(2\\psi^{n}-3\\psi^{n-1}+\\psi^{n-2})/\\delta t$; the same field extrapolation is applied to the stored auxiliary variable; the Crank-Nicolson average is $\\tfrac12(\\psi^{n+1}+\\psi^{n})$; the coefficient functions of step 4 are evaluated at the extrapolated state; the density subsystems go through step 5 with the signs of that step's convention; the mobilities are $\\mathcal M_\\phi$ for the densities and $\\mathcal M_m$ for the magnetization, and the stabilization constants $S_\\phi$ and $S_m$ likewise; every inner product is the grid sum times $h_xh_y$; and the ten returned planes are $[\\phi_1,\\ \\phi_2,\\ M_x,\\ M_y,\\ \\mu_1,\\ \\mu_2,\\ \\mu_{3x},\\ \\mu_{3y},\\ U,\\ Q^{n+1}]$, the last constant across the grid.

Returns
-------
`np.ndarray` of shape `(10,) + grid`, real and finite. The spatial means of the first two planes equal those of the corresponding planes of `state[0]` to round-off, whatever the time step; as $\\delta t\\to0$ the step reduces to the forward increment of the reformulated system and the last plane tends to $Q^{n}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pfc_time_step(state, q_n, cell, dt, params=None, H=None):
    """state: array of shape (3, 9, Nx, Ny); state[k] is layer n-k and holds
       [phi1, phi2, M_x, M_y, mu1, mu2, mu3_x, mu3_y, U] in that order.
    q_n: the nonlocal scalar at layer n, a float.
    cell: domain edge lengths. dt: the time step.
    params: as in pfc_free_energy. H: applied field, shape (2, Nx, Ny).
    Return the real array of shape (10, Nx, Ny) holding the nine fields of
    the new layer followed by a constant plane equal to Q^{n+1}.
    Raise ValueError if state is not a finite array of shape
    (3, 9, Nx, Ny), if q_n is not a finite scalar, if dt is not a finite
    positive scalar, if H is given with a shape other than (2, Nx, Ny),
    or if the equation for the nonlocal scalar is singular."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pfc_time_step(state, q_n, cell, dt, params=None, H=None):
    p = _check_params(params)
    S = np.asarray(state, float)
    if S.ndim != 4 or S.shape[0] != 3 or S.shape[1] != 9:
        raise ValueError("state must have shape (3, 9, Nx, Ny)")
    if not np.all(np.isfinite(S)):
        raise ValueError("state must be finite")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive scalar")
    q_n = float(q_n)
    if not np.isfinite(q_n):
        raise ValueError("q_n must be a finite scalar")
    grid = S.shape[2:]
    L = _check_cell(cell, grid)
    H = np.zeros((2,) + grid) if H is None else np.asarray(H, float)
    if H.shape != (2,) + grid:
        raise ValueError("H must have shape (2,) + grid")

    Mp, Mm = p["M_phi"], p["M_m"]
    Sp, Sm, a1, a2, a12, w0 = (p["S_phi"], p["S_m"], p["a1"], p["a2"],
                               p["a12"], p["omega0"])

    n0, n1, n2 = S[0], S[1], S[2]          # layers n, n-1, n-2
    p1n, p2n = n0[0], n0[1]
    Mn = n0[2:4]
    mu1n, mu2n = n0[4], n0[5]
    mu3n = n0[6:8]
    Un = n0[8]

    # second-order extrapolations
    p1s = 1.5 * p1n - 0.5 * n1[0]
    p2s = 1.5 * p2n - 0.5 * n1[1]
    Ms = 1.5 * Mn - 0.5 * n1[2:4]
    Us = 1.5 * Un - 0.5 * n1[8]
    p1ts = (2.0 * p1n - 3.0 * n1[0] + n2[0]) / dt
    p2ts = (2.0 * p2n - 3.0 * n1[1] + n2[1]) / dt
    Mts = (2.0 * Mn - 3.0 * n1[2:4] + n2[2:4]) / dt

    C = _oracle_pfc_ieq_coefficients(p1s, p2s, Ms, L, params, H)
    H1s, H2s = C[1], C[2]
    Rs = C[3:5]

    d1n = _oracle_pfc_spectral_derivatives(p1n, L, a12)
    d2n = _oracle_pfc_spectral_derivatives(p2n, L, a12)
    lap_p1s = _oracle_pfc_spectral_derivatives(p1s, L)[2]
    lap_p2s = _oracle_pfc_spectral_derivatives(p2s, L)[2]
    lapM = np.stack([_oracle_pfc_spectral_derivatives(Mn[j], L)[2]
                     for j in range(2)])

    nsite = p1n.size
    G1 = p1n / (Mp * dt) - 0.5 * (mu1n - np.sum(mu1n) / nsite)
    G3 = p2n / (Mp * dt) - 0.5 * (mu2n - np.sum(mu2n) / nsite)
    G2 = -mu1n + d1n[3] + 0.5 * d2n[4] + Sp * p1n + 4.0 * a1 * lap_p1s
    G4 = -mu2n + d2n[3] + 0.5 * d1n[4] + Sp * p2n + 4.0 * a2 * lap_p2s
    G5 = Mn / (Mm * dt) - 0.5 * mu3n
    G6 = -mu3n - w0 * lapM + Sm * Mn

    a = _oracle_pfc_split_solve(G1 + G3, G2 + G4, +1.0, L, dt, params)
    b = _oracle_pfc_split_solve(G1 - G3, G2 - G4, -1.0, L, dt, params)
    p11, mu11 = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    p21, mu21 = 0.5 * (a[0] - b[0]), 0.5 * (a[1] - b[1])

    zero = np.zeros(grid)
    c = _oracle_pfc_split_solve(zero, 2.0 * (H1s + H2s) * Us, +1.0, L, dt, params)
    d = _oracle_pfc_split_solve(zero, 2.0 * (H1s - H2s) * Us, -1.0, L, dt, params)
    p12, mu12 = 0.5 * (c[0] + d[0]), 0.5 * (c[1] + d[1])
    p22, mu22 = 0.5 * (c[0] - d[0]), 0.5 * (c[1] - d[1])

    k2 = _k2(grid, L)
    AM = 1.0 / (Mm * dt) + 0.5 * (w0 * k2 + Sm)

    def msolve(rhs):
        return np.stack([np.real(np.fft.ifft2(np.fft.fft2(rhs[j]) / AM))
                         for j in range(2)])

    M1 = msolve(G5 - 0.5 * G6)
    M2 = msolve(-Rs * Us)
    lapM1 = np.stack([_oracle_pfc_spectral_derivatives(M1[j], L)[2] for j in range(2)])
    lapM2 = np.stack([_oracle_pfc_spectral_derivatives(M2[j], L)[2] for j in range(2)])
    mu31 = -w0 * lapM1 + Sm * M1 + G6
    mu32 = -w0 * lapM2 + Sm * M2 + 2.0 * Rs * Us

    U1 = Un
    U2 = dt * 0.5 * (H1s * p1ts + H2s * p2ts + Rs[0] * Mts[0] + Rs[1] * Mts[1])

    RM = Rs[0] * Mts[0] + Rs[1] * Mts[1]
    xi1 = (_ip(H1s * Us, (p11 - p1n) / dt, L) - _ip(H1s * p1ts, Un, L)
           + _ip(H2s * Us, (p21 - p2n) / dt, L) - _ip(H2s * p2ts, Un, L)
           + _ip(Rs * Us, (M1 - Mn) / dt, L) - _ip(RM, Un, L))
    xi2 = (_ip(H1s * Us, p12 / dt, L) - _ip(H1s * p1ts, 0.5 * U2, L)
           + _ip(H2s * Us, p22 / dt, L) - _ip(H2s * p2ts, 0.5 * U2, L)
           + _ip(Rs * Us, M2 / dt, L) - _ip(RM, 0.5 * U2, L))

    den = 2.0 / dt - xi2
    if den == 0.0:
        raise ValueError("the ZEC scalar equation is singular")
    qbar = (2.0 / dt * q_n + xi1) / den
    q_new = 2.0 * qbar - q_n

    out = np.empty((10,) + grid)
    out[0] = p11 + qbar * p12
    out[1] = p21 + qbar * p22
    out[2:4] = M1 + qbar * M2
    out[4] = mu11 + qbar * mu12
    out[5] = mu21 + qbar * mu22
    out[6:8] = mu31 + qbar * mu32
    out[8] = U1 + qbar * U2
    out[9] = q_new
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    STIFF = "PS = {'gamma1': 1.0, 'gamma2': 1.0, 'eta1': -100.0, 'eta2': -100.0}\n"
    STATE = ("St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 2.0e-3)\n")
    return [
        # normal: the first step from the consistent initial state.
        {"setup": "import numpy as np\n" + STATE,
         "call": "pfc_time_step(St, 1.0, (24.0, 24.0), 2.0e-3, None, Hf)",
         "gold_call": "_oracle_pfc_time_step(St, 1.0, (24.0, 24.0), 2.0e-3, None, Hf)"},
        # normal: a later step, reached by advancing three times first, so the
        # three stored layers genuinely differ and every extrapolation matters.
        {"setup": "import numpy as np\n" + STATE +
                  "q0 = 1.0\n"
                  "o1 = _oracle_pfc_time_step(St, q0, (24.0, 24.0), 2.0e-3, None, Hf)\n"
                  "q1 = float(o1[9].flat[0]); St1 = np.stack([o1[:9], St[0], St[1]])\n"
                  "o2 = _oracle_pfc_time_step(St1, q1, (24.0, 24.0), 2.0e-3, None, Hf)\n"
                  "q2 = float(o2[9].flat[0]); St2 = np.stack([o2[:9], St1[0], St1[1]])\n"
                  "o3 = _oracle_pfc_time_step(St2, q2, (24.0, 24.0), 2.0e-3, None, Hf)\n"
                  "q = float(o3[9].flat[0]); St = np.stack([o3[:9], St2[0], St2[1]])\n",
         "call": "pfc_time_step(St, q, (24.0, 24.0), 2.0e-3, None, Hf)",
         "gold_call": "_oracle_pfc_time_step(St, q, (24.0, 24.0), 2.0e-3, None, Hf)"},
        # boundary: a nonlocal scalar far from one, which multiplies the whole
        # second half of the split and is the ZEC variable's whole purpose.
        {"setup": "import numpy as np\n" + STATE,
         "call": "pfc_time_step(St, 2.5, (24.0, 24.0), 2.0e-3, None, Hf)",
         "gold_call": "_oracle_pfc_time_step(St, 2.5, (24.0, 24.0), 2.0e-3, None, Hf)"},
        # boundary: no applied field at all, so the Zeeman contribution to the
        # magnetisation variation disappears from R.
        {"setup": "import numpy as np\n" + STATE,
         "call": "pfc_time_step(St, 1.0, (24.0, 24.0), 2.0e-3)",
         "gold_call": "_oracle_pfc_time_step(St, 1.0, (24.0, 24.0), 2.0e-3)"},
        # boundary: a RECTANGULAR box with a different grid in each direction and
        # a ten times larger time step.
        {"setup": "import numpy as np\n"
                  "St, Hf = _pfc_initial_state((16, 24), (16.0, 24.0), 2.0e-2)\n",
         "call": "pfc_time_step(St, 1.0, (16.0, 24.0), 2.0e-2, None, Hf)",
         "gold_call": "_oracle_pfc_time_step(St, 1.0, (16.0, 24.0), 2.0e-2, None, Hf)"},
        # edge: the high-stiffness parameter set, which the paper uses to stress
        # the scheme; the magneto-elastic coupling is a thousand times larger.
        {"setup": "import numpy as np\n" + STIFF +
                  "St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 1.0e-3, PS)\n",
         "call": "pfc_time_step(St, 1.0, (24.0, 24.0), 1.0e-3, PS, Hf)",
         "gold_call": "_oracle_pfc_time_step(St, 1.0, (24.0, 24.0), 1.0e-3, PS, Hf)"},
        # edge: a tiny time step, where the step must reduce to the explicit
        # forward increment of the reformulated system and Q must stay near one.
        {"setup": "import numpy as np\n"
                  "St, Hf = _pfc_initial_state((24, 24), (24.0, 24.0), 1.0e-7)\n",
         "call": "pfc_time_step(St, 1.0, (24.0, 24.0), 1.0e-7, None, Hf)",
         "gold_call": "_oracle_pfc_time_step(St, 1.0, (24.0, 24.0), 1.0e-7, None, Hf)"},
    ]
