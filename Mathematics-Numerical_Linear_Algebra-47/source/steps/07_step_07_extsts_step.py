"""
Advance the state by one full step of the partitioned integrator and return both the solution and the embedded solution.

The step uses the six-stage padded implicit-explicit Butcher pair of this integrator family, with shared abscissae c = (0, 2 gamma, 2 gamma, 1, 1, 1) where gamma = (2 - sqrt(2)) / 2, together with its embedding weight vectors, exactly as printed in the work that introduced the family. The pair is stiffly accurate and solve-decoupled.

Set the first stage value to the incoming state. For each stage i from 1 to 5, first assemble the coupling vector g for that stage row from the two tables against the stage values already computed, then branch on the tableau structure:

- if the implicit diagonal entry is zero and the abscissa increment is positive, advance from the previous stage value by one Runge-Kutta-Legendre super-step over an interval of length H = (c[i] - c[i-1]) h, with constant forcing g / H and with the stage count selected from H and lam_eff;

- if the implicit diagonal entry is zero and the abscissa increment is zero, set the stage value to the previous stage value plus g;

- otherwise solve the implicit reaction stage equation with coefficient h times the implicit diagonal entry, using the previous stage value as the initial iterate.

After each stage, evaluate the advective and reactive components at the new stage value; the diffusive component is not needed separately, since it enters through the super-step. The solution after the step is the sixth stage value. The embedded solution is the fifth stage value plus the coupling vector assembled from the embedding weights against the second-to-last tableau row, summed over all six stage values.

The reaction stage solves use a residual tolerance of 1e-13 in the maximum norm.

The function raises ValueError when y is not a one-dimensional array whose length is a multiple of 3; when the resulting number of nodes per species is fewer than 3; when D_nodes is not a one-dimensional array of length y.size // 3; when y or D_nodes contains a non-finite value; when h is not a finite positive scalar; when lam_eff is not a finite scalar; when lam_eff is negative; when dx is not a finite positive scalar; when eps is not a finite positive scalar; or when any of c, r, A, B is not a finite scalar.

This step is where the three treatments meet. Reading the tableau structure rather than the numbers makes the design visible: the abscissae repeat in two places, and every repetition is deliberate. Because a stage with a nonzero implicit diagonal must have a zero abscissa increment for the method to remain solve-decoupled, the repetitions are what allow implicit reaction stages to sit between diffusion sub-steps without either operation having to be solved simultaneously with the other. The padding of an existing implicit-explicit pair into this form is what converts a conventional additive Runge-Kutta method into one usable in the multirate-infinitesimal framework, and it changes the stage count without changing the order.

Stiff accuracy, meaning that the final row of each table equals its own solution weights, has a concrete consequence here: the step result is simply the last stage value, with no separate combination step. Together with the vanishing final abscissa increment, this also makes the embedded solution algebraic rather than requiring another sub-integration, so the error estimate costs one vector combination rather than another super-step.

The order of the resulting method is governed by two independent requirements. The coupling coefficients must satisfy their conditions through the desired order, which for the first-level-only structure used here reduces to internal consistency plus second-order conditions on both tables. Separately, the inner solver must itself be of at least that order. Since the stabilized explicit families available for the diffusive partition are second order, the composite method is second order, and improving it would require higher-order stabilized methods rather than a better tableau.

A property worth noticing is that only the advective and reactive components need to be evaluated and stored at each stage. The diffusive operator never appears in the coupling vectors at all; it is consumed entirely inside the sub-steps. This is the structural reason implicit solves stay local: diffusion, the operator that couples the whole domain, is the one operator never treated implicitly.

Returns
-------
np.ndarray of shape (2, 3n), row 0 the solution after the step and row 1 the embedded solution, both in the same species-stacked ordering as the input state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def extsts_step(y: np.ndarray, h: float, lam_eff: float, D_nodes: np.ndarray,
                dx: float, c: float, r: float, eps: float, A: float,
                B: float) -> np.ndarray:
    '''Advance one step of the partitioned integrator, returning solution and embedding.

    Parameters
    ----------
    y : np.ndarray
        Incoming state vector of shape (3n,), stacking the three species as
        [u; v; w].
    h : float
        Step size, positive.
    lam_eff : float
        Safeguarded dominant eigenvalue magnitude of the diffusion operator, used
        to select sub-step stage counts, non-negative.
    D_nodes : np.ndarray
        Nodal diffusion coefficient values, shape (n,).
    dx : float
        Uniform grid spacing, positive.
    c : float
        Advection speed.
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.

    Returns
    -------
    result : np.ndarray
        Array of shape (2, 3n). Row 0 is the solution after the step, row 1 is
        the embedded solution, both in the same species-stacked ordering as y.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_extsts_step(y: np.ndarray, h: float, lam_eff: float,
                        D_nodes: np.ndarray, dx: float, c: float, r: float,
                        eps: float, A: float, B: float) -> np.ndarray:
    """Reference implementation."""
    y = np.asarray(y, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y.ndim != 1 or y.size % 3 != 0:
        raise ValueError("y must be a 1D array whose length is a multiple of 3")
    n = y.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y.size // 3")
    if not (np.isfinite(y).all() and np.isfinite(D_nodes).all()):
        raise ValueError("y and D_nodes must contain only finite values")
    if not (np.isscalar(h) or np.ndim(h) == 0) or not np.isfinite(float(h)) \
            or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    if not (np.isscalar(lam_eff) or np.ndim(lam_eff) == 0) \
            or not np.isfinite(float(lam_eff)):
        raise ValueError("lam_eff must be a finite scalar")
    if float(lam_eff) < 0.0:
        raise ValueError("lam_eff must be a non-negative magnitude")
    if not (np.isscalar(dx) or np.ndim(dx) == 0) or not np.isfinite(float(dx)) \
            or float(dx) <= 0.0:
        raise ValueError("dx must be a finite positive scalar")
    if not (np.isscalar(eps) or np.ndim(eps) == 0) or not np.isfinite(float(eps)) \
            or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive scalar")
    for _name, _val in (("c", c), ("r", r), ("A", A), ("B", B)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) \
                or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
    h = float(h)
    lam_eff = float(lam_eff)
    dx = float(dx)
    c = float(c)
    r = float(r)
    eps = float(eps)
    A = float(A)
    B = float(B)

    sq2 = np.sqrt(2.0)
    gam = (2.0 - sq2) / 2.0
    dl = sq2 / 4.0
    Cv = np.array([0.0, 2.0 * gam, 2.0 * gam, 1.0, 1.0, 1.0])
    AI = np.zeros((6, 6))
    AE = np.zeros((6, 6))
    AE[1, 0] = 2.0 * gam
    AE[2, 0] = 2.0 * gam
    AE[3, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[3, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[4, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[4, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[5, 0] = dl
    AE[5, 2] = dl
    AE[5, 4] = gam
    AI[1, 0] = 2.0 * gam
    AI[2, 0] = gam
    AI[2, 2] = gam
    AI[3, 2] = 1.0
    AI[4, 0] = dl
    AI[4, 2] = dl
    AI[4, 4] = gam
    AI[5, 0] = dl
    AI[5, 2] = dl
    AI[5, 4] = gam
    dI = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
    dE = dI.copy()

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])

    def _advective(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            sp = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = -c * (sp[2:] - sp[:-2]) / (2.0 * dx)
            out[k * n:(k + 1) * n] = g
        return out

    def _diffusive(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            sp = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = (D_face[1:] * (sp[2:] - sp[1:-1])
                       - D_face[:-1] * (sp[1:-1] - sp[:-2])) / dx ** 2
            out[k * n:(k + 1) * n] = g
        return out

    def _reactive(vec):
        u = vec[:n]
        v = vec[n:2 * n]
        w = vec[2 * n:]
        out = np.concatenate([r * (A - (w + 1.0) * u + v * u ** 2),
                              r * (w * u - v * u ** 2),
                              r * ((B - w) / eps - w * u)])
        for k in range(3):
            out[k * n] = 0.0
            out[(k + 1) * n - 1] = 0.0
        return out

    def _stage_count(H):
        arg = 0.5 * (np.sqrt(9.0 + 8.0 * H * lam_eff) - 1.0)
        return max(2, int(np.ceil(arg)))

    def _super_step(y_in, H, s_cnt, forcing):
        b = np.zeros(s_cnt + 1)
        b[0] = 1.0 / 3.0
        b[1] = 1.0 / 3.0
        for j in range(2, s_cnt + 1):
            b[j] = (j * j + j - 2.0) / (2.0 * j * (j + 1.0))
        a_co = 1.0 - b
        w1 = 4.0 / (s_cnt * s_cnt + s_cnt - 2.0)
        slope = lambda vec: _diffusive(vec) + forcing
        f0 = slope(y_in)
        y_jm2 = y_in
        y_jm1 = y_in + b[1] * w1 * H * f0
        for j in range(2, s_cnt + 1):
            mu = (2.0 * j - 1.0) / j * b[j] / b[j - 1]
            nu = -(j - 1.0) / j * b[j] / b[j - 2]
            mu_t = mu * w1
            gamma_t = -a_co[j - 1] * mu_t
            y_new = (mu * y_jm1 + nu * y_jm2 + (1.0 - mu - nu) * y_in
                     + mu_t * H * slope(y_jm1) + gamma_t * H * f0)
            y_jm2 = y_jm1
            y_jm1 = y_new
        return y_jm1

    def _reaction_stage(rhs, hg, guess):
        mask = np.ones(n)
        mask[0] = 0.0
        mask[-1] = 0.0
        z = guess.copy()
        for _ in range(60):
            resid = z - hg * _reactive(z) - rhs
            if np.max(np.abs(resid)) < 1e-13:
                break
            u = z[:n]
            v = z[n:2 * n]
            w = z[2 * n:]
            J = np.zeros((n, 3, 3))
            J[:, 0, 0] = r * (-(w + 1.0) + 2.0 * v * u) * mask
            J[:, 0, 1] = r * u ** 2 * mask
            J[:, 0, 2] = -r * u * mask
            J[:, 1, 0] = r * (w - 2.0 * v * u) * mask
            J[:, 1, 1] = -r * u ** 2 * mask
            J[:, 1, 2] = r * u * mask
            J[:, 2, 0] = -r * w * mask
            J[:, 2, 2] = r * (-1.0 / eps - u) * mask
            M = np.tile(np.eye(3), (n, 1, 1)) - hg * J
            Fn = np.stack([resid[:n], resid[n:2 * n], resid[2 * n:]], axis=1)
            dz = np.linalg.solve(M, Fn[..., None])[..., 0]
            z = z - np.concatenate([dz[:, 0], dz[:, 1], dz[:, 2]])
        return z

    m = y.size
    Z = np.zeros((6, m))
    FR = np.zeros((6, m))
    FA = np.zeros((6, m))
    Z[0] = y
    FA[0] = _advective(y)
    FR[0] = _reactive(y)
    for i in range(1, 6):
        g = np.zeros(m)
        for j in range(i):
            g = g + ((AI[i, j] - AI[i - 1, j]) * FR[j]
                     + (AE[i, j] - AE[i - 1, j]) * FA[j])
        g = h * g
        dci = Cv[i] - Cv[i - 1]
        if AI[i, i] == 0.0 and dci > 0.0:
            H = dci * h
            Z[i] = _super_step(Z[i - 1], H, _stage_count(H), g / H)
        elif AI[i, i] == 0.0:
            Z[i] = Z[i - 1] + g
        else:
            Z[i] = _reaction_stage(Z[i - 1] + g, h * AI[i, i], Z[i - 1])
        FA[i] = _advective(Z[i])
        FR[i] = _reactive(Z[i])

    g_emb = np.zeros(m)
    for j in range(6):
        g_emb = g_emb + ((dI[j] - AI[4, j]) * FR[j] + (dE[j] - AE[4, j]) * FA[j])
    g_emb = h * g_emb

    return np.vstack([Z[5], Z[4] + g_emb])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration, first step from the initial state ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
h = 1.5 / 73.0
lam_eff = 3288.763196088486
""",
            "call": "extsts_step(y, h, lam_eff, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, h, lam_eff, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Normal: locked configuration from a perturbed state ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
rng = np.random.default_rng(13)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn]) + 0.03 * rng.standard_normal(3 * n)
h = 1.5 / 73.0
lam_eff = 3288.763196088486
""",
            "call": "extsts_step(y, h, lam_eff, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, h, lam_eff, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Normal: same state advanced with the unsafeguarded eigenvalue magnitude ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
h = 1.5 / 73.0
""",
            "call": "extsts_step(y, h, 2989.7847237168053, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, h, 2989.7847237168053, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Normal: halved step size on the locked configuration ---
        {
            "setup": """import numpy as np
n = 65
x = np.arange(n) / 64.0
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / 64.0
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
lam_eff = 3288.763196088486
""",
            "call": "extsts_step(y, 1.5 / 146.0, lam_eff, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, 1.5 / 146.0, lam_eff, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Boundary: minimal grid, one interior node per species ---
        {
            "setup": """import numpy as np
y = np.array([1.05, 1.2, 1.0, 2.76, 2.9, 2.7, 2.9, 3.0, 2.8])
D_nodes = np.array([0.08, 0.15, 0.05])
dx = 0.5
""",
            "call": "extsts_step(y, 0.02, 5.0, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, 0.02, 5.0, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Boundary: zero eigenvalue magnitude selects the two-stage floor ---
        {
            "setup": """import numpy as np
n = 17
x = np.arange(n) / (n - 1.0)
D_nodes = np.full(n, 0.02)
dx = 1.0 / (n - 1.0)
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
""",
            "call": "extsts_step(y, 0.001, 0.0, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, 0.001, 0.0, D_nodes, dx, 0.45, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Edge: zero advection speed leaves only diffusion and reaction ---
        {
            "setup": """import numpy as np
n = 33
x = np.arange(n) / (n - 1.0)
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
""",
            "call": "extsts_step(y, 0.01, 900.0, D_nodes, dx, 0.0, 1.1, 0.02, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, 0.01, 900.0, D_nodes, dx, 0.0, 1.1, 0.02, 1.05, 2.9)",
        },
        # --- Edge: stiffer relaxation channel ---
        {
            "setup": """import numpy as np
n = 33
x = np.arange(n) / (n - 1.0)
D_nodes = 0.1 * (1.0 + 0.9 * np.sin(2.0 * np.pi * x))
dx = 1.0 / (n - 1.0)
sn = 0.1 * np.sin(2.0 * np.pi * x)
y = np.concatenate([1.05 + sn, 2.9 / 1.05 + sn, 2.9 + sn])
""",
            "call": "extsts_step(y, 0.01, 900.0, D_nodes, dx, 0.45, 1.1, 1e-3, 1.05, 2.9)",
            "gold_call": "_oracle_extsts_step(y, 0.01, 900.0, D_nodes, dx, 0.45, 1.1, 1e-3, 1.05, 2.9)",
        },
        # --- Invalid: state length not a multiple of 3 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extsts_step(np.zeros(10), 0.02, 100.0, np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extsts_step(np.zeros(10), 0.02, 100.0, np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: diffusion coefficient array length mismatch ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extsts_step(np.zeros(9), 0.02, 100.0, np.ones(4), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extsts_step(np.zeros(9), 0.02, 100.0, np.ones(4), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive step size ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extsts_step(np.zeros(9), 0.0, 100.0, np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extsts_step(np.zeros(9), 0.0, 100.0, np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative eigenvalue magnitude ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extsts_step(np.zeros(9), 0.02, -1.0, np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extsts_step(np.zeros(9), 0.02, -1.0, np.ones(3), 1.0, 0.45, 1.1, 0.02, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive stiffness parameter ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        extsts_step(np.zeros(9), 0.02, 100.0, np.ones(3), 1.0, 0.45, 1.1, 0.0, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_extsts_step(np.zeros(9), 0.02, 100.0, np.ones(3), 1.0, 0.45, 1.1, 0.0, 1.05, 2.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
