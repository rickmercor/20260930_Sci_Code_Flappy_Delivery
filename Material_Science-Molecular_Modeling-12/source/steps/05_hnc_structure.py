"""
Solve the Ornstein-Zernike equation with the hypernetted-chain closure for a homogeneous fluid of number density rho whose pair potential is given, in units of k_B T, as the table betau of values beta u(r_i) on the uniform radial grid r_i = i dr, i = 1 .. N (N is the length of the table; there is no node at r = 0), and return the pair distribution function g(r_i) on the same grid. Work with the indirect correlation function gamma(r) = h(r) - c(r), where h = g - 1 is the total and c the direct correlation function: the closure is g(r) = exp(-beta u(r) + gamma(r)), and the Ornstein-Zernike relation in Fourier space reads gamma_hat(k) = rho c_hat(k)^2 / (1 - rho c_hat(k)). Use exactly the following discrete three-dimensional radial transform pair on the grid, with wavenumbers k_j = j pi / ((N + 1) dr), j = 1 .. N: forward f_hat(k_j) = (4 pi dr / k_j) sum_{i=1}^{N} r_i f(r_i) sin(k_j r_i), inverse f(r_i) = (dk / (2 pi^2 r_i)) sum_{j=1}^{N} k_j f_hat(k_j) sin(k_j r_i) with dk = pi / ((N + 1) dr) (these are type-I discrete sine transforms and are each other's inverse). Iterate to the fixed point of the map gamma -> inverse(rho c_hat^2 / (1 - rho c_hat)) with c = exp(-beta u + gamma) - 1 - gamma, starting from gamma = 0, with any convergence scheme (damped Picard, Ng acceleration, Newton) until the maximum absolute difference between gamma and its image under the map is below 1e-13; return g = exp(-beta u + gamma). Raise ValueError if the table is not one-dimensional with at least eight nodes or if rho or dr is not positive.

The HNC closure neglects the bridge function and is accurate for soft, bounded repulsions such as the kernel-shaped potentials of density-dependent models, which is why the source uses it to predict the structure of the mesoparticle fluid without simulation. With the potential tabulated on the grid the whole problem is discrete: the same sine transform maps radial functions to their three-dimensional Fourier transforms and back, and the fixed point is unique for the parameters of this task, so any solver that converges to the stated residual returns the same g to better than 1e-11.

Returns
-------
A (N,) float64 array g(r_i) on the input grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hnc_structure(betau, rho, dr):
    """Solve the Ornstein-Zernike equation with the hypernetted-chain closure for a homogeneous
    fluid of number density rho whose pair potential is given, in units of k_B T, as the
    table betau of values beta u(r_i) on the uniform radial grid r_i = i dr, i = 1 . A (N,)
    float64 array g(r_i) on the input grid."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def _fourier_pair(n_grid, dr):
    r, k = _grid(n_grid, dr)
    S = np.sin(np.outer(k, r))
    dk = k[0]
    fwd = lambda f: 4.0*np.pi*dr/k*(S @ (r*f))
    inv = lambda F: dk/(2.0*np.pi**2*r)*(S @ (k*F))
    return r, k, fwd, inv

def _oracle_hnc_structure(betau, rho, dr):
    betau = np.asarray(betau, dtype=float)
    if betau.ndim != 1 or betau.size < 8:
        raise ValueError("betau must be a one-dimensional table with at least eight nodes")
    if rho <= 0 or dr <= 0:
        raise ValueError("density and grid spacing must be positive")
    n_grid = betau.size
    r, k, fwd, inv = _fourier_pair(n_grid, dr)
    def picard(gam):
        c = np.exp(-betau + gam) - 1.0 - gam
        ch = fwd(c)
        den = 1.0 - rho*ch
        if np.any(den <= 0.0) or not np.all(np.isfinite(ch)):
            return None
        return inv(rho*ch*ch/den)
    gam = np.zeros(n_grid)
    mix = 0.1
    fin, fout = [], []
    res = np.inf
    for it in range(200000):
        out = picard(gam)
        if out is None:
            gam *= 0.5; mix *= 0.5; fin, fout = [], []
            if mix < 1.0e-6:
                raise RuntimeError("HNC iteration lost the physical branch")
            continue
        res = float(np.max(np.abs(out - gam)))
        if res < 1.0e-13:
            gam = out; break
        fin.append(gam.copy()); fout.append(out.copy())
        if len(fin) > 3:
            fin.pop(0); fout.pop(0)
        accepted = False
        if len(fin) == 3 and it > 5:                    # Ng (1974) two-vector acceleration
            f = [o - i for o, i in zip(fout, fin)]
            d1, d2 = f[2] - f[1], f[2] - f[0]
            a = np.array([[d1 @ d1, d1 @ d2], [d1 @ d2, d2 @ d2]])
            b = np.array([f[2] @ d1, f[2] @ d2])
            if np.linalg.cond(a) < 1.0e14:
                c1, c2 = np.linalg.solve(a, b)
                cand = (1.0 - c1 - c2)*fout[2] + c1*fout[1] + c2*fout[0]
                test = picard(cand)
                if test is not None and float(np.max(np.abs(test - cand))) < res:
                    gam = cand; accepted = True
        if not accepted:
            gam = (1.0 - mix)*gam + mix*out
    else:
        raise RuntimeError("HNC iteration did not converge")
    return np.exp(-betau + gam)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark-like repulsive kernel table\ndr = 0.025\nr = dr * np.arange(1, 101)\nrcut = 2.1564\nw = np.where(r < rcut, 15.0 / (2.0 * np.pi * rcut**3) * (1.0 - r / rcut)**2, 0.0)\nbetau = 2.0 * 0.1229 * w / 0.011054\nrho = 1.0\n',
         'call': 'hnc_structure(betau, rho, dr)',
         'gold_call': '_oracle_hnc_structure(betau, rho, dr)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted table length of eight nodes\ndr = 0.25\nr = dr * np.arange(1, 9)\nbetau = 0.2 * np.exp(-r)\nrho = 0.2\n',
         'call': 'hnc_structure(betau, rho, dr)',
         'gold_call': '_oracle_hnc_structure(betau, rho, dr)'},
        {'setup': 'import numpy as np\n# edge: zero pair potential, whose HNC solution is the structureless g(r)=1 limit\ndr = 0.2\nbetau = np.zeros(16)\nrho = 0.9\n',
         'call': 'hnc_structure(betau, rho, dr)',
         'gold_call': '_oracle_hnc_structure(betau, rho, dr)'},
    ]
