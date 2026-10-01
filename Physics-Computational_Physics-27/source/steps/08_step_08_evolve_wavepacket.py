"""
Evolve a Gaussian wave packet on the Hatano-Nelson chain over a fixed total time

split into equal Chebyshev steps, renormalising after each step, and accumulate the

total number of expansion terms summed.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)

momentum: float, the wave packet momentum k

sigma: float, the wave packet width, > 0

t_max: float, total evolution time, > 0

n_steps: int, number of equal time steps, >= 1

tol: float, term-norm stopping tolerance, > 0 (default 1e-14)

patience: int, consecutive sub-tolerance terms required to stop, >= 1 (default 5)



Returns

-------

psi_final: (n_sites,) complex ndarray, the normalised state at t_max

total_terms: int, Chebyshev terms summed across all steps



Raises

------

ValueError: if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, sigma or t_max is not positive, or n_steps or patience is below 1

The initial state is a Gaussian envelope carrying momentum k, centred on the middle of

the chain,



    psi_n(0) proportional to exp( -(n - n_0)^2 / (2 sigma^2) ) exp(i k n),

    n = 1 .. N,   n_0 = (N + 1) / 2,



normalised to unit Euclidean norm. Under the Hermitian part of the dynamics such a

packet propagates at the group velocity -2 gamma sin(k) while spreading diffusively.



Because the generator is non-Hermitian the norm is not conserved. The non-reciprocal

hoppings amplify one propagation direction and attenuate the other, so the raw norm of

the state grows step after step, by a fixed factor per step for a fixed step length.

Left unchecked this would overflow for long simulations, and it would also break the

adaptive stopping rule, because a rule that compares term norms against a fixed

absolute tolerance is only meaningful when the state it acts on has a controlled scale.

The state is therefore renormalised after every step, which keeps the vector entries at

order unity and keeps successive steps comparable. Renormalisation rescales the state

without rotating it, so the propagated direction - and hence all physical observables

built from the normalised state - is unaffected.



The routine accumulates the number of Chebyshev terms summed across all steps. That

total is the quantity entering the accumulated rounding-error bound of the whole

simulation, since every summed term contributes its own representation error.

Returns
-------
tuple (psi_final, total_terms), where psi_final is a unit-norm complex ndarray of shape (n_sites,) and total_terms is the native Python int total number of Chebyshev terms summed over all steps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evolve_wavepacket(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    n_steps: int,
    tol: float = 1e-14,
    patience: int = 5,
) -> tuple:
    '''Propagate a Gaussian packet on the Hatano-Nelson chain and count expansion terms.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.
    momentum : float
        Momentum k imprinted on the initial Gaussian envelope.
    sigma : float
        Width of the initial Gaussian envelope, must be > 0.
    t_max : float
        Total evolution time, must be > 0.
    n_steps : int
        Number of equal time steps, must be >= 1.
    tol : float
        Euclidean-norm tolerance a term must fall below to count toward stopping.
    patience : int
        Number of consecutive sub-tolerance terms required before stopping.

    Returns
    -------
    psi_final : np.ndarray
        Normalised complex state of shape (n_sites,) at time t_max.
    total_terms : int
        Total number of Chebyshev terms summed across every step.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, sigma or t_max is not positive, or n_steps or patience is below 1.
    '''
    return psi_final, total_terms

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _gaussian_packet(n_sites: int, momentum: float, sigma: float) -> np.ndarray:
    """Unit-norm Gaussian packet of width sigma centred at (N + 1) / 2."""
    site = np.arange(1, int(n_sites) + 1)
    centre = (int(n_sites) + 1) / 2.0
    psi = np.exp(-((site - centre) ** 2) / (2.0 * float(sigma) ** 2)) * np.exp(
        1j * float(momentum) * site
    )
    return psi / np.linalg.norm(psi)


def _oracle_evolve_wavepacket(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    n_steps: int,
    tol: float = 1e-14,
    patience: int = 5,
) -> tuple:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")
    if not np.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("sigma must be finite and > 0")
    if not np.isfinite(float(t_max)) or float(t_max) <= 0.0:
        raise ValueError("t_max must be finite and > 0")
    if not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and > 0")
    if not isinstance(patience, (int, np.integer)) or int(patience) < 1:
        raise ValueError("patience must be an integer >= 1")

    # --- step 01: the Hamiltonian ---
    try:
        H = _oracle_build_hatano_nelson(n_sites, gamma, p, alpha_bc)
    except NameError:
        H = build_hatano_nelson(n_sites, gamma, p, alpha_bc)

    psi = _gaussian_packet(n_sites, momentum, sigma)
    dt = float(t_max) / int(n_steps)

    # --- step 07: one adaptively truncated Chebyshev step, repeated ---
    total_terms = 0
    for _ in range(int(n_steps)):
        try:
            psi, used = _oracle_chebyshev_step(H, psi, dt, tol, patience, 4000)
        except NameError:
            psi, used = chebyshev_step(H, psi, dt, tol, patience, 4000)
        total_terms += used
        psi = psi / np.linalg.norm(psi)

    return psi, int(total_terms)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the full task configuration (first return value) ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
n_steps = 6
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
        },
        # --- Normal: the full task configuration (second return value) ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
n_steps = 6
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
        },
        # --- Normal: the same total time taken in more, shorter steps (first return value) ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
n_steps = 20
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
        },
        # --- Normal: the same total time taken in more, shorter steps (second return value) ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
n_steps = 20
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
        },
        # --- Normal: open boundary conditions, real spectrum (first return value) ---
        {
            "setup": """import numpy as np
n_sites = 60
gamma = 0.5
p = 0.2
alpha_bc = 0.0
momentum = np.pi / 3
sigma = 6.0
t_max = 10.0
n_steps = 4
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
        },
        # --- Normal: open boundary conditions, real spectrum (second return value) ---
        {
            "setup": """import numpy as np
n_sites = 60
gamma = 0.5
p = 0.2
alpha_bc = 0.0
momentum = np.pi / 3
sigma = 6.0
t_max = 10.0
n_steps = 4
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
        },
        # --- Boundary: a single step covering the whole interval (first return value) ---
        {
            "setup": """import numpy as np
n_sites = 40
gamma = 0.4
p = 0.1
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 5.0
t_max = 2.0
n_steps = 1
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
        },
        # --- Boundary: a single step covering the whole interval (second return value) ---
        {
            "setup": """import numpy as np
n_sites = 40
gamma = 0.4
p = 0.1
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 5.0
t_max = 2.0
n_steps = 1
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
        },
        # --- Edge: Hermitian limit p = 0, norm is conserved (first return value) ---
        {
            "setup": """import numpy as np
n_sites = 30
gamma = 0.6
p = 0.0
alpha_bc = 1.0
momentum = np.pi / 4
sigma = 4.0
t_max = 5.0
n_steps = 3
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[0]",
        },
        # --- Edge: Hermitian limit p = 0, norm is conserved (second return value) ---
        {
            "setup": """import numpy as np
n_sites = 30
gamma = 0.6
p = 0.0
alpha_bc = 1.0
momentum = np.pi / 4
sigma = 4.0
t_max = 5.0
n_steps = 3
""",
            "call": "evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
            "gold_call": "_oracle_evolve_wavepacket(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, 1e-14, 5)[1]",
        },
        # --- Invalid: zero steps ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = evolve_wavepacket
    try:
        _fn(20, 0.5, 0.2, 1.0, 1.0, 3.0, 5.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_evolve_wavepacket
    try:
        _fn(20, 0.5, 0.2, 1.0, 1.0, 3.0, 5.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-positive packet width ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = evolve_wavepacket
    try:
        _fn(20, 0.5, 0.2, 1.0, 1.0, 0.0, 5.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_evolve_wavepacket
    try:
        _fn(20, 0.5, 0.2, 1.0, 1.0, 0.0, 5.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
