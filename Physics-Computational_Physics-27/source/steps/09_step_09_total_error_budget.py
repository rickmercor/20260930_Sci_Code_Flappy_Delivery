"""
ORCHESTRATOR - final step. Chain every earlier step into the accumulated

rounding-error budget of a complete non-unitary Chebyshev simulation and return its

base-10 logarithm.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)

momentum: float, the wave packet momentum k

sigma: float, the wave packet width, > 0

t_max: float, total evolution time, > 0

delta_max: float, per-step tolerance driving the step selection, > 0

eps: float, machine precision, > 0 (default 1.11e-16)

tol: float, term-norm stopping tolerance, > 0 (default 1e-14)

patience: int, consecutive sub-tolerance terms required to stop, >= 1 (default 5)



Returns

-------

log10_budget: float, base-10 logarithm of the accumulated rounding-error budget



Composition

-----------

Steps 02, 03, 04, 05 and 08 are called directly from this step and their

results are consumed here. Steps 01, 06 and 07 are composed through step 08,

which builds the Hamiltonian, forms the expansion coefficients and applies the

vector recursion, returning the realised term count this step needs.



Raises

------

ValueError: if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, or t_max, delta_max or eps is not positive

The pipeline runs in the order the physics dictates:



  1. assemble the Hatano-Nelson Hamiltonian                       (step 01)

  2. evaluate its closed-form spectrum                            (step 02)

  3. find the Bernstein radius rho enclosing that spectrum        (step 03)

  4. convert rho and the tolerance into the largest legal step    (step 04)

  5. round the step count up so the steps tile t_max exactly      (here)

  6. propagate the packet, counting Chebyshev terms               (step 08, which

     composes the coefficients of step 06 with the vector recursion of step 07)

  7. turn the realised term count into an error budget            (step 05)



Step 5 is what makes the answer non-trivial. The tolerance fixes a ceiling on the

length of one step, but the steps must also tile the total evolution time exactly, so

the step count is the ceiling of t_max divided by that limit, and the step actually

used sits strictly below the limit whenever the ratio is not an integer. Evaluating

the bound at the ceiling would simply return the input tolerance, because the

selection rule is the exact algebraic inverse of the bound; evaluating it at the step

actually taken does not, and the gap between the two is set by how far the rounding-up

moved the step.



The reported quantity is the base-10 logarithm of the accumulated bound evaluated with

M, the total number of Chebyshev terms summed over the whole simulation, at the step

actually taken.

Returns
-------
float, the base-10 logarithm of the accumulated rounding-error budget of the whole simulation, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def total_error_budget(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    delta_max: float,
    eps: float = 1.11e-16,
    tol: float = 1e-14,
    patience: int = 5,
) -> float:
    '''Run the full pipeline and return log10 of the accumulated error budget.

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
    delta_max : float
        Per-step rounding-error tolerance driving the step selection, must be > 0.
    eps : float
        Machine precision used in the bounds, must be > 0.
    tol : float
        Euclidean-norm tolerance a term must fall below to count toward stopping.
    patience : int
        Number of consecutive sub-tolerance terms required before stopping.

    Returns
    -------
    log10_budget : float
        Base-10 logarithm of the accumulated rounding-error budget, evaluated with the
        total number of Chebyshev terms summed across the simulation.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, or t_max, delta_max or eps is not positive.
    '''
    return log10_budget

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_total_error_budget(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    delta_max: float,
    eps: float = 1.11e-16,
    tol: float = 1e-14,
    patience: int = 5,
) -> float:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")
    if not np.isfinite(float(t_max)) or float(t_max) <= 0.0:
        raise ValueError("t_max must be finite and > 0")
    if not np.isfinite(float(delta_max)) or float(delta_max) <= 0.0:
        raise ValueError("delta_max must be finite and > 0")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and > 0")

    # Every stage below is delegated to the earlier sub-problem, preferring the gold
    # binding when the harness provides one and falling back to the public name.

    # Step 01 (the Hamiltonian) and steps 06 and 07 (the expansion coefficients and the
    # single-step propagation) are reached through step 08, which consumes all three and
    # returns the realised term count. They are deliberately not called again here: the
    # orchestrator has no use for their return values of its own, and calling one only to
    # discard it would be a token call rather than a composition.

    # --- step 02: closed-form spectrum ---
    try:
        spectrum = _oracle_analytic_spectrum(n_sites, gamma, p, alpha_bc)
    except NameError:
        spectrum = analytic_spectrum(n_sites, gamma, p, alpha_bc)

    # --- step 03: smallest Bernstein ellipse containing the spectrum ---
    try:
        rho = _oracle_bernstein_radius(spectrum)
    except NameError:
        rho = bernstein_radius(spectrum)

    # --- step 04: tolerance -> ceiling on a single step ---
    try:
        dt_ceiling = _oracle_max_time_step(rho, delta_max, eps)
    except NameError:
        dt_ceiling = max_time_step(rho, delta_max, eps)

    # rounding the count up is what keeps the answer off the input tolerance
    n_steps = int(np.ceil(float(t_max) / dt_ceiling))
    dt = float(t_max) / n_steps

    # --- step 08: propagate, accumulating the realised Chebyshev term count ---
    try:
        _, total_terms = _oracle_evolve_wavepacket(
            n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, tol, patience
        )
    except NameError:
        _, total_terms = evolve_wavepacket(
            n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, n_steps, tol, patience
        )

    # --- step 05: accumulated rounding-error budget of the whole run ---
    try:
        budget = _oracle_rounding_error_bound(total_terms, dt, rho, eps)
    except NameError:
        budget = rounding_error_bound(total_terms, dt, rho, eps)

    return float(np.log10(budget))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the task configuration end to end ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
delta_max = 1e-12
""",
            "call": "total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
            "gold_call": "_oracle_total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
        },
        # --- Integration: a tighter tolerance forces more, shorter steps ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
delta_max = 1e-14
""",
            "call": "total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
            "gold_call": "_oracle_total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
        },
        # --- Integration: open boundary conditions shrink the enclosing ellipse ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 0.0
momentum = np.pi / 2
sigma = 10.0
t_max = 20.0
delta_max = 1e-12
""",
            "call": "total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
            "gold_call": "_oracle_total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
        },
        # --- Integration: Hermitian limit with a weaker hopping scale ---
        {
            "setup": """import numpy as np
n_sites = 50
gamma = 0.4
p = 0.0
alpha_bc = 1.0
momentum = np.pi / 3
sigma = 5.0
t_max = 12.0
delta_max = 1e-12
""",
            "call": "total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
            "gold_call": "_oracle_total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
        },
        # --- Edge: t_max shorter than the step ceiling, so a single step is taken ---
        {
            "setup": """import numpy as np
n_sites = 40
gamma = 0.3
p = 0.1
alpha_bc = 1.0
momentum = np.pi / 2
sigma = 4.0
t_max = 1.0
delta_max = 1e-12
""",
            "call": "total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
            "gold_call": "_oracle_total_error_budget(n_sites, gamma, p, alpha_bc, momentum, sigma, t_max, delta_max, 1.11e-16, 1e-14, 5)",
        },
        # --- Invalid: |p| >= 1 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = total_error_budget
    try:
        _fn(100, 0.85, 1.2, 1.0, 1.5, 10.0, 20.0, 1e-12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_total_error_budget
    try:
        _fn(100, 0.85, 1.2, 1.0, 1.5, 10.0, 20.0, 1e-12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-positive tolerance ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = total_error_budget
    try:
        _fn(100, 0.85, 0.35, 1.0, 1.5, 10.0, 20.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_total_error_budget
    try:
        _fn(100, 0.85, 0.35, 1.0, 1.5, 10.0, 20.0, 0.0)
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
