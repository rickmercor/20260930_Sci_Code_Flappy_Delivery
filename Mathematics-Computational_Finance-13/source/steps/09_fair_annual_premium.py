"""
Run the full pipeline and return the fair annual premium of the surrenderable contract. Build the guarantee and survival schedules, the equity innovation weights, and the deterministic shift increments; enumerate the reachable nodes of both factor lattices; and for each joint node assemble the transition stencils, the joint coupling table and the fund branch multipliers. Then run the backward recursion from the maturity benefit to inception, applying mortality at each policy-year close and the surrender obstacle at each admissible anniversary, with the value carried as a function of the fund on a grid that includes zero. Solve for the premium at which the time-zero value at zero fund vanishes, once on a uniform fund grid of n_fund nodes on [0, fund_max] and once on a uniform grid of 2*n_fund - 1 nodes on the same interval (half the spacing), and return (4*P_fine - P_coarse)/3, which removes the leading second-order discretisation error.

The valuation assembles into a single deterministic backward recursion over a lattice whose state is the pair of financial factors, with the accumulated fund carried as a function at each node.

Three objects are built once, before any node is touched. The contractual schedules fix the two guarantee levels and the survival probabilities by anniversary, from which the one-year death probabilities follow as ratios of consecutive survival entries. The equity innovation weights are fixed by the three correlations alone and do not vary over the lattice. The deterministic shift increments are calibrated on the rate lattice by forward propagation of state prices, one increment per step, and depend on the rate factor only.

The reachable state space is then enumerated. Each factor lattice starts at its anchor and grows by applying the transition stencil at every node already reached, one step at a time. Because the stencils may point away from the parent and because mean reversion pulls the outer nodes back, the reachable set does not simply widen by one node per step and must be built by iteration rather than assumed.

At each joint node the transition is assembled in three stages: the two marginal stencils give successor indices, probabilities and standardised innovations; the coupling table combines them into nine joint successors preserving both marginals and the factor correlation; and the fund transition expands those into branches by integrating the independent residual of the equity innovation, applying the branchwise discount and the martingale rescaling. These objects depend on the node and the step but not on the premium, so they are computed once and reused across the root search.

The backward recursion starts from the maturity benefit, the greater of the fund and the maturity guarantee, held at every terminal node. Each step maps child values through the branch multipliers, blending in the death benefit where the step closes a policy year and applying the surrender obstacle where the date admits it. At inception only the value at zero fund is required, since the contract begins before the first contribution is credited; the fund grid therefore includes zero as a node so that this value is read directly rather than extrapolated.

The fair premium is the level at which the time-zero value vanishes. That value is strictly decreasing in the premium: raising the premium reduces the continuation value at every node, and the surrender obstacle is non-expansive so it cannot reverse the ordering. Once the obstacle binds at the first admissible anniversary the dependence becomes exactly affine with unit slope, so the function remains strictly monotone over any bracket wide enough to contain the root and a safeguarded bracketing method converges to the unique solution. Because the surrender right destroys affinity in the premium, no closed form is available and the root must be found numerically.

Finally, the value computed on a fund grid of finite resolution carries a discretisation error which falls at second order in the grid spacing. Solving at two resolutions and combining the two premiums so that the leading term cancels removes it, leaving an error of higher order at the cost of one additional solve.

Returns
-------
float, the fair annual premium of the mixed-surrender contract as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fair_annual_premium(n_years: int, steps_per_year: int, contribution: float,
                        g_maturity: float, g_surrender: float, alpha_s: float, q: float,
                        issue_age: int, mk_a: float, mk_b: float, mk_c: float,
                        v0: float, kappa_v: float, theta_v: float, sigma_v: float,
                        x0: float, kappa_r: float, theta_r: float, sigma_r: float,
                        rho_sv: float, rho_sr: float, rho_vr: float,
                        lam_v: float, lam_x: float,
                        market_disc: np.ndarray, n_fund: int, fund_max: float) -> float:
    '''Fair annual premium of the surrenderable equity-linked contract.

    Parameters
    ----------
    n_years, steps_per_year : int
        Contract horizon in years and numerical steps per year.
    contribution, g_maturity, g_surrender, alpha_s, q : float
        Amount credited each anniversary, the maturity and surrender guarantee
        accumulation rates, the fund-based surrender fraction, and the dividend
        yield.
    issue_age, mk_a, mk_b, mk_c : int, float, float, float
        Issue age and the Gompertz-Makeham hazard parameters.
    v0, kappa_v, theta_v, sigma_v : float
        Initial level and parameters of the variance factor.
    x0, kappa_r, theta_r, sigma_r : float
        Initial level and parameters of the rate factor.
    rho_sv, rho_sr, rho_vr : float
        Equity-variance, equity-rate and variance-rate correlations.
    lam_v, lam_x : float
        Lattice scale parameters of the two factors.
    market_disc : np.ndarray
        Shape (n_years * steps_per_year,), market discount factors at the grid
        maturities.
    n_fund : int
        Number of fund nodes at the coarser of the two resolutions,
        at least three; the finer resolution has 2*n_fund - 1 nodes.
    fund_max : float
        Strictly positive upper end of the fund grid.

    Returns
    -------
    premium : float
        The fair annual premium as a native Python float.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid contract, model and grid.
    '''
    return premium

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_fair_annual_premium(n_years: int, steps_per_year: int, contribution: float,
                                g_maturity: float, g_surrender: float, alpha_s: float, q: float,
                                issue_age: int, mk_a: float, mk_b: float, mk_c: float,
                                v0: float, kappa_v: float, theta_v: float, sigma_v: float,
                                x0: float, kappa_r: float, theta_r: float, sigma_r: float,
                                rho_sv: float, rho_sr: float, rho_vr: float,
                                lam_v: float, lam_x: float,
                                market_disc: np.ndarray, n_fund: int, fund_max: float) -> float:
    if not isinstance(n_years, (int, np.integer)) or int(n_years) < 1:
        raise ValueError("n_years must be an integer of at least one")
    if not isinstance(steps_per_year, (int, np.integer)) or int(steps_per_year) < 1:
        raise ValueError("steps_per_year must be an integer of at least one")
    if not isinstance(n_fund, (int, np.integer)) or int(n_fund) < 3:
        raise ValueError("n_fund must be an integer of at least three")
    if not isinstance(fund_max, (int, float, np.integer, np.floating)) or not float(fund_max) > 0.0:
        raise ValueError("fund_max must be a positive real number")
    P = np.asarray(market_disc, dtype=float)
    T, Nyr, nF = int(n_years), int(steps_per_year), int(n_fund)
    N = T * Nyr
    if P.ndim != 1 or P.size != N or P.min() <= 0.0:
        raise ValueError("market_disc must hold one positive discount factor per numerical step")

    h = 1.0 / Nyr
    D, gm, gs = float(contribution), float(g_maturity), float(g_surrender)
    a_s, qy = float(alpha_s), float(q)

    sched = _oracle_guarantee_and_mortality_schedules(D, gm, gs, T, int(issue_age),
                                                      float(mk_a), float(mk_b), float(mk_c))
    G_mat, G_sur, surv = sched[0], sched[1], sched[2]
    death = np.array([1.0 - surv[i + 1] / surv[i] for i in range(T)])

    weights = _oracle_equity_innovation_weights(float(rho_sv), float(rho_sr), float(rho_vr))
    shift = _oracle_calibrate_lattice_shift(float(x0), float(kappa_r), float(theta_r),
                                            float(sigma_r), float(lam_x), h, P)

    def _lattice_node(root, sigma, lam, j):
        chi = 2.0 * np.sqrt(root) / sigma + j * lam * np.sqrt(h)
        return (sigma / 2.0 * max(chi, 0.0)) ** 2

    vv = float(v0), float(kappa_v), float(theta_v), float(sigma_v), float(lam_v)
    xx = float(x0), float(kappa_r), float(theta_r), float(sigma_r), float(lam_x)

    stv, stx = {}, {}
    def _cached_stencil(par, j, cache):
        if j not in cache:
            cache[j] = _oracle_cir_transition_stencil(par[0], par[1], par[2], par[3], par[4], h, j)
        return cache[j]

    span_v = _oracle_reachable_factor_nodes(vv[0], vv[1], vv[2], vv[3], vv[4], h, N)
    span_x = _oracle_reachable_factor_nodes(xx[0], xx[1], xx[2], xx[3], xx[4], h, N)
    reach_v = [list(range(int(span_v[0, n]), int(span_v[1, n]) + 1)) for n in range(N + 1)]
    reach_x = [list(range(int(span_x[0, n]), int(span_x[1, n]) + 1)) for n in range(N + 1)]

    def _step_moments(z, k, th, s):
        e = np.exp(-k * h)
        return (th + (z - th) * e,
                s * s * z / k * e * (1.0 - e) + th * s * s / (2.0 * k) * (1.0 - e) ** 2)

    def _node_innovations(par, j, cache):
        st = _cached_stencil(par, j, cache)
        Z = np.array([_lattice_node(par[0], par[3], par[4], int(j + r)) for r in st[0]])
        m, v = _step_moments(_lattice_node(par[0], par[3], par[4], j), par[1], par[2], par[3])
        return st[0].astype(int), st[1], Z, (Z - m) / np.sqrt(v)

    branch_cache = {}
    def _node_branches(n, jv, jx):
        key = (n, jv, jx)
        if key not in branch_cache:
            rv, pv, _, zv = _node_innovations(vv, jv, stv)
            rx, px, Zx, zx = _node_innovations(xx, jx, stx)
            J = _oracle_joint_factor_coupling(pv, zv, px, zx, float(rho_vr))
            B = _oracle_fund_branch_multipliers(_lattice_node(vv[0], vv[3], vv[4], jv),
                                                _lattice_node(xx[0], xx[3], xx[4], jx),
                                                Zx, J, zv, zx, weights, float(shift[n]), qy, h)
            cv = np.repeat(jv + rv, 3).repeat(3)
            cx = np.tile(jx + rx, 3).repeat(3)
            branch_cache[key] = (B, cv, cx)
        return branch_cache[key]

    def _net_value(prem, nfund):
        F = np.linspace(0.0, float(fund_max), nfund)
        nodes = [(jv, jx) for jv in sorted(reach_v[N]) for jx in sorted(reach_x[N])]
        pos = {kk: i for i, kk in enumerate(nodes)}
        U = np.repeat(np.maximum(F, G_mat[T])[None, :], len(nodes), axis=0)

        for n in range(N - 1, -1, -1):
            closes = ((n + 1) % Nyr == 0)
            i_next = (n + 1) // Nyr
            qd = float(death[i_next - 1]) if closes else 0.0
            gd = float(G_mat[i_next]) if closes else 0.0
            anniv = (n % Nyr == 0)
            i = n // Nyr
            pays = anniv and i <= T - 1
            surr = anniv and 1 <= i <= T - 1

            new_nodes = [(jv, jx) for jv in sorted(reach_v[n]) for jx in sorted(reach_x[n])]
            grid = np.array([0.0]) if n == 0 else F
            out = np.zeros((len(new_nodes), grid.size))
            for r, (jv, jx) in enumerate(new_nodes):
                B, cv, cx = _node_branches(n, jv, jx)
                ci = np.array([pos[(int(a), int(b))] for a, b in zip(cv, cx)], dtype=int)
                val = _oracle_backward_step_with_obstacle(
                    F, U, ci, B, D if pays else 0.0, prem if pays else 0.0,
                    qd, gd, float(G_sur[i]) if surr else 0.0, a_s, surr)
                if n == 0:
                    out[r, 0] = float(val[0])
                else:
                    out[r] = val
            U, nodes, pos = out, new_nodes, {kk: i2 for i2, kk in enumerate(new_nodes)}
        return float(U[pos[(0, 0)], 0])

    def _solve_premium(nfund):
        return brentq(lambda p: _net_value(p, nfund), 1.0, 10.0 * max(D, 1.0),
                      xtol=1e-10, rtol=1e-14)

    coarse = _solve_premium(nF)
    fine = _solve_premium(2 * nF - 1)
    return float((4.0 * fine - coarse) / 3.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = """import numpy as np
def _disc(u):
    return np.exp(-(0.03 * u - 0.02 * (1.0 - np.exp(-0.5 * u))))
def curve(T, Nyr):
    return np.array([_disc((n + 1) / Nyr) for n in range(T * Nyr)])
md34 = curve(3, 4)
md24 = curve(2, 4)
md14 = curve(1, 4)
md32 = curve(3, 2)
BASE = dict(n_years=3, steps_per_year=4, contribution=100.0, g_maturity=0.02,
            g_surrender=0.01, alpha_s=0.95, q=0.015, issue_age=50,
            mk_a=5e-4, mk_b=1e-5, mk_c=1.10,
            v0=0.04, kappa_v=2.0, theta_v=0.04, sigma_v=0.30,
            x0=0.018, kappa_r=0.45, theta_r=0.025, sigma_r=0.10,
            rho_sv=-0.70, rho_sr=-0.20, rho_vr=0.02, lam_v=1.0, lam_x=1.4,
            market_disc=md34, n_fund=2001, fund_max=1500.0)
A_base = dict(BASE)
A_two_year = dict(BASE, n_years=2, market_disc=md24)
A_no_dividend = dict(BASE, q=0.0)
A_one_year = dict(BASE, n_years=1, market_disc=md14)
A_coarse_time = dict(BASE, steps_per_year=2, market_disc=md32)
A_equal_rates = dict(BASE, g_surrender=0.02, alpha_s=1.0)
"""
    err = pre + """
def run_model(**kw):
    a = dict(BASE); a.update(kw)
    try:
        fair_annual_premium(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(BASE); a.update(kw)
    try:
        _oracle_fair_annual_premium(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        # --- normal: the contract instance ---
        {"setup": pre, "call": "fair_annual_premium(**A_base)",
         "gold_call": "_oracle_fair_annual_premium(**A_base)"},
        # --- normal: two-year horizon ---
        {"setup": pre, "call": "fair_annual_premium(**A_two_year)",
         "gold_call": "_oracle_fair_annual_premium(**A_two_year)"},
        # --- normal: no dividend yield ---
        {"setup": pre, "call": "fair_annual_premium(**A_no_dividend)",
         "gold_call": "_oracle_fair_annual_premium(**A_no_dividend)"},
        # --- boundary: single policy year, surrender never available ---
        {"setup": pre, "call": "fair_annual_premium(**A_one_year)",
         "gold_call": "_oracle_fair_annual_premium(**A_one_year)"},
        # --- boundary: coarser time grid ---
        {"setup": pre, "call": "fair_annual_premium(**A_coarse_time)",
         "gold_call": "_oracle_fair_annual_premium(**A_coarse_time)"},
        # --- boundary: equal guarantee rates and full surrender fraction ---
        {"setup": pre, "call": "fair_annual_premium(**A_equal_rates)",
         "gold_call": "_oracle_fair_annual_premium(**A_equal_rates)"},
        # --- edge: curve length inconsistent with the time grid ---
        {"setup": err, "call": "run_model(market_disc=md24)",
         "gold_call": "run_gold(market_disc=md24)"},
        # --- edge: fund grid too coarse to represent a function ---
        {"setup": err, "call": "run_model(n_fund=2)", "gold_call": "run_gold(n_fund=2)"},
        # --- edge: non-positive fund cap ---
        {"setup": err, "call": "run_model(fund_max=0.0)", "gold_call": "run_gold(fund_max=0.0)"},
    ]
