"""
Find the joint maximum likelihood estimates of population density and aggregation parameter under the right-censored negative-binomial distance model, together with the maximized log-likelihood.

The censored negative-binomial likelihood has no closed-form maximizer, so density and aggregation must be found by bivariate numerical optimization; this estimator was the most accurate across simulated clustered populations and mapped forest plots. Because downstream design calculations are evaluated at the fitted population, the maximizer has to be located to high precision rather than to a loose optimizer tolerance.

Returns
-------
return estimates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nbd_censored_mle(distances: list, C: float, ell: int) -> list:
    """Return [lam_hat, k_hat, max_log_likelihood] for the censored NBD model.
 
    (lam_hat, k_hat) is the interior maximizer over lam > 0 and k > 0 of the log-likelihood
    defined in nbd_censored_log_likelihood, and max_log_likelihood is its value there.
 
    Parameters
    ----------
    distances : sequence of sequences
        n sampling points (rows) by q sectors (columns) of ell-th nearest-neighbour distances;
        None or a value larger than C marks a censored sector.
    C : float
        Maximum search radius, finite and positive.
    ell : int
        Nearest-neighbour order, an integer >= 1.
 
    Returns
    -------
    estimates : list of float
        [lam_hat, k_hat, max_log_likelihood] as native Python floats. lam_hat and k_hat must be
        converged to at least 10 significant digits (the score equations solved to near machine
        precision, e.g. by Newton refinement), not merely to a default optimizer tolerance.
 
    Raises
    ------
    ValueError
        Under the conditions of nbd_censored_log_likelihood, or if the log-likelihood has no
        interior maximum with k <= 1e8 (for example when it keeps increasing as k grows), or if
        the maximization fails to converge.
    """
    return estimates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nbd_censored_mle(distances: list, C: float, ell: int) -> list:
    import math
 
    radius = float(C)
    order = int(float(ell)) if not isinstance(ell, bool) else ell
    # validates every input (raises ValueError for invalid surveys, radius or order)
    _oracle_nbd_censored_log_likelihood(distances, C, ell, 1.0, 1.0)
    rows = [list(row) for row in distances]
    q = len(rows[0])
    observed = []
    censored = 0
    for row in rows:
        for entry in row:
            if entry is None or float(entry) > radius:
                censored += 1
            else:
                observed.append(float(entry))
 
    def _loglik(s, t):
        return _oracle_nbd_censored_log_likelihood(distances, radius, order, math.exp(s), math.exp(t))
 
    def _gradient(s, t):
        lam = math.exp(s)
        k = math.exp(t)
        rate = math.pi * lam / q
        harmonic = math.fsum(1.0 / (k + i) for i in range(order))
        gs = []
        gt = []
        for r in observed:
            z = rate * r * r / k
            frac = z / (1.0 + z)
            gs.append(order - (order + k) * frac)
            gt.append(k * harmonic - order - k * math.log1p(z) + (order + k) * frac)
        if censored:
            mass = rate * radius * radius
            w = mass / (mass + k)
            log_w = math.log(w)
            log_1mw = math.log1p(-w)
            log_terms = [math.lgamma(k + j) - math.lgamma(k) - math.lgamma(j + 1) + j * log_w + k * log_1mw
                         for j in range(order)]
            top = max(log_terms)
            log_survival = top + math.log(math.fsum(math.exp(v - top) for v in log_terms))
            partial = [math.fsum(1.0 / (k + i) for i in range(j)) for j in range(order)]
            dk_fixed_w = math.fsum(math.exp(v - log_survival) * (partial[j] + log_1mw)
                                   for j, v in enumerate(log_terms))
            log_density = ((order - 1) * log_w + (k - 1.0) * log_1mw
                           - (math.lgamma(order) + math.lgamma(k) - math.lgamma(order + k)))
            density_ratio = math.exp(log_density - log_survival) * w * (1.0 - w)
            gs.append(-censored * density_ratio)
            gt.append(censored * (k * dk_fixed_w + density_ratio))
        return [math.fsum(gs), math.fsum(gt)]
 
    try:
        lam0 = _oracle_censored_shen_estimates(distances, radius, order)
        start_lam = lam0[0] if lam0[0] > 0.0 else _oracle_censored_poisson_densities(distances, radius, order)[1]
        start_k = min(max(lam0[1], 0.5), 1e3) if lam0[1] > 0.0 else 1.0
    except ValueError:
        start_lam = 0.0
        start_k = 1.0
    if not start_lam > 0.0:
        mean_sq = math.fsum(r * r for r in observed) / len(observed)
        start_lam = order * q / (math.pi * mean_sq)
 
    s = math.log(start_lam)
    t = math.log(start_k)
    f = _loglik(s, t)
    h = 1e-5
    t_max = math.log(1e8)
    converged = False
    for _ in range(2000):
        g = _gradient(s, t)
        gp = _gradient(s + h, t)
        gm = _gradient(s - h, t)
        hs0 = (gp[0] - gm[0]) / (2.0 * h)
        hs1 = (gp[1] - gm[1]) / (2.0 * h)
        gp = _gradient(s, t + h)
        gm = _gradient(s, t - h)
        ht0 = (gp[0] - gm[0]) / (2.0 * h)
        ht1 = (gp[1] - gm[1]) / (2.0 * h)
        h00 = hs0
        h11 = ht1
        h01 = 0.5 * (hs1 + ht0)
        det = h00 * h11 - h01 * h01
        if h00 < 0.0 and det > 0.0:
            ds = -(h11 * g[0] - h01 * g[1]) / det
            dt = -(-h01 * g[0] + h00 * g[1]) / det
            newton = True
        else:
            norm = max(abs(g[0]), abs(g[1]), 1e-300)
            ds = g[0] / norm
            dt = g[1] / norm
            newton = False
        longest = max(abs(ds), abs(dt))
        if longest > 1.0:
            ds /= longest
            dt /= longest
        if newton and longest < 1e-11:
            converged = True
            break
        step = 1.0
        accepted = False
        while step > 1e-14:
            sn = s + step * ds
            tn = t + step * dt
            try:
                fn = _loglik(sn, tn)
            except ValueError:
                fn = -math.inf
            if fn >= f - 1e-12 * (1.0 + abs(f)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        s, t, f = sn, tn, fn
        if t > t_max:
            raise ValueError("the log-likelihood has no interior maximum with k <= 1e8")
    if not converged:
        g = _gradient(s, t)
        if not (max(abs(g[0]), abs(g[1])) <= 1e-7 * (1.0 + len(observed) + censored)):
            raise ValueError("censored NBD maximization did not converge")
    if t > t_max:
        raise ValueError("the log-likelihood has no interior maximum with k <= 1e8")
    return [float(math.exp(s)), float(math.exp(t)), float(_loglik(s, t))]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
 
    def _expect_value_error(model_expr, gold_expr):
        setup = (
            "def run_model():\n"
            "    try:\n"
            f"        {model_expr}\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            f"        {gold_expr}\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
        )
        return {"setup": setup, "call": "run_model()", "gold_call": "run_gold()"}
 
    return [
        # normal: 20 points, second-nearest distances, 21 of 80 sectors censored
        {"setup": "D = [[9.48, None, None, 9.33], [None, None, 6.39, 9.67], [7.19, 2.46, None, 3.84], [3.67, 7.9, 6.59, 6.66], [5.34, None, 9.26, None], [None, 3.23, 2.28, 3.16], [7.56, 6.13, 6.31, 4.69], [3.5, 6.96, 2.75, 4.17], [5.35, 4.85, 6.22, 6.77], [8.09, 6.74, 5.75, 4.74], [None, 3.72, 5.38, None], [7.15, None, None, None], [None, 4.61, None, 8.5], [7.47, 8.3, None, 4.04], [3.48, 3.11, 3.33, 3.56], [8.04, 5.74, 5.87, 9.85], [6.99, 7.68, None, 7.82], [6.73, 5.58, 6.32, 4.4], [7.8, 8.96, None, None], [None, 4.84, 5.75, None]]",
         "call": "nbd_censored_mle(D, 10.0, 2)",
         "gold_call": "_oracle_nbd_censored_mle(D, 10.0, 2)"},
        # normal: nearest-neighbour distances, 14 of 80 sectors censored
        {"setup": "D = [[9.07, 1.72, 1.47, 4.42], [7.42, 5.36, 3.48, 6.48], [7.38, 5.53, 4.28, 5.0], [4.28, 3.49, 8.01, 5.5], [5.77, 5.72, 3.88, 3.87], [3.47, 7.87, 3.37, 0.99], [4.64, 4.64, 7.73, None], [1.29, 3.82, 2.84, 2.72], [None, None, 3.96, 8.88], [4.65, 5.07, 5.04, 1.96], [None, 2.62, 7.11, None], [None, 5.82, None, 4.06], [None, 9.13, 9.82, 9.21], [None, 2.93, 4.16, 2.51], [2.1, 5.99, 1.59, 0.62], [9.94, None, 1.49, 2.17], [7.59, 1.27, 4.49, None], [None, 6.95, 5.12, 7.45], [6.41, 3.38, 6.09, None], [None, 5.02, 2.46, 8.98]]",
         "call": "nbd_censored_mle(D, 10.0, 1)",
         "gold_call": "_oracle_nbd_censored_mle(D, 10.0, 1)"},
        # edge: third-nearest distances with a 12 m radius
        {"setup": "D = [[11.15, 8.57, 4.46, 4.9], [6.1, 6.53, 7.84, None], [7.16, 7.28, 6.24, 4.78], [None, 5.55, 4.21, 6.08], [5.48, 11.53, None, 8.5], [11.89, 9.65, None, 6.93], [9.9, 10.24, 4.61, 5.58], [10.27, 5.48, 4.87, 10.02], [11.91, 7.5, 9.46, None], [10.28, 9.91, None, None], [4.76, 5.99, 5.58, 5.89], [None, 11.94, 7.17, 5.91], [6.96, 8.77, 5.11, 6.3], [None, None, None, 6.17], [5.96, 5.9, 3.06, 7.24]]",
         "call": "nbd_censored_mle(D, 12.0, 3)",
         "gold_call": "_oracle_nbd_censored_mle(D, 12.0, 3)"},
        # boundary: only 10 sampling points, weakly identified aggregation
        {"setup": "D = [[9.48, None, None, 9.33], [None, None, 6.39, 9.67], [7.19, 2.46, None, 3.84], [3.67, 7.9, 6.59, 6.66], [5.34, None, 9.26, None], [None, 3.23, 2.28, 3.16], [7.56, 6.13, 6.31, 4.69], [3.5, 6.96, 2.75, 4.17], [5.35, 4.85, 6.22, 6.77], [8.09, 6.74, 5.75, 4.74]]",
         "call": "nbd_censored_mle(D, 10.0, 2)",
         "gold_call": "_oracle_nbd_censored_mle(D, 10.0, 2)"},
        # invalid: every sector censored
        _expect_value_error("nbd_censored_mle([[None, None], [12.0, None]], 10.0, 1)",
                           "_oracle_nbd_censored_mle([[None, None], [12.0, None]], 10.0, 1)"),
        # invalid: nearly identical distances, likelihood increases without bound in k
        _expect_value_error("nbd_censored_mle([[5.0, 5.1, 4.9, 5.0], [5.05, 4.95, 5.0, 5.02], [4.98, 5.0, 5.01, 4.97]], 10.0, 1)",
                           "_oracle_nbd_censored_mle([[5.0, 5.1, 4.9, 5.0], [5.05, 4.95, 5.0, 5.02], [4.98, 5.0, 5.01, 4.97]], 10.0, 1)"),
    ]
