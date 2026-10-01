"""
The deterministic state supplies the settled numbers; the explorer numbers follow from the exact explorer balance, which is linear in the explorers,

[diag(D_a q + gamma_a + lambda_a) - D_a W^T] X_a = C_a^T (c_a * S_a),

because every attempt consumes an explorer whether or not it succeeds. The state vector is ordered as the settled numbers of species 1 to S, patch by patch, followed by the explorer numbers in the same order.

The occupancy dynamics describe the limit of infinitely many sites. With finite site numbers the landscape is a continuous-time Markov chain on the numbers of settled individuals S_ai and of explorers X_ai of every species, and its fluctuations about the deterministic steady state are what the finite size produces. To leading order in the inverse site numbers, with every M_i grown in its stated proportion, the chain is described by its drift and by the covariance of its jumps, both evaluated at the deterministic state and both written in numbers of individuals.

Every elementary reaction r has a propensity a_r and a jump vector nu_r. For species a in patch i:

- death S_ai -> empty site, rate e_ai S_ai, jump -1 in S_ai;
- release of an explorer into patch j, rate c_ai C_a,ij S_ai, jump +1 in X_aj;
- movement to patch j, rate D_a w_ij X_ai, jumps -1 in X_ai and +1 in X_aj;
- explorer loss, rate gamma_a X_ai, jump -1 in X_ai;
- successful colonisation, rate lambda_a X_ai (M_i - sum_b S_bi) / M_i, jumps -1 in X_ai and +1 in S_ai;
- an attempt on a held site, rate lambda_a X_ai sum_b S_bi / M_i, jump -1 in X_ai.

The drift is A = sum_r nu_r a_r and the jump covariance is B = sum_r nu_r nu_r^T a_r. Two features of B are easy to lose and both move the fluctuations: a successful colonisation moves one explorer into the settled class, so it contributes a negative covariance between X_ai and S_ai; and a movement removes an explorer from one patch and adds it to another, so it contributes negative covariances between the explorer numbers of linked patches. The Jacobian of the drift carries the competition: settled individuals of any species reduce the success of every species' explorers in the same patch.

Returns
-------
dict, the linearised chain at the deterministic state, keyed by explorers (S by N explorer numbers from the exact balance), jacobian and jump_covariance (2SN by 2SN), and drift_residual (the largest modulus of the explorer balance residual).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def chain_linearisation(
    weights: np.ndarray,
    sites: np.ndarray,
    settled: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
) -> dict:
    """Build the drift Jacobian and the jump covariance of the settled-and-explorer chain at a deterministic state.

    Parameters
    ----------
    weights : np.ndarray
        Link weights w_ij.
    sites : np.ndarray
        Site numbers M_i.
    settled : np.ndarray
        Deterministic settled numbers, one row per species.
    fecundity : np.ndarray
        Fecundities, one row per species.
    extinction : np.ndarray
        Extinction rates, one row per species.
    max_explorability : np.ndarray
        Maximal explorability of each species.
    exploration_rate : np.ndarray
        Explorer movement rate of each species.
    colonisation_rate : np.ndarray
        Colonisation attempt rate of each species.
    explorer_death_rate : np.ndarray
        Explorer death rate of each species.

    Returns
    -------
    dict
        Under the keys explorers, jacobian, jump_covariance and drift_residual.

    Raises
    ------
    ValueError
        When the network, the site numbers, the settled numbers or any species rate is invalid, or when the arrays disagree in size.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_network(weights, sites):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("weights must be a square array of at least two patches")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(np.diag(w) != 0.0):
        raise ValueError("weights must be finite, non-negative and zero on the diagonal")
    m = np.asarray(sites, dtype=float)
    if m.shape != (w.shape[0],) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return w, m


def _species_rates(n_species, *rates, allow_zero_last=True):
    out = []
    for k, r in enumerate(rates):
        arr = np.asarray(r, dtype=float)
        floor_ok = (k == len(rates) - 1) and allow_zero_last
        if arr.shape != (n_species,) or not np.all(np.isfinite(arr)) or (np.any(arr < 0.0) if floor_ok else np.any(arr <= 0.0)):
            raise ValueError("species rates must be finite, one per species, and above zero (gamma not below zero)")
        out.append(arr)
    return out


def _oracle_chain_linearisation(
    weights: np.ndarray,
    sites: np.ndarray,
    settled: np.ndarray,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
) -> dict:
    """Reference implementation."""
    w, m = _check_network(weights, sites)
    n = m.size
    s_mat = np.asarray(settled, dtype=float)
    if s_mat.ndim != 2 or s_mat.shape[1] != n or s_mat.shape[0] < 1:
        raise ValueError("settled must be an S by N array")
    s = s_mat.shape[0]
    if not np.all(np.isfinite(s_mat)) or np.any(s_mat < 0.0) or np.any(s_mat.sum(axis=0) > m * (1.0 + 1e-12)):
        raise ValueError("settled numbers must be finite, not below zero and fit their sites")
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    for arr in (c, e):
        if arr.shape != (s, n) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("fecundity and extinction must be finite, above zero and S by N")
    xi, d, lam, gamma = _species_rates(s, max_explorability, exploration_rate, colonisation_rate, explorer_death_rate)

    q = w.sum(axis=1)
    occupied = s_mat.sum(axis=0)
    free = 1.0 - occupied / m
    dim = 2 * s * n
    jac = np.zeros((dim, dim))
    cov = np.zeros((dim, dim))
    explorers = np.zeros((s, n))
    residual = 0.0

    def _sl(block, a):
        start = (block * s + a) * n
        return slice(start, start + n)

    for a in range(s):
        feasibility = _oracle_effective_colonisation_kernel(  # noqa: F821
            w, m, xi[a], d[a], lam[a], gamma[a])["release_feasibility"]
        outflow = d[a] * q + gamma[a] + lam[a]
        operator = np.diag(outflow) - d[a] * w.T
        births = feasibility.T @ (c[a] * s_mat[a])
        x = np.linalg.solve(operator, births)
        explorers[a] = x
        residual = max(residual, float(np.max(np.abs(operator @ x - births))))
        success = lam[a] * x * free

        for b in range(s):
            block = -np.diag(lam[a] * x / m)
            if a == b:
                block = block - np.diag(e[a])
            jac[_sl(0, a), _sl(0, b)] = block
        jac[_sl(0, a), _sl(1, a)] = np.diag(lam[a] * free)
        jac[_sl(1, a), _sl(0, a)] = feasibility.T * c[a][None, :]
        jac[_sl(1, a), _sl(1, a)] = d[a] * w.T - np.diag(outflow)

        cov[_sl(0, a), _sl(0, a)] = np.diag(e[a] * s_mat[a] + success)
        cov[_sl(0, a), _sl(1, a)] = np.diag(-success)
        cov[_sl(1, a), _sl(0, a)] = np.diag(-success)
        moves = w * x[:, None]
        xx = -d[a] * (moves + moves.T)
        xx[np.diag_indices(n)] = births + d[a] * q * x + d[a] * (w.T @ x) + (gamma[a] + lam[a]) * x
        cov[_sl(1, a), _sl(1, a)] = xx

    return {
        "explorers": explorers,
        "jacobian": jac,
        "jump_covariance": cov,
        "drift_residual": residual,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
        if isinstance(x, (tuple, list)):
            out = []
            for v in x:
                out.extend(flat(v))
            return tuple(out)
        if hasattr(x, "tolist"):
            return flat(x.tolist())
        if isinstance(x, bool):
            return (int(x),)
        return (x,)
    """

    SETUP = """
    import numpy as np
    rng = np.random.default_rng(88)
    n = 5
    W = rng.uniform(0.2, 1.2, (n, n)) * (rng.random((n, n)) < 0.6)
    np.fill_diagonal(W, 0.0)
    for i in range(n):
        W[i, (i + 1) % n] = max(W[i, (i + 1) % n], 0.3)
    M = rng.integers(40, 200, n).astype(float)
    S = np.array([0.3 * M * rng.uniform(0.5, 1.0, n), 0.35 * M * rng.uniform(0.5, 1.0, n)])
    C = rng.uniform(0.6, 1.4, (2, n)); E = rng.uniform(0.1, 0.4, (2, n))
    XI = np.array([0.8, 0.7]); D = np.array([1.6, 2.5]); LAM = np.array([1.0, 1.3]); GAM = np.array([0.25, 0.0])
    def digest(out):
        return (np.round(out["explorers"], 8), np.round(out["jacobian"], 9), np.round(out["jump_covariance"], 8),
                int(out["drift_residual"] < 1e-9))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # two species with different explorer rates on a seeded directed network, one lossless explorer
            "setup": SETUP + FLAT,
            "call": "flat(digest(chain_linearisation(W, M, S, C, E, XI, D, LAM, GAM)))",
            "gold_call": "flat(digest(_oracle_chain_linearisation(W, M, S, C, E, XI, D, LAM, GAM)))",
        },
        {
            # the Jacobian must equal central differences of the drift written from the reactions, and the jump
            # covariance must equal the sum over reactions of nu nu^T times the propensity, both built here
            # reaction by reaction
            "setup": SETUP + """
def reactions(state, s=2):
    St = state[:s * n].reshape(s, n); Xt = state[s * n:].reshape(s, n)
    occ = St.sum(axis=0)
    out = []
    for a in range(s):
        feas = XI[a] * W / (1.0 + LAM[a] / D[a])
        for i in range(n):
            v = np.zeros(2 * s * n); v[a * n + i] = -1; out.append((v, E[a, i] * St[a, i]))
            for j in range(n):
                if feas[i, j] > 0:
                    v = np.zeros(2 * s * n); v[(s + a) * n + j] = 1; out.append((v, C[a, i] * feas[i, j] * St[a, i]))
                if W[i, j] > 0:
                    v = np.zeros(2 * s * n); v[(s + a) * n + i] = -1; v[(s + a) * n + j] = 1; out.append((v, D[a] * W[i, j] * Xt[a, i]))
            v = np.zeros(2 * s * n); v[(s + a) * n + i] = -1; out.append((v, GAM[a] * Xt[a, i]))
            v = np.zeros(2 * s * n); v[(s + a) * n + i] = -1; v[a * n + i] = 1; out.append((v, LAM[a] * Xt[a, i] * (M[i] - occ[i]) / M[i]))
            v = np.zeros(2 * s * n); v[(s + a) * n + i] = -1; out.append((v, LAM[a] * Xt[a, i] * occ[i] / M[i]))
    return out
def check(fn):
    out = fn(W, M, S, C, E, XI, D, LAM, GAM)
    y = np.concatenate([S.ravel(), out["explorers"].ravel()])
    drift = lambda z: sum(v * r for v, r in reactions(z))
    fd = np.zeros((y.size, y.size))
    for k in range(y.size):
        h = np.zeros(y.size); h[k] = 1e-5
        fd[:, k] = (drift(y + h) - drift(y - h)) / 2e-5
    cov = sum(np.outer(v, v) * r for v, r in reactions(y))
    xdrift = drift(y)[2 * n:]
    return (int(float(np.max(np.abs(fd - out["jacobian"]))) < 1e-6), int(float(np.max(np.abs(cov - out["jump_covariance"]))) < 1e-9),
            int(float(np.max(np.abs(xdrift))) < 1e-9))
""" + FLAT,
            "call": "flat(check(chain_linearisation))",
            "gold_call": "flat(check(_oracle_chain_linearisation))",
        },
        {
            # boundary: a single species with no settled individuals has no explorers, a jump covariance that
            # vanishes identically, and a Jacobian whose settled block is -diag(e)
            "setup": SETUP + """
def empty(fn):
    out = fn(W, M, np.zeros((1, n)), C[:1], E[:1], XI[:1], D[:1], LAM[:1], GAM[:1])
    return (float(np.max(np.abs(out["explorers"]))), float(np.max(np.abs(out["jump_covariance"]))),
            np.round(np.diag(out["jacobian"])[:n] + E[0], 12))
""" + FLAT,
            "call": "flat(empty(chain_linearisation))",
            "gold_call": "flat(empty(_oracle_chain_linearisation))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(weights=W, sites=M, settled=S, fecundity=C, extinction=E, max_explorability=XI, exploration_rate=D,
                colonisation_rate=LAM, explorer_death_rate=GAM)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "(verdict(chain_linearisation, settled=np.array([M, M])), verdict(chain_linearisation, settled=-S), verdict(chain_linearisation, fecundity=C[:1]), verdict(chain_linearisation, exploration_rate=D[:1]), verdict(chain_linearisation, colonisation_rate=np.array([1.0, 0.0])), verdict(chain_linearisation, explorer_death_rate=np.array([0.25, -0.1])), verdict(chain_linearisation, sites=M[:4]), verdict(chain_linearisation))",
            "gold_call": "(verdict(_oracle_chain_linearisation, settled=np.array([M, M])), verdict(_oracle_chain_linearisation, settled=-S), verdict(_oracle_chain_linearisation, fecundity=C[:1]), verdict(_oracle_chain_linearisation, exploration_rate=D[:1]), verdict(_oracle_chain_linearisation, colonisation_rate=np.array([1.0, 0.0])), verdict(_oracle_chain_linearisation, explorer_death_rate=np.array([0.25, -0.1])), verdict(_oracle_chain_linearisation, sites=M[:4]), verdict(_oracle_chain_linearisation))",
        },
    ]
