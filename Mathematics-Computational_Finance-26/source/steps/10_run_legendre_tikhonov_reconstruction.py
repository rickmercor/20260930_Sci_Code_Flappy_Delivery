"""
Chain the nine earlier steps into the whole reconstruction and return the recovered price at the requested asset value. Build the synthetic observation and corrupt it, project it onto the basis, assemble the convection block once and the diffusion block at each interval midpoint, combine them into the coefficient matrices, then sweep the candidate regularisation weights: for each one assemble and solve the least-squares system and evaluate the two L-curve quantities the selection rule compares. Hand those to the selection step, take the candidate it returns, and expand that solution's terminal coefficients back into a price. This is the orchestrator step: it chains the nine earlier public functions rather than reproducing any of them inline.

The regularisation weight is not supplied. It is chosen from the candidate ladder by the source paper's parameter-choice criterion, which is what makes the answer a property of the method rather than of a number somebody picked. The two quantities are the ones that section defines, a residual quantity and a regularisation norm, and they are evaluated on the solution with the same quadrature and the same difference rules that the assembled system uses for the corresponding terms of the functional, so that each is exactly what the least-squares rows measure rather than a separately discretised approximation of it. A pipeline that skips the sweep and uses a single plausible weight produces a number that looks entirely reasonable and is wrong, because the selected candidate is a discrete choice and the reconstruction jumps between neighbouring candidates rather than drifting.

What the method returns is not the true payoff and is not meant to be. It is the output of a regularised reconstruction at the weight the criterion selects, and every choice in the procedure leaves a fingerprint on the number: how the data were manufactured, how the basis is scaled, which direction the reduced generator runs in, which norm the penalty is taken in, where the noise is applied, and how the corner of the L-curve is located. Asking for the exact payoff would erase all of that, because every reasonable variant converges to the payoff as the noise and the weight go to zero.

The arrangement is free of adaptivity beyond that one selection. The truncation level, the grid counts, the quadrature size, the candidate ladder and the noise seed are all supplied, so nothing depends on a convergence test that could terminate differently on a different machine. The coefficient matrix is evaluated at interval midpoints, not at grid nodes, and the same midpoint value is used for both endpoint contributions of its block.

A reconstruction pipeline for an ill-posed problem has a characteristic shape: generate or accept data, reduce to a manageable representation, assemble the forward operator on that representation, and solve a regularised inverse problem against it. Each stage is individually unremarkable and the difficulty is that errors in any of them are invisible until the end, because a wrong operator still produces a plausible-looking reconstruction rather than a failure.

Adding a parameter-choice rule changes the character of the pipeline. Without one, the answer is a smooth function of everything that goes into it, and a small implementation error produces a small error in the output. With one, the output passes through a discrete decision, and a small error in either L-curve quantity can move the decision to a neighbouring candidate and the answer by far more than the error that caused it. That sensitivity is the price of not having to supply the weight, and it is why the two quantities have to be the ones the method intends rather than any reasonable-looking pair.

That is also why the quantity to report is the procedure's output rather than the truth it approximates. In a well-posed problem one can check an implementation by refining until it converges. Here refinement is not available in the same way: reducing the weight does not converge to the truth, it converges to noise amplification. The configuration and the answer have to be pinned together.

The gap between the reconstruction and the true payoff at the end is the honest cost of the regularisation. A nine-mode polynomial expansion cannot represent a kink, and data carrying ten per cent relative noise do not determine one.

Returns
-------
float, the reconstructed price at sstar. The reconstruction stays bounded by the price interval, and the selected candidate index is always an interior index of the supplied ladder.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_legendre_tikhonov_reconstruction(market, numerics):
    """Chain the nine earlier steps and return the reconstructed price.

    Builds the synthetic observation and corrupts it, projects it onto the
    reduction basis, assembles the reduced convection block once and the
    reduced diffusion block at each of the nt interval midpoints, combines
    them into the reduced coefficient matrices, then for each candidate
    weight alpha_j = 10**(lo + j), j = 0..nalpha-1, assembles and solves the
    Tikhonov system and evaluates the two L-curve quantities of the source
    paper's parameter-choice section for that candidate, discretised with the
    same quadrature and difference rules as the assembled system.  The corner
    of the resulting L-curve selects the candidate, and the terminal
    coefficient vector of that candidate's solution is evaluated against the
    basis at sstar.

    This is the orchestrator step: chain the nine earlier public functions
    rather than reproducing any of them inline.

    Args:
        market: (9,) array [smax, r, sigma0, eta, sref, T, K1, K2, K3].
        numerics: (9,) array
            [N, nt, delta, seed, ns, nq, sstar, lo, nalpha], with the
            candidate ladder alpha_j = 10**(lo + j) for j = 0..nalpha-1.

    Expected return:
        float: the reconstructed price at sstar.

    Raises:
        ValueError: if market and numerics do not both have length 9, if
        smax <= 0, if sstar lies outside [0, smax], or if nalpha < 3.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_legendre_tikhonov_reconstruction(market, numerics):
    def _lcurve_quantities(sol, data, c_blocks, dt, nt, m):
        V = sol.reshape(nt + 1, m)
        res2 = 0.0
        for k in range(nt):
            Ck = c_blocks[k * m * m:(k + 1) * m * m].reshape(m, m)
            d = (V[k + 1] - V[k]) / dt - Ck @ (0.5 * (V[k] + V[k + 1]))
            res2 += dt * float(d @ d)
        e = V[0] - data
        res2 += float(e @ e)
        d1 = (V[1:] - V[:-1]) / dt
        d2 = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dt ** 2
        q2 = dt * (float(np.sum(V * V)) + float(np.sum(d1 * d1))
                   + float(np.sum(d2 * d2)))
        return np.sqrt(res2), np.sqrt(q2)

    market = np.asarray(market, dtype=float).reshape(-1)
    numerics = np.asarray(numerics, dtype=float).reshape(-1)
    if market.size != 9 or numerics.size != 9:
        raise ValueError("market and numerics must both have size 9")
    smax, r, sigma0, eta, sref, T, K1, K2, K3 = market
    N, nt, delta, seed, ns, nq, sstar, lo, nalpha = numerics
    N = int(N)
    nt = int(nt)
    ns = int(ns)
    nq = int(nq)
    seed = int(seed)
    nalpha = int(nalpha)
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if not (0.0 <= sstar <= smax):
        raise ValueError("sstar must lie in [0, smax]")
    if nalpha < 3:
        raise ValueError("at least three candidate weights are required")

    prof = _oracle_observed_price_profile(ns, smax, np.array([K1, K2, K3]), T,
                                          sigma0, eta, sref, r, delta, seed)
    noisy = prof[ns + 1:]
    grid = np.linspace(0.0, smax, ns + 1)
    u0d = _oracle_project_profile_onto_basis(grid, noisy, N, smax)

    b_flat = _oracle_reduced_convection_matrix(N, smax, nq)
    dt = T / nt
    blocks = []
    for k in range(nt):
        a_flat = _oracle_reduced_diffusion_matrix((k + 0.5) * dt, N, smax, nq,
                                                  sigma0, eta, sref, T)
        blocks.append(_oracle_reduced_coefficient_matrix(a_flat, b_flat, r))
    c_stack = np.concatenate(blocks)

    m = N + 1
    ncol = (nt + 1) * m
    alphas = np.array([10.0 ** (lo + 1.0 * j) for j in range(nalpha)])
    sols = []
    resq = np.zeros(nalpha)
    regn = np.zeros(nalpha)
    for j in range(nalpha):
        sys_flat = _oracle_assemble_tikhonov_system(u0d, c_stack, T,
                                                    float(alphas[j]))
        sol = _oracle_solve_tikhonov_system(sys_flat, ncol)
        sols.append(sol)
        resq[j], regn[j] = _lcurve_quantities(sol, u0d, c_stack, dt, nt, m)

    jstar = int(_oracle_lcurve_corner_index(resq, regn))
    vT = sols[jstar].reshape(nt + 1, m)[-1]
    tab = _oracle_shifted_legendre_basis_table(np.array([sstar]), N, smax)
    return float(np.dot(vT, tab[:m]))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    MARKET = "np.array([10.0, 0.05, 0.2, 0.25, 5.0, 1.5, 3.0, 5.0, 7.0])"
    NUMERICS = "np.array([8, 40, 0.10, 2026, 100, 40, 6.25, -9.0, 11])"
    return [
        {
            "setup": "m = " + MARKET + "\nq = " + NUMERICS,
            "call": "run_legendre_tikhonov_reconstruction(m, q)",
            "gold_call": "_oracle_run_legendre_tikhonov_reconstruction(m, q)",
            "tol": 1e-08,
        },
        {
            "setup": "m = " + MARKET + "\nq = " + NUMERICS + "\nq = q.copy()\nq[6] = 5.0",
            "call": "run_legendre_tikhonov_reconstruction(m, q)",
            "gold_call": "_oracle_run_legendre_tikhonov_reconstruction(m, q)",
            "tol": 1e-08,
        },
        {
            "setup": "m = " + MARKET + "\nq = " + NUMERICS + "\nq = q.copy()\nq[2] = 0.0",
            "call": "run_legendre_tikhonov_reconstruction(m, q)",
            "gold_call": "_oracle_run_legendre_tikhonov_reconstruction(m, q)",
            "tol": 1e-08,
        },
        {
            "setup": "m = " + MARKET + "\nq = " + NUMERICS + "\nq13 = q.copy()\nq13[8] = 13",
            "call": "run_legendre_tikhonov_reconstruction(m, q13)",
            "gold_call": "_oracle_run_legendre_tikhonov_reconstruction(m, q)",
            "tol": 1e-08,
        },
        {
            "setup": "m = np.array([10.0, 0.05, 0.2, 0.25, 5.0, 1.0, 2.5, 5.0, 7.5])\nq = np.array([6, 30, 0.05, 17, 100, 30, 4.0, -8.0, 9])",
            "call": "run_legendre_tikhonov_reconstruction(m, q)",
            "gold_call": "_oracle_run_legendre_tikhonov_reconstruction(m, q)",
            "tol": 1e-08,
        },
        {
            "setup": "m = " + MARKET + "\n" + "def run_model():\n    try:\n        run_legendre_tikhonov_reconstruction(m, np.zeros(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_run_legendre_tikhonov_reconstruction(m, np.zeros(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "m = " + MARKET + "\nq = " + NUMERICS + "\nq = q.copy()\nq[6] = 25.0\n" + "def run_model():\n    try:\n        run_legendre_tikhonov_reconstruction(m, q)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_run_legendre_tikhonov_reconstruction(m, q)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
