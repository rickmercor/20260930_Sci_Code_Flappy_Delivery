"""
Build the observation the reconstruction starts from: the synthetic option-price profile at today's date on a uniform price grid, and the corrupted copy that stands in for market data. Return both, packed end to end, so that later steps and tests can compare them.

The clean profile is not a closed-form price. It is manufactured the way the source paper manufactures the data for its numerical experiments: a butterfly-spread payoff is imposed at the horizon, and the ordinary terminal-value pricing problem, carrying the same state-dependent volatility that the reconstruction uses later, is marched back to today with the explicit finite-difference scheme of the paper's data-generation section. The stencil, the instant at which the coefficients are frozen within a step, the rule that fixes the number of time steps and the way the two endpoint values are filled are all the paper's, and none of them is supplied here. Each one moves the profile by more than the downstream tolerance, so a profile produced by a more accurate solver of the same equation is not a better answer to this step, it is a different one.

The corruption follows the rule the source paper uses to generate noisy data in its numerical experiments. How the noise level enters, and what it scales against, is the paper's convention and it is load bearing: option prices across a strike range span orders of magnitude, so a rule that treats the whole grid on one footing and a rule that treats each point on its own give perturbations that differ by orders of magnitude in the wings. Where the rule is applied matters as much as its form, because the profile is projected onto a small basis immediately afterwards and a perturbation applied before that projection is not the same object as one applied after it.

The draws are consumed once, in grid order, from a single seeded generator. Drawing them in a different order, or drawing more than the grid needs and slicing, gives a different perturbation and therefore a different reconstruction, so the order is part of the specification rather than an implementation detail.

An inverse problem needs data that are wrong in a controlled way, otherwise the regularisation has nothing to regularise and the exercise measures only discretisation error. The standard construction is to take a profile that is known exactly, or produced by a stated procedure, and perturb it at a prescribed level, so the reconstruction can be compared against something and the noise level can be swept.

Producing the clean profile with a solver rather than a formula has a purpose. A closed-form price exists only for flat volatility, and a reconstruction that assumes a volatility smile but is fed flat-volatility data is solving a problem that is inconsistent with its own model before any noise is added. Marching the pricing equation itself back from a known payoff keeps data and model consistent, at the price of making the data a property of the scheme: an explicit method for a parabolic equation is only conditionally stable, the first-order drift term admits more than one reasonable difference, and the interior stencil never reaches the grid ends, so some closing rule has to supply them. Studies of this kind fix all three and state them.

How the perturbation is formed is a second modelling choice with real consequences. Option prices span orders of magnitude across the price interval, and a perturbation whose size is set once for the whole grid behaves completely differently from one whose size follows the value it disturbs. There is also the stage at which the corruption is applied: data can be perturbed in the space they were measured in, or after they have been reduced to a handful of coefficients, and projection is a smoothing operator, so the two are not interchangeable.

The order in which pseudo-random draws are consumed is part of a deterministic specification. A modern generator is a deterministic function of its seed and the sequence of calls made against it, so two implementations agree only if they make the same calls in the same order.

Returns
-------
np.ndarray of shape (2*(ns+1),). The first ns+1 entries are the clean synthetic profile at t = 0 on the uniform grid from 0 to smax, the next ns+1 the same profile after corruption.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def observed_price_profile(ns, smax, strikes, T, sigma0, eta, sref, r, delta,
                           seed):
    """Synthetic and noise-corrupted option-price profile on a uniform grid.

    The clean profile is the value at t = 0 of a butterfly spread whose
    payoff at t = T is (S - K1)^+ - 2 (S - K2)^+ + (S - K3)^+, computed on the
    grid S_i = i*smax/ns, i = 0..ns, by the explicit finite-difference
    data-generation scheme of the source paper, with that scheme's time-step
    rule and its treatment of the two endpoint values, under the local
    volatility sigma(t,S) = sigma0*sqrt(1 + eta*exp(-t/T)*((S - sref)/sref)**2)
    and rate r.  The noisy profile applies the paper's data-generation noise
    rule at level delta, using the draws of a single seeded generator consumed
    in grid order.

    Args:
        ns (int): number of grid intervals, so ns+1 grid points.
        smax (float): right end of the price grid.
        strikes (np.ndarray): shape (3,), the strikes K1 < K2 < K3.
        T (float): horizon at which the payoff is imposed.
        sigma0 (float): base volatility level.
        eta (float): smile curvature parameter.
        sref (float): reference price of the smile.
        r (float): risk-free rate.
        delta (float): noise level of the paper's data-generation rule.
        seed (int): seed of the generator supplying the draws.

    Expected return:
        np.ndarray of shape (2*(ns+1),).  The first ns+1 entries are the
        clean profile at t = 0, the next ns+1 the corrupted profile.

    Raises:
        ValueError: if ns < 2, if smax <= 0, if strikes is not three strictly
        increasing positive values, if T <= 0, if sigma0 <= 0, if sref <= 0,
        if eta < 0, or if delta < 0.
    """
    return np.zeros(2 * (ns + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_observed_price_profile(ns, smax, strikes, T, sigma0, eta, sref, r,
                                   delta, seed):
    def _vol(t, s, sig0, curv, ref, hor):
        return sig0 * np.sqrt(1.0 + curv * np.exp(-t / hor)
                              * ((s - ref) / ref) ** 2)

    def _march(grid, payoff, hor, sig0, curv, ref, rate):
        nt = max(1, int(round(hor / 5.0e-4)))
        dt = hor / nt
        ds = grid[1] - grid[0]
        s_in = grid[1:-1]
        u = payoff.copy()
        for n in range(nt, 0, -1):
            sig = _vol(n * dt, s_in, sig0, curv, ref, hor)
            new = np.empty_like(u)
            new[1:-1] = (u[1:-1]
                         + dt * 0.5 * sig ** 2 * s_in ** 2
                         * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / ds ** 2
                         + dt * rate * s_in * (u[2:] - u[1:-1]) / ds
                         - rate * dt * u[1:-1])
            new[0] = 2.0 * new[1] - new[2]
            new[-1] = 2.0 * new[-2] - new[-3]
            u = new
        return u

    ns = int(ns)
    smax = float(smax)
    K = np.asarray(strikes, dtype=float).reshape(-1)
    T = float(T)
    sigma0 = float(sigma0)
    eta = float(eta)
    sref = float(sref)
    r = float(r)
    delta = float(delta)
    if ns < 2:
        raise ValueError("ns must be at least 2")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    if K.size != 3 or not (0.0 < K[0] < K[1] < K[2]):
        raise ValueError("strikes must be three increasing positive values")
    if T <= 0.0 or sigma0 <= 0.0 or sref <= 0.0:
        raise ValueError("T, sigma0 and sref must be positive")
    if eta < 0.0 or delta < 0.0:
        raise ValueError("eta and delta must be non-negative")
    grid = np.linspace(0.0, smax, ns + 1)
    payoff = (np.maximum(grid - K[0], 0.0) - 2.0 * np.maximum(grid - K[1], 0.0)
              + np.maximum(grid - K[2], 0.0))
    clean = _march(grid, payoff, T, sigma0, eta, sref, r)
    xi = np.random.default_rng(int(seed)).uniform(-1.0, 1.0, size=ns + 1)
    noisy = clean * (1.0 + delta * xi)
    return np.concatenate((clean, noisy))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "pass",
            "call": "observed_price_profile(100, 10.0, np.array([3.0, 5.0, 7.0]), 1.5, 0.2, 0.25, 5.0, 0.05, 0.10, 2026)[[0, 30, 50, 70, 100]]",
            "gold_call": "_oracle_observed_price_profile(100, 10.0, np.array([3.0, 5.0, 7.0]), 1.5, 0.2, 0.25, 5.0, 0.05, 0.10, 2026)[[0, 30, 50, 70, 100]]",
        },
        {
            "setup": "pass",
            "call": "observed_price_profile(100, 10.0, np.array([3.0, 5.0, 7.0]), 1.5, 0.2, 0.25, 5.0, 0.05, 0.10, 2026)[[131, 151, 171, 201]]",
            "gold_call": "_oracle_observed_price_profile(100, 10.0, np.array([3.0, 5.0, 7.0]), 1.5, 0.2, 0.25, 5.0, 0.05, 0.10, 2026)[[131, 151, 171, 201]]",
        },
        {
            "setup": "pass",
            "call": "observed_price_profile(40, 10.0, np.array([3.0, 5.0, 7.0]), 1.0, 0.2, 0.25, 5.0, 0.05, 0.0, 7)",
            "gold_call": "_oracle_observed_price_profile(40, 10.0, np.array([3.0, 5.0, 7.0]), 1.0, 0.2, 0.25, 5.0, 0.05, 0.0, 7)",
        },
        {
            "setup": "pass",
            "call": "observed_price_profile(20, 10.0, np.array([2.0, 4.0, 6.0]), 0.01, 0.2, 0.0, 5.0, 0.05, 0.10, 3)",
            "gold_call": "_oracle_observed_price_profile(20, 10.0, np.array([2.0, 4.0, 6.0]), 0.01, 0.2, 0.0, 5.0, 0.05, 0.10, 3)",
        },
        {
            "setup": "pass",
            "call": "observed_price_profile(50, 12.0, np.array([3.0, 6.0, 9.0]), 0.5, 0.25, 0.4, 6.0, 0.03, 0.05, 11)",
            "gold_call": "_oracle_observed_price_profile(50, 12.0, np.array([3.0, 6.0, 9.0]), 0.5, 0.25, 0.4, 6.0, 0.03, 0.05, 11)",
        },
        {
            "setup": "def run_model():\n    try:\n        observed_price_profile(1, 10.0, np.array([3.0, 5.0, 7.0]), 1.0, 0.2, 0.25, 5.0, 0.05, 0.1, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_observed_price_profile(1, 10.0, np.array([3.0, 5.0, 7.0]), 1.0, 0.2, 0.25, 5.0, 0.05, 0.1, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        observed_price_profile(10, 10.0, np.array([5.0, 3.0, 7.0]), 1.0, 0.2, 0.25, 5.0, 0.05, 0.1, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_observed_price_profile(10, 10.0, np.array([5.0, 3.0, 7.0]), 1.0, 0.2, 0.25, 5.0, 0.05, 0.1, 1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
