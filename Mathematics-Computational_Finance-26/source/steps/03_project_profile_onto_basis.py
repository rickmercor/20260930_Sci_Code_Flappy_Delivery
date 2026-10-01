"""
Project a profile sampled on a price grid onto the reduced basis, returning its Legendre coefficients. This is the step that turns a hundred-and-one grid values into nine numbers, and it is where the noise the previous step injected stops being pointwise and becomes a perturbation of the reduced state.

The coefficient of a mode is the integral of the profile against that mode over the price interval, with no weight function in the integrand and nothing to invert afterwards. Whether the vector that comes out has the same length as the function it represents depends on how the basis was scaled upstream, which is settled in the step that builds it and not here; this step simply integrates against whatever table it is given.

The integral is evaluated by the composite trapezoidal rule on the grid that carries the data, not by a high-order rule. This is deliberate and it is not a compromise: the data are only known at those points, so a rule that samples elsewhere would be integrating an interpolant rather than the data, and the interpolation error would enter the reconstruction as an unmodelled perturbation on top of the noise the experiment prescribes.

The projection is where the reduction does its stabilising work. High-frequency content in the noisy profile has small inner products against the first few modes and is simply not represented in the output, which is exactly the content that the unstable forward evolution would otherwise amplify.

Projecting onto a truncated orthogonal basis is the cleanest form of spectral cutoff. The projection is the best approximation to the profile from the span of the retained modes, and the part that is discarded is precisely the part with no component along those modes. When the retained modes are the smoothest ones, as they are for a low-order Legendre truncation, the discarded part is the rough part.

That is why the reduction is a stabiliser and not just an economy. An ill-posed evolution amplifies components at a rate that grows with their frequency, so the damage a perturbation does is concentrated in its roughest part. Removing that part before the evolution starts bounds the amplification by whatever the highest retained mode can do, and the truncation level becomes a regularisation parameter in its own right, acting alongside the penalty weight rather than instead of it.

The choice of quadrature for a projection is constrained by what is actually known. When the integrand is an analytic function, a Gauss rule is the obvious choice and converges very fast. When the integrand is data, sampled at fixed points and carrying noise, the sampled points are the only honest information available, and a composite rule on exactly those points is the faithful choice. Using a higher-order rule on interpolated data does not recover accuracy that the samples never contained.

Returns
-------
np.ndarray of shape (N+1,), the Legendre coefficients of the profile, coefficient m being the trapezoidal approximation to the integral of the profile against mode m over the grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def project_profile_onto_basis(S, profile, N, smax):
    """Legendre coefficients of a profile sampled on a price grid.

    Coefficient m is the integral of profile(S) * ell_m(S) over the price
    interval, evaluated by the composite trapezoidal rule on the supplied
    grid, with ell_m the reduction basis of the source paper.

    Args:
        S (np.ndarray): grid points, increasing, spanning the interval.
        profile (np.ndarray): profile values at those points.
        N (int): highest mode index, so modes n = 0..N.
        smax (float): right end of the price interval.

    Expected return:
        np.ndarray of shape (N+1,), the Legendre coefficients.

    Raises:
        ValueError: if S and profile have different lengths, if fewer than
        two grid points are supplied, if N < 0, or if smax <= 0.
    """
    return np.zeros(N + 1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_project_profile_onto_basis(S, profile, N, smax):
    def _values(nmax, pts, right):
        x = 2.0 * pts / right - 1.0
        P = np.zeros((nmax + 1, x.size))
        P[0] = 1.0
        if nmax >= 1:
            P[1] = x
        for n in range(1, nmax):
            P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
        nrm = np.sqrt((2.0 * np.arange(nmax + 1) + 1.0) / right).reshape(-1, 1)
        return nrm * P

    S = np.asarray(S, dtype=float).reshape(-1)
    profile = np.asarray(profile, dtype=float).reshape(-1)
    N = int(N)
    smax = float(smax)
    if S.size != profile.size:
        raise ValueError("S and profile must have the same length")
    if S.size < 2:
        raise ValueError("at least two grid points are required")
    if N < 0:
        raise ValueError("N must be non-negative")
    if smax <= 0.0:
        raise ValueError("smax must be positive")
    L = _values(N, S, smax)
    integ = L * profile.reshape(1, -1)
    dS = (S[1:] - S[:-1]).reshape(1, -1)
    return np.sum(0.5 * (integ[:, 1:] + integ[:, :-1]) * dS, axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "S = np.linspace(0.0, 10.0, 101)",
            "call": "project_profile_onto_basis(S, np.maximum(S - 5.0, 0.0), 4, 10.0)",
            "gold_call": "_oracle_project_profile_onto_basis(S, np.maximum(S - 5.0, 0.0), 4, 10.0)",
        },
        {
            "setup": "S = np.linspace(0.0, 10.0, 101)",
            "call": "project_profile_onto_basis(S, np.ones(101), 2, 10.0)",
            "gold_call": "_oracle_project_profile_onto_basis(S, np.ones(101), 2, 10.0)",
        },
        {
            "setup": "S = np.linspace(0.0, 6.0, 2001)\nx = 2.0 * S / 6.0 - 1.0\nl3 = np.sqrt(7.0 / 6.0) * (5.0 * x ** 3 - 3.0 * x) / 2.0",
            "call": "project_profile_onto_basis(S, l3, 4, 6.0)",
            "gold_call": "_oracle_project_profile_onto_basis(S, l3, 4, 6.0)",
        },
        {
            "setup": "S = np.linspace(0.0, 10.0, 51)\ndef run_model():\n    a = project_profile_onto_basis(S, S ** 2, 3, 10.0)\n    b = project_profile_onto_basis(S, 3.0 * S ** 2 + 1.0, 3, 10.0)\n    return b - 3.0 * a\ndef run_gold():\n    a = _oracle_project_profile_onto_basis(S, S ** 2, 3, 10.0)\n    b = _oracle_project_profile_onto_basis(S, 3.0 * S ** 2 + 1.0, 3, 10.0)\n    return b - 3.0 * a",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "pass",
            "call": "project_profile_onto_basis(np.array([0.0, 10.0]), np.array([0.0, 1.0]), 0, 10.0)",
            "gold_call": "_oracle_project_profile_onto_basis(np.array([0.0, 10.0]), np.array([0.0, 1.0]), 0, 10.0)",
        },
        {
            "setup": "S = np.linspace(0.0, 10.0, 11)\ndef run_model():\n    try:\n        project_profile_onto_basis(S, np.ones(5), 3, 10.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_project_profile_onto_basis(S, np.ones(5), 3, 10.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "S = np.linspace(0.0, 10.0, 11)\ndef run_model():\n    try:\n        project_profile_onto_basis(S, np.ones(11), -2, 10.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_project_profile_onto_basis(S, np.ones(11), -2, 10.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
