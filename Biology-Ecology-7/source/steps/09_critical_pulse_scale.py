"""
Determine the unique pulse-strength scale at which the signed normalized permanence margin changes sign. The spray interval and ecological parameters remain fixed, while a single dimensionless scale modifies the logarithmic strength of every multiplicative pulse relative to the reference pulse vector. For every trial scale, independently rebuild the pulsed boundary dynamics, identify the existing periodic boundary orbits, reconstruct the finest Morse decomposition, recompute all pulse-adjusted invasion rates, solve the associated weighted max-min certificates, and evaluate the resulting permanence margin. The Morse decomposition is compared with the reference decomposition at each trial value; a change in decomposition invalidates the supplied fixed-bracket calculation. Locate the zero of the fully recomputed margin within the supplied bracket to the required absolute tolerance.

The permanence criterion for periodically pulsed ecological systems depends on weighted long-term growth rates evaluated on the invariant components of the extinction boundary. Pulse effects enter these rates through logarithmic multiplicative contributions, while simultaneously altering the boundary trajectories on which the continuous ecological growth is evaluated. Therefore, changing pulse strength generally changes both direct pulse contributions and the underlying periodic boundary dynamics. An inverse threshold problem must consequently recompute the complete boundary certificate as the pulse scale varies. A positive permanence margin provides the sufficient permanence condition, while a sign change identifies the critical intervention scale at which this certificate vanishes.

Returns
-------
the unique critical pulse-strength scale q* in the supplied bracket at which the fully recomputed permanence margin is zero, returned as a single finite decimal.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def critical_pulse_scale(params: dict, tau: float, q_lo: float, q_hi: float) -> float:
    """Locate the unique q where the fully recomputed permanence margin crosses zero.

    Parameters
    ----------
    params : dict
        Base model parameters. The entries in ``h`` are the reference pulse effects h_i^0.
    tau : float
        Fixed spray interval (weeks), tau > 0.
    q_lo, q_hi : float
        Positive bracket with opposite signs of the recomputed permanence margin.

    Returns
    -------
    float
        Critical pulse-strength scale q*, located to absolute tolerance 1e-12.

    Raises
    ------
    ValueError
        If inputs are invalid, the bracket has no sign change, the boundary Morse
        decomposition changes inside the bracket, or the assembled chain is internally
        inconsistent.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq


def _s09_chain(params, tau, pieces_ref=None):
    """Assemble the boundary orbits, invasion rates, Morse decomposition,
    and permanence margin using the earlier pipeline oracles."""
    P = _s08_params(params)
    a, B, e, c, om, d, h = P

    # Always rebuild the boundary orbit census and invasion rates for the
    # supplied parameter set.  A reference decomposition may only be used
    # for an independent consistency check; it must never supply the orbit
    # census or rates themselves.
    orbits, rates = _s08_census(tau, P)

    # Cross-check every orbit with the individual earlier-step oracles.
    for key, z in orbits.items():
        T = list(key)

        if len(T) == 1 and T[0] < 3:
            if abs(
                z[T[0]]
                - _oracle_single_pest_orbit(
                    a[T[0]],
                    B[T[0], T[0]],
                    h[T[0]],
                    tau,
                )
            ) > 1e-10:
                raise ValueError(
                    "single-strain orbit disagrees with the closed form"
                )

        if len(T) >= 2:
            for i in T:
                if abs(
                    _oracle_face_orbit_post_pulse(
                        params,
                        T,
                        z[T],
                        tau,
                        i,
                    )
                    - z[i]
                ) > 1e-10:
                    raise ValueError(
                        "face orbit is not reproduced by the shooting step"
                    )

            if (
                _oracle_stroboscopic_floquet_radius(
                    params,
                    T,
                    z,
                    tau,
                )
                >= 1.0
                and 3 in T
            ):
                raise ValueError(
                    "a pest-parasitoid orbit does not attract within its face"
                )

        if len(T) == 1:
            if abs(
                _oracle_stroboscopic_map(
                    params,
                    z,
                    tau,
                    T[0],
                )
                - z[T[0]]
            ) > 1e-10:
                raise ValueError(
                    "single-species orbit is not a fixed point "
                    "of the one-interval map"
                )

        if all(i < 3 for i in T):
            mean = _s09_mean(z, T, tau, P)

            for i in T:
                if abs(
                    _oracle_pest_face_mean_density(
                        params,
                        T,
                        tau,
                        i,
                    )
                    - mean[T.index(i)]
                ) > 1e-9:
                    raise ValueError(
                        "period average on a pest-only orbit "
                        "disagrees with the averages identity"
                    )

        for i in range(4):
            if abs(
                _oracle_pulsed_invasion_rate(
                    params,
                    z,
                    i,
                    tau,
                )
                - rates[key][i]
            ) > 1e-10:
                raise ValueError(
                    "rate along an orbit disagrees with "
                    "the invasion-rate step"
                )

            if i in T and abs(rates[key][i]) > 1e-9:
                raise ValueError(
                    "a resident species must have zero "
                    "long-term growth on its own orbit"
                )

    # Recompute the complete boundary census for this parameter set.
    if float(len(orbits)) != _oracle_boundary_census(params, tau):
        raise ValueError("boundary census mismatch")

    # Build the Morse decomposition for this parameter set unless a
    # reference decomposition was explicitly supplied for validation.
    #
    # IMPORTANT:
    # pieces_ref is never used to calculate the margin.  When supplied,
    # it is only checked against the freshly rebuilt decomposition.
    pieces = _s08_pieces(
        tau,
        P,
        orbits,
        rates,
    )

    if pieces_ref is not None:
        current_set = {
            tuple(piece)
            for piece in pieces
        }

        reference_set = {
            tuple(piece)
            for piece in pieces_ref
        }

        if current_set != reference_set:
            raise ValueError(
                "Morse decomposition changes across the q bracket"
            )

    # Calculate the signed normalized permanence margin from the freshly
    # reconstructed Morse decomposition.
    m = math.inf

    for piece in pieces:
        val, w = _s08_piece_margin(
            piece,
            tau,
            P,
            rates,
        )
        m = min(m, val)

    return float(m), pieces


def _s09_mean(z, T, tau, P):
    """Period averages of present species along a boundary orbit."""
    def _rhs(t, y):
        zz = y[:4]
        return np.concatenate([
            zz * _s08_rates(zz, P),
            zz,
        ])

    sol = solve_ivp(
        _rhs,
        (0.0, tau),
        np.concatenate([z, np.zeros(4)]),
        method="DOP853",
        rtol=1e-13,
        atol=1e-16,
    )

    return sol.y[4:8, -1][T] / tau


def _s09_scaled_params(params, q):
    """Apply h_i(q) = (1 + h_i^0)^q - 1 componentwise."""
    if not isinstance(params, dict) or "h" not in params:
        raise ValueError(
            "params must contain base pulse vector h"
        )

    q = float(q)

    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(
            "q must be finite and positive"
        )

    h0 = np.asarray(
        params["h"],
        dtype=float,
    ).ravel()

    if (
        h0.shape != (4,)
        or not np.all(np.isfinite(h0))
        or np.any(h0 <= -1.0)
    ):
        raise ValueError(
            "base pulse vector must have four finite "
            "entries greater than -1"
        )

    out = dict(params)

    out["h"] = np.expm1(
        q * np.log1p(h0)
    ).tolist()

    return out


def _s09_margin_at_scale(params, tau, q):
    """Recompute the complete permanence certificate at trial q.

    No reference orbit census or Morse decomposition is reused.
    """
    p = _s09_scaled_params(
        params,
        q,
    )

    # _s09_chain independently rebuilds:
    #   1. the boundary orbit census,
    #   2. all invasion rates,
    #   3. the Morse decomposition,
    #   4. the weighted component margins,
    #   5. the global permanence margin.
    m, pieces = _s09_chain(
        p,
        tau,
    )

    return float(m), pieces


def _oracle_critical_pulse_scale(
    params: dict,
    tau: float,
    q_lo: float,
    q_hi: float,
) -> float:
    """Locate the q value where the fully recomputed permanence
    margin crosses zero.
    """
    tau = float(tau)
    q_lo = float(q_lo)
    q_hi = float(q_hi)

    if not (
        math.isfinite(tau)
        and math.isfinite(q_lo)
        and math.isfinite(q_hi)
    ):
        raise ValueError(
            "tau and bracket endpoints must be finite"
        )

    if tau <= 0.0 or not (0.0 < q_lo < q_hi):
        raise ValueError(
            "require tau > 0 and 0 < q_lo < q_hi"
        )

    # ------------------------------------------------------------------
    # Reference decomposition at q = 1.
    #
    # This decomposition is used only as a validation reference.
    # It is NOT reused to calculate margins at other q values.
    # ------------------------------------------------------------------
    p_ref = _s09_scaled_params(
        params,
        1.0,
    )

    m_ref, pieces_ref = _s09_chain(
        p_ref,
        tau,
    )

    # ------------------------------------------------------------------
    # Independently rebuild the full certificate at both endpoints.
    # ------------------------------------------------------------------
    m_lo, pieces_lo = _s09_margin_at_scale(
        params,
        tau,
        q_lo,
    )

    m_hi, pieces_hi = _s09_margin_at_scale(
        params,
        tau,
        q_hi,
    )

    # Canonicalize the decomposition so ordering of the components
    # does not affect the comparison.
    reference_set = {
        tuple(piece)
        for piece in pieces_ref
    }

    lo_set = {
        tuple(piece)
        for piece in pieces_lo
    }

    hi_set = {
        tuple(piece)
        for piece in pieces_hi
    }

    if lo_set != reference_set:
        raise ValueError(
            "Morse decomposition changes between q=1 and q_lo"
        )

    if hi_set != reference_set:
        raise ValueError(
            "Morse decomposition changes between q=1 and q_hi"
        )

    # The supplied bracket must contain a sign change.
    if not (
        math.isfinite(m_lo)
        and math.isfinite(m_hi)
        and m_lo * m_hi < 0.0
    ):
        raise ValueError(
            "q bracket does not enclose a sign change "
            "of the permanence margin"
        )

    # ------------------------------------------------------------------
    # Root function.
    #
    # Every evaluation independently rebuilds the complete ecological
    # certificate.  No cached orbit census, invasion rates, or Morse
    # decomposition from q=1 is reused.
    # ------------------------------------------------------------------
    def _margin_at_q(q):
        m_q, pieces_q = _s09_margin_at_scale(
            params,
            tau,
            q,
        )

        pieces_set = {
            tuple(piece)
            for piece in pieces_q
        }

        if pieces_set != reference_set:
            raise ValueError(
                "Morse decomposition changes across the q bracket"
            )

        if not math.isfinite(m_q):
            raise ValueError(
                "non-finite permanence margin"
            )

        return float(m_q)

    q_star = brentq(
        _margin_at_q,
        q_lo,
        q_hi,
        xtol=1e-12,
        maxiter=200,
    )

    # Verify the returned root using another complete independent
    # evaluation rather than relying on Brent's internal value.
    m_star, pieces_star = _s09_margin_at_scale(
        params,
        tau,
        q_star,
    )

    pieces_star_set = {
        tuple(piece)
        for piece in pieces_star
    }

    if pieces_star_set != reference_set:
        raise ValueError(
            "Morse decomposition changes at the computed q_star"
        )

    if abs(m_star) > 1e-9:
        raise ValueError(
            "margin at q_star is not zero"
        )

    # Independent reference-scale consistency check.
    m_ref_independent = _oracle_permanence_margin(
        p_ref,
        tau,
    )

    if abs(
        m_ref - m_ref_independent
    ) > 1e-9:
        raise ValueError(
            "reference margin disagrees with "
            "independent margin evaluation"
        )

    return float(q_star)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": "critical_pulse_scale(PZ, 2.0, 1.15, 1.20)",
            "gold_call": "_oracle_critical_pulse_scale(PZ, 2.0, 1.15, 1.20)",
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.3]}",
            "call": "critical_pulse_scale(PZ, 2.0, 1.20, 1.30)",
            "gold_call": "_oracle_critical_pulse_scale(PZ, 2.0, 1.20, 1.30)",
        },
        {
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.45, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": "critical_pulse_scale(PZ, 2.0, 1.15, 1.30)",
            "gold_call": "_oracle_critical_pulse_scale(PZ, 2.0, 1.15, 1.30)",
        },
    ]
