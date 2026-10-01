"""
Return the exact ground-state energy of the two-electron

asymmetric Hubbard dimer, its site-0 occupation, and the static response of

that occupation to the potential difference.

The asymmetric Hubbard dimer is a minimal

site-occupation model for studying density functionals from weak to strong

correlation. The site-0 occupation n plays the role of the density, and the

static response chi = dn/d(Delta v) plays the role of the density-density

response function. The two-electron ground state of the model, and these two

quantities, can be obtained exactly.

Returns
-------
a tuple ``(E_N, n_0, chi)`` of three floats, with     ``chi > 0``. Raise ``ValueError`` if ``t <= 0`` or ``U < 0``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_electron_ground_state(t: float, U: float, dv: float) -> tuple[float, float, float]:
    """Return the two-electron ground-state energy, site-0 occupation and response.

    Expected return: a tuple ``(E_N, n_0, chi)`` of three floats, with
    ``chi > 0``. Raise ``ValueError`` if ``t <= 0`` or ``U < 0``.
    """
    return (0.0, 0.0, 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_two_electron_ground_state(t: float, U: float, dv: float) -> tuple[float, float, float]:
    if t <= 0.0:
        raise ValueError("The hopping t must be positive.")
    if U < 0.0:
        raise ValueError("The on-site repulsion U must be non-negative.")

    linear = 4.0 * t * t - U * U + dv * dv
    energy = -(abs(dv) + 2.0 * (2.0**0.5) * t + 1.0)
    for _ in range(100):
        value = energy**3 - 2.0 * U * energy**2 - linear * energy + 4.0 * t * t * U
        slope = 3.0 * energy**2 - 4.0 * U * energy - linear
        energy = energy - value / slope

    slope = 3.0 * energy**2 - 4.0 * U * energy - linear
    first = 2.0 * dv * energy / slope
    slope_derivative = 6.0 * energy * first - 4.0 * U * first - 2.0 * dv
    second = ((2.0 * energy + 2.0 * dv * first) * slope - 2.0 * dv * energy * slope_derivative) / slope**2
    return (energy, 1.0 - first, -second)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return canonical, non-interacting, symmetric and strongly asymmetric cases."""
    return [
        # Normal: canonical fixture of the task.
        {
            "setup": "",
            "call": "two_electron_ground_state(1.0, 2.5, 1.5)",
            "gold_call": "_oracle_two_electron_ground_state(1.0, 2.5, 1.5)",
        },
        # Boundary: non-interacting symmetric dimer.
        {
            "setup": "",
            "call": "two_electron_ground_state(1.0, 0.0, 0.0)",
            "gold_call": "_oracle_two_electron_ground_state(1.0, 0.0, 0.0)",
        },
        # Boundary: interacting symmetric dimer.
        {
            "setup": "",
            "call": "two_electron_ground_state(1.0, 1.0, 0.0)",
            "gold_call": "_oracle_two_electron_ground_state(1.0, 1.0, 0.0)",
        },
        # Normal: hopping different from 1.
        {
            "setup": "",
            "call": "two_electron_ground_state(2.0, 3.0, 4.0)",
            "gold_call": "_oracle_two_electron_ground_state(2.0, 3.0, 4.0)",
        },
        # Edge: strongly correlated and strongly asymmetric, negative dv.
        {
            "setup": "",
            "call": "two_electron_ground_state(1.0, 5.0, -8.0)",
            "gold_call": "_oracle_two_electron_ground_state(1.0, 5.0, -8.0)",
        },
    ]
