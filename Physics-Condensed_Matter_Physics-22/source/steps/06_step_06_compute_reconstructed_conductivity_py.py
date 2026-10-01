"""
Convert the fitted zero-frequency heat-current spectral density into the reconstructed isotropic thermal conductivity.

For the isotropic reduced system, the three equivalent Cartesian contributions cancel the conventional factor of one third in the Green--Kubo expression. With $k_B=V=1$, the reconstructed conductivity is

$$

\kappa_{\mathrm{rec}}=\beta^2\Lambda_x(0).

$$

Returns
-------
float, the reconstructed isotropic thermal conductivity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_reconstructed_conductivity(
    beta: float,
    lambda_zero: float,
) -> float:
    """Compute the reconstructed isotropic thermal conductivity.

    Parameters
    ----------
    beta : float
        Positive inverse temperature.
    lambda_zero : float
        Fitted total zero-frequency heat-current spectral density.

    Returns
    -------
    kappa_rec : float
        Reconstructed thermal conductivity in the dimensionless
        units of the task.
    """
    return kappa_rec

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_reconstructed_conductivity(
    beta: float,
    lambda_zero: float,
) -> float:
    return float(
        float(beta) ** 2
        * float(lambda_zero)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """
beta = 1.5
lambda_zero = 6.48556825465
""",
            "call": "round(compute_reconstructed_conductivity(beta, lambda_zero), 10)",
            "gold_call": "round(_oracle_compute_reconstructed_conductivity(beta, lambda_zero), 10)",
        },
        {
            "setup": """
beta = 1.0
lambda_zero = 2.5
""",
            "call": "round(compute_reconstructed_conductivity(beta, lambda_zero), 10)",
            "gold_call": "round(_oracle_compute_reconstructed_conductivity(beta, lambda_zero), 10)",
        },
        {
            "setup": """
beta = 0.25
lambda_zero = 0.0
""",
            "call": "round(compute_reconstructed_conductivity(beta, lambda_zero), 10)",
            "gold_call": "round(_oracle_compute_reconstructed_conductivity(beta, lambda_zero), 10)",
        },
    ]
