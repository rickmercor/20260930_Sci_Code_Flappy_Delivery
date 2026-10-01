"""
Compute the PCKAN price prediction and the Black–Scholes physics residual, together with the first derivative with respect to physical spot, the second derivative with respect to physical spot, and the first derivative with respect to time to maturity.



Use the initialized PCKAN coefficients and readout parameters supplied as inputs, preserving the Step-03 tanh transformation and second-kind Chebyshev expansion. Evaluate all required derivatives with PyTorch automatic differentiation in float64 arithmetic.



The network spot coordinate is S0/100, so convert derivatives with respect to that scaled coordinate into derivatives with respect to physical spot using the appropriate chain-rule factors. Treat T as time to maturity and use sigma_phys = sqrt(v0) = 0.2 in the Black–Scholes residual.



This step performs forward evaluation and physics-residual computation only. It must not update the PCKAN parameters, construct a training loss, run Adam optimization, or perform differential-evolution calibration.

The PCKAN uses the Black–Scholes partial differential equation only as a physics constraint; the Black–Scholes model is not used to generate the FVSJ option-price targets.



For a European option written as a function of physical spot S and time to maturity T, use the benchmark residual



R = rV - V_T - r S V_S - (1/2) sigma_phys^2 S^2 V_SS,



where V is the PCKAN prediction, V_S is its first derivative with respect to physical spot, V_SS is its second derivative with respect to physical spot, and V_T is its derivative with respect to time to maturity. The sign of V_T follows from treating T as time to maturity rather than calendar time.



For this benchmark, r = 0.05 and sigma_phys = sqrt(v0) = sqrt(0.04) = 0.2. The physics volatility is therefore distinct from the FVSJ volatility-of-variance parameter sigma_1 = 0.30 used by the offline pricing model.



The PCKAN receives the scaled spot coordinate x_S = S/100. Automatic differentiation therefore initially produces derivatives with respect to x_S. The physical derivatives required by the PDE residual are obtained by the chain rule:



V_S = (1/100) V_xS,



V_SS = (1/10000) V_xSxS.



The maturity coordinate T is not rescaled, so its automatic derivative is already V_T.



Evaluate the supplied frozen PCKAN parameters using the same forward construction as Step 03: apply tanh coordinatewise, evaluate the second-kind Chebyshev basis through order 3, sum the CKAN edge contributions, and apply the scalar linear readout. Use PyTorch float64 automatic differentiation to obtain the first- and second-order derivatives.



This step establishes the differentiable physical quantities needed by the later physics-informed training stage, but it does not itself optimize any network parameters.

Returns
-------
return predictions, dV_dS, d2V_dS2, dV_dT, residuals
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_pckan_physics_residual(
    inputs: np.ndarray,
    W_hat: np.ndarray,
    W_out: np.ndarray,
    b_out: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCKAN predictions, physical spot/time derivatives, and the
    benchmark Black-Scholes physics residual.

    Parameters
    ----------
    inputs : np.ndarray
        Benchmark-scaled PCKAN inputs with shape (n, 6).
    W_hat : np.ndarray
        CKAN Chebyshev coefficient tensor with shape (8, 6, 4).
    W_out : np.ndarray
        Linear readout weights with shape (8,).
    b_out : float
        Scalar output bias.

    Returns
    -------
    predictions : np.ndarray
        PCKAN price predictions with shape (n,).
    dV_dS : np.ndarray
        First derivatives with respect to physical spot, shape (n,).
    d2V_dS2 : np.ndarray
        Second derivatives with respect to physical spot, shape (n,).
    dV_dT : np.ndarray
        Derivatives with respect to time to maturity, shape (n,).
    residuals : np.ndarray
        Black-Scholes physics residuals with shape (n,).
    """
    n = inputs.shape[0]
    return (
        np.empty((n,), dtype=np.float64),
        np.empty((n,), dtype=np.float64),
        np.empty((n,), dtype=np.float64),
        np.empty((n,), dtype=np.float64),
        np.empty((n,), dtype=np.float64),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_pckan_physics_residual(inputs, W_hat, W_out, b_out):
    import numpy as np

    x = np.asarray(inputs, dtype=np.float64)
    Wh = np.asarray(W_hat, dtype=np.float64)
    Wo = np.asarray(W_out, dtype=np.float64)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    if x.ndim != 2 or x.shape[1] != 6:
        raise ValueError("inputs must have shape (n, 6)")

    if Wh.shape != (8, 6, 4):
        raise ValueError("W_hat must have shape (8, 6, 4)")

    if Wo.shape != (8,):
        raise ValueError("W_out must have shape (8,)")

    try:
        b = float(b_out)
    except (TypeError, ValueError):
        raise ValueError("b_out must be finite")

    if not np.isfinite(b):
        raise ValueError("b_out must be finite")

    if not np.all(np.isfinite(x)):
        raise ValueError("inputs must be finite")

    if not np.all(np.isfinite(Wh)):
        raise ValueError("W_hat must be finite")

    if not np.all(np.isfinite(Wo)):
        raise ValueError("W_out must be finite")

    # ------------------------------------------------------------
    # PCKAN forward map
    #
    # z = tanh(x)
    #
    # Second-kind Chebyshev basis:
    # U0 = 1
    # U1 = 2z
    # U2 = 4z^2 - 1
    # U3 = 8z^3 - 4z
    # ------------------------------------------------------------
    z = np.tanh(x)

    u0 = np.ones_like(z)
    u1 = 2.0 * z
    u2 = 4.0 * z**2 - 1.0
    u3 = 8.0 * z**3 - 4.0 * z

    U = np.stack(
        (u0, u1, u2, u3),
        axis=-1,
    )

    h = np.einsum("qpn,bpn->bq", Wh, U)
    V = np.einsum("bq,q->b", h, Wo) + b

    # ------------------------------------------------------------
    # Collapse hidden/readout weights.
    #
    # For coordinate p and Chebyshev order n:
    #
    # C[p,n] = sum_q W_out[q] * W_hat[q,p,n]
    #
    # Therefore
    #
    # V = b + sum_{p,n} C[p,n] U_n(tanh(x_p)).
    #
    # This makes the coordinate derivatives explicit.
    # ------------------------------------------------------------
    C = np.einsum("q,qpn->pn", Wo, Wh)

    # ------------------------------------------------------------
    # First derivatives of U_n(z) with respect to z
    #
    # dU0/dz = 0
    # dU1/dz = 2
    # dU2/dz = 8z
    # dU3/dz = 24z^2 - 4
    # ------------------------------------------------------------
    dU_dz = np.stack(
        (
            np.zeros_like(z),
            2.0 * np.ones_like(z),
            8.0 * z,
            24.0 * z**2 - 4.0,
        ),
        axis=-1,
    )

    # tanh derivative
    dz_dx = 1.0 - z**2

    # dU/dx = dU/dz * dz/dx
    dU_dx = dU_dz * dz_dx[..., None]

    # dV/dx_p
    dV_dx = np.einsum("pn,bpn->bp", C, dU_dx)

    dV_dx0 = dV_dx[:, 0]
    dV_dx2 = dV_dx[:, 2]

    # ------------------------------------------------------------
    # Second derivative with respect to x_0.
    #
    # d2U/dx2 =
    #     d2U/dz2 * (dz/dx)^2
    #     + dU/dz * d2z/dx2
    #
    # where
    #
    # d2z/dx2 = -2 z (1-z^2)
    #
    # and
    #
    # d2U0/dz2 = 0
    # d2U1/dz2 = 0
    # d2U2/dz2 = 8
    # d2U3/dz2 = 48z
    # ------------------------------------------------------------
    d2U_dz2 = np.stack(
        (
            np.zeros_like(z),
            np.zeros_like(z),
            8.0 * np.ones_like(z),
            48.0 * z,
        ),
        axis=-1,
    )

    d2z_dx2 = -2.0 * z * (1.0 - z**2)

    d2U_dx2 = (
        d2U_dz2 * (dz_dx[..., None] ** 2)
        + dU_dz * d2z_dx2[..., None]
    )

    d2V_dx2_all = np.einsum(
        "pn,bpn->bp",
        C,
        d2U_dx2,
    )

    d2V_dx0 = d2V_dx2_all[:, 0]

    # ------------------------------------------------------------
    # Convert benchmark-scaled derivatives to physical derivatives.
    #
    # x_0 = S / 100
    #
    # dV/dS = dV/dx_0 / 100
    # d2V/dS2 = d2V/dx_0^2 / 10000
    #
    # x_2 is time to maturity directly.
    # ------------------------------------------------------------
    dV_dS = dV_dx0 / 100.0
    d2V_dS2 = d2V_dx0 / 10000.0
    dV_dT = dV_dx2

    # ------------------------------------------------------------
    # Physical Black-Scholes residual
    #
    # R = r V
    #     - V_T
    #     - r S V_S
    #     - 0.5 sigma_phys^2 S^2 V_SS
    #
    # Benchmark physical volatility:
    # sigma_phys = sqrt(v0) = sqrt(0.04) = 0.2
    # ------------------------------------------------------------
    S = 100.0 * x[:, 0]
    r = x[:, 3]

    sigma_phys = 0.2

    residuals = (
        r * V
        - dV_dT
        - r * S * dV_dS
        - 0.5 * (sigma_phys**2) * (S**2) * d2V_dS2
    )

    return (
        np.asarray(V, dtype=np.float64),
        np.asarray(dV_dS, dtype=np.float64),
        np.asarray(d2V_dS2, dtype=np.float64),
        np.asarray(dV_dT, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return Step-04 differential and frozen-regression test cases."""

    return [
        # CASE 1 — normal benchmark row
        {
            "setup": (
                "import numpy as np\n"
                "_, W_hat, W_out, b_out = "
                "initialize_and_evaluate_pckan("
                "np.zeros((1, 6), dtype=np.float64))\n"
                "row = np.array([[\n"
                "    1.0,\n"
                "    0.9,\n"
                "    0.25,\n"
                "    0.05,\n"
                "    0.2,\n"
                "    0.16666666666666669,\n"
                "]], dtype=np.float64)"
            ),
            "call": (
                "compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)"
            ),
            "gold_call": (
                "_oracle_compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)"
            ),
        },

        # CASE 2 — boundary/differential case
        {
            "setup": (
                "import numpy as np\n"
                "_, W_hat, W_out, b_out = "
                "initialize_and_evaluate_pckan("
                "np.zeros((1, 6), dtype=np.float64))\n"
                "row = np.array([[\n"
                "    1.0,\n"
                "    0.9,\n"
                "    0.25,\n"
                "    0.05,\n"
                "    0.2,\n"
                "    0.16666666666666669,\n"
                "]], dtype=np.float64)\n"
                "b_out = -6.681805872082e-05"
            ),
            "call": (
                "compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)"
            ),
            "gold_call": (
                "_oracle_compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)"
            ),
        },

        # CASE 3 — edge/differential case
        {
            "setup": (
                "import numpy as np\n"
                "_, W_hat, W_out, b_out = "
                "initialize_and_evaluate_pckan("
                "np.zeros((1, 6), dtype=np.float64))\n"
                "row = np.array([[\n"
                "    1.0,\n"
                "    0.9,\n"
                "    0.0,\n"
                "    0.05,\n"
                "    0.2,\n"
                "    0.16666666666666669,\n"
                "]], dtype=np.float64)"
            ),
            "call": (
                "compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)"
            ),
            "gold_call": (
                "_oracle_compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)"
            ),
        },

        # CASE 4 — frozen representative regression.
        #
        # Use np.isclose explicitly because the public implementation
        # and NumPy-only QC oracle can differ by float64 roundoff.
        {
            "setup": (
                "import numpy as np\n"
                "_, W_hat, W_out, b_out = "
                "initialize_and_evaluate_pckan("
                "np.zeros((1, 6), dtype=np.float64))\n"
                "row = np.array([[\n"
                "    1.0,\n"
                "    0.9,\n"
                "    0.25,\n"
                "    0.05,\n"
                "    0.2,\n"
                "    0.16666666666666669,\n"
                "]], dtype=np.float64)\n"
                "out = compute_pckan_physics_residual("
                "row, W_hat, W_out, b_out)\n"
                "expected = (\n"
                "    -0.0017843316853409753,\n"
                "    -2.009663083399474e-05,\n"
                "    9.360305285346383e-08,\n"
                "    0.0008842860537378545,\n"
                "    -0.0008917400944056223,\n"
                ")\n"
                "result = tuple(\n"
                "    bool(np.isclose(\n"
                "        float(v[0]),\n"
                "        expected[i],\n"
                "        rtol=1.0e-12,\n"
                "        atol=1.0e-14,\n"
                "    ))\n"
                "    for i, v in enumerate(out)\n"
                ")"
            ),
            "call": "result",
            "gold_call": "(True, True, True, True, True)",
        },
    ]
