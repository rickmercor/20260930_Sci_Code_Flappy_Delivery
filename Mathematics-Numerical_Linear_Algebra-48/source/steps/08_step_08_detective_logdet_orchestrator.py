"""
detective Nyström+SLQ estimate plus scaled nuclear certificate.

End-to-end pipeline: build the instance, sketch it, score the residual diagnostic at the two widths, apply the adaptive rule, then assemble the preconditioner term and the stochastic residual term on whichever branch was selected and combine the estimate with the scaled certificate.

Returns
-------
float, detective estimate E_hat plus cert_scale times R as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def detective_logdet_certificate(
    n: int = 32,
    mu: float = 1e-2,
    ell: int = 16,
    m: int = 5,
    beta: float = 0.75,
    seed: int = 0,
    p: int = 2,
    cert_scale: float = 1e-3,
) -> float:
    """Run the detective Nyström+SLQ estimator and return E_hat + cert_scale * R.

    Parameters
    ----------
    n : int
        Matrix order for the algebraic diagonal instance.
    mu : float
        Regularization scale in A_ii = i^{-2}/mu.
    ell : int
        Nyström matvec budget parameter.
    m : int
        Lanczos depth / residual matvec block size.
    beta : float
        Detective fraction in (0, 1).
    seed : int
        NumPy Generator seed; all Gaussians drawn in algorithmic order.
    p : int
        Oversampling for the nuclear-norm certificate (k = r - p).
    cert_scale : float
        Multiplier for the certificate R in the returned scalar.

    Returns
    -------
    float
        E_hat + cert_scale * R.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``mu <= 0``; if ``beta`` is not strictly inside
        ``(0, 1)``; if ``ell < 1`` or ``m < 1``; if either nested rank
        ``floor(beta * ell)`` or ``floor(beta**2 * ell)`` is below 2, so the
        leave-one-out diagnostic cannot be formed; if the alpha-rank branch
        is selected and ``floor((ell + m - floor(beta * ell)) / m) < 1``
        leaves no residual probe; or if the assembled preconditioner
        violates the Loewner-order consequences of ``0 <= Ahat <= A``, namely
        ``||Ahat||_F > ||A||_F`` or ``tr log(Ahat + I) > tr log(A + I)``
        beyond a relative tolerance of 1e-8, or either quantity is
        non-finite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_detective_logdet_certificate(
    n: int = 32,
    mu: float = 1e-2,
    ell: int = 16,
    m: int = 5,
    beta: float = 0.75,
    seed: int = 0,
    p: int = 2,
    cert_scale: float = 1e-3,
) -> float:
    import numpy as np

    if n < 1 or mu <= 0:
        raise ValueError("invalid n or mu")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0, 1)")
    if ell < 1 or m < 1:
        raise ValueError("ell and m must be positive")

    logdet_exact = _oracle_construct_scaled_logdet(n, mu)

    idx = np.arange(1, n + 1, dtype=float)
    lam = (idx ** (-2)) / mu
    A = np.diag(lam)
    fro_exact = float(np.sqrt(np.sum(lam ** 2)))

    beta_ell = int(np.floor(beta * ell))
    beta2_ell = int(np.floor((beta ** 2) * ell))
    if beta_ell < 2 or beta2_ell < 2:
        raise ValueError("ranks too small for leave-one-out")

    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((n, beta_ell))

    # The coarse diagnostic reuses the leading beta2_ell columns of the same
    # sketch; drawing a fresh block would advance the generator stream.
    err_fine = _oracle_leave_one_out_errF2(A, Omega)
    err_coarse = _oracle_leave_one_out_errF2(A, Omega[:, :beta2_ell])
    flag = _oracle_detective_one_sample_flag(err_fine, err_coarse, ell, m, beta)

    if flag >= 0.5:
        Psi = rng.standard_normal((n, ell - beta_ell))
        Omega_use = np.hstack([Omega, Psi])
        r = ell
        fro_hat = _oracle_nystrom_frobenius_norm(A, Omega_use)
        t1 = _oracle_preconditioner_logdet(A, Omega_use)
        w = rng.standard_normal(n)
        t2 = _oracle_slq_preconditioned_quadratic(A, Omega_use, w, m)
    else:
        Omega_use = Omega
        r = beta_ell
        fro_hat = _oracle_nystrom_frobenius_norm(A, Omega_use)
        t1 = _oracle_preconditioner_logdet(A, Omega_use)
        N = int(np.floor((ell + m - beta_ell) / m))
        if N < 1:
            raise ValueError("no SLQ probes available on alpha-rank branch")
        acc = 0.0
        for _i in range(N):
            w = rng.standard_normal(n)
            acc += _oracle_slq_preconditioned_quadratic(A, Omega_use, w, m)
        t2 = acc / N

    # Loewner-order consequences of 0 <= Ahat <= A (see module docstring).
    tol_f = 1e-8 * max(1.0, abs(fro_exact))
    tol_l = 1e-8 * max(1.0, abs(logdet_exact))
    if not np.isfinite(fro_hat) or fro_hat < -tol_f or fro_hat > fro_exact + tol_f:
        raise ValueError(
            "Nystrom Frobenius norm violates 0 <= ||Ahat||_F <= ||A||_F"
        )
    if not np.isfinite(t1) or t1 < -tol_l or t1 > logdet_exact + tol_l:
        raise ValueError(
            "preconditioner log-determinant exceeds exact tr log(A+I)"
        )

    E_hat = t1 + t2
    R = _oracle_nuclear_residual_certificate(lam, r, p)
    return float(E_hat + cert_scale * R)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "detective_logdet_certificate(32, 1e-2, 16, 5, 0.75, 0, 2, 1e-3)",
            "gold_call": "_oracle_detective_logdet_certificate(32, 1e-2, 16, 5, 0.75, 0, 2, 1e-3)",
        },
        {
            "setup": "import numpy as np",
            "call": "detective_logdet_certificate(16, 1e-2, 8, 4, 0.75, 1, 2, 1e-3)",
            "gold_call": "_oracle_detective_logdet_certificate(16, 1e-2, 8, 4, 0.75, 1, 2, 1e-3)",
        },
        {
            "setup": "import numpy as np",
            "call": "detective_logdet_certificate(24, 1e-3, 12, 3, 0.5, 2, 2, 0.0)",
            "gold_call": "_oracle_detective_logdet_certificate(24, 1e-3, 12, 3, 0.5, 2, 2, 0.0)",
        },
    ]
