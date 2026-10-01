"""
Numerical-error covariances for the two spherical multipoles.

For each mode in the resolution difference let $A=\max|\Delta h|$. The trained standard GP pair supplied from the source is $(\lambda_{\rm GP},\mu_{\rm GP})=(6.92,1.68)$. Use $P=2\pi\mu_{\rm GP}/\Re\omega$, $\tau=-1/\Im\omega$, and $\sigma(t)=\operatorname{smin}(\lambda_{\rm GP}Ae^{-t/\tau},1.1A;10^{-3})$, where $\operatorname{smin}(x,c;s)=[x+c(1-\sqrt{(x/c-1)^2+s})]/2$. For `error_model="anchor_gp"` use



$$K_{ab}=\sigma_a\sigma_b(1-r)_+^7(1+7r+19r^2+21r^3)+0.02A^2\delta_{ab},\qquad r=|t_a-t_b|/P.$$



The $generic_exponential$ option substitutes $e^{-r}$ for the compact correlation while retaining the same envelope. The $white_noise$ option uses $K=0.20A I$, where $0.20A$ is the covariance entry rather than a standard deviation to square. The `0.02A^2` finite-record regularizer applies only to the correlated models. The amplitude scale multiplies standard deviation and the period scale multiplies $P$.



Let $d$ be the complete observed highest-minus-next-highest residual and $e$ the additive highest-resolution waveform error, both in `(Re22,Re64,Im22,Im64)` block order. For each real or imaginary pair, take lower-Cholesky factors $K_{22}=L_{22}L_{22}^T$ and $K_{64}=L_{64}L_{64}^T$, define the task-supplied cross-mode block $C=\gamma L_{22}L_{64}^T$, and set



$$

D=\operatorname{diag}\!\left(

\begin{bmatrix}K_{22}\&C\\C^T\&K_{64}\end{bmatrix},

\begin{bmatrix}K_{22}\&C\\C^T\&K_{64}\end{bmatrix}

\right).

$$



The task-defined synthetic joint calibration hierarchy is



$$

\operatorname{Cov}\!\begin{pmatrix}e\\d\end{pmatrix}

=\begin{pmatrix}

qD&\rho\sqrt q\,D\\

\rho\sqrt q\,D\&D

\end{pmatrix},

\qquad q=0.50,\quad \rho=0.85,\quad \gamma=0.08.

$$



Real and imaginary processes are independent, but the 22 and 64 processes are correlated. Require finite $q>0$, $-1<\rho<1$, and $-1<\gamma<1$.

Returns
-------
`numpy.ndarray` of shape `(n,56,56)` for $z=(e,d)$, where each 28-vector uses `(Re22,Re64,Im22,Im64)` order. Every cell, including each exact-zero omitted cross-block, is compared at `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_correlated_covariances(
    data: np.ndarray,
    block_frequencies: np.ndarray,
    error_model: str = "anchor_gp",
    error_amplitude_scale: float = 1.0,
    error_period_scale: float = 1.0,
    highest_error_variance_ratio: float = 0.5,
    error_residual_correlation: float = 0.85,
    cross_mode_correlation: float = 0.08,
) -> np.ndarray:
    """Construct the joint waveform-error and residual covariance.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite record containing times and both complex resolution differences.
    block_frequencies : numpy.ndarray, shape (n, 4)
        `(Re omega220, Im omega220, Re omega640, Im omega640)` for each state.
    error_model : str, default="anchor_gp"
        One of `"anchor_gp"`, `"generic_exponential"`, or `"white_noise"`.
        For `"white_noise"`, every unscaled mode-covariance diagonal equals
        `0.20*A`; this quantity is the covariance entry and is not squared.
    error_amplitude_scale : float, default=1.0
        Finite positive multiplier on standard deviation; its square multiplies
        the complete covariance.
    error_period_scale : float, default=1.0
        Finite positive multiplier on correlated-model periods. It is checked
        but inactive for white noise.
    highest_error_variance_ratio : float, default=0.5
        Finite positive ratio `q` between the marginal highest-resolution
        error covariance and the resolution-residual covariance, common to the
        two modes.
    error_residual_correlation : float, default=0.85
        Finite common correlation coefficient `rho` between the highest-
        resolution waveform error and observed residual, strictly between
        -1 and 1.
    cross_mode_correlation : float, default=0.08
        Finite residual correlation coefficient `gamma` between the spherical
        22 and 64 modes, strictly between -1 and 1. Its oriented cross block is
        `gamma * chol(K22) @ chol(K64).T` for both real and imaginary parts.

    Returns
    -------
    numpy.ndarray, shape (n, 56, 56)
        Joint covariance of `z=(e,d)`, with each 28-vector in
        `(Re22,Re64,Im22,Im64)` block order. Every cell is compared at
        tolerance `1e-9`.

    Raises
    ------
    ValueError
        If an array has an incompatible shape or nonfinite entry; times are not
        strictly increasing; a carrier has an invalid sign; a scale is not
        finite and positive; either correlation is outside the open interval
        `(-1, 1)`; a resolution-difference block has zero amplitude; the model
        name is unsupported; or a covariance produced within the stated input
        domain is not finite, symmetric, and positive definite.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""Joint numerical-error covariances for the two spherical multipoles.

For each mode let $A$ be the maximum magnitude of its supplied resolution
difference. The source-derived trained standard-GP pair is
$(\lambda_{\rm GP},\mu_{\rm GP})=(6.92,1.68)$. Use
$P=2\pi\mu_{\rm GP}/\Re\omega$, $\tau=-1/\Im\omega$, and

$$\sigma(t)=\operatorname{smin}(\lambda_{\rm GP}Ae^{-t/\tau},1.1A;10^{-3}),$$

where
$\operatorname{smin}(x,c;s)=[x+c(1-\sqrt{(x/c-1)^2+s})]/2$.
For `error_model="anchor_gp"`, use

$$K_{ab}=\sigma_a\sigma_b(1-r)_+^7(1+7r+19r^2+21r^3)
+0.02A^2\delta_{ab},\qquad r=|t_a-t_b|/P.$$

The `generic_exponential` counterfactual replaces only the compact
correlation by $e^{-r}$; `white_noise` uses $0.20A\,I$, meaning that $0.20A$
is the diagonal covariance entry rather than a standard deviation to be
squared. The `0.02A^2` term is a task-defined finite-record regularizer and is
used by the two correlated models only.

Let $d$ be the 28-vector of observed highest-minus-next-highest residuals and
$e$ the additive numerical error in the highest-resolution waveform, both in
`(Re22,Re64,Im22,Im64)` block order. For each real or imaginary pair, let
$K_{22}=L_{22}L_{22}^T$ and $K_{64}=L_{64}L_{64}^T$ be lower-Cholesky
factorizations and define the task-supplied cross-mode residual block
$C=\gamma L_{22}L_{64}^T$. The residual covariance is

$$
D=\operatorname{diag}\!\left(
\begin{bmatrix}K_{22}&C\\C^T&K_{64}\end{bmatrix},
\begin{bmatrix}K_{22}&C\\C^T&K_{64}\end{bmatrix}
\right).
$$

The task-defined joint hierarchy is

$$
\operatorname{Cov}\!\begin{pmatrix}e\\d\end{pmatrix}
=\begin{pmatrix}
qD&\rho\sqrt qD\\
\rho\sqrt qD&D
\end{pmatrix}.
$$

The default uses $q=0.5$, $\rho=0.85$, and $\gamma=0.08$. Real and imaginary
processes remain independent. The Cholesky construction fixes the orientation
of the synthetic cross-mode block and guarantees a positive-definite residual
covariance for $|\gamma|<1$.

Returns
-------
`numpy.ndarray` of shape `(n, 56, 56)` for `z=(e,d)`, where both 28-vectors
use order `(Re22[0:7], Re64[0:7], Im22[0:7], Im64[0:7])`. Every cell is a
specified covariance entry; only real-imaginary cross-blocks are exact zero.
Components are compared at `1e-9`.
"""
import numpy as np
import numpy as np
_ERROR_MODELS = {"anchor_gp", "generic_exponential", "white_noise"}
def _covariance_positive_definite(matrix: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(label + " must be finite and symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(label + " must be positive definite") from exc
def _covariance_smooth_min(value: np.ndarray, ceiling: float) -> np.ndarray:
    ratio = value / ceiling
    return 0.5 * (value + ceiling * (1.0 - np.sqrt((ratio - 1.0) ** 2 + 1.0e-3)))
def _mode_covariance(
    time: np.ndarray,
    difference: np.ndarray,
    frequency: complex,
    model: str,
    period_scale: float,
) -> np.ndarray:
    peak = float(np.max(np.abs(difference)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("each resolution-difference block must have positive amplitude")
    if peak > np.sqrt(np.finfo(float).max):
        raise ValueError("resolution-difference amplitude is too large for finite covariance")
    if model == "white_noise":
        covariance = np.eye(time.size) * 0.20 * peak
    else:
        period = 2.0 * np.pi * 1.68 * period_scale / frequency.real
        decay_time = -1.0 / frequency.imag
        envelope = _covariance_smooth_min(
            6.92 * peak * np.exp(-time / decay_time), 1.1 * peak
        )
        separation = np.abs(time[:, None] - time[None, :]) / period
        if model == "anchor_gp":
            clipped = np.maximum(1.0 - separation, 0.0)
            correlation = clipped ** 7 * (
                1.0 + 7.0 * separation + 19.0 * separation ** 2
                + 21.0 * separation ** 3
            )
        else:
            correlation = np.exp(-separation)
        covariance = envelope[:, None] * envelope[None, :] * correlation
        covariance += np.eye(time.size) * 0.02 * peak ** 2
    _covariance_positive_definite(covariance, model + " mode covariance")
    return covariance
def _oracle_build_correlated_covariances(
    data: np.ndarray,
    block_frequencies: np.ndarray,
    error_model: str = "anchor_gp",
    error_amplitude_scale: float = 1.0,
    error_period_scale: float = 1.0,
    highest_error_variance_ratio: float = 0.5,
    error_residual_correlation: float = 0.85,
    cross_mode_correlation: float = 0.08,
) -> np.ndarray:
    record = np.asarray(data, dtype=float)
    frequencies = np.asarray(block_frequencies, dtype=float)
    if record.shape != (7, 9) or not np.all(np.isfinite(record)):
        raise ValueError("data must have finite shape (7, 9)")
    if np.any(np.diff(record[:, 0]) <= 0.0):
        raise ValueError("sample times must be strictly increasing")
    if frequencies.ndim != 2 or frequencies.shape[1] != 4 or frequencies.shape[0] == 0 or not np.all(np.isfinite(frequencies)):
        raise ValueError("block_frequencies must have nonempty finite shape (n, 4)")
    if np.any(frequencies[:, (0, 2)] <= 0.0) or np.any(frequencies[:, (1, 3)] >= 0.0):
        raise ValueError("carrier frequencies require positive real and negative imaginary parts")
    if not isinstance(error_model, str) or error_model not in _ERROR_MODELS:
        raise ValueError("error_model is unsupported")
    amplitude_scale = float(error_amplitude_scale)
    period_scale = float(error_period_scale)
    variance_ratio = float(highest_error_variance_ratio)
    error_residual_rho = float(error_residual_correlation)
    cross_mode_gamma = float(cross_mode_correlation)
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
        raise ValueError("error_amplitude_scale must be finite and positive")
    if amplitude_scale > np.sqrt(np.finfo(float).max):
        raise ValueError("error_amplitude_scale is too large for finite covariance")
    if not np.isfinite(period_scale) or period_scale <= 0.0:
        raise ValueError("error_period_scale must be finite and positive")
    if not np.isfinite(variance_ratio) or variance_ratio <= 0.0:
        raise ValueError("highest_error_variance_ratio must be finite and positive")
    if not np.isfinite(error_residual_rho) or not -1.0 < error_residual_rho < 1.0:
        raise ValueError("error_residual_correlation must be finite and strictly between -1 and 1")
    if not np.isfinite(cross_mode_gamma) or not -1.0 < cross_mode_gamma < 1.0:
        raise ValueError("cross_mode_correlation must be finite and strictly between -1 and 1")

    time = record[:, 0]
    differences = (
        record[:, 5] + 1j * record[:, 6],
        record[:, 7] + 1j * record[:, 8],
    )
    output = np.zeros((frequencies.shape[0], 56, 56), dtype=float)
    for index, row in enumerate(frequencies):
        mode22 = _mode_covariance(
            time, differences[0], row[0] + 1j * row[1], error_model, period_scale
        )
        mode64 = _mode_covariance(
            time, differences[1], row[2] + 1j * row[3], error_model, period_scale
        )
        factor22 = np.linalg.cholesky(mode22)
        factor64 = np.linalg.cholesky(mode64)
        cross_mode = cross_mode_gamma * factor22 @ factor64.T
        complex_block = np.block([
            [mode22, cross_mode],
            [cross_mode.T, mode64],
        ])
        residual_covariance = (amplitude_scale * amplitude_scale) * np.block([
            [complex_block, np.zeros((14, 14))],
            [np.zeros((14, 14)), complex_block],
        ])
        joint = output[index]
        joint[:28, :28] = variance_ratio * residual_covariance
        cross_block = error_residual_rho * np.sqrt(variance_ratio) * residual_covariance
        joint[:28, 28:] = cross_block
        joint[28:, :28] = cross_block.T
        joint[28:, 28:] = residual_covariance
        _covariance_positive_definite(joint, "joint numerical covariance")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data):\n    import numpy as np\n    data = fixture_load_ringdown_data(0.89, 1.07)\n    freq = np.array([[0.54, -0.08, 1.57, -0.09], [0.61, -0.07, 1.68, -0.08]])\n    return fn_under_test(data, freq, 'anchor_gp', 0.82, 1.17)",
            'call': 'run_case(build_correlated_covariances, load_ringdown_data)',
            'gold_call': 'run_case(_oracle_build_correlated_covariances, _oracle_load_ringdown_data)',
        },
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data):\n    import numpy as np\n    data = fixture_load_ringdown_data(1.14, 0.93)\n    freq = np.array([[0.49, -0.11, 1.31, -0.06]])\n    return np.concatenate((fn_under_test(data, freq, 'generic_exponential', 1.21, 0.78, 0.63, 0.4, -0.3).ravel(), fn_under_test(data, freq, 'white_noise', 0.91, 1.4, 1.2, -0.2, 0.55).ravel()))",
            'call': 'run_case(build_correlated_covariances, load_ringdown_data)',
            'gold_call': 'run_case(_oracle_build_correlated_covariances, _oracle_load_ringdown_data)',
        },
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data):\n    import numpy as np\n    data = fixture_load_ringdown_data(0.96, 1.02)\n    freq = np.array([[0.52, -0.09, 1.48, -0.07], [0.63, -0.06, 1.72, -0.1]])\n    return fn_under_test(data, freq, 'anchor_gp', 1.0, 1.0, 0.5, 0.0, -0.25)",
            'call': 'run_case(build_correlated_covariances, load_ringdown_data)',
            'gold_call': 'run_case(_oracle_build_correlated_covariances, _oracle_load_ringdown_data)',
        },
        {
            'setup': "def run_case(fn_under_test, fixture_load_ringdown_data):\n    import numpy as np\n\n    def catches(fn):\n        try:\n            fn()\n        except ValueError:\n            return 1\n        except Exception:\n            return 2\n        return 0\n    data = fixture_load_ringdown_data()\n    freq = np.array([[0.55, -0.08, 1.6, -0.09]])\n    zero22 = data.copy()\n    zero22[:, 5:7] = 0.0\n    zero64 = data.copy()\n    zero64[:, 7:9] = 0.0\n    bad_time = data.copy()\n    bad_time[2, 0] = bad_time[1, 0]\n    bad_data = data.copy()\n    bad_data[0, 5] = np.nan\n    bad_freq = freq.copy()\n    bad_freq[0, 3] = 0.0\n    nonfinite_freq = freq.copy()\n    nonfinite_freq[0, 0] = np.inf\n    return np.array([catches(lambda: fn_under_test(zero22, freq)), catches(lambda: fn_under_test(zero64, freq)), catches(lambda: fn_under_test(bad_time, freq)), catches(lambda: fn_under_test(bad_data, freq)), catches(lambda: fn_under_test(data, bad_freq)), catches(lambda: fn_under_test(data, nonfinite_freq)), catches(lambda: fn_under_test(data, np.zeros((1, 3)))), catches(lambda: fn_under_test(data, freq, 'matern')), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 0.0, 1.0)), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 1.0, np.nan)), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 1.0, 1.0, 0.0)), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 1.0, 1.0, np.nan)), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 1.0, 1.0, 0.5, 1.0)), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 1.0, 1.0, 0.5, np.nan)), catches(lambda: fn_under_test(data, freq, 'anchor_gp', 1.0, 1.0, 0.5, 0.0, -1.0))])",
            'call': 'run_case(build_correlated_covariances, load_ringdown_data)',
            'gold_call': 'run_case(_oracle_build_correlated_covariances, _oracle_load_ringdown_data)',
        },
    ]
