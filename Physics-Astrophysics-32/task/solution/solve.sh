#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""
Recorded spherical 22 and 64 ringdown multipoles and differences.

Return the seven rows in the Problem Statement in the stated order. `amplitude_scale` and `time_scale` are finite and strictly positive; the first multiplies every non-time column and the second every time entry. The task instance uses one for both.

Returns
-------
`numpy.ndarray` of floating dtype and shape `(7, 9)`.

Every returned component is compared at numerical tolerance `1e-9`.
"""
"""Recorded spherical 22 and 64 ringdown multipoles and differences."""
import numpy as np
"""Recorded spherical 22 and 64 ringdown multipoles and differences."""
import numpy as np
_BASE_DATA = np.array([
    [0.00, 0.607954649691, 0.295051751110, 0.000628378919, -0.000337532774, -0.037867242596, -0.001909207733, 0.000044617819, -0.000067295330],
    [3.00, 0.147374778600, -0.432772573681, 0.000144374885, 0.000297055983, -0.013031289479, 0.008698702522, -0.000040102718, -0.000121982286],
    [6.50, -0.303886427800, 0.075363262462, 0.000046548106, 0.000234305339, 0.029983620352, 0.019511427616, -0.000092589566, -0.000105810338],
    [10.50, 0.206496858961, 0.222099527098, 0.000017845774, 0.000196568821, 0.048606458833, 0.001108395423, 0.000046898531, -0.000025239149],
    [15.00, -0.060781083081, -0.123253180953, 0.000032917024, 0.000059030411, 0.059110971375, -0.059988566549, 0.000061689148, -0.000106702268],
    [20.00, 0.087855912970, 0.131424789334, -0.000048979985, -0.000053579051, 0.031337457207, 0.013158966817, -0.000027079682, 0.000062232067],
    [26.00, -0.044012797140, -0.023897529367, 0.000015531811, 0.000054570561, -0.034883268283, 0.023852694832, -0.000009366161, -0.000023349673],
], dtype=float)
def load_ringdown_data(amplitude_scale: float = 1.0, time_scale: float = 1.0) -> np.ndarray:
    scale = float(amplitude_scale)
    time_multiplier = float(time_scale)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(time_multiplier) or time_multiplier <= 0.0:
        raise ValueError("amplitude_scale and time_scale must be finite and positive")
    data = _BASE_DATA.copy()
    data[:, 0] *= time_multiplier
    data[:, 1:] *= scale
    return data

r"""
Componentwise interpolation of the supplied Kerr-mode state table and pairing with remnant mass.

Linearly interpolate every real state-table column independently at requested spins in $[0.70,0.86]$, with no extrapolation. Pair each spin with its supplied finite positive remnant-mass ratio $M_f/M_{\rm ref}$. The tabulated frequencies are the intrinsic dimensionless products $\widehat\omega=M_f\omega$; do not divide them by the mass ratio in this stage.

Returns
-------
`numpy.ndarray` of shape `(n, 16)` with columns

$$
(\chi_f,M_f/M_{\rm ref},\Re\widehat\omega_{220},\Im\widehat\omega_{220},\Re\widehat\omega_{320},\Im\widehat\omega_{320},\Re\widehat\omega_{640},\Im\widehat\omega_{640},
\Re\mu_{22},\Im\mu_{22},\Re\mu_{32},\Im\mu_{32},
\Re \rho_4,\Im \rho_4,\Re \rho_5,\Im \rho_5).
$$

The $\rho_l$ values are task-defined synthetic 220-by-220 radial transfers.
Step 03 also uses the task-defined mixed-parent transfer
$\kappa_l=-0.20\rho_l$.

Every returned component is compared at numerical tolerance `1e-9`.
"""
"""Componentwise interpolation of the supplied Kerr-mode state table."""
import numpy as np
"""Componentwise interpolation of the supplied Kerr-mode state table."""
import numpy as np
_RAW_TABLE = np.array([
    [0.70, 0.532600243597, -0.080792873136, 0.759174723145, -0.084189645815, 1.541269014176, -0.086754879659, 0.997417280443, -0.009984915659, -0.100258037153, 0.008919401626, -0.0146100910845, 0.2685306133210, -0.0000431583755, 0.0057182161510],
    [0.72, 0.541793731483, -0.079990805121, 0.767600177103, -0.083373330711, 1.558104443344, -0.085893956678, 0.997176808792, -0.010342555366, -0.104222397641, 0.008993049256, -0.0135624224605, 0.2594809831015, -0.0000440202685, 0.0058059579770],
    [0.74, 0.551630332971, -0.079092730022, 0.776508654282, -0.082467481710, 1.575854074157, -0.084937290752, 0.996913703333, -0.010696053255, -0.108307676796, 0.009045655703, -0.0125189913410, 0.2504395563260, -0.0000496081975, 0.0058990191325],
    [0.76, 0.562200718998, -0.078081685942, 0.785958791816, -0.081457397033, 1.594626097338, -0.083869410692, 0.996624850763, -0.011042407814, -0.112526148250, 0.009074892257, -0.0114774776610, 0.2413981510995, -0.0000538512135, 0.0059813295865],
    [0.78, 0.573616428376, -0.076936363177, 0.796021310470, -0.080324671683, 1.614550655886, -0.082671004345, 0.996306431311, -0.011377667052, -0.116892351938, 0.009077706061, -0.0104350677610, 0.2323464427360, -0.0000564632375, 0.0060520126375],
    [0.80, 0.586016974889, -0.075629552389, 0.806782753193, -0.079045861888, 1.635786572623, -0.081317547659, 0.995953682283, -0.011696506766, -0.121423923726, 0.009050514766, -0.0093882979190, 0.2232712080030, -0.0000633699605, 0.0061239641660],
    [0.82, 0.599580345727, -0.074125837403, 0.818350853396, -0.077590486692, 1.658530992051, -0.079777259961, 0.995560546571, -0.011991671098, -0.126142547285, 0.008988807405, -0.0083328258905, 0.2141552033065, -0.0000680304500, 0.0061813114210],
    [0.84, 0.614539083929, -0.072378038263, 0.830862496923, -0.075917929225, 1.683033655782, -0.078007941274, 0.995119134957, -0.012253083617, -0.131075225463, 0.008886290872, -0.0072630847515, 0.2049754391135, -0.0000761402500, 0.0062223026810],
    [0.86, 0.631206038694, -0.070321478391, 0.844496006954, -0.073972411405, 1.709618885024, -0.075951859914, 0.994618869600, -0.012466362882, -0.136256982615, 0.008735425951, -0.0061717338255, 0.1957004072510, -0.0000810477215, 0.0062578720305],
], dtype=float)
def interpolate_spin_tables(spins: np.ndarray, remnant_mass_ratios: np.ndarray | None = None) -> np.ndarray:
    query = np.asarray(spins, dtype=float)
    if query.ndim != 1 or query.size == 0 or not np.all(np.isfinite(query)):
        raise ValueError("spins must be a nonempty finite one-dimensional array")
    nodes = _RAW_TABLE[:, 0]
    if np.any(query < nodes[0]) or np.any(query > nodes[-1]):
        raise ValueError("spins must lie in the supplied interpolation interval")
    masses = np.ones_like(query) if remnant_mass_ratios is None else np.asarray(remnant_mass_ratios, dtype=float)
    if masses.shape != query.shape or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("remnant_mass_ratios must have finite positive shape (n,)")
    interpolated = np.column_stack([query, masses, *[np.interp(query, nodes, _RAW_TABLE[:, j]) for j in range(1, 15)]])
    return interpolated

r"""
Angular construction of two spherical quadratic responses.

The table contains the intrinsic parent frequency and task-defined synthetic
radial transfer coefficients $\rho_4,\rho_5$. Construct normalized
spin-weighted spheroidal harmonics in a spin-weighted spherical basis through
$L=12$. Project both the squared raised 220 parent and the bilinear raised
220-by-320 source onto their respective driven children, project each child
into spherical 64, and apply the stated strain conversions. The unequal-parent
transfer is $\kappa_l=-0.20\rho_l$, where `0.20` is a supplied synthetic scale
for the unequal-parent channel. The
exact matrix, phase, Gaunt, and response
conventions are in the ordered specification.

Returns
-------
`numpy.ndarray` of shape `(n, 4)` containing real and imaginary parts of
$(\mathcal R_{64}^{22},\mathcal R_{64}^{23})$. Every component is compared at
numerical tolerance `1e-9`.
"""
"""Angular construction of the spherical quadratic responses."""
import numpy as np
"""Angular construction of the spherical quadratic responses."""
import math
import numpy as np
_ANGULAR_LMAX = 12
def _wigner_3j_integer(j1, j2, j3, m1, m2, m3):
    if (
        m1 + m2 + m3 != 0
        or abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3
        or j3 < abs(j1 - j2) or j3 > j1 + j2
    ):
        return 0.0
    triangle = (
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )
    factorial_product = triangle
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        factorial_product *= math.factorial(j + m) * math.factorial(j - m)
    lower = max(0, j2 - j3 - m1, j1 - j3 + m2)
    upper = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    series = 0.0
    for z in range(lower, upper + 1):
        denominator = (
            math.factorial(z)
            * math.factorial(j1 + j2 - j3 - z)
            * math.factorial(j1 - m1 - z)
            * math.factorial(j2 + m2 - z)
            * math.factorial(j3 - j2 + m1 + z)
            * math.factorial(j3 - j1 - m2 + z)
        )
        series += (-1) ** z / denominator
    return (-1) ** (j1 - j2 - m3) * math.sqrt(factorial_product) * series
def _angular_gaunt_tensor(parent_l, child_l):
    tensor = np.zeros((parent_l.size, parent_l.size, child_l.size), dtype=float)
    for a, l1 in enumerate(parent_l):
        for b, l2 in enumerate(parent_l):
            for d, j in enumerate(child_l):
                prefactor = math.sqrt(
                    (2 * l1 + 1) * (2 * l2 + 1) * (2 * j + 1) / (4.0 * math.pi)
                )
                tensor[a, b, d] = prefactor * _wigner_3j_integer(
                    int(l1), int(l2), int(j), 1, 1, -2
                ) * _wigner_3j_integer(int(l1), int(l2), int(j), 2, 2, -4)
    return tensor
def _spheroidal_vector(spin_weight, azimuthal, spheroidicity, target_l):
    lmin = max(abs(spin_weight), abs(azimuthal))
    extended_l = np.arange(lmin, _ANGULAR_LMAX + 2, dtype=int)
    cosine = np.zeros((extended_l.size, extended_l.size), dtype=float)
    for index, ell in enumerate(extended_l):
        cosine[index, index] = -spin_weight * azimuthal / (ell * (ell + 1))
        if index + 1 < extended_l.size:
            upper = ell + 1
            coupling = math.sqrt(
                (upper ** 2 - azimuthal ** 2)
                * (upper ** 2 - spin_weight ** 2)
                / (upper ** 2 * (4 * upper ** 2 - 1))
            )
            cosine[index, index + 1] = coupling
            cosine[index + 1, index] = coupling
    cosine_squared = (cosine @ cosine)[:-1, :-1]
    cosine = cosine[:-1, :-1]
    ell = extended_l[:-1]
    spherical_eigenvalues = ell * (ell + 1) - spin_weight * (spin_weight + 1)
    angular_matrix = (
        np.diag(spherical_eigenvalues.astype(complex))
        + 2.0 * spheroidicity * spin_weight * cosine
        - spheroidicity ** 2 * cosine_squared
    )
    eigenvalues, eigenvectors = np.linalg.eig(angular_matrix)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("angular eigensystem is nonfinite")
    target = target_l * (target_l + 1) - spin_weight * (spin_weight + 1)
    distances = np.abs(eigenvalues - target)
    order = np.argsort(distances)
    if order.size < 2 or distances[order[1]] - distances[order[0]] <= 1e-10:
        raise ValueError("angular eigenbranch is ambiguous")
    vector = eigenvectors[:, order[0]].astype(complex)
    norm = math.sqrt(float(np.vdot(vector, vector).real))
    target_component = vector[target_l - lmin]
    if not np.isfinite(norm) or norm <= 0.0 or abs(target_component) <= 1e-12:
        raise ValueError("angular eigenvector cannot be normalized")
    vector /= norm
    vector *= np.exp(-1j * np.angle(vector[target_l - lmin]))
    return ell, vector
def _angular_response_at_spin(spin, omega220, radial4, radial5):
    parent_l, parent = _spheroidal_vector(-2, 2, spin * omega220, 2)
    raised_parent = -np.sqrt((parent_l + 2) * (parent_l - 1)) * parent
    response = 0.0j
    for target_l, radial in ((4, radial4), (5, radial5)):
        child_l, child = _spheroidal_vector(-2, 4, spin * (2.0 * omega220), target_l)
        gaunt = _angular_gaunt_tensor(parent_l, child_l)
        angular_source = np.einsum(
            "a,b,c,abc->", raised_parent, raised_parent, np.conjugate(child), gaunt
        )
        spherical64_overlap = child[6 - 4]
        conversion = (
            -1j * omega220 / 48.0
            * math.sqrt(math.factorial(target_l + 2) / math.factorial(target_l - 2))
        )
        response += spherical64_overlap * angular_source * conversion * radial
    return response
def _mixed_angular_response_at_spin(spin, omega220, omega320, radial4, radial5):
    parent_l, parent220 = _spheroidal_vector(-2, 2, spin * omega220, 2)
    other_l, parent320 = _spheroidal_vector(-2, 2, spin * omega320, 3)
    if not np.array_equal(parent_l, other_l):
        raise ValueError("mixed-parent angular bases do not agree")
    raised220 = -np.sqrt((parent_l + 2) * (parent_l - 1)) * parent220
    raised320 = -np.sqrt((parent_l + 2) * (parent_l - 1)) * parent320
    response = 0.0j
    parent_factor = math.sqrt(
        math.factorial(2 + 2) / math.factorial(2 - 2)
        * math.factorial(3 + 2) / math.factorial(3 - 2)
    )
    for target_l, radial in ((4, radial4), (5, radial5)):
        child_l, child = _spheroidal_vector(
            -2, 4, spin * (omega220 + omega320), target_l
        )
        gaunt = _angular_gaunt_tensor(parent_l, child_l)
        angular_source = np.einsum(
            "a,b,c,abc->", raised220, raised320, np.conjugate(child), gaunt
        )
        spherical64_overlap = child[6 - 4]
        conversion = (
            -1j * np.sqrt(omega220 * omega320) / (2.0 * parent_factor)
            * math.sqrt(math.factorial(target_l + 2) / math.factorial(target_l - 2))
        )
        unequal_parent_scale = 0.20
        response += (
            spherical64_overlap * angular_source * conversion
            * (-unequal_parent_scale * radial)
        )
    return response
def synthesize_quadratic_response(table: np.ndarray) -> np.ndarray:
    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or values.shape[1] != 16 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("table must be a nonempty finite array with shape (n, 16)")
    if np.any(values[:, 0] < 0.70) or np.any(values[:, 0] > 0.86):
        raise ValueError("spin must lie in [0.70, 0.86]")
    if np.any(values[:, 1] <= 0.0):
        raise ValueError("mass ratio must be positive")
    omega220 = values[:, 2] + 1j * values[:, 3]
    omega320 = values[:, 4] + 1j * values[:, 5]
    if np.any(omega220.real <= 0.0) or np.any(omega220.imag >= 0.0):
        raise ValueError("220 frequencies require positive real and negative imaginary parts")
    if np.any(omega320.real <= 0.0) or np.any(omega320.imag >= 0.0):
        raise ValueError("320 frequencies require positive real and negative imaginary parts")
    radial4 = values[:, 12] + 1j * values[:, 13]
    radial5 = values[:, 14] + 1j * values[:, 15]
    cache = {}
    result = np.empty((values.shape[0], 2), dtype=complex)
    for row, (spin, frequency220, frequency320, coefficient4, coefficient5) in enumerate(
        zip(values[:, 0], omega220, omega320, radial4, radial5)
    ):
        key = (
            float(spin), complex(frequency220), complex(frequency320),
            complex(coefficient4), complex(coefficient5),
        )
        if key not in cache:
            cache[key] = (
                _angular_response_at_spin(
                    spin, frequency220, coefficient4, coefficient5
                ),
                _mixed_angular_response_at_spin(
                    spin, frequency220, frequency320, coefficient4, coefficient5
                ),
            )
        result[row] = cache[key]
    if not np.all(np.isfinite(result)):
        raise ValueError("quadratic response is nonfinite")
    return np.column_stack((
        result[:, 0].real, result[:, 0].imag,
        result[:, 1].real, result[:, 1].imag,
    ))

r"""Build the mixed-mode parent design used by the analytic amplitude posterior.

For each trial state divide the intrinsic frequencies by the paired remnant-mass
ratio and form the spherical-22 columns

$$x_{22}=\mu_{22}e^{-i\omega_{220}t},\qquad
x_{32}=\mu_{32}e^{-i\omega_{320}t}.$$

A complex column ``x`` maps amplitudes ``(Re A, Im A)`` to the real data order
``(Re h[0:7], Im h[0:7])`` through columns ``(Re x, Im x)`` and
``(-Im x, Re x)``. The nonlinear response is carried as metadata for the later
quadratic-moment calculation; it is not linearized here.

Returns
-------
`numpy.ndarray` of shape `(n, 18, 4)`. Rows `0:14` are the real parent design.
Row 14 is `(spin, mass_ratio, Re omega220, Im omega220)`, row 15 is
`(Re omega320, Im omega320, Re omega640, Im omega640)`, row 16 is
`(Re R64^22, Im R64^22, Re R64^23, Im R64^23)`, and row 17 is
`(Re mu22, Im mu22, Re mu32, Im mu32)`. With a 220-only parent, design
columns `2:4` are exact zero while all physical metadata remain populated.
Every returned component is compared at tolerance `1e-9`.
"""
import numpy as np
import numpy as np
def _template_real_design(columns: list[np.ndarray]) -> np.ndarray:
    packed = []
    for column in columns:
        packed.extend((
            np.concatenate((column.real, column.imag)),
            np.concatenate((-column.imag, column.real)),
        ))
    return np.column_stack(packed)
def build_linearized_templates(
    data: np.ndarray,
    table: np.ndarray,
    quadratic_response: np.ndarray,
    include_mixed_parent: bool = True,
) -> np.ndarray:
    record = np.asarray(data, dtype=float)
    states = np.asarray(table, dtype=float)
    responses = np.asarray(quadratic_response, dtype=float)
    if record.shape != (7, 9) or not np.all(np.isfinite(record)):
        raise ValueError("data must have finite shape (7, 9)")
    if np.any(np.diff(record[:, 0]) <= 0.0):
        raise ValueError("sample times must be strictly increasing")
    if states.ndim != 2 or states.shape[1] != 16 or states.shape[0] == 0 or not np.all(np.isfinite(states)):
        raise ValueError("table must have nonempty finite shape (n, 16)")
    if responses.shape != (states.shape[0], 4) or not np.all(np.isfinite(responses)):
        raise ValueError("quadratic_response must have finite shape (n, 4)")
    if type(include_mixed_parent) is not bool:
        raise ValueError("include_mixed_parent must be an exact boolean")
    if np.any(states[:, 1] <= 0.0):
        raise ValueError("remnant-mass ratios must be positive")
    frequency_pairs = np.stack((states[:, 2:8:2], states[:, 3:8:2]), axis=2)
    if np.any(frequency_pairs[:, :, 0] <= 0.0) or np.any(frequency_pairs[:, :, 1] >= 0.0):
        raise ValueError("frequencies require positive real and negative imaginary parts")

    time = record[:, 0]
    output = np.zeros((states.shape[0], 18, 4), dtype=float)
    for index, state in enumerate(states):
        spin, mass = state[:2]
        omega220 = (state[2] + 1j * state[3]) / mass
        omega320 = (state[4] + 1j * state[5]) / mass
        omega640 = (state[6] + 1j * state[7]) / mass
        mu22 = state[8] + 1j * state[9]
        mu32 = state[10] + 1j * state[11]
        parent220 = mu22 * np.exp(-1j * omega220 * time)
        parent320 = mu32 * np.exp(-1j * omega320 * time)
        design = _template_real_design([parent220, parent320])
        if not include_mixed_parent:
            design[:, 2:4] = 0.0
        output[index, :14] = design
        output[index, 14] = (spin, mass, omega220.real, omega220.imag)
        output[index, 15] = (
            omega320.real, omega320.imag, omega640.real, omega640.imag
        )
        output[index, 16] = responses[index]
        output[index, 17] = (mu22.real, mu22.imag, mu32.real, mu32.imag)
    if not np.all(np.isfinite(output)):
        raise ValueError("template construction produced nonfinite output")
    return output

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

r"""Residual-conditioned analytic posterior for the spherical-22 parent.

Let $e$ be the complete 28-coordinate highest-resolution numerical error and
$d$ the complete observed highest-minus-next-highest residual in step-05
order. Condition the full correlated system before selecting its parent block:

$$m_e=K_{ed}K_{dd}^{-1}d,\qquad
K_{e|d}=K_{ee}-K_{ed}K_{dd}^{-1}K_{de}.$$

For the conditioned real data vector $y-m_e$, design $X$, covariance
$K_{e|d}$, and task-defined proper Cartesian prior
$a\sim N(0,s_a^2I_d)$ on the $d=2$ or $d=4$ active coordinates, the posterior
is Gaussian:

$$F=X^TK^{-1}X+s_a^{-2}I_d,\qquad
V=F^{-1},\qquad \bar a=VX^TK^{-1}y.$$

The state score is the amplitude-marginalized log likelihood with common
state-independent constants omitted,

$$\log Z_p=-\tfrac12[y^TK^{-1}y-\bar a^TF\bar a
+\log|K|+\log|F|+d\log(s_a^2)].$$

The $d\log(s_a^2)$ prior-normalization term must be retained because the
220-only and mixed-parent models have different active dimensions. The default
uses $s_a=0.75$. Only the common $14\log(2\pi)$ data-normalization constant is
omitted.

All positive-definite applications use checked Cholesky factors. The complex
mode significance is $1-\exp(-d^2/2)$ using the two-real-dimensional
marginal mean and covariance.

Returns
-------
`numpy.ndarray` of shape `(n, 25)`: spin, mass, four amplitude-mean
coordinates, the row-major `4 by 4` posterior covariance, conditional parent
log score, and the 220 and 320 significances. In a 220-only fit the 320 mean,
covariance rows and columns, and significance are exact zero. Components are
compared at `1e-9`.
"""
import numpy as np
import numpy as np
_ALL_ERROR_INDICES = np.arange(28)
_ALL_RESOLUTION_INDICES = 28 + _ALL_ERROR_INDICES
_PARENT_COVARIANCE_INDICES = np.r_[0:7, 14:21]
def _posterior_cholesky(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(label + " must be finite and symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(label + " must be positive definite") from exc
def _posterior_factor_solve(factor: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor.T, np.linalg.solve(factor, right_hand_side))
def _condition_parent_error(
    joint_covariance: np.ndarray,
    resolution_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error_covariance = joint_covariance[:28, :28]
    cross_covariance = joint_covariance[
        np.ix_(_ALL_ERROR_INDICES, _ALL_RESOLUTION_INDICES)
    ]
    resolution_covariance = joint_covariance[
        np.ix_(_ALL_RESOLUTION_INDICES, _ALL_RESOLUTION_INDICES)
    ]
    resolution_factor = _posterior_cholesky(
        resolution_covariance, "parent resolution covariance"
    )
    conditional_mean = cross_covariance @ _posterior_factor_solve(
        resolution_factor, resolution_observed
    )
    conditional_covariance = error_covariance - cross_covariance @ (
        _posterior_factor_solve(resolution_factor, cross_covariance.T)
    )
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    _posterior_cholesky(conditional_covariance, "conditional numerical covariance")
    return (
        conditional_mean[_PARENT_COVARIANCE_INDICES],
        conditional_covariance[
            np.ix_(_PARENT_COVARIANCE_INDICES, _PARENT_COVARIANCE_INDICES)
        ],
    )
def evaluate_conditional_posteriors(
    data: np.ndarray,
    templates: np.ndarray,
    covariances: np.ndarray,
    amplitude_prior_scale: float = 0.75,
) -> np.ndarray:
    record = np.asarray(data, dtype=float)
    design_pack = np.asarray(templates, dtype=float)
    covariance_pack = np.asarray(covariances, dtype=float)
    prior_scale = float(amplitude_prior_scale)
    if record.shape != (7, 9) or not np.all(np.isfinite(record)):
        raise ValueError("data must have finite shape (7, 9)")
    if design_pack.ndim != 3 or design_pack.shape[1:] != (18, 4) or design_pack.shape[0] == 0 or not np.all(np.isfinite(design_pack)):
        raise ValueError("templates must have nonempty finite shape (n, 18, 4)")
    if covariance_pack.shape != (design_pack.shape[0], 56, 56) or not np.all(np.isfinite(covariance_pack)):
        raise ValueError("covariances must have finite shape (n, 56, 56)")
    if np.any(design_pack[:, 14, 1] <= 0.0):
        raise ValueError("template masses must be positive")
    if (
        not np.isfinite(prior_scale)
        or prior_scale <= 0.0
        or prior_scale > np.sqrt(np.finfo(float).max)
    ):
        raise ValueError("amplitude_prior_scale must be finite, positive, and squareable")
    y_complex = record[:, 1] + 1j * record[:, 2]
    observed = np.concatenate((y_complex.real, y_complex.imag))
    resolution_observed = np.concatenate(
        (record[:, 5], record[:, 7], record[:, 6], record[:, 8])
    )
    output = np.zeros((design_pack.shape[0], 25), dtype=float)

    for index, (template, full_covariance) in enumerate(zip(design_pack, covariance_pack)):
        full_factor = _posterior_cholesky(full_covariance, "joint numerical covariance")
        del full_factor
        design = template[:14]
        first_active = np.linalg.norm(design[:, :2], axis=0) > 0.0
        mixed_active = np.linalg.norm(design[:, 2:4], axis=0) > 0.0
        if not np.all(first_active) or mixed_active[0] != mixed_active[1]:
            raise ValueError("active complex-mode columns must occur in real-imaginary pairs")
        active = np.array([0, 1, 2, 3] if np.all(mixed_active) else [0, 1])
        selected_design = design[:, active]
        if np.linalg.matrix_rank(selected_design) != active.size:
            raise ValueError("active parent design must have full column rank")
        error_mean, covariance = _condition_parent_error(
            full_covariance, resolution_observed
        )
        conditioned_observed = observed - error_mean
        covariance_factor = _posterior_cholesky(covariance, "parent numerical covariance")
        inverse_design = _posterior_factor_solve(covariance_factor, selected_design)
        inverse_observed = _posterior_factor_solve(
            covariance_factor, conditioned_observed
        )
        prior_variance = prior_scale * prior_scale
        fisher = selected_design.T @ inverse_design
        fisher += np.eye(active.size) / prior_variance
        fisher = 0.5 * (fisher + fisher.T)
        fisher_factor = _posterior_cholesky(
            fisher, "active amplitude posterior precision"
        )
        posterior_covariance = _posterior_factor_solve(
            fisher_factor, np.eye(active.size)
        )
        posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
        _posterior_cholesky(posterior_covariance, "active posterior covariance")
        mean = posterior_covariance @ selected_design.T @ inverse_observed
        quadratic = conditioned_observed @ inverse_observed - mean @ fisher @ mean
        logdet_covariance = 2.0 * np.log(np.diag(covariance_factor)).sum()
        logdet_fisher = 2.0 * np.log(np.diag(fisher_factor)).sum()
        score = -0.5 * (
            quadratic
            + logdet_covariance
            + logdet_fisher
            + active.size * np.log(prior_variance)
        )

        full_mean = np.zeros(4, dtype=float)
        full_mean[active] = mean
        full_posterior_covariance = np.zeros((4, 4), dtype=float)
        full_posterior_covariance[np.ix_(active, active)] = posterior_covariance
        significances = np.zeros(2, dtype=float)
        for mode in range(active.size // 2):
            mode_slice = slice(2 * mode, 2 * mode + 2)
            marginal = posterior_covariance[mode_slice, mode_slice]
            marginal_factor = _posterior_cholesky(
                marginal, "complex-amplitude marginal covariance"
            )
            whitened = np.linalg.solve(marginal_factor, mean[mode_slice])
            significances[mode] = -np.expm1(-0.5 * (whitened @ whitened))

        output[index, :2] = template[14, :2]
        output[index, 2:6] = full_mean
        output[index, 6:22] = full_posterior_covariance.ravel(order="C")
        output[index, 22] = score
        output[index, 23:25] = np.clip(significances, 0.0, 1.0)
    if not np.all(np.isfinite(output)):
        raise ValueError("analytic posterior produced nonfinite output")
    return output

r"""Residual-conditioned prediction of the driven spherical-64 child.

Let the real parent coordinates be $a=(\Re C,\Im C,\Re E,\Im E)$ with the
analytic Gaussian posterior from step 06.  Each real child coordinate has the
quadratic form $f_i(a)=a^TA_i a$ generated by

$$h_{64}^{(2)}=\mathcal R_{64}^{22}C^2e^{-2i\omega_{220}t}
+2\mathcal R_{64}^{23}CEe^{-i(\omega_{220}+\omega_{320})t}.$$

For symmetric real matrices $A_i$, use the exact Gaussian moments

$$E[f_i]=\bar a^TA_i\bar a+\operatorname{tr}(A_iV),$$
$$\operatorname{Cov}(f_i,f_j)=2\operatorname{tr}(A_iVA_jV)
+4\bar a^TA_iVA_j\bar a.$$

The task-defined predictive closure treats those exact first two moments as a
Gaussian. First condition the complete 28-coordinate highest-resolution error
$e=(e_p,e_c)$ on the complete observed residual $d$ using

$$m_e=K_{ed}K_{dd}^{-1}d,\qquad
K_{e|d}=K_{ee}-K_{ed}K_{dd}^{-1}K_{de}.$$

Partition the result into parent and child blocks $(m_p,m_c,K_{pp},K_{pc},
K_{cc})$. The spherical-22 observation also informs the child numerical error.
For parent design $X$ and conditioned parent observation $y_p$, define

$$G=K_{cp}K_{pp}^{-1},\qquad B=GX,\qquad
b=m_c+G(y_p-m_p),\qquad R=K_{cc}-GK_{pc}.$$

Given the step-06 amplitude posterior $a\sim N(\bar a,V)$, the total child is
$f(a)+b-Ba+\eta$ with $\eta\sim N(0,R)$. Therefore its mean is
$E[f]+b-B\bar a$. If row $i$ of
$Q=\operatorname{Cov}(f,a)$ is $2\bar a^TA_iV$, its covariance is

$$\Sigma=\operatorname{Cov}(f)+R+BVB^T-QB^T-BQ^T.$$

The child score is $-[r^T\Sigma^{-1}r+\log|\Sigma|]/2$. Cholesky factors are
used for every positive-definite application. The complete supplied residual
record is one correlated 28-coordinate calibration observation with score

$$\ell_\Delta=-\frac12[d^TK_{dd}^{-1}d+\log|K_{dd}|].$$

No density for $A_\beta$ is added. The factorization is
$p(d_{22},d_{64})p(h_{22},h_{64}\mid d_{22},d_{64})$, so the total state score
is the sum of the conditional parent, conditional child, and residual scores.

Returns
-------
`numpy.ndarray` of shape `(n, 8)`: spin, mass, parent score, child score,
their sum plus the resolution score, the quadratic child Mahalanobis value
$r^T\Sigma^{-1}r$ (not its square root), and the real and imaginary total
conditional predictive mean at the first time. Components are compared at
`1e-9`.
"""
import numpy as np
import numpy as np
_PARENT_COVARIANCE_INDICES = np.r_[0:7, 14:21]
_CHILD_COVARIANCE_INDICES = np.r_[7:14, 21:28]
_ALL_ERROR_INDICES = np.arange(28)
_ALL_RESOLUTION_INDICES = 28 + _ALL_ERROR_INDICES
def _child_cholesky(matrix: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, matrix.T, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(label + " must be finite and symmetric")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(label + " must be positive definite") from exc
def _child_factor_solve(factor: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor.T, np.linalg.solve(factor, right_hand_side))
def _condition_complete_error(
    joint_covariance: np.ndarray,
    resolution_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error_covariance = joint_covariance[:28, :28]
    cross_covariance = joint_covariance[
        np.ix_(_ALL_ERROR_INDICES, _ALL_RESOLUTION_INDICES)
    ]
    resolution_covariance = joint_covariance[
        np.ix_(_ALL_RESOLUTION_INDICES, _ALL_RESOLUTION_INDICES)
    ]
    resolution_factor = _child_cholesky(
        resolution_covariance, "complete resolution covariance"
    )
    conditional_mean = cross_covariance @ _child_factor_solve(
        resolution_factor, resolution_observed
    )
    conditional_covariance = error_covariance - cross_covariance @ (
        _child_factor_solve(resolution_factor, cross_covariance.T)
    )
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    _child_cholesky(
        conditional_covariance, "conditional complete numerical covariance"
    )
    return conditional_mean, conditional_covariance
def _quadratic_coordinate_matrices(
    omega220: complex,
    omega320: complex,
    response22: complex,
    response23: complex,
    time: np.ndarray,
) -> np.ndarray:
    self_coefficients = response22 * np.exp(-2j * omega220 * time)
    mixed_coefficients = 2.0 * response23 * np.exp(
        -1j * (omega220 + omega320) * time
    )
    complex_matrices = []
    for self_coefficient, mixed_coefficient in zip(
        self_coefficients, mixed_coefficients
    ):
        matrix = np.zeros((4, 4), dtype=complex)
        matrix[0, 0] = self_coefficient
        matrix[1, 1] = -self_coefficient
        matrix[0, 1] = matrix[1, 0] = 1j * self_coefficient
        matrix[0, 2] = matrix[2, 0] = 0.5 * mixed_coefficient
        matrix[1, 3] = matrix[3, 1] = -0.5 * mixed_coefficient
        matrix[0, 3] = matrix[3, 0] = 0.5j * mixed_coefficient
        matrix[1, 2] = matrix[2, 1] = 0.5j * mixed_coefficient
        complex_matrices.append(matrix)
    return np.stack(
        [matrix.real for matrix in complex_matrices]
        + [matrix.imag for matrix in complex_matrices]
    )
def _quadratic_gaussian_moments(
    mean: np.ndarray,
    covariance: np.ndarray,
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_mean = np.einsum("i,kij,j->k", mean, matrices, mean)
    predicted_mean += np.einsum("kij,ji->k", matrices, covariance)
    predicted_covariance = np.empty((matrices.shape[0], matrices.shape[0]))
    for left_index, left in enumerate(matrices):
        for right_index, right in enumerate(matrices):
            predicted_covariance[left_index, right_index] = (
                2.0 * np.trace(left @ covariance @ right @ covariance)
                + 4.0 * mean @ left @ covariance @ right @ mean
            )
    predicted_covariance = 0.5 * (
        predicted_covariance + predicted_covariance.T
    )
    return predicted_mean, predicted_covariance
def score_heldout_h64(
    data: np.ndarray,
    templates: np.ndarray,
    covariances: np.ndarray,
    posteriors: np.ndarray,
    response_scale: float = 1.0,
) -> np.ndarray:
    record = np.asarray(data, dtype=float)
    design_pack = np.asarray(templates, dtype=float)
    covariance_pack = np.asarray(covariances, dtype=float)
    posterior_pack = np.asarray(posteriors, dtype=float)
    scale = float(response_scale)
    if record.shape != (7, 9) or not np.all(np.isfinite(record)):
        raise ValueError("data must have finite shape (7, 9)")
    if design_pack.ndim != 3 or design_pack.shape[1:] != (18, 4) or design_pack.shape[0] == 0 or not np.all(np.isfinite(design_pack)):
        raise ValueError("templates must have nonempty finite shape (n, 18, 4)")
    if covariance_pack.shape != (design_pack.shape[0], 56, 56) or not np.all(np.isfinite(covariance_pack)):
        raise ValueError("covariances must have finite shape (n, 56, 56)")
    if posterior_pack.shape != (design_pack.shape[0], 25) or not np.all(np.isfinite(posterior_pack)):
        raise ValueError("posteriors must have finite shape (n, 25)")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("response_scale must be finite and nonnegative")
    if not np.allclose(posterior_pack[:, :2], design_pack[:, 14, :2], rtol=0.0, atol=1.0e-12):
        raise ValueError("posterior and template state rows must agree")
    if np.any(posterior_pack[:, 23:25] < 0.0) or np.any(posterior_pack[:, 23:25] > 1.0):
        raise ValueError("amplitude significances must lie in [0, 1]")

    parent_complex = record[:, 1] + 1j * record[:, 2]
    parent_observed = np.concatenate((parent_complex.real, parent_complex.imag))
    child_complex = record[:, 3] + 1j * record[:, 4]
    child_observed = np.concatenate((child_complex.real, child_complex.imag))
    resolution_observed = np.concatenate(
        (record[:, 5], record[:, 7], record[:, 6], record[:, 8])
    )
    time = record[:, 0]
    output = np.empty((design_pack.shape[0], 8), dtype=float)
    for index, (template, full_covariance, posterior) in enumerate(
        zip(design_pack, covariance_pack, posterior_pack)
    ):
        _child_cholesky(full_covariance, "joint numerical covariance")
        mean = posterior[2:6]
        amplitude_covariance = posterior[6:22].reshape((4, 4), order="C")
        active_mixed = np.any(template[:14, 2:4] != 0.0)
        if not active_mixed and (
            np.any(mean[2:4] != 0.0)
            or np.any(amplitude_covariance[2:, :] != 0.0)
            or np.any(amplitude_covariance[:, 2:] != 0.0)
            or posterior[24] != 0.0
        ):
            raise ValueError("inactive mixed-parent packing must be exact zero")
        active_count = 4 if active_mixed else 2
        _child_cholesky(
            amplitude_covariance[:active_count, :active_count],
            "active parent posterior covariance",
        )
        omega220 = template[14, 2] + 1j * template[14, 3]
        omega320 = template[15, 0] + 1j * template[15, 1]
        response22 = scale * (template[16, 0] + 1j * template[16, 1])
        response23 = scale * (template[16, 2] + 1j * template[16, 3])
        matrices = _quadratic_coordinate_matrices(
            omega220, omega320, response22, response23, time
        )
        physical_mean, response_covariance = _quadratic_gaussian_moments(
            mean, amplitude_covariance, matrices
        )
        complete_error_mean, complete_error_covariance = _condition_complete_error(
            full_covariance, resolution_observed
        )
        parent_error_mean = complete_error_mean[_PARENT_COVARIANCE_INDICES]
        child_error_mean = complete_error_mean[_CHILD_COVARIANCE_INDICES]
        parent_error_covariance = complete_error_covariance[
            np.ix_(_PARENT_COVARIANCE_INDICES, _PARENT_COVARIANCE_INDICES)
        ]
        child_error_covariance = complete_error_covariance[
            np.ix_(_CHILD_COVARIANCE_INDICES, _CHILD_COVARIANCE_INDICES)
        ]
        child_parent_covariance = complete_error_covariance[
            np.ix_(_CHILD_COVARIANCE_INDICES, _PARENT_COVARIANCE_INDICES)
        ]
        parent_factor = _child_cholesky(
            parent_error_covariance, "conditional parent numerical covariance"
        )
        gain = child_parent_covariance @ _child_factor_solve(
            parent_factor, np.eye(parent_error_covariance.shape[0])
        )
        linear_coupling = gain @ template[:14]
        child_intercept = child_error_mean + gain @ (
            parent_observed - parent_error_mean
        )
        predicted_mean = physical_mean + child_intercept - linear_coupling @ mean
        residual_numerical_covariance = child_error_covariance - (
            gain @ child_parent_covariance.T
        )
        quadratic_parent_covariance = np.stack(
            [2.0 * mean @ matrix @ amplitude_covariance for matrix in matrices]
        )
        child_covariance = (
            response_covariance
            + residual_numerical_covariance
            + linear_coupling @ amplitude_covariance @ linear_coupling.T
            - quadratic_parent_covariance @ linear_coupling.T
            - linear_coupling @ quadratic_parent_covariance.T
        )
        child_covariance = 0.5 * (child_covariance + child_covariance.T)
        child_factor = _child_cholesky(child_covariance, "child predictive covariance")
        residual = child_observed - predicted_mean
        mahalanobis = residual @ _child_factor_solve(child_factor, residual)
        logdet = 2.0 * np.log(np.diag(child_factor)).sum()
        child_score = -0.5 * (mahalanobis + logdet)
        resolution_covariance = full_covariance[28:, 28:]
        resolution_factor = _child_cholesky(
            resolution_covariance, "complete resolution calibration covariance"
        )
        resolution_quadratic = resolution_observed @ _child_factor_solve(
            resolution_factor, resolution_observed
        )
        resolution_logdet = 2.0 * np.log(np.diag(resolution_factor)).sum()
        resolution_score = -0.5 * (
            resolution_quadratic + resolution_logdet
        )
        output[index] = (
            posterior[0], posterior[1], posterior[22], child_score,
            posterior[22] + child_score + resolution_score, mahalanobis,
            predicted_mean[0], predicted_mean[7],
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("child prediction produced nonfinite output")
    return output

r"""Marginalize remnant mass and normalize the spin posterior.

For state scores $\ell_{ka}$ at spin $k$ and mass node $a$, normalize the
provided mass quadrature weights inside each spin and form

$$L_k=\operatorname{LSE}_a(\log w_a+\ell_{ka}),\qquad
W_k=\frac{e^{L_k}}{\sum_j e^{L_j}}.$$

The task uses a uniform prior over its discrete spin grid. Conditional scalar
summaries use the same normalized mass-node weights.

Returns
-------
`numpy.ndarray` of shape `(s, 7)` with spin, marginalized log score, normalized
spin weight, conditional mean mass, parent score, child score, and child
Mahalanobis residual. Components are compared at `1e-9`.
"""
import numpy as np
import numpy as np
def _spin_logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    total = float(np.sum(np.exp(values - maximum)))
    if not np.isfinite(maximum) or not np.isfinite(total) or total <= 0.0:
        raise ValueError("log weights cannot be normalized")
    return maximum + np.log(total)
def normalize_spin_weights(
    predictive_scores: np.ndarray,
    mass_log_weights: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(predictive_scores, dtype=float)
    mass_prior = np.asarray(mass_log_weights, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 8 or scores.shape[0] == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("predictive_scores must have nonempty finite shape (n, 8)")
    if mass_prior.shape != (scores.shape[0],) or not np.all(np.isfinite(mass_prior)):
        raise ValueError("mass_log_weights must have finite shape (n,)")
    if np.any(scores[:, 1] <= 0.0):
        raise ValueError("mass ratios must be positive")
    spins = np.unique(scores[:, 0])
    groups = [np.flatnonzero(scores[:, 0] == spin) for spin in spins]
    counts = np.array([group.size for group in groups])
    if np.any(counts == 0) or np.any(counts != counts[0]):
        raise ValueError("every spin must have the same positive mass-node count")
    output = np.empty((spins.size, 7), dtype=float)
    log_marginals = np.empty(spins.size, dtype=float)
    for spin_index, (spin, indices) in enumerate(zip(spins, groups)):
        masses = scores[indices, 1]
        if np.unique(masses).size != masses.size:
            raise ValueError("mass nodes must be unique within each spin")
        local_log_prior = mass_prior[indices]
        local_log_prior -= _spin_logsumexp(local_log_prior)
        state_log_weights = local_log_prior + scores[indices, 4]
        log_marginal = _spin_logsumexp(state_log_weights)
        local_weights = np.exp(state_log_weights - log_marginal)
        local_weights /= local_weights.sum()
        log_marginals[spin_index] = log_marginal
        output[spin_index] = (
            spin, log_marginal, 0.0,
            local_weights @ masses,
            local_weights @ scores[indices, 2],
            local_weights @ scores[indices, 3],
            local_weights @ scores[indices, 5],
        )
    normalization = _spin_logsumexp(log_marginals)
    output[:, 2] = np.exp(log_marginals - normalization)
    output[:, 2] /= output[:, 2].sum()
    if (
        not np.all(np.isfinite(output))
        or np.any(output[:, 2] < 0.0)
        or not np.isclose(output[:, 2].sum(), 1.0, rtol=0.0, atol=1.0e-14)
    ):
        raise ValueError("spin posterior failed normalization")
    return output

r"""Final binary-black-hole remnant-spin inference.

Use spins `0.738, 0.748, ..., 0.848` with a uniform discrete prior and a
uniform remnant-mass-ratio prior on `[0.96, 1.04]`. Integrate the mass prior
with the seven-point Gauss-Legendre rule mapped to that interval and normalized
to unit total weight. The default path uses `error_model="anchor_gp"`, the
joint residual-conditioned numerical-error hierarchy, together with the full
correlated residual likelihood and parent-conditioned child prediction from
step 07.
Evaluate two parent-content hypotheses: `M0` contains only the 220 parent and
its self-coupled child, while `M1` contains the 220 and 320 parents and both
self and mixed child responses. Also marginalize three task-supplied coupled
calibration regimes `(response scale, amplitude-prior scale, q, rho, gamma)`
equal to `(0.75,0.55,0.40,0.82,0.10)`, `(1.00,0.75,0.50,0.85,0.08)`, and
`(1.25,1.20,0.48,0.82,0.30)`, with prior probabilities `0.75`, `0.15`, and
`0.10`. These are paired regimes, not independent parameter grids, and each
requires its own covariance and conditional likelihood. Within every regime,
`M0` and `M1` have equal conditional probability. Marginalize mass separately
in all six regime-model cells, combine each cell with prior weight `p_r/2`,
then normalize the spin posterior. Do not average already-normalized spin
posteriors. The optional scale arguments multiply the three response or
amplitude-prior entries and provide alternate configurations.

Returns
-------
A native finite `float`, the posterior mean remnant spin, compared at `1e-9`.
"""
import numpy as np
import numpy as np
_INFERENCE_SPINS = np.array([
    0.738, 0.748, 0.758, 0.768, 0.778, 0.788,
    0.798, 0.808, 0.818, 0.828, 0.838, 0.848,
])
_CALIBRATION_REGIMES = (
    (0.75, 0.75, 0.55, 0.40, 0.82, 0.10),
    (0.15, 1.00, 0.75, 0.50, 0.85, 0.08),
    (0.10, 1.25, 1.20, 0.48, 0.82, 0.30),
)
def infer_remnant_spin(
    response_scale: float = 1.0,
    error_model: str = "anchor_gp",
    amplitude_scale: float = 1.0,
    time_scale: float = 1.0,
    amplitude_prior_scale: float = 1.0,
) -> float:
    scale = float(response_scale)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("response_scale must be finite and nonnegative")
    data = load_ringdown_data(amplitude_scale, time_scale)
    mass_nodes_standard, mass_weights_standard = np.polynomial.legendre.leggauss(7)
    mass_nodes = 1.0 + 0.04 * mass_nodes_standard
    state_spins = np.repeat(_INFERENCE_SPINS, mass_nodes.size)
    state_masses = np.tile(mass_nodes, _INFERENCE_SPINS.size)
    mass_log_weights = np.tile(np.log(mass_weights_standard / 2.0), _INFERENCE_SPINS.size)
    table = interpolate_spin_tables(state_spins, state_masses)
    response = synthesize_quadratic_response(table)
    templates_m0 = build_linearized_templates(data, table, response, False)
    templates_m1 = build_linearized_templates(data, table, response, True)
    block_frequencies = np.column_stack((
        templates_m1[:, 14, 2:4], templates_m1[:, 15, 2:4]
    ))
    cell_rows = []
    cell_log_weights = []
    for (
        regime_weight,
        regime_response,
        regime_prior_scale,
        regime_q,
        regime_rho,
        regime_gamma,
    ) in _CALIBRATION_REGIMES:
        covariances = build_correlated_covariances(
            data,
            block_frequencies,
            error_model,
            1.0,
            1.0,
            regime_q,
            regime_rho,
            regime_gamma,
        )
        for templates in (templates_m0, templates_m1):
            posteriors = evaluate_conditional_posteriors(
                data,
                templates,
                covariances,
                amplitude_prior_scale * regime_prior_scale,
            )
            predictive_scores = score_heldout_h64(
                data,
                templates,
                covariances,
                posteriors,
                scale * regime_response,
            )
            cell_rows.append(
                normalize_spin_weights(
                    predictive_scores, mass_log_weights
                )
            )
            cell_log_weights.append(np.log(regime_weight / 2.0))
    reference_spins = cell_rows[0][:, 0]
    if any(
        not np.array_equal(rows[:, 0], reference_spins)
        for rows in cell_rows[1:]
    ):
        raise ValueError("regime-model spin grids must agree")
    cell_log_evidence = np.stack([rows[:, 1] for rows in cell_rows])
    weighted_log_evidence = cell_log_evidence + np.asarray(
        cell_log_weights
    )[:, None]
    cell_maximum = np.max(weighted_log_evidence, axis=0)
    combined_log_evidence = cell_maximum + np.log(
        np.sum(np.exp(weighted_log_evidence - cell_maximum), axis=0)
    )
    maximum = float(np.max(combined_log_evidence))
    spin_weights = np.exp(combined_log_evidence - maximum)
    spin_weights /= spin_weights.sum()
    answer = float(reference_spins @ spin_weights)
    if not np.isfinite(answer):
        raise ValueError("posterior mean is nonfinite")
    return answer
SCICODE_GOLD_EOF
