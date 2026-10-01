# Continuous-phase error of an unrefined matter spectrum

## Background

# Scientific background

Neutrinos are produced and detected in flavor states, while their propagation is governed by states with different masses. Interference between these components changes the probability of detecting each flavor. Coherent forward scattering from electrons modifies the propagation in matter, making atmospheric neutrinos sensitive to the density structure along their passage through Earth as well as to the mixing parameters and mass ordering.

Repeated probability calculations are needed when comparing oscillation models with experimental data. A varying matter profile makes these calculations more demanding because flavor amplitudes retain their phases across density interfaces. Efficient propagation methods exploit the structure of small Hermitian matrices and the symmetries of the density profile while preserving the interference responsible for flavor conversion.

## Problem

Quantify the worst error in coherent three-flavor neutrino propagation caused by leaving an atmospheric matter-eigenvalue initializer unrefined, with the same resolved Earth profile used for both the approximate and exact calculations. Use normal mass ordering, column flavor amplitudes ordered $(e,\mu,\tau)$, no atmosphere, absorption or decoherence, and

$$
s_{12}^2=0.307,\quad s_{13}^2=0.02195,\quad s_{23}^2=0.561,\quad
m_{21}=7.49\times10^{-5}\,\mathrm{eV}^2,\quad m_{31}=2.534\times10^{-3}\,\mathrm{eV}^2,
$$

with $E\in\{2.1,3.8,6.2,9.7\}\,\mathrm{GeV}$, zenith cosine $c\in\{-0.97,-0.84,-0.41\}$, both neutrino signs $\sigma\in\{+1,-1\}$, and continuous physical CP phase $0\leq\delta<2\pi$. The detector is $2\,\mathrm{km}$ below a sphere of radius $6371\,\mathrm{km}$, $c=-1$ means a diametrical upward-going path, the source is the first surface intersection backward from the detector, and the inner-to-outer shell model is

$$
r_{\mathrm{out}}=(1221.5,3480,5701,6371)\,\mathrm{km},\qquad
Y_e=(0.467,0.467,0.495,0.495),
$$

$$
\rho_j(r)=\sum_{k=0}^3 b_{jk}\left(\frac{r}{6371\,\mathrm{km}}\right)^k,\qquad
b=\begin{pmatrix}
13.0&0&-2.0&0\\
12.0&-1.2&-2.5&0.3\\
7.5&-2.0&-0.4&0\\
5.2&-0.7&0&0
\end{pmatrix}\,\mathrm{g\,cm^{-3}}.
$$

Split every positive-length shell-crossing interval into three equal path-length pieces and replace each piece's density-electron-fraction product by its mean at 16 equally spaced path midpoints; this finite piecewise-constant model defines both propagation calculations exactly. Define $U_\sigma=R_{23}\operatorname{diag}(1,1,e^{i\sigma\delta})R_{13}R_{12}$, with each $R_{ij}$ having the positive-sine block $\begin{pmatrix}c_{ij}&s_{ij}\\-s_{ij}&c_{ij}\end{pmatrix}$ and positive square roots for sines and cosines, and take

$$
K=U_\sigma\operatorname{diag}(0,m_{21},m_{31})U_\sigma^\dagger+
\operatorname{diag}(a,0,0),\qquad
 a=\sigma(1.526493231029146\times10^{-4})\overline{\rho Y_e}E\;\mathrm{eV}^2,
\qquad \kappa=\frac{10^{-9}10^3}{2(1.97327\times10^{-7})},
$$

where a layer of length $L$ evolves by $\exp(-i\kappa KL/E)$ and $P_{\alpha\to\beta}=|S_{\beta\alpha}|^2$ for the ordered total amplitude. For the approximate calculation, factor out the common $R_{23}D_3$ rotation, use the square-root atmospheric eigenvalue initializer based on $\Delta m_{ee}^2=m_{31}-s_{12}^2m_{21}$ with its absolute vacuum normalization $\lambda_3^{(0)}(a=0)=m_{31}$ and zero characteristic-polynomial corrections, recover the lower pair from the trace and determinant of the unshifted mass-squared Hamiltonian, and form the spectral matrices $Q_2,Q_3$ for the middle and highest ordered estimated eigenvalues from the electron-row cofactors using the rank-one formulas for the $\mu\mu$ and $\mu\tau$ entries and fixing $\tau\tau$ by unit trace. Remove each layer's lowest estimated eigenvalue phase and eliminate its spectral matrix by $Q_1=I-Q_2-Q_3$, so the approximate layer amplitude is $I+\sum_{i=2}^3 Q_i[\exp(-i\kappa(\lambda_i^{(0)}-\lambda_1^{(0)})L/E)-1]$; use it without orthogonalizing spectral matrices, restoring unitarity, or clipping probabilities, and compare it with exact three-flavor propagation through precisely the same pieces. Report the single finite dimensionless error statistic

$$
\mathcal E=10^6\max_{E,c,\sigma}\max_{0\leq\delta<2\pi}
\max_{\beta\in\{e,\mu\}}
\left|P_{\mu\to\beta}^{(0)}(E,c,\sigma,\delta)
-P_{\mu\to\beta}^{\mathrm{exact}}(E,c,\sigma,\delta)\right|
$$

within absolute tolerance $10^{-4}$, resolving the continuous-phase extremum and justifying the unrefined spectrum, approximate projector convention, ordered propagation, finite harmonic representation of the error and completeness of the phase search, with the maximizing energy, zenith cosine, sign, channel, phase, phase-average error and first- and second-harmonic magnitudes as supporting evidence.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_trace_shell_segments

Goal
----
Trace a directed neutrino path through concentric spherical shells.

```python
def trace_shell_segments(
    radii_km: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    subdivisions: int = 1,
) -> "np.ndarray":
    r"""Trace a directed neutrino path through concentric spherical shells.

    Parameters
    ----------
    radii_km : np.ndarray
        Nonempty finite positive strictly increasing shell outer radii, shape (S,).
    cos_zenith : float
        Finite direction cosine in [-1, 1]; -1 points through the center.
    detector_depth_km : float
        Finite depth d in [0, R), where R is the final shell radius.
    subdivisions : int, default 1
        Positive integer number of equal path-length pieces per shell interval.

    Returns
    -------
    segments : np.ndarray
        Real shape (N, 3) array in source-to-detector order. Columns contain
        backward distance at entry, backward distance at exit, and the
        zero-based integer shell index stored as a float. Entry exceeds exit.
        A zero-length path returns shape (0, 3).

    Raises
    ------
    ValueError
        If radii, direction, depth, or subdivisions violate their domains.
        Boolean values are not accepted for subdivisions.
    """
    return
```

### Step 2

02_average_segment_densities

Goal
----
Compute path-averaged electron densities inside the traced shell pieces.

```python
def average_segment_densities(
    segments: "np.ndarray",
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    cos_zenith: float,
    detector_depth_km: float,
    quadrature_order: int = 16,
) -> "np.ndarray":
    r"""Compute path-averaged electron densities inside the traced shell pieces.

    Parameters
    ----------
    segments : np.ndarray
        Real shape (N, 3) entries from the shell trace: backward entry, exit,
        and integer-valued shell index. Require entry > exit >= 0.
    radii_km : np.ndarray
        Finite positive increasing outer radii, shape (S,).
    density_coefficients : np.ndarray
        Finite real shape (S, 4) ascending coefficients for powers of r/R,
        in grams per cubic centimeter.
    electron_fractions : np.ndarray
        Finite real shape (S,) fractions between zero and one inclusive.
    cos_zenith : float
        Finite direction cosine in [-1, 1].
    detector_depth_km : float
        Finite depth in [0, R), with R the final radius.
    quadrature_order : int, default 16
        Positive integer midpoint sample count per segment.

    Returns
    -------
    layers : np.ndarray
        Real shape (N, 2) array of path lengths in kilometers and mean density
        times electron fraction in grams per cubic centimeter.

    Raises
    ------
    ValueError
        If input shapes, finiteness, geometry, segment ordering, indices,
        fraction ranges, or quadrature count are invalid, or a sampled
        density is negative. Segment shell membership is an upstream
        precondition established by the trace function.
    """
    return
```

### Step 3

03_compute_matter_spectrum

Goal
----
Compute matter eigenvalue estimates together with reusable cofactor coefficients.

```python
def compute_matter_spectrum(
    matter_ev2: float,
    s12_sq: float,
    s13_sq: float,
    dm21_ev2: float,
    dm31_ev2: float,
    n_newton: int = 2,
) -> "np.ndarray":
    r"""Compute matter eigenvalue estimates together with reusable cofactor coefficients.

    Parameters
    ----------
    matter_ev2 : float
        Finite signed matter potential in eV squared.
    s12_sq, s13_sq : float
        Squared sines strictly between zero and one.
    dm21_ev2, dm31_ev2 : float
        Finite mass-squared differences with 0 < dm21_ev2 < dm31_ev2.
    n_newton : int, default 2
        Number of cubic corrections, between zero and eight inclusive.

    Returns
    -------
    spectrum : np.ndarray
        Real shape (3, 3) array. Row zero contains increasing eigenvalues in
        eV squared; rows one and two contain the electron-row S coefficients
        in eV squared and T coefficients in eV to the fourth, respectively.

    Raises
    ------
    ValueError
        If inputs violate their domains, a correction has a zero denominator,
        the recovery radicand is materially negative, or recovered roots are
        nonfinite or not strictly increasing. Negative roundoff up to 64
        machine epsilons times the radicand term scale is clipped to zero.
    """
    return
```

### Step 4

04_compute_eigenprojectors

Goal
----
Construct the two real spectral matrices from electron-row cofactors.

```python
def compute_eigenprojectors(
    eigenvalues_ev2: "np.ndarray", coefficients: "np.ndarray"
) -> "np.ndarray":
    r"""Construct the two real spectral matrices from electron-row cofactors.

    Parameters
    ----------
    eigenvalues_ev2 : np.ndarray
        Finite real shape (3,) array of strictly increasing eigenvalues.
    coefficients : np.ndarray
        Finite real shape (2, 3) array of electron-row S and T coefficients.

    Returns
    -------
    projectors : np.ndarray
        Real dimensionless array of shape (2, 3, 3), ordered as Q_2, Q_3.

    Raises
    ------
    ValueError
        If shapes, reality, finiteness, or strict eigenvalue ordering fail,
        if an electron diagonal entry is nonpositive, or if the completed
        matrices have nonfinite entries. The inputs must describe simple
        eigenvalues with nonzero electron projections.
    """
    return
```

### Step 5

05_propagate_spectral_pair

Goal
----
Propagate an unrefined spectral approximation and an exact reference through the same layers.

```python
def propagate_spectral_pair(
    layers: "np.ndarray", energy_gev: float, mixing: "np.ndarray", charge: int = 1
) -> "np.ndarray":
    r"""Propagate an unrefined spectral approximation and an exact reference through the same layers.

    Parameters
    ----------
    layers : np.ndarray
        Finite real shape (N, 2) of nonnegative lengths in km and density
        products q in grams per cubic centimeter, in propagation order.
    energy_gev : float
        Finite positive energy in GeV.
    mixing : np.ndarray
        Finite real shape (4,) vector s12_sq, s13_sq, m21, m31; squared
        sines in (0, 1), mass-squared differences 0 < m21 < m31 in eV squared.
    charge : int, default 1
        +1 for neutrinos or -1 for antineutrinos; Boolean values are invalid.

    Returns
    -------
    amplitudes : np.ndarray
        Complex shape (2, 3, 3), approximate then exact, with the lowest
        layer eigenvalue phases removed separately in each branch.

    Raises
    ------
    ValueError
        If input shapes, finiteness, ranges or charge are invalid, or the
        approximate spectrum or electron-row pivot cannot be recovered.
    """
    return
```

### Step 6

06_compute_error_harmonics

Goal
----
Extract the continuous CP-phase harmonics of appearance and survival probability errors.

```python
def compute_error_harmonics(
    amplitudes: "np.ndarray", s23_sq: float, charge: int = 1
) -> "np.ndarray":
    r"""Extract the continuous CP-phase harmonics of appearance and survival probability errors.

    Parameters
    ----------
    amplitudes : np.ndarray
        Finite complex shape (2, 3, 3), approximate then exact, in the real
        propagation basis. Exact unitarity is not required by this function.
    s23_sq : float
        Finite squared sine in [0, 1].
    charge : int, default 1
        +1 or -1; determines the sign of the physical CP phase.

    Returns
    -------
    coefficients : np.ndarray
        Finite real shape (2, 5). Rows are muon-to-electron and muon-to-muon
        errors; columns are constant, cos(delta), sin(delta), cos(2 delta),
        sin(2 delta). Errors are unscaled probability differences.

    Raises
    ------
    ValueError
        If amplitude shape or finiteness, the squared sine or charge is
        invalid, or squared amplitudes overflow to nonfinite coefficients.
    """
    return
```

### Step 7

07_maximize_phase_error

Goal
----
Find the global absolute extremum of a second-harmonic CP-phase error.

```python
def maximize_phase_error(coefficients: "np.ndarray") -> float:
    r"""Find the global absolute extremum of a second-harmonic CP-phase error.

    Parameters
    ----------
    coefficients : np.ndarray
        Finite real shape (5,) coefficients ordered as constant, cos(delta),
        sin(delta), cos(2 delta), sin(2 delta).

    Returns
    -------
    maximum : float
        Finite nonnegative maximum absolute value over a complete phase period.

    Raises
    ------
    ValueError
        If coefficient shape, reality or finiteness is invalid, or the
        requested finite maximum overflows floating-point representation.
    """
    return
```

### Step 8

08_compute_worst_spectral_error

Goal
----
Return the worst continuous-phase error of the unrefined matter-spectrum propagation.

```python
def compute_worst_spectral_error(
    radii_km: "np.ndarray",
    density_coefficients: "np.ndarray",
    electron_fractions: "np.ndarray",
    energies_gev: "np.ndarray",
    cos_zeniths: "np.ndarray",
    mixing: "np.ndarray",
    detector_depth_km: float = 2.0,
    subdivisions: int = 3,
    quadrature_order: int = 16,
) -> float:
    r"""Return the worst continuous-phase error of the unrefined matter-spectrum propagation.

    Parameters
    ----------
    radii_km : np.ndarray
        Finite positive increasing outer shell radii, shape (S,), in km.
    density_coefficients : np.ndarray
        Finite real shape (S, 4) ascending polynomial coefficients in r/R,
        in grams per cubic centimeter; sampled densities must be nonnegative.
    electron_fractions : np.ndarray
        Finite real shape (S,) fractions in [0, 1].
    energies_gev : np.ndarray
        Nonempty finite real positive energy vector, shape (J,), in GeV.
    cos_zeniths : np.ndarray
        Nonempty finite real direction cosines, shape (K,), in [-1, 1].
    mixing : np.ndarray
        Finite real shape (5,) vector s12_sq, s13_sq, s23_sq, m21, m31;
        first two entries in (0, 1), third in [0, 1], and 0 < m21 < m31.
        Mass-squared entries are in eV squared; CP phase is optimized.
    detector_depth_km : float, default 2.0
        Finite detector depth in [0, R), in km.
    subdivisions : int, default 3
        Positive integer pieces per geometric shell interval.
    quadrature_order : int, default 16
        Positive integer midpoint samples per piece.

    Returns
    -------
    error : float
        Finite nonnegative dimensionless error scaled by one million.

    Raises
    ------
    ValueError
        If an input violates its domain, sampled density is negative,
        an approximate spectrum or projector pivot cannot be recovered,
        or a probability harmonic or final maximum is nonfinite.
    """
    return
```
