# Physics-Particle_Physics-23

## Background

Near a two-body threshold, repeated long-range interactions can amplify dark-matter annihilation strongly enough that an undressed short-distance rate violates partial-wave unitarity. The loss of flux into relativistic annihilation products feeds back on the scattering state. In a multistate system the long-range and short-range channel matrices generally have different eigenvectors.

Use natural units, \(\hbar=c=1\). The incident energy relative to channel 0 is \(E=p^2/(2\mu)\), \(v_{\rm rel}=p/\mu\), and channel momenta are \(k_i=\sqrt{p^2-d_i}\), with positive real roots for open channels and positive imaginary roots for closed channels. Radial waves and their derivatives are continuous at each shell boundary. At infinity, an open outgoing s-wave is proportional to \(e^{ik_i r}\); a closed-channel wave decays. If \(w\) is the regular scattering solution with an incident sine wave in each open channel, its origin coefficient is \(w(r)\sim\operatorname{diag}(k_i r)Q\), and the ordinary Sommerfeld matrix is \(\Sigma_0=\operatorname{diag}(k_i)QP^{-1}\), where \(P\) contains only open momenta. Rows retain the stated physical channel order and columns retain increasing open-channel order. The leading resonant prescription applies to the already matched amplitude \(\bar f_s\); it retains the constant near-resonance relation between the regular and irregular radial families. The two-shell potential and the halo mixture are specified benchmark instances.

## Problem

A three-state dark sector has reduced mass \(\mu=1000\,\mathrm{GeV}\), orbital angular momentum \(\ell=0\), and real two-body potential \(2\mu V(r)=\operatorname{diag}(d)-gW_j\) in successive spherical shells of widths \((0.43,0.57)\,\mathrm{GeV}^{-1}\), with \(2\mu V=\operatorname{diag}(d)\) outside, where the ordered channel data are
\[
d=(0,0.004^2,0.011^2)\,\mathrm{GeV}^2,\qquad
W_1=\begin{pmatrix}2.9&-1.3&0.55\\-1.3&4.2&-0.8\\0.55&-0.8&1.5\end{pmatrix}\mathrm{GeV}^2,\quad
W_2=\begin{pmatrix}1.4&0.6&-0.45\\0.6&2.7&1.1\\-0.45&1.1&3.2\end{pmatrix}\mathrm{GeV}^2.
\]
Let \(g_*\) be the unique zero-energy scattering pole in \([0.9,1.2]\), characterized by a regular radial solution that is constant in channel 0 and decays in channels 1 and 2 outside the potential, and use \(g=(1+10^{-4})g_*\).
The finite, already matched short-distance amplitude in this same basis is \(\bar f_s=3\times10^{-4}(H+iG)\,\mathrm{GeV}^{-1}\), with
\[
H=\begin{pmatrix}0.2&-0.13&0.09\\-0.13&-0.11&0.05\\0.09&0.05&0.07\end{pmatrix},\qquad
G=\begin{pmatrix}1&0.45&-0.2\\0.45&0.7&0.15\\-0.2&0.15&0.6\end{pmatrix}.
\]
For an incident pair in channel 0, compare the regular-wave, leading resonant multistate unitarization prescription, which reconstructs the dispersive outgoing-wave regulator from the long-range scattering matrix, with the otherwise identical prescription using \(\bar Z=0\); denote them by F and A, respectively.
For each prescription define \(\mathcal A=\langle\sigma_{\rm ann}v_{\rm rel}\rangle\), \(\mathcal C=\sum_{j>0}\langle\sigma_{0\to j}v_{\rm rel}\rangle\), and \(B=\mathcal A/(\mathcal A+\mathcal C)\), using distinguishable-particle normalization and the open-channel scattering matrix normalized to unit flux.
The channel-0 momentum \(p\) has density proportional to \(0.7M(p;0.0002)+0.3M(p;0.006)\) on \([10^{-5},0.03]\,\mathrm{GeV}\), where \(M(p;t)=4p^2e^{-(p/t)^2}/(\sqrt\pi t^3)\); define every average by 64-point Gauss–Legendre quadrature on each interval with boundaries \((10^{-5},0.001,0.004,0.011,0.03)\,\mathrm{GeV}\), followed by one normalization of the combined weighted density.
Determine the dimensionless scalar \(B_F/B_A-1\), and report \(g_*\) and the four dimensionless rates \((a_F,c_F,a_A,c_A)=\mu p_{\rm ref}(\mathcal A_F,\mathcal C_F,\mathcal A_A,\mathcal C_A)/\pi\), with \(p_{\rm ref}=0.001\,\mathrm{GeV}\), giving the regulator relation, its behavior at a multistate pole, and the flux-loss relation supporting the calculation.
Report the scalar to six decimal places and the five diagnostics to five significant figures, with absolute tolerance \(10^{-6}\) on the scalar and relative tolerance \(3\times10^{-4}\) on each diagnostic; all propagation and quadrature use unrounded values, and the numerical setup is a constructed application of the resonant prescription.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

propagate_shells

Goal
----
Propagate a regular coupled radial basis through successive constant spherical shells.

```python
import numpy as np

def propagate_shells(p: float, gaps: np.ndarray, wells: np.ndarray, widths: np.ndarray, depth: float) -> np.ndarray:
    """Propagate a regular coupled radial basis through successive constant spherical shells.

    p : float
        Reference momentum in GeV, p >= 0.
    gaps : ndarray, shape (N,)
        Nonnegative threshold offsets 2*mu*Delta_i in GeV^2, nondecreasing, gaps[0]=0.
    wells : ndarray, shape (J,N,N)
        Real symmetric attraction matrices W_j in GeV^2; shell potential
        2*mu*V_j = diag(gaps)-depth*W_j. Repulsive eigenvalues are permitted.
    widths : ndarray, shape (J,)
        Positive successive shell widths in GeV^-1, ordered from the origin.
    depth : float
        Dimensionless interaction multiplier.
    Returns
    -------
    ndarray, shape (2*N,N), real
        Top N rows are U(R) in GeV^-1; bottom N rows are U'(R), dimensionless,
        at R=sum(widths), with U(0)=0 and U'(0)=I. The radial equation is
        -U''+(2*mu*V-p^2*I)U=0; column order is the origin basis order.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Negative momentum, unordered thresholds, nonpositive widths, inconsistent shell counts or nonsymmetric wells.
        Symmetry/Hermiticity is checked with rtol=1e-10 and atol=1e-12.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 2

threshold_pole

Goal
----
Locate a zero-energy pole with decaying closed channels and constant neutral asymptotics.

```python
import numpy as np

def threshold_pole(gaps: np.ndarray, wells: np.ndarray, widths: np.ndarray, bracket: tuple) -> float:
    """Locate a zero-energy pole with decaying closed channels and constant neutral asymptotics.

    gaps : ndarray, shape (N,)
        Strictly increasing threshold offsets in GeV^2, gaps[0]=0; other entries positive.
    wells : ndarray, shape (J,N,N)
        Real symmetric attraction matrices in GeV^2 as in propagate_shells.
    widths : ndarray, shape (J,)
        Positive shell widths in GeV^-1.
    bracket : tuple of two floats
        Positive ordered dimensionless depths, bracketing one simple pole.
    Returns
    -------
    float
        Dimensionless depth where a nonzero regular zero-energy solution is
        constant in channel 0 and decays in the other channels outside R.
        Resolve the root to absolute accuracy 1e-11. Use propagate_shells.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Non-strict threshold ordering, invalid shell data, or a bracket without two positive ordered endpoints and a sign change (an endpoint root is allowed).
        Symmetry/Hermiticity is checked with rtol=1e-10 and atol=1e-12.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return value
```

### Step 3

match_channels

Goal
----
Match a regular radial basis to incident and outgoing waves and closed-channel decay.

```python
import numpy as np

def match_channels(p: float, gaps: np.ndarray, radius: float, fundamental: np.ndarray) -> np.ndarray:
    """Match a regular radial basis to incident and outgoing waves and closed-channel decay.

    p : float
        Positive reference momentum in GeV; p^2 differs from every threshold.
    gaps : ndarray, shape (N,)
        Nondecreasing nonnegative threshold offsets in GeV^2, gaps[0]=0.
    radius : float
        Positive outer radius R in GeV^-1.
    fundamental : ndarray, shape (2*N,N)
        Real U(R) above U'(R), normalized by U(0)=0, U'(0)=I, as in propagate_shells.
    Returns
    -------
    ndarray, complex, shape (N+M,M)
        Top N rows: dimensionless ordinary Sommerfeld matrix Sigma_0.
        Bottom M rows: dimensionless unit-flux elastic matrix S_0.
        M counts gaps[i]<p^2. Open channels occur in increasing physical order;
        columns are incident channels. For k_i=sqrt(p^2-gaps[i]), use positive
        real or positive imaginary roots. Open outgoing waves are exp(i*k_i*r).
        The regular solution has w~diag(k_i*r)Q at the origin and incident sine
        waves in open channels; Sigma_0=diag(k_i)Q P^-1. Closed waves decay.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momentum/radius, exact channel threshold, unordered gaps, inconsistent fundamental shape or singular matching system.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 4

resonant_regulator

Goal
----
Reconstruct the leading resonant dispersive regulator from regular scattering data.

```python
import numpy as np

def resonant_regulator(p_open: np.ndarray, matched: np.ndarray) -> np.ndarray:
    """Reconstruct the leading resonant dispersive regulator from regular scattering data.

    p_open : ndarray, shape (M,)
        Positive open-channel momenta in GeV, in physical channel order.
    matched : ndarray, shape (N+M,M)
        Sigma_0 above the unit-flux S_0, in the match_channels format.
        S_0 is unitary. The smallest singular value of S_0-I exceeds
        M*eps, with eps the float64 machine precision; eigenvalue +1 is excluded.
    Returns
    -------
    ndarray, complex, shape (N,N)
        Hermitian dispersive outgoing-wave regulator Z in GeV from the leading
        near-resonance, regular-wave multistate prescription. An eigenvalue -1
        of S_0 is permitted. Return the Hermitian part to remove roundoff.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent (N+M,M) shape with N>=M>=1, nonunitary S_0 or singular S_0-I.
        Elastic unitarity is checked with rtol=atol=1e-8.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 5

dress_channels

Goal
----
Dress the ordinary Sommerfeld matrix with short-distance absorptive and dispersive feedback.

```python
import numpy as np

def dress_channels(p_open: np.ndarray, matched: np.ndarray, regulator: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    """Dress the ordinary Sommerfeld matrix with short-distance absorptive and dispersive feedback.

    p_open : ndarray, shape (M,)
        Positive open momenta in GeV.
    matched : ndarray, shape (N+M,M)
        Ordinary Sigma_0 above S_0 as in match_channels.
    regulator : ndarray, shape (N,N)
        Hermitian outgoing-wave regulator Z in GeV; zero is an allowed comparison.
    short_amplitude : ndarray, shape (N,N)
        Finite matched complex amplitude f in GeV^-1, with positive semidefinite
        (f-f^dagger)/(2i). Singular f is allowed; the dressing system is nonsingular.
    Returns
    -------
    ndarray, complex, shape (N,M)
        Dimensionless dressed regular-wave enhancement, in physical state rows
        and incident open-channel columns, to contract with the absorptive part
        of f. It is the continuous extension of the resonant prescription to
        singular f. Inputs retain their original values.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent shapes, nonunitary S_0, non-Hermitian regulator, negative absorptive eigenvalue or singular dressing system.
        Symmetry/Hermiticity is checked with rtol=1e-10 and atol=1e-12.
        Elastic unitarity is checked with rtol=atol=1e-8.
        Absorptive eigenvalues may be negative only within 1e-12*max(1,||Im_H(f)||_2).
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 6

annihilation_loss

Goal
----
Compute the inclusive annihilation loss operator in the incident unit-flux basis.

```python
import numpy as np

def annihilation_loss(p_open: np.ndarray, dressed: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    """Compute the inclusive annihilation loss operator in the incident unit-flux basis.

    p_open : ndarray, shape (M,)
        Positive incident open momenta in GeV.
    dressed : ndarray, shape (N,M)
        Dimensionless dressed regular-wave enhancement from dress_channels.
    short_amplitude : ndarray, shape (N,N)
        Finite amplitude f in GeV^-1 with positive semidefinite absorptive part.
    Returns
    -------
    ndarray, complex, shape (M,M)
        Dimensionless Hermitian inclusive loss operator L=I-S^dagger S.
        L[i,i] gives sigma_ann*v_rel multiplied by mu*p_open[i]/pi for
        distinguishable particles in the s-wave. Off-diagonal entries retain
        the coherence of incident channels. Evaluate from the absorptive
        contraction of the dressed wave. Return its Hermitian part.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent shapes with N>=M>=1, or a negative absorptive eigenvalue.
        Absorptive eigenvalues may be negative only within 1e-12*max(1,||Im_H(f)||_2).
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 7

conversion_strength

Goal
----
Compute state-conversion probabilities from the full regulated scattering matrix.

```python
import numpy as np

def conversion_strength(p_open: np.ndarray, matched: np.ndarray, dressed: np.ndarray, short_amplitude: np.ndarray) -> np.ndarray:
    """Compute state-conversion probabilities from the full regulated scattering matrix.

    p_open : ndarray, shape (M,)
        Positive open momenta in GeV, ordered by physical channel.
    matched : ndarray, shape (N+M,M)
        Ordinary Sigma_0 above S_0 as in match_channels.
    dressed : ndarray, shape (N,M)
        Dressed regular-wave enhancement for this same configuration.
    short_amplitude : ndarray, shape (N,N)
        Finite matched amplitude f in GeV^-1, allowing a singular matrix,
        with positive semidefinite absorptive part (f-f^dagger)/(2i).
    Returns
    -------
    ndarray, real, shape (M,M)
        Dimensionless transition strengths: entry [j,i] is |S[j,i]|^2 for j!=i,
        and every diagonal entry is zero. Rows are outgoing open channels;
        columns are incident ones. Thus column i summed and multiplied by
        pi/(mu*p_open[i]) is the conversion rate coefficient in GeV^-2.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Nonpositive momenta, inconsistent shapes with N>=M>=1, nonunitary S_0 or a negative absorptive eigenvalue.
        Elastic unitarity is checked with rtol=atol=1e-8.
        Absorptive eigenvalues may be negative only within 1e-12*max(1,||Im_H(f)||_2).
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 8

halo_branching

Goal
----
Form the branching fraction after averaging incident-channel reaction rates over a momentum distribution.

```python
import numpy as np

def halo_branching(nodes: np.ndarray, weights: np.ndarray, losses: np.ndarray, conversions: np.ndarray, p_ref: float) -> np.ndarray:
    """Form the branching fraction after averaging incident-channel reaction rates over a momentum distribution.

    nodes : ndarray, shape (Q,)
        Positive channel-0 momenta p in GeV.
    weights : ndarray, shape (Q,)
        Nonnegative quadrature weights times momentum density, with positive sum.
    losses : ndarray, shape (Q,)
        Nonnegative dimensionless inclusive neutral-channel flux losses L[0,0].
    conversions : ndarray, shape (Q,)
        Nonnegative dimensionless sums of off-diagonal neutral-incident transition strengths.
    p_ref : float
        Positive reference momentum in GeV, used to scale rate coefficients.
    Returns
    -------
    ndarray, real, length 3
        Entries [a,c,B], all dimensionless and in this order. a and c are
        mu*p_ref/pi times the averaged annihilation and conversion rate
        coefficients. B is their branching fraction a/(a+c). The weights
        describe one combined distribution and a+c is positive.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        Unequal or empty vector shapes, nonpositive momenta, negative weights/strengths, zero total weight or zero total averaged reaction rate.
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return result
```

### Step 9

resonant_branching_shift

Goal
----
Orchestrate the pole search, coupled scattering, two regulator prescriptions, and halo branching comparison.

```python
import numpy as np

def resonant_branching_shift(detuning: float = 1e-4, eta: float = 3e-4, cold_weight: float = 0.7, order: int = 64) -> float:
    """Orchestrate the pole search, coupled scattering, two regulator prescriptions, and halo branching comparison.

    detuning : float
        Fractional change of the potential depth relative to the pole in [.9,1.2];
        supported range [-0.001,0.001].
    eta : float
        Positive amplitude scale in GeV^-1, f=eta*(H+iG); supported range
        [1e-5,0.003]. H and G are the prompt matrices.
    cold_weight : float
        Mixture weight of M(p;0.0002), in [0,1]; M(p;0.006) has weight 1-cold_weight.
    order : int
        Gauss-Legendre order per stated interval, 16 through 256 inclusive.
    Returns
    -------
    float
        Dimensionless B_F/B_A-1 for the prompt's fixed three-state matrices,
        shell widths, thresholds, pole bracket, momentum interval and mixture
        scales. F uses the leading resonant regulator; A uses Z=0. Both retain
        the full complex f. The defaults specify the main task.
        Call and combine every preceding public function: propagate_shells,
        threshold_pole, match_channels, resonant_regulator, dress_channels,
        annihilation_loss, conversion_strength, and halo_branching. Valid parameters
        use nonsingular linear systems throughout this benchmark.
    Raises
    ------
    ValueError
        Invalid rank, shape, nonnumeric or nonfinite input. All dimensions are nonzero.
        An input outside its supported interval, a nonscalar parameter or an order that is not an integer (Boolean orders are invalid).
        Numerical linear systems must be nonsingular and outputs finite.
    """
    return value
```
