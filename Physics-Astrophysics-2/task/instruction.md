# Physics-Astrophysics-2

## Background

Relativistic spin precession in binary black holes has a celebrated closed-form solution at 1.5PN order, where the conservation of the orbital angular momentum magnitude and of $\mathbf{l}\cdot\mathbf{s}_{\rm eff}$ renders the dynamics integrable. At 2PN order the spin–spin couplings enter and this structure collapses: neither quantity is conserved, and earlier published versions of the 2PN precession equations even propagate a sign error in a spin–spin term. Building on a conserved quantity first identified for the orbit-averaged system by earlier work, and on secular solution techniques developed for spin-precessing binaries over the preceding decade, the orbit-averaged 2PN dynamics admits a closed-form solution: one nutation degree of freedom obeying a Jacobi-elliptic law, with azimuthal precession given by incomplete elliptic integrals of the third kind.

Orbit averaging, however, discards real physics: on each orbit the separation varies strongly for an eccentric binary, and the spins respond with fast oscillations on the orbital timescale that the averaged solution cannot represent. The central new construction of a 2026 analysis is a hybrid model that repairs this: it recovers the fast orbital-timescale oscillations of the unaveraged dynamics while inheriting the correct 2PN secular evolution of the orbit-averaged system by construction. Remarkably, the hybrid equations still admit a closed-form solution, with the nutation and azimuthal phases carrying an orbital modulation. Against direct numerical integration of the full 2PN equations, the analysis finds the hybrid solution substantially more accurate than both the 1.5PN solution and the orbit-averaged 2PN solution.

This task poses one concrete, deterministic instance of that construction: a pinned binary configuration evolved under the hybrid model to a fixed reduced time, from which a single scalar — the azimuthal precession phase of the primary spin about the conserved total angular momentum — must be computed, together with the orbital-phase checkpoints that certify the hybrid evolution was actually performed.

## Problem

Spin precession in eccentric binary black holes at second post-Newtonian (2PN) order can be solved in closed form after orbit averaging, but averaging suppresses the fast oscillations that the spins undergo on the orbital timescale. A recent analysis of the conservative 2PN spin dynamics introduced a hybrid construction that restores those orbital-timescale oscillations while preserving the orbit-averaged secular evolution, and showed that it still admits a closed-form solution. Your input is a fully specified binary configuration; your output is the azimuthal precession phase accumulated by the primary spin about the conserved total angular momentum under that hybrid evolution.

Work in the reduced variables of that analysis with $G = c = 1$: masses $m_1 = 2/3$ and $m_2 = 1/3$ (total mass $m = 1$), dimensionless spin magnitudes $\chi_1 = 0.9$ and $\chi_2 = 0.8$ entering through the reduced spins $s_a = \chi_a m_a^2/(\mu m)$ with $\mu$ the reduced mass, reduced orbital energy $h = -1/256$, and reduced orbital angular momentum $l = 8.965$. The initial angles are $\langle\mathbf{l},\mathbf{s}_1\rangle = 32^{\circ}$, $\langle\mathbf{l},\mathbf{s}_2\rangle = 82^{\circ}$, and $\langle\mathbf{s}_1,\mathbf{s}_2\rangle = 54^{\circ}$, with $\mathbf{l}\cdot(\mathbf{s}_1\times\mathbf{s}_2) > 0$ at $t = 0$, and the binary is at periapsis at $t = 0$.

Construct the analysis's hybrid model and evaluate its closed-form solution, tracking the orbital phase through the 1PN quasi-Keplerian parametrization and accumulating every anomaly and phase continuously across all orbits and nutation cycles (do not reduce any angle modulo an orbit or a cycle). Report the azimuthal phase $\Delta\phi_{S_1}$ of the primary spin about the conserved total angular momentum $\mathbf{j}$, accumulated from $t = 0$ to the reduced time $t = 5\times10^{6}$, in radians. In the reasoning, also report four checkpoint scalars: the continuous eccentric anomaly $u$ and the continuous angular anomaly $v_\theta$ at $t = 5\times10^{6}$, the 1PN averaging radius $d$, and the accumulated azimuthal phase $\Delta\phi_L$ of the orbital angular momentum over the same interval.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

binary_invariants

Goal
----
Mass parameters, reduced spins, and conserved invariants.

```python
import numpy as np


# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def binary_invariants(m1: float, m2: float, chi1: float, chi2: float,
                      l: float, kappa1_0: float, kappa2_0: float,
                      gamma_0: float, c: float = 1.0) -> np.ndarray:
    '''Mass parameters, reduced spin magnitudes, and conserved invariants.

    Parameters
    ----------
    m1 : float
        Primary mass, must satisfy m1 > m2 > 0.
    m2 : float
        Secondary mass.
    chi1 : float
        Dimensionless spin magnitude of the primary, in (0, 1].
    chi2 : float
        Dimensionless spin magnitude of the secondary, in (0, 1].
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    kappa1_0 : float
        Initial angle between l and s1 in radians, in (0, pi).
    kappa2_0 : float
        Initial angle between l and s2 in radians, in (0, pi).
    gamma_0 : float
        Initial angle between s1 and s2 in radians, in (0, pi).
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    inv : np.ndarray
        Array of shape (12,):
        [nu, mu, delta1, delta2, sigma1, sigma2, s1, s2, j, lam,
         Sigma1, Sigma2].

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if the masses do not satisfy
        m1 > m2 > 0; if chi1 or chi2 lies outside (0, 1] (both spins must be
        nonvanishing for the conserved-quantity construction); if l <= 0 or
        c <= 0; or if any initial angle lies outside (0, pi).
    '''
    return None  # placeholder
```

### Step 2

quasi_keplerian_elements

Goal
----
1PN quasi-Keplerian orbital elements and the averaging radius

```python
import numpy as np

def quasi_keplerian_elements(h: float, l: float, nu: float,
                             c: float = 1.0) -> np.ndarray:
    '''1PN quasi-Keplerian elements from reduced energy and angular momentum.

    Parameters
    ----------
    h : float
        Reduced orbital energy, must be < 0 (bound orbit).
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    nu : float
        Symmetric mass ratio, in (0, 0.25].
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    elements : np.ndarray
        Array of shape (6,): [a_r, e_r, n, e_t, e_theta, d].

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if h >= 0; if l <= 0 or
        c <= 0; if nu lies outside (0, 0.25]; or if (h, l) do not
        describe an eccentric bound orbit (e_r^2 <= 0, e_t^2 <= 0, or
        |e_theta| >= 1).
    '''
    return None  # placeholder
```

### Step 3

nutation_cubic_roots

Goal
----
Nutation cubic and its ordered roots.

```python
import numpy as np

def nutation_cubic_roots(m1: float, m2: float, l: float, s1: float,
                         s2: float, lam: float, big_sigma1: float,
                         big_sigma2: float) -> np.ndarray:
    '''Ordered roots of the nutation cubic and its overall factor A.

    Parameters
    ----------
    m1 : float
        Primary mass, must satisfy m1 > m2 > 0.
    m2 : float
        Secondary mass.
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    s1 : float
        Reduced spin magnitude of the primary, must be > 0.
    s2 : float
        Reduced spin magnitude of the secondary, must be > 0.
    lam : float
        Conserved quantity lambda = (l . s0) / l^2, must satisfy lam != 1.
    big_sigma1 : float
        Conserved quantity Sigma1.
    big_sigma2 : float
        Conserved quantity Sigma2.

    Returns
    -------
    out : np.ndarray
        Array of shape (4,): [x_minus, x_plus, x_3, A] with
        x_minus < x_plus < x_3 and A > 0.

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if the masses do not satisfy
        m1 > m2 > 0; if l, s1 or s2 <= 0; if lam == 1; or if the cubic
        does not have three real roots.
    '''
    return None  # placeholder
```

### Step 4

kepler_eccentric_anomaly

Goal
----
Eccentric anomaly of the 1PN quasi-Keplerian orbit

```python
import numpy as np

def kepler_eccentric_anomaly(t_grid: np.ndarray, n: float, e_t: float,
                             t0: float = 0.0) -> np.ndarray:
    '''Continuous eccentric anomaly u(t) from the quasi-Keplerian
    Kepler equation n (t - t0) = u - e_t sin u.

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of reduced times, entries finite.
    n : float
        Mean motion in reduced units, must be > 0.
    e_t : float
        Time eccentricity, must satisfy 0 <= e_t < 1.
    t0 : float
        Epoch of periapsis passage, must be finite.

    Returns
    -------
    u : np.ndarray
        1-D array, same shape as t_grid, the continuous (unreduced)
        eccentric anomaly in radians. For the computed mean anomaly
        ell = n * (t - t0), the absolute error in u must be at most
        1e-12 + 8 * eps * max(1, abs(ell)) per entry, where eps is
        the float64 machine epsilon. This allows for the resolution
        of large unreduced angles. Any sufficiently accurate solver
        is acceptable.

    Raises
    ------
    ValueError
        If t_grid is not a nonempty 1-D array of finite times; if n is
        not a finite scalar > 0; if e_t is not a finite scalar in
        [0, 1); if t0 is not a finite scalar; or if computing
        ell = n * (t - t0) in float64 produces a nonfinite value.
    RuntimeError
        If the numerical solver cannot establish convergence.
    '''
    return None  # placeholder
```

### Step 5

angular_anomaly

Goal
----
Continuous angular anomaly of the 1PN quasi-Keplerian orbit

```python
import numpy as np

def angular_anomaly(u: np.ndarray, e_theta: float) -> np.ndarray:
    '''Continuous angular anomaly v_theta from the eccentric anomaly.

    Parameters
    ----------
    u : np.ndarray
        1-D array of continuous eccentric-anomaly values in radians,
        entries finite.
    e_theta : float
        Angular eccentricity, must satisfy 0 <= e_theta < 1.

    Returns
    -------
    v_theta : np.ndarray
        1-D array, same shape as u, the continuous angular anomaly in
        radians.

    Raises
    ------
    ValueError
        If u is not a nonempty 1-D array with finite entries, or if
        e_theta is not a finite scalar in [0, 1).
    '''
    return None  # placeholder
```

### Step 6

nutation_angle

Goal
----
Jacobi-elliptic nutation angle under the hybrid evolution

```python
import numpy as np

def nutation_angle(t_grid: np.ndarray, x_minus: float, x_plus: float,
                   x_3: float, big_a: float, d: float, n: float,
                   e_t: float, e_theta: float, x0: float,
                   sign0: float, c: float = 1.0) -> np.ndarray:
    '''Nutation cosine cos kappa1(t) under the hybrid evolution.

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of reduced times, entries finite.
    x_minus, x_plus, x_3 : float
        Ordered nutation-cubic roots, x_minus < x_plus < x_3.
    big_a : float
        Overall cubic factor A, must be > 0.
    d : float
        1PN averaging radius, must be > 0.
    n : float
        1PN mean motion, must be > 0.
    e_t : float
        Time eccentricity of the Kepler equation, in [0, 1).
    e_theta : float
        Angular eccentricity of the anomaly and its modulation, in [0, 1).
    x0 : float
        Initial value cos kappa1(0), in [x_minus, x_plus].
    sign0 : float
        Sign of d cos kappa1 / dt at t = 0; must be +1.0 or -1.0.
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    cos_kappa1 : np.ndarray
        1-D array, same shape as t_grid, the values of cos kappa1(t).

    Raises
    ------
    ValueError
        If t_grid is not a nonempty 1-D array of finite times; if the
        roots do not satisfy x_minus < x_plus < x_3; if big_a, d, n or
        c <= 0; if e_t or e_theta lies outside [0, 1); if x0 lies
        outside [x_minus, x_plus]; or if sign0 is not +1.0 or -1.0.
    '''
    return None  # placeholder
```

### Step 7

companion_angles

Goal
----
Companion angles from the conserved linear combinations

```python
import numpy as np

def companion_angles(cos_kappa1: np.ndarray, m1: float, m2: float, l: float,
                     s1: float, s2: float, big_sigma1: float,
                     big_sigma2: float) -> np.ndarray:
    '''cos kappa2 and cos gamma from cos kappa1 and the invariants.

    Parameters
    ----------
    cos_kappa1 : np.ndarray
        1-D array of cos kappa1 values, entries finite in [-1, 1].
    m1 : float
        Primary mass, must satisfy m1 > m2 > 0.
    m2 : float
        Secondary mass.
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    s1 : float
        Reduced spin magnitude of the primary, must be > 0.
    s2 : float
        Reduced spin magnitude of the secondary, must be > 0.
    big_sigma1 : float
        Conserved quantity Sigma1.
    big_sigma2 : float
        Conserved quantity Sigma2.

    Returns
    -------
    angles : np.ndarray
        Array of shape (2, N): row 0 holds cos kappa2, row 1 holds
        cos gamma.

    Raises
    ------
    ValueError
        If cos_kappa1 is not a nonempty 1-D array with finite entries in
        [-1, 1]; if the masses do not satisfy m1 > m2 > 0; or if l, s1
        or s2 <= 0.
    '''
    return None  # placeholder
```

### Step 8

azimuthal_coefficients

Goal
----
Coefficients of the azimuthal precession equations

```python
import numpy as np

def azimuthal_coefficients(m1: float, m2: float, l: float, s1: float,
                           s2: float, j: float, lam: float,
                           big_sigma1: float, big_sigma2: float) -> np.ndarray:
    '''Coefficient sets of the azimuthal equations for l and s1.

    Parameters
    ----------
    m1 : float
        Primary mass, must satisfy m1 > m2 > 0.
    m2 : float
        Secondary mass.
    l : float
        Reduced orbital angular momentum magnitude, must be > 0.
    s1 : float
        Reduced spin magnitude of the primary, must be > 0.
    s2 : float
        Reduced spin magnitude of the secondary, must be > 0.
    j : float
        Magnitude of the total reduced angular momentum, must be > 0.
    lam : float
        Conserved quantity lambda.
    big_sigma1 : float
        Conserved quantity Sigma1.
    big_sigma2 : float
        Conserved quantity Sigma2.

    Returns
    -------
    coeffs : np.ndarray
        Array of shape (10,):
        [alpha1L, beta1L, alpha2L, beta2L, beta3L,
         alpha1S1, beta1S1, alpha2S1, beta2S1, beta3S1].

    Raises
    ------
    ValueError
        If any input is not a finite scalar; if the masses do not satisfy
        m1 > m2 > 0; or if l, s1, s2 or j <= 0.
    '''
    return None  # placeholder
```

### Step 9

azimuthal_phase

Goal
----
Accumulated azimuthal phase under the hybrid evolution

```python
import numpy as np

def azimuthal_phase(t_grid: np.ndarray, x_minus: float, x_plus: float,
                    x_3: float, big_a: float, d: float, n: float,
                    e_t: float, e_theta: float, x0: float, sign0: float,
                    alpha1: float, beta1: float, alpha2: float,
                    beta2: float, beta3: float,
                    c: float = 1.0) -> np.ndarray:
    '''Azimuthal phase accumulated since t = 0 under the hybrid evolution.

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of reduced times, entries finite.
    x_minus, x_plus, x_3 : float
        Ordered nutation-cubic roots, x_minus < x_plus < x_3.
    big_a : float
        Overall cubic factor A, must be > 0.
    d : float
        1PN averaging radius, must be > 0.
    n : float
        1PN mean motion, must be > 0.
    e_t : float
        Time eccentricity of the Kepler equation, in [0, 1).
    e_theta : float
        Angular eccentricity of the anomaly and its modulation, in [0, 1).
    x0 : float
        Initial value cos kappa1(0), in [x_minus, x_plus].
    sign0 : float
        Sign of d cos kappa1 / dt at t = 0; must be +1.0 or -1.0.
    alpha1, beta1, alpha2, beta2, beta3 : float
        Finite azimuthal coefficients of the vector being tracked. For
        each active term (beta_i != 0, i = 1, 2), alpha_i + x must
        be nonzero for every x in [x_minus, x_plus]. Equivalently,
        alpha_i + x_minus and alpha_i + x_plus must have the same
        strict sign. Terms with beta_i == 0 are omitted entirely
        and impose no restriction on alpha_i beyond finiteness.
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    dphi : np.ndarray
        1-D array, same shape as t_grid, the phase accumulated since
        t = 0, in radians.

    Raises
    ------
    ValueError
        If t_grid is not a nonempty 1-D array of finite times; if the
        roots do not satisfy x_minus < x_plus < x_3; if big_a, d, n or
        c <= 0; if e_t or e_theta lies outside [0, 1); if x0 lies
        outside [x_minus, x_plus]; if sign0 is not +1.0 or -1.0; or if
        any coefficient is nonfinite; or if an active denominator
        vanishes anywhere in [x_minus, x_plus].
    FloatingPointError
        If numerical evaluation cannot produce finite phase values.
    '''
    return None  # placeholder
```

### Step 10

run_hybrid_precession

Goal
----
Orchestrator: end-to-end hybrid 2PN precession solution

```python
import numpy as np

def run_hybrid_precession(m1: float, m2: float, chi1: float, chi2: float,
                          h: float, l: float, kappa1_0: float,
                          kappa2_0: float, gamma_0: float,
                          triple_sign: float, t_final: float,
                          c: float = 1.0) -> np.ndarray:
    '''Run the full hybrid-precession pipeline and report its diagnostics.

    Parameters
    ----------
    m1, m2 : float
        Component masses with m1 > m2 > 0 (reduced units, m1 + m2 = m).
    chi1, chi2 : float
        Dimensionless spin magnitudes, each in (0, 1].
    h : float
        Reduced orbital energy, must be < 0.
    l : float
        Reduced orbital angular momentum, must be > 0.
    kappa1_0, kappa2_0, gamma_0 : float
        Initial angles between (l, s1), (l, s2) and (s1, s2), in
        radians, each in (0, pi).
    triple_sign : float
        Sign of l . (s1 x s2) at t = 0; must be +1.0 or -1.0.
    t_final : float
        Final reduced time, must be a finite scalar > 0.
    c : float
        Speed of light in reduced units, must be > 0.

    Returns
    -------
    diagnostics : np.ndarray
        Array of shape (7,):
        [0] continuous eccentric anomaly u at t_final, in radians;
        [1] continuous angular anomaly v_theta at t_final, in radians;
        [2] 1PN averaging radius d, in reduced units;
        [3] cos kappa1 at t_final;
        [4] cos kappa2 at t_final;
        [5] accumulated azimuthal phase of the orbital angular momentum,
            Delta phi_L, in radians;
        [6] accumulated azimuthal phase of the primary spin,
            Delta phi_S1, in radians.

    Raises
    ------
    ValueError
        If any configuration parameter is invalid (for example
        m1 <= m2, a spin magnitude outside (0, 1], h >= 0, or
        triple_sign not +1.0 or -1.0) -- such errors propagate from the
        pipeline stages.
    '''
    return None  # placeholder
```
