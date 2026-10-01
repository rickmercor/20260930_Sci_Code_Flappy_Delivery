# Material_Science-Semiconductor_Materials-15

## Background

The task concerns the variational theory of a Frohlich polaron in a two-dimensional polar monolayer: a charge carrier that locally polarises the lattice and becomes dressed by a cloud of longitudinal-optical phonons, forming a quasiparticle with a lowered energy and an enhanced effective mass. In a genuinely two-dimensional polar material the electron-phonon coupling and the longitudinal-optical phonon frequency are wave-vector dependent, in contrast to the three-dimensional Frohlich problem where they are constants. The weak-coupling (Landau-Pekar/LLP) limit is obtained from second-order perturbation theory, while the full solution for arbitrary coupling uses a variational path-integral treatment; the two are compared, with emphasis on the strong-coupling enhancement of the polaron effective mass relative to the weak-coupling estimate.

## Problem

Compute the effective mass of a strongly coupled polaron in a two-dimensional polar monolayer, treating the carrier-phonon problem at arbitrary coupling strength with a two-parameter variational path-integral treatment, following the conventions fixed by the supplied paper.

A charge carrier in a polar monolayer locally polarises the lattice and becomes dressed by a cloud of longitudinal-optical (LO) phonons, forming a polaron with a lowered energy and an enhanced effective mass. Unlike the three-dimensional Fröhlich problem, a genuinely two-dimensional polar material screens non-locally, so the electron-phonon coupling and the LO-phonon frequency both acquire a wave-vector dependence set by the static and high-frequency two-dimensional screening lengths $r_0$ and $r_\infty$. The problem is fully determined by experimentally accessible parameters: the carrier band mass $m_e$, the TO phonon energy $\hbar\omega_t$, the two screening lengths, and the background dielectric constant $\epsilon$. A single effective dimensionless Fröhlich constant $\alpha_m$, built from these parameters, measures the coupling strength, and the remaining dependence enters through two dimensionless ratios $\sigma_0$ and $\sigma_t$. At weak coupling, second-order perturbation theory (the Landau-Pekar/LLP limit) fixes a binding energy $E_0$ and a polaron mass $m_{\mathrm{LLP}}$; for arbitrary coupling a two-parameter variational path-integral bound $E(v,w)$ over $v\ge w>0$ is minimised, and the polaron effective mass is read off at the energy-minimising $(v^\ast,w^\ast)$. Because the mass weights the long-time (adiabatic) response, it departs strongly from the weak-coupling estimate as the coupling grows, so the full variational solve at the energy minimum is required.

Build the following nine functions. Each returns a fresh numpy value of the shape declared below, computed deterministically with no randomness. Invalid input must raise ValueError, and each signature docstring states the conditions that trigger it. Work in units of nm (length), meV (energy), and the free-electron mass $m_0$; take $\hbar^2/2m_0 = 38.0998$ meV nm$^2$ and $e^2/4\pi\epsilon_0 = 1439.96$ meV nm.

1. characteristic_lengths(m_e, omega_t, eps) -> np.ndarray. The length-2 array [a_B, r_t] (nm): the effective Bohr radius and the TO-phonon oscillator length.

2. coupling_constant(m_e, omega_t, r0, rinf, eps) -> float. The effective dimensionless Fröhlich coupling constant $\alpha_m$.

3. dimensionless_ratios(m_e, omega_t, r0, rinf) -> np.ndarray. The length-2 array [sigma_0, sigma_t] of the two ratios that parameterise the momentum integrals.

4. weak_coupling_integral(sigma_0, sigma_t) -> float. The dimensionless weak-coupling energy integral $I_0(\sigma_0,\sigma_t)$.

5. weak_coupling_mass_integral(sigma_0, sigma_t) -> float. The dimensionless weak-coupling (LLP) mass integral $M_{\mathrm{LLP}}(\sigma_0,\sigma_t)$.

6. weak_coupling_observables(m_e, omega_t, r0, rinf, eps) -> np.ndarray. The length-2 array [E0, m_LLP]: the weak-coupling binding energy (meV) and LLP polaron mass ($m_0$ units). It must reuse the earlier functions.

7. feynman_integral(sigma_0, sigma_t, v, w) -> float. The dimensionless variational electron-phonon integral $I_F(\sigma_0,\sigma_t;v,w)$ that enters the variational energy bound.

8. feynman_mass_integral(sigma_0, sigma_t, v, w) -> float. The dimensionless variational mass integral $M_F(\sigma_0,\sigma_t;v,w)$ that renormalises the effective mass.

9. polaron_effective_mass(m_e, omega_t, r0, rinf, eps) -> float. The orchestrator. It must call the earlier functions rather than reimplementing them: form the coupling constant and ratios, minimise the variational energy bound over $v\ge w>0$, and return the full variational polaron effective mass ($m_0$ units) evaluated at the energy-minimising parameters.

Evaluation case

Consider a freestanding (background dielectric $\epsilon = 1$) two-dimensional polar monolayer whose hole carrier is representative of monolayer ZrSe$_2$: band mass $m_e = 0.32\,m_0$, TO phonon energy $\hbar\omega_t = 18.2$ meV, static screening length $r_0 = 25.9$ nm and high-frequency screening length $r_\infty = 4.82$ nm.

Report, as the single number, the full variational polaron effective mass $m_F$ (in units of $m_0$) of this carrier, obtained from the complete all-coupling variational solve: minimise the energy bound over $(v,w)$ and evaluate the mass at the minimum. It must be the full variational value, not the weak-coupling (LLP) estimate and not the bare band mass.

Report also, as evidence that the chain was executed: the effective Fröhlich coupling constant $\alpha_m$; the weak-coupling binding energy $E_0$; the weak-coupling (LLP) polaron mass $m_{\mathrm{LLP}}$; the Feynman ground-state (binding) energy $E_F$; and the energy-minimising variational parameters $(v^\ast,w^\ast)$.

Alongside the implementation and the number, explain the steps you took. State the modelling conventions you adopted at each step and justify each from the source; in particular, state how the wave-vector dependence of the coupling and of the LO-phonon frequency arises in a genuinely 2D polar material, how the effective Bohr radius, the oscillator length and the coupling constant $\alpha_m$ are defined, the form of the variational energy bound, and how the effective mass is obtained from the mass integral at the energy-minimising variational parameters. Then discuss the model's behaviour: why the polaron effective mass departs strongly from the weak-coupling (LLP) estimate as the coupling increases, so that the full all-coupling solve is required.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, report the requested quantities above and the scalars that determine them, together with the requested conventions and discussion. Do not paste momentum-grid arrays, full coefficient vectors, per-iteration minimiser paths or integrand tables.
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
In <reasoning>, report the requested quantities above and the scalars that determine them, together with the requested conventions and discussion. Do not paste momentum-grid arrays, full coefficient vectors, per-iteration minimiser paths or integrand tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

characteristic_lengths

Goal
----
Compute the effective Bohr radius and the TO-phonon oscillator length of a 2D polar monolayer.

```python
import numpy as np

def characteristic_lengths(m_e: float, omega_t: float, eps: float) -> np.ndarray:
    r"""m_e: carrier band mass in units of the free-electron mass m0 (> 0).
    omega_t: transverse-optical (TO) phonon energy hbar*omega_t in meV (> 0).
    eps: surrounding/background dielectric constant (> 0; use 1.0 for a freestanding monolayer).

    Return a length-2 numpy array [a_B, r_t] (nm): the effective Bohr radius
    a_B = 4 pi eps0 eps hbar^2/(e^2 m_e) and the oscillator length r_t = sqrt(hbar/(2 m_e omega_t))
    set by the TO phonon frequency.

    Raises:
        ValueError: if any argument is not positive and finite.
    """
    return None
```

### Step 2

coupling_constant

Goal
----
Evaluate the effective 2D Frohlich coupling constant that measures the electron-phonon interaction strength.

```python
def coupling_constant(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    r"""m_e: carrier band mass (m0 units, > 0). omega_t: TO phonon energy hbar*omega_t (meV, > 0).
    r0: static 2D screening length (nm, > 0). rinf: high-frequency 2D screening length (nm, > 0,
    with r0 > rinf). eps: background dielectric constant (> 0).

    Return the effective dimensionless Frohlich coupling constant alpha_m of the 2D polaron: the
    upper bound on the wave-vector-dependent coupling, obtained by replacing the LO frequency with
    the TO frequency and evaluating the dielectric factor at the wave vector q = 1/sqrt(r0*rinf)
    that maximises it. It is not a q-average and not the coupling evaluated at that wave vector.

    Raises:
        ValueError: for non-positive arguments or if r0 <= rinf.
    """
    return None
```

### Step 3

dimensionless_ratios

Goal
----
Form the two dimensionless screening/oscillator ratios that parameterise the 2D polaron integrals.

```python
import numpy as np

def dimensionless_ratios(m_e: float, omega_t: float, r0: float, rinf: float) -> np.ndarray:
    r"""m_e: carrier band mass (m0 units, > 0). omega_t: TO phonon energy (meV, > 0).
    r0, rinf: static and high-frequency 2D screening lengths (nm, > 0, r0 > rinf).

    Return a length-2 numpy array [sigma_0, sigma_t] of the two dimensionless ratios that fix the
    shape of the 2D polaron integrals: sigma_0 = r0/rinf and sigma_t = r_t/rinf, with
    r_t = sqrt(hbar/(2 m_e omega_t)) the oscillator length.

    Raises:
        ValueError: for non-positive arguments or if r0 <= rinf.
    """
    return None
```

### Step 4

weak_coupling_integral

Goal
----
Evaluate the dimensionless weak-coupling momentum integral of the 2D polaron energy.

```python
def weak_coupling_integral(sigma_0: float, sigma_t: float) -> float:
    r"""sigma_0: dimensionless ratio r0/rinf (> 0). sigma_t: dimensionless ratio r_t/rinf (> 0).

    Return the dimensionless weak-coupling (second-order perturbation-theory) integral
    I_0(sigma_0, sigma_t) that sets the leading polaron energy shift E_0 = -alpha_m hbar omega_t I_0.
    The integrand runs over the dimensionless momentum p in [0, inf) and includes the wave-vector
    dependence of the screening and of the LO frequency through Omega_p = sqrt((1+sigma_0 p)/(1+p)).

    Raises:
        ValueError: for non-positive arguments.
    """
    return None
```

### Step 5

weak_coupling_mass_integral

Goal
----
Evaluate the dimensionless weak-coupling (LLP) mass integral of the 2D polaron.

```python
def weak_coupling_mass_integral(sigma_0: float, sigma_t: float) -> float:
    r"""sigma_0: dimensionless ratio r0/rinf (> 0). sigma_t: dimensionless ratio r_t/rinf (> 0).

    Return the dimensionless weak-coupling (LLP) mass integral M_LLP(sigma_0, sigma_t) that sets the
    weak-coupling polaron mass m_LLP = m_e (1 + alpha_m M_LLP). It is the mass integral in the limit
    of coincident variational parameters (v = w).

    Raises:
        ValueError: for non-positive arguments.
    """
    return None
```

### Step 6

weak_coupling_observables

Goal
----
Assemble the weak-coupling polaron binding energy and LLP effective mass from the coupling constant and the dimensionless integrals.

```python
import numpy as np

def weak_coupling_observables(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> np.ndarray:
    r"""m_e, omega_t, r0, rinf, eps: as in the earlier steps (all > 0; r0 > rinf).

    Return a length-2 numpy array [E0, m_LLP]: the weak-coupling polaron binding energy
    E0 = -alpha_m hbar omega_t I_0 (meV, a negative energy shift) and the weak-coupling (LLP) polaron
    mass m_LLP = m_e (1 + alpha_m M_LLP) (in m0 units). It must reuse the earlier functions.

    Raises:
        ValueError: whenever any function it calls would raise for these arguments.
    """
    return None
```

### Step 7

feynman_integral

Goal
----
Evaluate the dimensionless electron-phonon integral of the all-coupling Feynman variational energy.

```python
def feynman_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
    r"""sigma_0, sigma_t: dimensionless ratios (> 0). v, w: the two Feynman variational parameters
    (dimensionless, with v >= w > 0).

    Return the dimensionless Feynman electron-phonon integral I_F(sigma_0, sigma_t; v, w) that enters
    the all-coupling variational energy bound E(v,w) = hbar omega_t[(v-w)^2/(2v) - alpha_m I_F]. The
    integral is over the imaginary-time variable and the dimensionless momentum p in [0, inf); the
    memory kernel of the Feynman trial action enters through the momentum- and time-dependent
    exponent.

    Raises:
        ValueError: for non-positive sigma or if not v >= w > 0.
    """
    return None
```

### Step 8

feynman_mass_integral

Goal
----
Evaluate the dimensionless Feynman mass integral that renormalises the 2D polaron effective mass.

```python
def feynman_mass_integral(sigma_0: float, sigma_t: float, v: float, w: float) -> float:
    r"""sigma_0, sigma_t: dimensionless ratios (> 0). v, w: Feynman variational parameters
    (dimensionless, v >= w > 0).

    Return the dimensionless Feynman mass integral M_F(sigma_0, sigma_t; v, w) that sets the polaron
    effective-mass renormalisation m_F = m_e (1 + alpha_m M_F) once evaluated at the energy-minimising
    variational parameters. It is a double integral over imaginary time and dimensionless momentum;
    the imaginary-time part carries an extra quadratic weight relative to the energy integral.

    Raises:
        ValueError: for non-positive sigma or if not v >= w > 0.
    """
    return None
```

### Step 9

polaron_effective_mass

Goal
----
Assemble the all-coupling pipeline and return the full Feynman polaron effective mass of a 2D polar monolayer.

```python
def polaron_effective_mass(m_e: float, omega_t: float, r0: float, rinf: float, eps: float) -> float:
    r"""m_e: carrier band mass (m0 units, > 0). omega_t: TO phonon energy hbar*omega_t (meV, > 0).
    r0, rinf: static and high-frequency 2D screening lengths (nm, > 0, r0 > rinf).
    eps: background dielectric constant (> 0; 1.0 for a freestanding monolayer).

    The orchestrator. It must call the earlier functions rather than reimplementing them: form the
    coupling constant and dimensionless ratios, minimise the all-coupling Feynman variational energy
    bound E(v,w) = hbar omega_t[(v-w)^2/(2v) - alpha_m I_F(sigma_0,sigma_t;v,w)] over v >= w > 0, and
    return the full Feynman polaron effective mass m_F = m_e (1 + alpha_m M_F) (in m0 units) evaluated
    at the energy-minimising variational parameters (v*, w*).

    Raises:
        ValueError: whenever any function it calls would raise for these arguments.
    """
    return None
```
