# Material_Science-Molecular_Modeling-34

## Background

The supplied paper studies additive binary square-well mixtures within classical density functional theory. Its hard-sphere reference is the White-Bear version of fundamental measure theory, and it recasts the mean-field square-well attraction as fundamental-measure weighted densities. The construction is applied to the liquid-vapor interface of a square-well solvent and to the effective one-body interaction that this interface exerts on a dilute nano-particle solute, which is strongly adsorbed at the interface. The task asks for the strength of that adsorption, expressed as the peak of the normalised solute density profile, for a specified solute at the solvent liquid-vapor interface. The conventions that fix the answer -- the square-well range convention, the White-Bear functional and its planar weight functions, the Lorentz-Berthelot mixing rules for the cross interaction, and the equal-well-width rule relating the solute range to its size -- are all set by the paper.

## Problem

Implement the classical density-functional theory of a dilute nano-particle at the liquid-vapor interface of a square-well solvent, and evaluate how strongly the interface adsorbs the nano-particle, following the conventions fixed by the supplied paper.

A pure square-well fluid (the solvent, component 1) phase-separates into liquid and vapor. A second, dilute component (the solute, or nano-particle) is added at vanishing concentration. The solvent is treated with the White-Bear version of fundamental measure theory for the hard-sphere reference, plus the paper's mean-field (random-phase) square-well perturbation written in the fundamental-measure style. The dilute solute feels the inhomogeneous solvent through its one-body direct correlation; the resulting effective one-body interaction adsorbs the solute at the interface, and its strength is measured by the peak of the normalised solute density profile. Work in the planar geometry with the interface normal along $z$, in units $kT = 1$ and solvent diameter $\sigma_1 = 1$ (so the solvent hard-sphere radius is $R_1 = 0.5$).

Build the following nine functions. Each returns a fresh numpy value of the shape declared below, computed deterministically with no randomness. Invalid input must raise ValueError, and each signature docstring states the conditions that trigger it.

1. sw_bulk_coexistence(beps, lam) -> shape $(2,)$. Return the coexisting liquid and vapor densities $[\rho_l, \rho_v]$ of the pure square-well fluid at reduced depth $\beta\varepsilon$ and range $\lambda$, from equal pressure and equal chemical potential of the two bulk phases. Take the hard-sphere reference and the mean-field square-well bulk terms from the paper.

2. fmt_weight_functions(R, dz) -> shape $(6, M)$. Return the six planar fundamental-measure weight functions of a hard sphere of radius $R$ on a grid of spacing $dz$, in the row order $[w_0, w_1, w_2, w_3, w_{v1}, w_{v2}]$. The scalar weights must reproduce the exact three-dimensional geometric moments of the sphere.

3. sw_meanfield_kernel(beps, rng, dz) -> shape $(M_2,)$. Return the planar-projected mean-field square-well kernel of depth $\beta\varepsilon$ and outer range $\text{rng}$. Its continuum integral must equal the full three-dimensional integrated strength of the well; the finite-grid sum is the unrenormalized quadrature approximation specified in Step 03; take the range convention (which part of the pair separation the well covers) from the paper.

4. solvent_density_profile(rho_l, rho_v, weights, kernel, beps, lam, dz, L) -> shape $(2, N)$. Solve the equilibrium solvent profile $\rho_1(z)$ across the planar interface by damped iteration of the Euler-Lagrange equation $\rho(z) = \exp(\beta\mu + c^{(1)}(z))$, with the FMT weights and mean-field kernel supplied. Row 0 is the grid $z \in [-L/2, L/2]$ (spacing $dz$), row 1 is $\rho_1(z)$, with vapor at $z \to -\infty$ and liquid at $z \to +\infty$. Every weighted density and convolution must carry the correct bulk behaviour beyond the ends of the box.

5. surface_tension(profile, weights, kernel, beps, lam, dz) -> float. Return the reduced surface tension $\beta\gamma\sigma_1^2$ of the solvent interface as the excess grand potential per unit area, $\int dz\,[\beta\omega(z) + \beta P]$.

6. lorentz_berthelot_cross(sig1, lam1, beps1, sig2, lam2, beps2) -> shape $(2,)$. Return $[r_{12}, \beta\varepsilon_{12}]$, the outer range $r_{12}$ (a length) and depth of the cross square-well interaction between solvent and solute under the additive mixing rules of the model. Recover from the paper how the cross depth, cross diameter and cross range are combined, and which product sets the outer range that enters the kernel.

7. solute_one_body_correlation(profile, weights1, weights2, cross_kernel, dz) -> shape $(N,)$. Return the inhomogeneous one-body direct correlation $c_2^{(1)}(z)$ of a single dilute solute in the solvent field: the sum of its hard-sphere fundamental-measure contribution (the solvent's FMT derivative fields convolved with the solute's own weight functions, which carry the solute size) and its cross square-well contribution.

8. solute_profile_peak(c1_2) -> float. Return the peak interfacial enhancement $P = \max_z \rho_2(z)/\rho_{2,0}$, where the normalised solute profile is $\rho_2(z)/\rho_{2,0} = \exp(c_2^{(1)}(z) - c_2^{(1)}(+\infty))$ referred to the liquid reservoir at $z \to +\infty$.

9. interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L) -> float. The orchestrator. It must call the eight earlier functions rather than reimplementing them. For a solute of size ratio $\sigma_2/\sigma_1 = \text{ratio}$ (so $\sigma_2 = \text{ratio}$, $R_2 = \sigma_2/2$) and depth $\beta\varepsilon_2$, with the solute range fixed by the equal-well-width rule $\lambda_2 = 1 + (\lambda_1 - 1)/\text{ratio}$, assemble bulk coexistence, weights and kernel, solvent profile and surface tension, cross parameters and solute correlation, and return the peak interfacial enhancement $P$.

Evaluation case

Take the solvent to be the square-well fluid with $\beta\varepsilon_1 = 1.0$ and $\lambda_1 = 1.5$. Add a dilute nano-particle solute of size ratio $\sigma_2/\sigma_1 = 2.0$ and square-well depth $\beta\varepsilon_2 = 1.20$; its range follows the equal-well-width rule above. Solve the solvent interface on the grid $dz = 0.0025$ over a box of length $L = 44$ (in units of $\sigma_1$).

The nano-particle is attracted to the liquid-vapor interface and its density develops a pronounced maximum there. The strength of that adsorption is the peak enhancement $P = \max_z \rho_2(z)/\rho_{2,0}$ of the normalised solute density profile.

Report, as the single number, that peak interfacial enhancement $P$ of the reference nano-particle. It must be the full inhomogeneous cDFT value obtained from the solute's one-body correlation in the interfacial solvent field, not the leading-order local-density (bulk chemical-potential) estimate.

Report also, as evidence that the chain was executed, the peak enhancement for two further nano-particles of the same size $\sigma_2/\sigma_1 = 2.0$ inserted at the same solvent interface but with different well depths: a more-solvophobic shallower well $\beta\varepsilon_2 = 1.0$, and the symmetric-solution depth $\beta\varepsilon_2 = 1.3918$. Report as well the solvent coexisting densities $\rho_l$ and $\rho_v$, the reduced solvent surface tension $\beta\gamma\sigma_1^2$, and the cross-well outer range $r_{12}$ of the reference solute.

Alongside the implementation and the number, explain the steps you took to get the final answer. State the modelling conventions you adopted at each step and justify each one from the source; in particular, state the hard-sphere reference and mean-field square-well bulk terms that fix coexistence, the range convention of the square-well kernel and why its integrated strength is what it is, how the equilibrium profile is iterated and how the correct bulk behaviour is maintained beyond the box, how the surface tension is obtained from the grand potential, the mixing rules that set the cross depth and cross range and which product enters the kernel, how the solute one-body correlation combines a hard-sphere part built from the solute's own weight functions with the cross square-well part, and how the normalised solute profile and its peak follow. In the explanation, also identify the paper's signed square-well weight construction and its quadratic weighted-density free-energy expression, with source attribution. Implementing the mathematically equivalent projected pair kernel is acceptable. Explain how the hard-sphere functional derivative determines the contraction of the scalar and vector weights. Then discuss the model's behaviour: why the solute is driven to the interface even when it is neither strongly solvophilic nor solvophobic, and how the peak enhancement grows as the solute becomes larger relative to the solvent.

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

sw_bulk_coexistence

Goal
----
Locate the bulk liquid-vapor coexistence of a one-component square-well fluid.

```python
def sw_bulk_coexistence(beps: float, lam: float) -> np.ndarray:
    r"""beps: float reduced square-well depth $\beta\varepsilon$ of the one-component fluid, > 0.
    lam: float square-well range $\lambda$ in units of the hard-sphere diameter, > 1.

    Returns a numpy float64 array of shape $(2,)$: the coexisting liquid and vapor number
    densities $[\rho_l, \rho_v]$ (in units of $\sigma^{-3}$, $\sigma = 1$) of the bulk
    square-well fluid at the given temperature, with $\rho_l > \rho_v$.

    Raises:
        ValueError: if beps is not a finite positive scalar, or lam is not a finite scalar > 1.
    """
    return None
```

### Step 2

fmt_weight_functions

Goal
----
Build the planar fundamental-measure (FMT) weight functions of a hard sphere.

```python
import numpy as np


def fmt_weight_functions(R: float, dz: float) -> np.ndarray:
    r"""Build the six planar hard-sphere FMT weight arrays.

    R: finite positive hard-sphere radius.
    dz: finite positive grid spacing, with dz <= R.

    Numerical representation:
        Use float64 offsets t = np.arange(-R, R + dz / 2, dz).
        Let M = t.size; M is not required to be odd, and t is not
        required to contain zero. Return rows in the order
        [w0, w1, w2, w3, wv1, wv2], with shape (6, M).

        Evaluate the planar sphere polynomials on this offset array.
        Use equal rectangular quadrature weights, not endpoint weights.
        Uniformly rescale w2 and w3 separately so that
        dz * sum(w2) = 4 * pi * R**2 and
        dz * sum(w3) = 4 * pi * R**3 / 3.
        Obtain w1 and w0 from the corrected w2 by the standard radial
        rescalings. Use wv2 = 2 * pi * t and wv1 = wv2 / (4 * pi * R).
        Do not symmetrize, recenter, clip, or add grid points.
        This fixed sampling convention also applies when R / dz is
        nonintegral; the last offset is the one selected by np.arange.

    Returns:
        A fresh numpy float64 array of shape (6, M).

    Raises:
        ValueError: if R or dz is not finite and positive, or dz > R.
    """
    return None
```

### Step 3

sw_meanfield_kernel

Goal
----
Build the planar-projected mean-field kernel of a square-well attraction.

```python
import numpy as np


def sw_meanfield_kernel(
    beps: float, rng: float, dz: float
) -> np.ndarray:
    r"""Build the planar-projected mean-field square-well kernel.

    beps: finite positive reduced well depth, beta * epsilon.
    rng: finite positive outer interaction range, in sigma1 units.
    dz: finite positive spacing, with dz <= rng.

    Numerical representation:
        Use float64 offsets t = np.arange(-rng, rng + dz / 2, dz).
        Let M2 = t.size; M2 is not required to be odd, and t is not
        required to contain zero. Evaluate the transverse-integrated
        square-well polynomial at every offset selected by this array.
        Do not recenter, clip, or renormalize the sampled values.

        The continuum kernel integral equals the three-dimensional
        well strength. The discrete rectangular sum approximates that
        integral; it is not constrained to equal it exactly at finite dz.
        Recover the physical separation-range convention from the paper.

    Returns:
        A fresh numpy float64 array of shape (M2,).

    Raises:
        ValueError: if any argument is not finite and positive,
            or if dz > rng.
    """
    return None
```

### Step 4

solvent_density_profile

Goal
----
Solve the planar solvent density profile at liquid-vapor coexistence by Picard iteration.

```python
import numpy as np


def solvent_density_profile(
    rho_l: float,
    rho_v: float,
    weights: np.ndarray,
    kernel: np.ndarray,
    beps: float,
    lam: float,
    dz: float,
    L: float,
) -> np.ndarray:
    r"""Solve the finite-box solvent interface by damped Picard iteration.

    rho_l, rho_v: finite positive coexistence densities, rho_l > rho_v.
    weights: solvent FMT weights, shape (6, M), for radius R = 0.5.
    kernel: one-dimensional solvent mean-field kernel.
    beps, lam: finite positive solvent reduced depth and range.
    dz, L: finite positive spacing and box length.

    Numerical convention:
        Use z = np.arange(-L / 2, L / 2 + dz / 2, dz).
        Initialize rho = rho_v + (rho_l - rho_v) *
                         (1 + np.tanh(z)) / 2.
        Keep the array alignment supplied by Steps 02 and 03.

        For a kernel w, set h = len(w) // 2, extend the input by
        h endpoint-value cells on each side, take the same-mode
        discrete convolution, multiply by dz, and retain indices
        h through h + len(input) - 1.

        Use the liquid-reservoir bulk chemical potential. For each
        unmixed Euler-Lagrange update, pin int(lam / dz) + 10 cells
        on the left to rho_v and on the right to rho_l. Start with the
        initialization above; do not translate the returned profile.
        The reference mixes 15 percent of the updated density with
        85 percent of the previous density and stops after an unmixed
        maximum absolute density residual below 1e-6, or 25000 updates.
        More tightly converged profiles are acceptable within the
        declared density comparison tolerance.

    Returns:
        A fresh numpy float64 array of shape (2, N), where N = z.size.
        Row 0 is z and row 1 is rho. Vapor is on the left, liquid on
        the right. Density tests use rtol = atol = 5e-4, while grid
        coordinates must agree within an absolute tolerance of 1e-10.

    Raises:
        ValueError: on malformed weights or kernel, non-finite or
            non-positive scalar inputs, or rho_l <= rho_v.
    """
    return None
```

### Step 5

surface_tension

Goal
----
Compute the reduced surface tension of the solvent liquid-vapor interface.

```python
def surface_tension(profile: np.ndarray, weights: np.ndarray, kernel: np.ndarray, beps: float, lam: float, dz: float) -> float:
    r"""profile: $(2, N)$ array [$z$, $\rho_1(z)$] from the solvent interface solve.
    weights: $(6, M)$ solvent FMT weights; kernel: $(M_2,)$ solvent mean-field kernel.
    beps: float solvent depth $\beta\varepsilon_1$; lam: float range $\lambda_1$; dz: float spacing.

    Returns a Python float: the reduced interfacial surface tension
    $\beta\gamma\sigma_1^2 = \int dz\,[\beta\omega(z) + \beta P]$, where $\omega$ is the grand
    potential density of the profile and $P$ the coexistence pressure.

    Raises:
        ValueError: on malformed profile/weights/kernel or non-finite scalar inputs.
    """
    return None
```

### Step 6

lorentz_berthelot_cross

Goal
----
Apply the model's mixing rules to obtain the cross square-well interaction parameters.

```python
def lorentz_berthelot_cross(sig1: float, lam1: float, beps1: float, sig2: float, lam2: float, beps2: float) -> np.ndarray:
    r"""sig1, lam1, beps1: solvent (component 1) diameter, square-well range, depth.
    sig2, lam2, beps2: solute (component 2) diameter, square-well range, depth.

    Returns a numpy float64 array of shape $(2,)$: $[\,r_{12},\ \beta\varepsilon_{12}\,]$, the
    outer range $r_{12}$ (a length) and depth $\beta\varepsilon_{12}$ of the cross square-well
    interaction between the two components under the mixing rules of the model.

    Raises:
        ValueError: if any argument is not a finite positive scalar.
    """
    return None
```

### Step 7

solute_one_body_correlation

Goal
----
Compute the inhomogeneous one-body direct correlation of a dilute solute in the solvent field.

```python
def solute_one_body_correlation(profile: np.ndarray, weights1: np.ndarray, weights2: np.ndarray, cross_kernel: np.ndarray, dz: float) -> np.ndarray:
    r"""profile: $(2, N)$ array [$z$, $\rho_1(z)$] of the solvent interface.
    weights1: $(6, M_1)$ FMT weights of the solvent hard sphere ($R_1$).
    weights2: $(6, M_2)$ FMT weights of the solute hard sphere ($R_2$).
    cross_kernel: $(M_c,)$ mean-field kernel of the cross square-well interaction.
    dz: float grid spacing.

    Returns a numpy float64 array of shape $(N,)$: the inhomogeneous one-body direct correlation
    $c_2^{(1)}(z)$ of a single dilute solute particle in the solvent field $\rho_1(z)$, i.e. the
    sum of its hard-sphere (FMT) and cross square-well contributions.

    Raises:
        ValueError: on malformed profile/weights/kernel or non-finite dz.
    """
    return None
```

### Step 8

solute_profile_peak

Goal
----
Convert the solute one-body correlation into the normalised adsorption profile and its peak.

```python
def solute_profile_peak(c1_2: np.ndarray) -> float:
    r"""c1_2: $(N,)$ inhomogeneous one-body direct correlation of the solute across the interface,
    with the liquid reservoir at the last grid point ($z \to +\infty$).

    Returns a Python float: the peak interfacial enhancement
    $P = \max_z\, \rho_2(z)/\rho_{2,0}$ of the dilute solute, where
    $\rho_2(z)/\rho_{2,0} = \exp\!\big(c_2^{(1)}(z) - c_2^{(1)}(+\infty)\big)$ is normalised so
    that the solute density approaches its liquid-reservoir value $\rho_{2,0}$ as $z \to +\infty$.

    Raises:
        ValueError: if c1_2 is not a finite 1D array of length >= 3.
    """
    return None
```

### Step 9

interface_adsorption_audit

Goal
----
Assemble the full pipeline and report the peak interfacial adsorption of the dilute solute.

```python
def interface_adsorption_audit(beps1: float, lam1: float, ratio: float, beps2: float, dz: float, L: float) -> float:
    r"""beps1, lam1: solvent square-well depth $\beta\varepsilon_1$ and range $\lambda_1$.
    ratio: float solute/solvent size ratio $\sigma_2/\sigma_1$ (with $\sigma_1 = 1$).
    beps2: float solute square-well depth $\beta\varepsilon_2$.
    dz: float grid spacing; L: float box length.

    Returns a Python float: the peak interfacial enhancement $P = \max_z \rho_2(z)/\rho_{2,0}$ of
    the dilute solute at the solvent liquid-vapor interface. The solute range obeys the
    equal-well-width rule $\lambda_2 = 1 + (\lambda_1 - 1)/\text{ratio}$.

    Numerical accuracy:
        End-to-end peak comparisons use rtol = atol = 1e-2.
        This accommodates the finite-box profile convergence tolerance
        declared in Step 04; intermediate analytic checks remain separate.

    Raises:
        ValueError: whenever any of the eight earlier functions it calls would raise.
    """
    return None
```
