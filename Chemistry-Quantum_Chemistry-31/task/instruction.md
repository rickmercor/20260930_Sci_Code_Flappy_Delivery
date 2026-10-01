# Chemistry-Quantum_Chemistry-31

## Background

Hydrogen and deuterium transfer rates can differ strongly when nuclear quantum effects are important. A useful rate construction must account for the transfer coordinate, the coupling between localized vibronic states, and thermal fluctuations of the donor-acceptor separation without reducing the process to a conventional zero-point-energy correction.

At a crossing of two diabatic vibronic surfaces, the effective reaction-coordinate inertia depends on both the surface gradients and the relaxed nuclear curvature. Quantum-nuclear basis-centre coordinates respond to motion of the classical nuclei and can therefore be eliminated before the remaining Hessian is mass weighted and projected onto the crossing direction.

Different transmission approximations emphasize different physics. A recrossing-corrected crossing formula connects weak and strong vibronic coupling above the barrier, whereas a weak-coupling Airy construction also admits tunnelling by the remaining nuclei and requires care when its approximate probability exceeds unity. Both branches must be combined with the same distance-dependent activation and normalized reactant distribution before an isotope ratio is formed.

Numerically, the benchmark couples Schur complements, projected masses, exponentially varying couplings, semi-infinite quadrature, a sharp physical distance gate, and ratios of very small rates. Stable exponential evaluation and consistent energy references are therefore part of the scientific calculation rather than cosmetic implementation details.

## Problem

A pair of isotope-resolved nuclear-electronic snapshots is available for a deep-tunnelling hydrogen-transfer model; quantify how replacing its recrossing-aware positive-energy treatment by its tunnelling-capable weak-coupling treatment changes the predicted H/D kinetic isotope effect. Execute the deterministic NumPy block once; its two records are ordered H then D, the two Hessians and gradients within each record are ordered reactant then product, all energies and gradients are in atomic units, all distances are in bohr, both crossings use the peaked branch, each energy coefficient vector is ordered from the constant through the quartic term, and every energy and coupling power uses the offset `y=distance-reference_distance`.

```python
import numpy as np

seed=26091312
temperature=82.0
n_distance=61
lz_order=48
wc_order=64
kb_au=3.1668114e-6

def build_snapshot(seed,isotope_index,n_distance):
    rng=np.random.default_rng(int(seed)+7919*int(isotope_index))
    classical_dim,quantum_dim=6,3
    matrices=[]
    for surface in range(2):
        factor=rng.normal(scale=.055+.004*surface,size=(classical_dim,classical_dim))
        h_cc=factor.T@factor+np.diag(np.linspace(.075,.145,classical_dim))
        h_qc=rng.normal(scale=.0055,size=(quantum_dim,classical_dim))
        qfactor=rng.normal(scale=.035,size=(quantum_dim,quantum_dim))
        h_qq=qfactor.T@qfactor+(.27+.02*surface)*np.eye(quantum_dim)
        matrices.append(np.block([[h_cc,h_qc.T],[h_qc,h_qq]]))
    base_a=np.array([.031,-.025,.019,-.014,.012,-.009])
    base_b=np.array([-.027,.021,-.016,.017,-.010,.011])
    gradient_a=(1+.035*isotope_index)*base_a+rng.normal(scale=.0018,size=classical_dim)
    gradient_b=(1-.025*isotope_index)*base_b+rng.normal(scale=.0018,size=classical_dim)
    masses=1822.888486217313*np.array([12.,12.,15.99491462,15.99491462,14.003074,14.003074])
    reference_distance=4.94+.022*isotope_index+rng.normal(scale=.004)
    distance=np.linspace(4.42,5.50,int(n_distance))
    reactant=np.array([0.,.00015*(-1. if isotope_index==0 else 1.),.0125+.0012*isotope_index,-.0018+.0002*isotope_index,.0021])
    mecp=np.array([.00485+.00058*isotope_index,-.0016+.00018*isotope_index,.0088+.0007*isotope_index,.0011,.0015])
    product=np.array([.00055+.00016*isotope_index,-.00035,.0112+.0008*isotope_index,.0014,.0018])
    coupling=np.array([(.00048 if isotope_index==0 else .000215)*(1+rng.normal(scale=.025)),-2.20-.16*isotope_index,-.55-.06*isotope_index])
    critical_distance=4.505+.018*isotope_index
    return matrices,gradient_a,gradient_b,masses,distance,reactant,mecp,product,coupling,reference_distance,critical_distance

snapshots=[build_snapshot(seed,isotope,n_distance) for isotope in (0,1)]
```

Use the primary article and its pinned public implementation to reconstruct the complete source rate construction for each isotope and evaluate both competing transmission treatments from the supplied snapshots. Preserve the source's physical domains, branch choices, fixed-distance partition-function cancellation, WC energy lower bound, and probability cap; after the natural semi-infinite Boltzmann substitutions, apply Gauss-Laguerre rules of the stated orders as a deterministic benchmark convention supplied here rather than inferred from the sources. Define `KIE_LZ=k_H_LZ/k_D_LZ` and `KIE_WC=k_H_WC/k_D_WC`, and return `100*(KIE_WC/KIE_LZ-1)` with its sign retained.

In `<reasoning>`, cite the sources and state the article's low-temperature comparison with ring-polymer instanton theory and its stated regime preference between the two transmission treatments. Compactly report the H and D reactant-state reduced-Hessian traces in `E_h bohr^-2`, reduced masses in `m_e`, gradient gaps in `E_h bohr^-1`, active-distance counts, four reduced thermal rates in `E_h/hbar`, both dimensionless KIEs, and the signed percentage shift; also identify the normal-crossing Holstein relation, the strict critical-distance inequality, and the WC lower-bound and capping conventions. Report floating values to at least ten significant digits, but do not paste matrices, quadrature tables, or distance-resolved integrands.

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

reduce_neo_hessian

Goal
----
Reduce an extended nuclear-electronic Hessian to classical coordinates.

```python
def reduce_neo_hessian(
    extended_hessian: 'np.ndarray',
    quantum_dim: int = 3,
) -> 'np.ndarray':
    """Return the relaxed classical-coordinate Hessian.

    Parameters
    ----------
    extended_hessian
        Extended Hessian in ``E_h bohr^-2`` with shape
        ``(n + quantum_dim, n + quantum_dim)``.
    quantum_dim
        Number of trailing quantum-coordinate rows and columns.

    Returns
    -------
    np.ndarray
        Symmetric reduced Hessian in ``E_h bohr^-2`` with shape ``(n, n)``.

    Raises
    ------
    ValueError
        If the matrix is invalid, ``quantum_dim`` is invalid, or the quantum
        block is not positive definite.
    """
    return result
```

### Step 2

compute_crossing_descriptor

Goal
----
Compute the reaction-coordinate mass and crossing diagnostics.

```python
def compute_crossing_descriptor(
    hessian_a: 'np.ndarray',
    hessian_b: 'np.ndarray',
    gradient_a: 'np.ndarray',
    gradient_b: 'np.ndarray',
    masses: 'np.ndarray',
    peaked: bool = True,
) -> 'np.ndarray':
    """Return reduced mass, two gradient invariants, and projected curvature.

    Parameters
    ----------
    hessian_a, hessian_b
        Finite real symmetric Hessians in ``E_h bohr^-2`` with common shape
        ``(n, n)``.
    gradient_a, gradient_b
        Finite real gradient vectors in ``E_h bohr^-1`` with shape ``(n,)``.
    masses
        Strictly positive finite masses in ``m_e`` with shape ``(n,)``.
    peaked
        Select the plus-sign effective Hessian when true and the minus-sign
        branch when false.

    Returns
    -------
    np.ndarray
        ``[mu, gradient_gap, gradient_product_root, curvature]``, with ``mu``
        in ``m_e``, the two gradient quantities in ``E_h bohr^-1``, and the
        mass-weighted curvature in ``E_h m_e^-1 bohr^-2``.

    Raises
    ------
    ValueError
        If shapes or values are invalid, the gradient difference is zero, or
        the projected mode is physically singular.
    """
    return result
```

### Step 3

evaluate_distance_profiles

Goal
----
Evaluate the donor-acceptor-distance profiles used by the rate model.

```python
def evaluate_distance_profiles(
    distance: 'np.ndarray',
    reactant_coefficients: 'np.ndarray',
    mecp_coefficients: 'np.ndarray',
    product_coefficients: 'np.ndarray',
    coupling_parameters: 'np.ndarray',
    reference_distance: float,
    critical_distance: float,
) -> 'np.ndarray':
    '''Return the three energy profiles and critical-distance-masked coupling.

    Parameters
    ----------
    distance
        Finite, strictly increasing donor-acceptor distances in bohr with
        shape ``(n,)``.
    reactant_coefficients, mecp_coefficients, product_coefficients
        Finite quartic coefficients ``[E0, a1, a2, a3, a4]`` in the
        corresponding ``E_h bohr^-k`` units.
    coupling_parameters
        Finite parameters ``[V0, c1, c2]`` in
        ``[E_h, bohr^-1, bohr^-2]`` with ``V0 >= 0``.
    reference_distance
        Finite real origin in bohr used in
        ``y = distance - reference_distance``.
    critical_distance
        Finite real strict lower gate in bohr for the coupling.

    Returns
    -------
    np.ndarray
        Array in ``E_h`` with columns
        ``[V_reactant, V_MECP, V_product, |V_ab|]``.

    Raises
    ------
    ValueError
        If an input has the wrong shape or domain, is non-finite, or makes the
        exponential coupling overflow.
    '''
    return result
```

### Step 4

compute_holstein_transmission

Goal
----
Evaluate the recrossing-corrected normal-crossing transmission probability.

```python
def compute_holstein_transmission(
    velocity: 'np.ndarray',
    coupling: float,
    gradient_gap: float,
) -> 'np.ndarray':
    """Return the Holstein-corrected transmission at each velocity.

    Parameters
    ----------
    velocity
        Nonempty finite real vector of nonnegative crossing velocities in
        bohr per atomic unit of time.
    coupling
        Finite real nonnegative diabatic coupling in ``E_h``.
    gradient_gap
        Finite real strictly positive gradient-difference norm in
        ``E_h bohr^-1``.

    Returns
    -------
    np.ndarray
        Dimensionless Holstein-corrected probabilities with the same shape as
        ``velocity``.

    Raises
    ------
    ValueError
        If the vector or either scalar lies outside its stated domain.
    """
    return result
```

### Step 5

integrate_lz_coefficient

Goal
----
Evaluate the declared finite-order benchmark for the positive-energy thermal
transmission average over crossing velocity.

```python
def integrate_lz_coefficient(
    coupling: float,
    gradient_gap: float,
    reduced_mass: float,
    beta: float,
    quadrature_order: int,
) -> float:
    """Return the thermally averaged LZ/Holstein coefficient.

    Parameters
    ----------
    coupling
        Finite nonnegative diabatic coupling in ``E_h``.
    gradient_gap
        Finite strictly positive gradient gap in ``E_h bohr^-1``.
    reduced_mass
        Finite strictly positive mass in ``m_e``.
    beta
        Finite strictly positive inverse temperature in ``E_h^-1``.
    quadrature_order
        Integer Gauss-Laguerre order from eight through 100. It defines the
        finite deterministic benchmark rather than an adaptive convergence
        target.

    Returns
    -------
    float
        Finite nonnegative reduced rate coefficient in ``E_h/hbar``.

    Raises
    ------
    ValueError
        If a scalar is non-finite or outside its physical domain, or if the
        quadrature order is not an integer from eight through 100, or if the
        transformed quadrature is not finite.
    """
    return result
```

### Step 6

integrate_wc_coefficient

Goal
----
Evaluate the weak-coupling Airy transmission coefficient.

```python
def integrate_wc_coefficient(
    coupling: float,
    gradient_gap: float,
    gradient_product_root: float,
    reduced_mass: float,
    beta: float,
    energy_floor: float,
    quadrature_order: int,
    cap_probability: bool = True,
) -> float:
    """Return the energy-integrated weak-coupling transmission coefficient.

    Parameters
    ----------
    coupling
        Finite nonnegative diabatic coupling in ``E_h``.
    gradient_gap, gradient_product_root
        Finite strictly positive gradient quantities in ``E_h bohr^-1``.
    reduced_mass
        Finite strictly positive mass in ``m_e``.
    beta
        Finite strictly positive inverse temperature in ``E_h^-1``.
    energy_floor
        Finite lower integration energy in ``E_h`` satisfying
        ``-beta * energy_floor < 600``.
    quadrature_order
        Integer Gauss-Laguerre order from eight through 100.
    cap_probability
        Boolean selecting whether Airy probabilities above one are capped.

    Returns
    -------
    float
        Finite nonnegative reduced weak-coupling rate coefficient in
        ``E_h/hbar``.

    Raises
    ------
    ValueError
        If an input is non-finite, has an invalid type or physical domain, or
        violates the transformed-energy bound, or if the transformed
        probability or quadrature is not finite.
    """
    return result
```

### Step 7

compute_thermal_isotope_rates

Goal
----
Thermally average fixed-distance LZ and weak-coupling rate kernels under the
declared finite-order benchmark convention.

```python
def compute_thermal_isotope_rates(
    distance: "np.ndarray",
    profile_values: "np.ndarray",
    beta: float,
    crossing_descriptor: "np.ndarray",
    lz_order: int,
    wc_order: int,
) -> "np.ndarray":
    """Return LZ and WC thermal rates plus two distance-distribution audits.

    Parameters
    ----------
    distance
        Finite strictly increasing distance grid in bohr with shape ``(n,)``.
    profile_values
        Finite array in ``E_h`` with shape ``(n, 4)`` and columns
        ``[V_r, V_MECP, V_product, |V_ab|]`` with nonnegative couplings.
    beta
        Finite strictly positive inverse temperature in ``E_h^-1``.
    crossing_descriptor
        Finite vector ``[mu, gradient_gap, gradient_product_root, curvature]``
        whose four entries are strictly positive; the first three units are
        ``m_e``, ``E_h bohr^-1``, and ``E_h bohr^-1`` respectively.
    lz_order, wc_order
        Integer Gauss-Laguerre orders from eight through 100.

    Returns
    -------
    np.ndarray
        ``[k_LZ, k_WC, shifted_partition, mean_distance]``; the two reduced
        rates are finite and nonnegative in ``E_h/hbar`` and the last two
        entries are in bohr.

    Raises
    ------
    ValueError
        If shapes, orders, or physical domains are invalid, or if an
        activation exponent reaches magnitude 600, or if a thermal rate is
        negative or non-finite.
    """
    return result
```

### Step 8

compute_neogrt_kie_shift

Goal
----
Run the complete finite-order two-isotope benchmark and compare rate-model
KIEs.

```python
def compute_neogrt_kie_shift(
    seed: int,
    temperature: float,
    n_distance: int = 61,
    lz_order: int = 48,
    wc_order: int = 64,
) -> float:
    """Return the signed WC-versus-LZ percentage shift in the H/D KIE.

    Parameters
    ----------
    seed
        Integer seed for the deterministic H and D snapshots.
    temperature
        Finite strictly positive temperature in kelvin.
    n_distance
        Odd integer number of distance nodes, at least nine.
    lz_order, wc_order
        Integer Gauss-Laguerre orders from eight through 100 defining the
        finite deterministic benchmark.

    Returns
    -------
    float
        Finite relative percentage ``100 * (KIE_WC / KIE_LZ - 1)``.

    Raises
    ------
    ValueError
        If the seed or grid controls are not valid integers, the temperature
        is non-finite or nonpositive, or either isotope rate is nonphysical.
    """
    return result
```
