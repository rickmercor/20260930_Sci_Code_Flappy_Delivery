# Physics-Optics-15

## Background

Intrinsic anomalous Hall response can produce directional thermal nonreciprocity without an applied magnetic field. The anchor work combines first-principles DFT-Kubo dielectric tensors with Maxwell propagation and derives separate universal ENZ laws for response strength and bandwidth. Reconstructing the passive surface mode from the tensor keeps the Maxwell stage load-bearing instead of accepting a precomputed contrast surface. A realizable fixed-band detector needs one spectral interval and one collection geometry to remain useful as gate state and alignment drift. A nominal fidelity screen separately checks the anchor paper's universal strength and signed-lobe-spacing laws before the exact Maxwell rectangle is certified. Intersecting the state-specific peak-signed half-maximum supports makes response strength and robust broadband operation load-bearing, while the same fixed-alignment Maxwell construction keeps magnetized InAs a subordinate but physically matched reference.

## Problem

## Setup

Design a fixed-geometry intrinsic Co3Sn2S2 emitter in the transverse Voigt optical geometry under a closed +/-0.026 eV gate window and a closed +/-4 degree collection-alignment window about one fixed nominal positive observation angle.

## Inputs

The compact JSON block below supplies all reasoning-stage target inputs: seven chemical-potential cells, 17 photon-energy nodes, six propagation-frame tensor components in the declared order, eV units, and raw-data provenance; preserve the supplied precision and do not use a stored contrast or summary.

```json
{
"provenance":{"source":"https://okongoyango.github.io/Nonreciprocal_Thermal_Material_Database/","material_index":0,"selection":"Seven source chemical-potential cells and the 17 source photon-energy nodes from 0.0688 through 0.2256 eV, copied without rounding from the public companion database."},

"chemical_potential_eV":[-0.333,-0.3,-0.267,-0.233,-0.2,-0.167,-0.133],

"photon_energy_eV":[0.0688,0.0786,0.0884,0.0982,0.108,0.1178,0.1276,0.1374,0.1472,0.157,0.1668,0.1766,0.1864,0.1962,0.206,0.2158,0.2256],

"tensor_component_order":["rexx","imxx","rezz","imzz","rexz","imxz"],

"tensor_values":[

[[-36.3,29.46,-34.1,30.12,-2.821,17.9],[-11.17,23.63,-8.684,24.17,-3.049,15.68],[5.906,20.36,8.592,20.8,-3.604,13.76],[17.92,17.92,20.77,18.29,-3.923,11.87],[26.84,16.08,29.83,16.41,-3.846,10.03],[33.67,15.06,36.77,15.35,-3.644,8.635],[38.38,13.83,41.57,14.12,-3.2,7.286],[42.81,12.52,46.06,12.81,-2.756,6.47],[46.45,11.73,49.75,12.03,-2.221,5.462],[49.76,11.36,53.09,11.67,-1.35,5.197],[51.97,11.3,55.31,11.6,-0.9812,5.3],[53.46,10.92,56.84,11.21,-1.077,5.121],[55.05,9.696,58.41,10.0,-0.7066,4.91],[57.46,9.182,60.83,9.442,-0.68,5.134],[58.96,9.228,62.39,9.499,-1.004,5.031],[60.18,8.781,63.58,9.041,-1.103,4.593],[61.69,8.271,65.12,8.516,-0.9109,4.377]],

[[-102.2,35.88,-100.2,36.46,-0.5649,21.56],[-61.99,27.18,-59.79,27.65,-1.179,19.59],[-34.39,22.38,-32.01,22.77,-2.16,17.72],[-14.99,19.58,-12.47,19.9,-3.11,15.76],[-1.194,17.31,1.44,17.59,-3.453,13.56],[9.484,15.49,12.21,15.75,-3.445,11.91],[17.57,13.84,20.36,14.09,-3.192,10.37],[24.49,12.43,27.33,12.67,-2.908,9.454],[29.91,11.36,32.8,11.59,-2.672,8.392],[34.83,10.42,37.76,10.65,-2.244,7.886],[38.81,9.889,41.77,10.15,-2.181,7.48],[41.92,9.483,44.88,9.761,-2.254,6.781],[44.82,8.484,47.74,8.75,-1.709,6.15],[48.03,8.22,50.97,8.436,-1.459,6.206],[50.21,8.384,53.17,8.59,-1.603,5.985],[51.95,8.012,54.92,8.193,-1.568,5.508],[53.94,7.525,56.94,7.702,-1.276,5.25]],

[[-178.2,45.49,-175.8,45.92,-0.07571,21.88],[-121.5,32.82,-119.0,33.19,-0.1103,19.44],[-82.18,25.02,-79.61,25.34,-0.1789,17.61],[-53.73,20.03,-51.11,20.31,-0.3141,16.25],[-32.44,16.81,-29.79,17.07,-0.6386,15.25],[-16.38,14.95,-13.69,15.19,-1.326,14.18],[-4.226,13.32,-1.515,13.56,-1.594,12.82],[5.735,12.11,8.439,12.37,-1.815,11.91],[13.38,11.16,16.08,11.39,-1.963,10.66],[20.04,9.949,22.77,10.16,-1.618,9.911],[25.97,9.328,28.71,9.532,-1.635,9.568],[30.53,9.133,33.29,9.34,-1.935,8.833],[34.37,8.294,37.12,8.518,-1.571,8.063],[38.42,7.833,41.17,8.039,-1.389,7.984],[41.69,7.878,44.44,8.082,-1.589,7.71],[44.3,7.717,47.05,7.9,-1.635,7.165],[46.96,7.365,49.72,7.53,-1.419,6.855]],

[[-248.4,55.27,-246.0,55.64,-0.1558,23.53],[-175.9,39.28,-173.5,39.6,-0.0955,20.74],[-125.7,29.4,-123.2,29.68,-0.05355,18.59],[-89.47,22.96,-87.0,23.22,-0.02032,16.9],[-62.49,18.55,-60.01,18.79,9.127e-06,15.55],[-41.81,15.4,-39.32,15.62,0.004155,14.46],[-25.57,13.09,-23.06,13.3,-0.003431,13.56],[-12.53,11.36,-10.01,11.55,-0.0214,12.83],[-1.86,10.05,0.6702,10.23,-0.05217,12.22],[7.042,9.053,9.587,9.237,-0.09959,11.74],[14.61,8.329,17.18,8.515,-0.1794,11.38],[21.18,7.886,23.76,8.089,-0.3322,11.12],[26.91,7.799,29.48,8.016,-0.6037,10.92],[31.86,8.092,34.42,8.315,-1.025,10.71],[35.82,8.626,38.39,8.873,-1.547,10.3],[38.97,9.084,41.5,9.327,-1.983,9.614],[41.69,9.135,44.2,9.383,-2.003,8.976]],

[[-303.1,63.9,-300.8,64.23,-0.2161,28.9],[-217.5,45.2,-215.2,45.49,-0.1786,25.55],[-158.1,33.66,-155.8,33.92,-0.1598,22.99],[-115.2,26.16,-112.9,26.4,-0.1533,21.0],[-83.13,21.07,-80.83,21.3,-0.1676,19.44],[-58.47,17.52,-56.15,17.74,-0.215,18.21],[-38.97,15.03,-36.64,15.27,-0.3172,17.27],[-23.15,13.4,-20.86,13.68,-0.5293,16.55],[-10.16,12.76,-7.895,12.94,-0.9835,15.98],[-0.213,12.65,2.183,12.92,-1.584,15.05],[8.261,11.97,10.49,12.31,-1.665,14.21],[15.83,12.11,18.05,12.33,-1.978,13.75],[21.84,12.38,24.04,12.62,-2.322,13.1],[27.14,12.95,29.31,13.13,-2.69,12.56],[31.12,13.62,33.31,13.79,-3.044,11.83],[34.6,14.04,36.79,14.14,-3.215,11.19],[37.41,14.39,39.64,14.5,-3.366,10.57]],

[[-326.3,70.49,-324.8,70.86,-0.7981,41.73],[-232.7,50.55,-231.0,50.85,-1.031,37.56],[-167.2,38.72,-165.5,38.97,-1.504,34.64],[-119.5,32.21,-117.7,32.42,-2.631,32.55],[-85.04,29.17,-83.12,29.5,-4.351,29.93],[-59.33,26.61,-57.6,26.98,-5.023,27.2],[-38.84,25.66,-37.09,25.8,-5.907,25.32],[-23.98,25.2,-22.16,25.42,-6.728,22.8],[-11.73,24.14,-9.967,24.26,-6.73,20.77],[-1.647,23.85,0.2289,23.94,-6.922,19.16],[6.592,23.44,8.408,23.57,-6.958,17.71],[13.53,23.79,15.34,23.77,-7.209,16.46],[18.48,23.74,20.39,23.68,-7.248,14.98],[23.25,23.41,25.24,23.33,-7.077,13.88],[27.06,23.35,29.11,23.36,-7.003,12.83],[30.61,23.17,32.61,23.15,-6.806,11.97],[33.5,23.28,35.53,23.23,-6.719,11.19]],

[[-290.8,92.94,-289.3,93.33,-19.3,73.84],[-205.7,76.53,-204.2,76.65,-21.62,61.32],[-148.6,65.32,-146.9,65.54,-21.78,50.8],[-105.7,56.93,-104.1,57.01,-20.51,44.33],[-75.13,53.53,-73.34,53.59,-21.27,38.28],[-53.86,49.19,-52.13,49.29,-19.93,31.85],[-35.51,45.21,-33.67,45.2,-17.99,28.25],[-21.35,43.35,-19.5,43.51,-17.24,25.16],[-10.38,41.59,-8.678,41.57,-16.29,22.37],[-1.853,40.23,0.1357,40.16,-15.56,19.85],[5.033,38.42,7.003,38.49,-14.48,17.55],[11.42,36.88,13.41,36.95,-13.39,15.96],[16.52,35.77,18.46,35.93,-12.59,14.57],[21.11,34.92,22.99,35.01,-11.92,13.42],[24.54,34.03,26.46,34.11,-11.24,12.26],[28.08,33.18,30.02,33.21,-10.62,11.44],[30.93,32.68,32.93,32.76,-10.13,10.62]]

]
}
```

## Physical model

Recover the anchor paper's passive semi-infinite transverse-TM reconstruction and its separate universal ENZ nonreciprocity-strength and signed-lobe-spacing laws from the registered anchor source, then independently PCHIP each supplied target tensor component in chemical potential onto 329 uniform nodes from -0.315 through -0.151 eV and reconstruct the response on 81 uniform positive angles from 5 through 85 degrees with epsilon_zx=-epsilon_xz, positive q corresponding to positive angle, and signed contrast D=R(+q)-R(-q).

At every interpolated chemical-potential state, independently PCHIP each of the six tensor components in photon energy; use those energy interpolants to locate the diagonal ENZ crossing and to evaluate its diagonal slope, loss, imaginary gyrotropy, real gyrotropy, and all quantities derived from them.

On the supplied 0.0688-through-0.2256 eV target window, PCHIP each fixed-state fixed-angle signed contrast over energy, evaluate both endpoints and every derivative root, reject spectra without significant positive and negative lobes using scale-aware floating-point roundoff, define B_pair as the separation of the continuous signed extrema, choose the largest absolute extremum with exact energy ties going to lower energy, and define B_half as the closed connected peak-signed half-maximum component, including any internal threshold contact.

At each unique x-directed diagonal ENZ crossing, evaluate both recovered laws. For each target nominal node, evaluate Delta_law and Delta_Maxwell at that same fixed chemical-potential state and the same fixed nominal angle; do not replace either quantity by an angle-optimized or global angular maximum. Define the relative strength error as abs(Delta_law-Delta_Maxwell)/abs(Delta_Maxwell) and the relative signed-lobe-spacing error as abs(B_pair,law-B_pair,Maxwell)/abs(B_pair,Maxwell), where both reconstructed Maxwell denominators are positive; retain a target nominal node only when these separate errors are at most 0.35 and 0.0085, respectively, while the real-to-imaginary gyrotropy ratio is at most 0.16 throughout the closed gate interval.

For the subordinate denominator, retrieve the fitted parameters for the FTIR reflectance sample shown as Figure 4 in the published InAs article (Figure 3 in arXiv v1), which is distinct from the ellipsometry sample, reconstruct its positive-field B=0.16 T tensor using positive elementary charge, and evaluate its signed contrast at exactly the 101 uniform nodes from 0.04 through 0.14 eV rather than on the target's 17-energy window. PCHIP those 101 benchmark contrast values in energy and apply the same continuous-extremum, exact-tie, significant-signed-lobe, and connected half-maximum-support construction specified for the target; do not densify the grid or optimize the analytic Drude-Fresnel curve directly.

## Task

Return the largest law-screened fixed-geometry common-passband guarantee ratio whose numerator is the minimum absolute reconstructed Co3Sn2S2 peak in a full closed gate/alignment node rectangle times the width of one interval shared by every peak-signed half-maximum support in that rectangle and whose denominator is the largest reconstructed 0.16-T InAs fixed-geometry common-passband product over the same angle grid and closed +/-4-degree alignment window.

Intersect supports by their largest lower and smallest upper endpoints, omit empty intersections and windows containing a rejected spectrum, maximize minimum peak times common width, and break target ties by lower nominal chemical potential then angle and benchmark or limiter ties by the lower available coordinate.

## Numerical conventions

Report the four fitted InAs parameters; the target nominal chemical potential and angle, both target law values and their separate relative errors, worst-peak coordinates, common endpoints, both endpoint-limiter coordinates, guaranteed peak, common bandwidth, and target product; the benchmark nominal angle, worst-peak and endpoint-limiter angles, common endpoints, guaranteed peak, common bandwidth, and benchmark product; and the final ratio, accepting an equivalent final scalar within 0.5 percent.

## Output format

```
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

01_material_fields

Goal
----
Interpolate the first-principles dielectric-tensor field.

```python
import numpy as np
from scipy.interpolate import PchipInterpolator

def material_fields(
    mu_nodes: np.ndarray,
    tensors: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Return a (Q,E,6) tensor field in query order.

    ``mu_nodes`` contains at least three nodes. ``tensors`` has shape
    (J,E,6), ordered Re xx, Im xx, Re zz,
    Im zz, Re xz, Im xz. Interpolate every scalar independently by PCHIP
    along the strictly increasing chemical-potential axis. ``queries`` must
    lie in the closed node range and may be unsorted. Reject malformed,
    nonfinite, nonincreasing, or out-of-domain input with ``ValueError``.
    Absolute accuracy: 1e-8.
    """
    return None
```

### Step 2

02_voigt_surface_responses

Goal
----
Solve the semi-infinite anisotropic Voigt boundary problem.

```python
import numpy as np

def voigt_surface_responses(
    tensors: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Return the signed TM surface contrast with shape (...,A,E).

    The final two axes of ``tensors`` are (E,6), ordered Re xx, Im xx,
    Re zz, Im zz, Re xz, Im xz, for the x-z gyrotropic block with
    eps_zx=-eps_xz. For each positive angle in degrees set q=sin(theta),
    c=cos(theta), and solve the transmitted TM mode

      kz^2 = eps_x + eps_xz^2/eps_z - eps_x*q^2/eps_z.

    Select the square root with Im(kz)>0, or Re(kz)>0 when Im(kz)=0.
    Its normalized tangential admittance is
    Y=(eps_z*kz-q*eps_xz)/(eps_z-q^2), and the tangential-electric-field
    reflection amplitude is r=(1-c*Y)/(1+c*Y). Evaluate this expression
    separately at +q and -q and return |r(+q)|^2-|r(-q)|^2. Treat
    abs(eps_z)<1e-14 or abs(eps_z-q^2)<1e-14 as singular. Reject malformed,
    nonfinite, nonpositive-angle, grazing, or singular input with
    ``ValueError``. Absolute accuracy: 2e-10.
    """
    return None
```

### Step 3

03_full_maxwell_metrics

Goal
----
Extract signed Maxwell strength, support, and lobe separation.

```python
import numpy as np
from scipy.interpolate import PchipInterpolator

def full_maxwell_metrics(
    energies: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    """Return (...,7) rows (Delta,E0,Eleft,Eright,Bhalf,C,Bpair).

    ``responses`` has shape (D1,...,Dk,E), where k>=1, every leading
    dimension is nonempty, and the final axis is sampled on at least five
    strictly increasing photon energies. Independently PCHIP every fixed-state,
    fixed-angle signed spectrum and evaluate both endpoints and every
    interior derivative root. The spectrum must have a strictly positive
    maximum and a strictly negative minimum in its shape-preserving sampled
    range, using a roundoff floor of 64*machine_epsilon*max(1,max(abs(row))).
    ``Bpair`` is the absolute
    energy separation between those signed extrema, with exact same-sign
    ties going to the smaller energy.

    Select the largest absolute signed extremum for ``Delta`` and ``E0``;
    an exact positive/negative magnitude tie also goes to the smaller energy.
    Multiply the spectrum by that peak's sign. ``Eleft`` and ``Eright`` bound
    the closed connected component containing ``E0`` on which the adjusted
    response is at least ``Delta/2``. Set ``Bhalf=Eright-Eleft`` and
    ``C=Delta*Bhalf``. Preserve every leading response dimension. Reject
    malformed, nonfinite, non-double-lobed input with ``ValueError``.
    Absolute accuracy: 2e-8 for values and 2e-7 eV for locations.
    """
    return None
```

### Step 4

04_enz_design_coordinates

Goal
----
Recover the anchor paper's local ENZ design coordinates.

```python
import numpy as np
from scipy.interpolate import PchipInterpolator

def enz_design_coordinates(spectra: list) -> np.ndarray:
    """Return rows (beta,loss,g,x,W,r_o).

    ``spectra`` must be nonempty. Each input is a finite (K,7) table with K>=5 containing energy, Re xx,
    Im xx, Re zz, Im zz, Re xz and Im xz; K may differ between inputs and
    each energy column is strictly increasing. Independently PCHIP every
    tensor component in energy. Re xx must have exactly one crossing,
    including a possible endpoint zero. At that crossing set
    beta=abs(d Re xx/dE), loss=Im xx, g=abs(Im xz), x=g/loss,
    W=2/sqrt(3)*sqrt(x*x-1+2*sqrt(x**4+x*x+1)) and
    r_o=abs(Re xz)/g.
    Reject a malformed or nonfinite table, a nonincreasing energy column,
    Re xx with other than one crossing, or nonpositive beta, loss, or
    gyrotropy with ``ValueError``. Absolute accuracy: 1e-8.
    """
    return None
```

### Step 5

05_universal_law_metrics

Goal
----
Evaluate the anchor paper's universal strength and bandwidth laws.

```python
import numpy as np

def universal_law_metrics(
    coordinates: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    """Return an (M,A,3) array (Delta_law,Bpair_law,C_law).

    ``coordinates`` contains Step 4 rows (beta,loss,g,x,W,r_o), and
    ``angles`` contains finite observation angles strictly between 0 and 90
    degrees. Require x=g/loss and the stated W(x) relation to satisfy
    abs(actual-expected) <= 1e-12 + 1e-10*abs(expected). For each material
    row set

      kappa=sqrt(2/(loss*(1+x*x))),
      G=sin(theta)*cos(theta)/(cos(theta)+kappa)^2,
      d=loss*W/2,
      f=-2*g*loss*d/((d*d+loss*loss-g*g)^2+4*g*g*loss*loss).

    Return Delta_law=8*abs(f)*G, the universal signed-lobe peak spacing
    Bpair_law=loss*W/beta, and their product. Reject malformed, nonfinite,
    inconsistent, or nonphysical input with ``ValueError``. Absolute
    accuracy: 1e-10.
    """
    return None
```

### Step 6

06_window_guarantees

Goal
----
Apply universal-law validity and guarantee one Maxwell passband.

```python
import numpy as np

def window_guarantees(
    mu: np.ndarray,
    angles: np.ndarray,
    metrics: np.ndarray,
    law_metrics: np.ndarray,
    validity: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
    spectrum_valid: np.ndarray = None,
) -> np.ndarray:
    """Return finite common-band guarantees with nominal law agreement.

    ``mu`` and ``angles`` are nonempty one-dimensional increasing uniform
    grids with at least two nodes. ``metrics`` has shape
    (N,A,7) with Step 3 columns (Delta,E0,Eleft,Eright,Bhalf,C,Bpair),
    ``law_metrics`` has shape (N,A,3) with Step 5 columns
    (Delta_law,Bpair_law,C_law), and ``validity`` is the length-N Step 4
    real-to-imaginary gyrotropy ratio. Optional ``spectrum_valid`` is a Boolean
    (N,A) mask. A false cell may contain nonfinite metric placeholders and
    invalidates only uncertainty rectangles containing that cell; every true
    cell must satisfy the complete finite metric contract. If omitted, every
    spectrum is valid. ``drifts`` is a finite length-2 vector.
    Each nonnegative drift is an exact
    integer multiple of its grid spacing. ``validity_limit`` must be finite and
    strictly positive. ``agreement_limits`` must be a finite length-2 vector,
    and each of its two entries must lie strictly between zero and one; the
    boundary values zero and one are outside this public contract.

    A nominal node passes the universal-law fidelity screen only when its
    relative strength error abs(Delta_law-Delta)/Delta is no greater than
    ``agreement_limits[0]`` and its relative signed-lobe-spacing error
    abs(Bpair_law-Bpair)/Bpair is no greater than ``agreement_limits[1]``.
    The screen is evaluated at the nominal material state and fixed angle;
    gate and alignment robustness are then certified by the full Maxwell
    rectangle. Every gate node must also satisfy ``validity_limit``.

    Within each full closed uncertainty rectangle, set Dmin to the minimum
    Maxwell Delta, common_left to the largest Eleft, and common_right to the
    smallest Eright. Omit nonpositive intersections. Ties for Dmin and the
    two limiting endpoints select smaller chemical potential, then angle.
    Return rows
    (nominal_mu,nominal_angle,Dmin,Bcommon,worst_D_mu,worst_D_angle,
    common_left,common_right,left_mu,left_angle,right_mu,right_angle,
    strength_error,spacing_error,Delta_law,Bpair_law), ordered by nominal
    chemical potential and then angle. Reject malformed, nonfinite,
    inconsistent, or nonphysical input with ``ValueError``. Absolute
    accuracy: 1e-10.
    """
    return None
```

### Step 7

07_select_operating_point

Goal
----
Select the strongest law-screened fixed-geometry common passband.

```python
import numpy as np

def select_operating_point(guarantees: np.ndarray) -> np.ndarray:
    """Return the unique tie-broken best length-17 operating-point row.

    ``guarantees`` has finite shape (K,16), K>0, and the Step 6 column
    contract, with positive Dmin, Bcommon, Delta_law, and Bpair_law and
    nonnegative agreement errors. Maximize Dmin*Bcommon. Exact product ties
    select smaller nominal chemical potential, then smaller nominal angle.
    Return (nominal_mu,nominal_angle,worst_D_mu,worst_D_angle,left_mu,
    left_angle,right_mu,right_angle,Dmin,common_left,common_right,Bcommon,
    product,strength_error,spacing_error,Delta_law,Bpair_law). Reject
    malformed, empty, nonfinite, or nonphysical input with ``ValueError``.
    Absolute accuracy: 1e-10.
    """
    return None
```

### Step 8

08_robust_surface_advantage

Goal
----
Compose the law-screened target and robust full-Maxwell benchmark.

```python
import numpy as np

def robust_surface_advantage(
    mu_nodes: np.ndarray,
    energies: np.ndarray,
    tensors: np.ndarray,
    mu_grid: np.ndarray,
    angle_grid: np.ndarray,
    benchmark: np.ndarray,
    enz_window: np.ndarray,
    drifts: np.ndarray,
    validity_limit: float,
    agreement_limits: np.ndarray,
) -> float:
    """Return the best law-screened common-passband guarantee ratio.

    ``mu_nodes``, ``tensors``, and ``mu_grid`` retain their Step 1 input
    roles, ``angle_grid`` retains its Step 2 role, and ``benchmark`` is the
    supplied increasing energy-plus-six-tensor-column table with shape (K,7).
    The target tensor energy-axis length must equal ``len(energies)``.

    ``energies`` is a finite strictly increasing one-dimensional supplied-sample
    grid with at least five nodes. Interpolate target tensors and reconstruct
    the signed semi-infinite Maxwell response. The closed ``enz_window`` selects
    supplied samples; it does not create clipped boundary samples or evaluate a
    continuous interpolant there. Expand each finite comparison boundary by
    exactly one outward IEEE-754 ``nextafter`` step so a decimal endpoint that
    differs from its intended grid node by one representable value remains
    included. Then extract the retained spectrum's signed peak,
    connected half-maximum support, and positive-to-negative lobe spacing.
    Extract target metrics spectrum by spectrum. A rejected spectrum omits only
    target gate/alignment windows containing that spectrum; it must not abort
    the remaining target calculation, and unaffected windows remain eligible.
    Build target ENZ coordinates, evaluate the universal strength and
    lobe-spacing laws on ``angle_grid``, and apply the nominal agreement and
    interval-wide gyrotropy-validity screens. Intersect every target Maxwell
    support in each gate/alignment rectangle, multiply the common width by
    the minimum peak, and select one fixed nominal setting.

    Independently reconstruct the benchmark response from exactly its supplied
    energy-plus-six tensor rows, with no densification or direct analytic-curve
    optimization. At each angle, PCHIP the resulting contrast values and extract
    Step 3 metrics when both signed lobes exist. For each fixed benchmark nominal angle whose complete closed
    alignment window contains valid metrics, intersect all peak-signed
    half-maximum supports, multiply the common width by the minimum peak, and
    maximize this product with exact ties going to the smaller nominal angle.
    Divide the selected target product by that robust 0.16-T benchmark
    product. ``drifts[0]`` is a closed physical chemical-potential half-width:
    include exactly the supplied grid nodes within it. ``drifts[1]`` is an exact
    integer multiple of the uniform angle spacing. Compose all seven prior public
    functions; private reference-implementation names are not public. Raise ``ValueError``
    for malformed, nonfinite, unordered, incompatible, or inherited-contract
    inputs, including tensor/energy-axis mismatch, invalid windows or drifts,
    or the absence of an admissible target or benchmark passband. Absolute
    accuracy: 2e-6.
    """
    return 0.0
```
