# Physics-Optics-25

## Background

Complete Mueller matrices record diattenuation, retardance and depolarization acting together, so a single measured element or a conventional chiroptical reading can mix linear and circular effects. Differential Mueller analysis treats the matrix logarithm of a homogeneous medium as its generator and separates material anisotropies from depolarization, which yields linear and circular diattenuation and birefringence as simultaneous intrinsic parameters.

The 2026 study applies this analysis to plasmonic gammadion arrays of both handedness and to momentum-resolved scattering from an inhomogeneous anisotropic medium, and it contrasts the intrinsic parameters with conventional readings of circular dichroism and circular birefringence. It reports how the intrinsic parameters of opposite enantiomers compare, and it reports which signatures in momentum space it attributes to geometric phase rather than to intrinsic chirality.

This task uses synthetic matrices to compare a nonintrinsic circular response in momentum space with the intrinsic circular birefringence of an enantiomer pair.

## Problem

## Setup

A 2026 differential Mueller polarimetry study of chiral plasmonic gammadion arrays contrasts conventional chiroptical readings with intrinsic anisotropy parameters, both for arrays of opposite handedness and for momentum-resolved scattering. Six anonymous homogeneous samples were measured at normal incidence and one wavelength in one shared laboratory mounting, and three anonymous rings in momentum space were measured on one common irregular azimuth grid, all as complete Mueller matrices in Stokes order (I, Q, U, V) with sample and ring indices starting at 0.

## Inputs

The matrices below are synthetic data defined for this task.

```python
import numpy as np

azimuths_rad = np.array([0.00, 0.47, 1.02, 1.49, 2.07, 2.61, 3.12, 3.71, 4.18, 4.77, 5.29, 5.83])

sample_mueller = np.array([
  # sample 0
  [
    [0.8653071300, -0.0781494224, 0.0488998759, 0.0170688722],
    [-0.0781961616, 0.8014568415, 0.0266321865, -0.1575368577],
    [0.0467076555, -0.0915734338, 0.7307419235, -0.3088412607],
    [0.0221856238, 0.1311203789, 0.3211293276, 0.7066762949],
  ],

  # sample 1
  [
    [0.8369387404, -0.1019701827, -0.0265960428, -0.0007625336],
    [-0.0978828378, 0.7343722252, 0.0748843101, 0.2733262659],
    [-0.0257695196, 0.0047065636, 0.7380209591, -0.2080095968],
    [0.0292723376, -0.2847973616, 0.1943005605, 0.6936486943],
  ],

  # sample 2
  [
    [0.8839121960, -0.0174714253, -0.0755386764, 0.0225782172],
    [-0.0177267863, 0.7936682171, 0.0868161055, 0.2508542420],
    [-0.0727035705, -0.0937442341, 0.8019370618, 0.0123126447],
    [0.0304532733, -0.2482925601, -0.0424726363, 0.7903528110],
  ],

  # sample 3
  [
    [0.8149627860, -0.0691646849, 0.0514860506, 0.0152629721],
    [-0.0759144422, 0.7360078662, -0.0860014591, -0.1251641702],
    [0.0385940982, 0.0247740301, 0.6912961065, -0.3059531908],
    [0.0202632143, 0.1466912139, 0.2964736972, 0.6830655887],
  ],

  # sample 4
  [
    [0.7615189412, 0.0181308638, -0.0396764486, -0.0176111526],
    [0.0109318487, 0.6053042830, -0.0699467856, 0.3485889215],
    [-0.0372602967, 0.0813469017, 0.7119725197, -0.0025055924],
    [-0.0265408048, -0.3464158835, 0.0437542074, 0.6127781360],
  ],

  # sample 5
  [
    [0.7967607310, -0.0941083378, -0.0253649268, -0.0271613833],
    [-0.0976340481, 0.6925395521, 0.0021124603, 0.2670542101],
    [-0.0266257497, 0.0718611750, 0.7143589360, -0.1809770644],
    [0.0013548452, -0.2560528809, 0.1944486737, 0.6514171916],
  ],
])

ring_mueller = np.array([
  # ring 0
  [
    # phi = 0.00 rad
    [
      [0.8535232862, -0.1158290805, -0.2012661370, 0.0020499657],
      [-0.1283819905, 0.5579715731, 0.1901728107, 0.4969655511],
      [-0.1935538212, 0.0978817589, 0.6940975638, -0.3290046156],
      [0.0024794954, -0.5202989366, 0.2910261541, 0.4657611464],
    ],

    # phi = 0.47 rad
    [
      [0.8802477307, -0.2403765875, -0.0364326575, 0.0022367311],
      [-0.2421150091, 0.8138226721, 0.0878262008, 0.0554595304],
      [-0.0222656307, 0.0000337850, 0.4483903913, -0.6303353073],
      [0.0024004702, -0.1001652548, 0.6248495130, 0.4524275721],
    ],

    # phi = 1.02 rad
    [
      [0.9028237687, -0.1495738309, 0.1944771324, 0.0024642943],
      [-0.1393474140, 0.5962068523, -0.1373847002, -0.5456117611],
      [0.2018507666, -0.2157612555, 0.6970168848, -0.3635586761],
      [0.0016329352, 0.5220689430, 0.3962099669, 0.4674006854],
    ],

    # phi = 1.49 rad
    [
      [0.9171940436, 0.0696059207, 0.2320101493, 0.0013738928],
      [0.0798229507, 0.5432501866, 0.1349546085, -0.6121034402],
      [0.2287383301, 0.0658568241, 0.8047593587, 0.2106281677],
      [0.0019252209, 0.6222953117, -0.1788016355, 0.5093953663],
    ],

    # phi = 2.07 rad
    [
      [0.9241318056, 0.2353866016, 0.0224360895, 0.0011420849],
      [0.2360495031, 0.8679709221, 0.0530470748, -0.0358163059],
      [0.0139685633, -0.0049768886, 0.5408323668, 0.6136497002],
      [0.0012853783, 0.0616414219, -0.6116279164, 0.5538715720],
    ],

    # phi = 2.61 rad
    [
      [0.9236640020, 0.1164104133, -0.1912693909, 0.0013993844],
      [0.1102050677, 0.6379747934, -0.1086514849, 0.5484420316],
      [-0.1948631869, -0.1599983085, 0.7707672349, 0.2995430953],
      [0.0006575380, -0.5369138806, -0.3194485549, 0.5464615559],
    ],

    # phi = 3.12 rad
    [
      [0.9195593711, -0.1101176627, -0.1859859251, 0.0007136875],
      [-0.1160268308, 0.6165482684, 0.1773054465, 0.5530884062],
      [-0.1824006238, 0.1265224485, 0.7561204735, -0.3452038704],
      [0.0014024485, -0.5657455223, 0.3243428545, 0.5075983814],
    ],

    # phi = 3.71 rad
    [
      [0.9105142108, -0.2094225806, 0.0394446272, 0.0014011987],
      [-0.2078233341, 0.8433022168, -0.0466720995, -0.1531505652],
      [0.0470858566, -0.1042563817, 0.4842020751, -0.6521115224],
      [0.0011469981, 0.1241087454, 0.6581897366, 0.4693586406],
    ],

    # phi = 4.18 rad
    [
      [0.9055750170, -0.0876898083, 0.1943163899, 0.0016897873],
      [-0.0790935214, 0.5364921172, -0.0986020290, -0.6233605085],
      [0.1979453214, -0.1665465153, 0.7834335820, -0.2420681496],
      [0.0013345553, 0.6098398006, 0.2740962783, 0.4732303757],
    ],

    # phi = 4.77 rad
    [
      [0.8996872946, 0.1517535814, 0.1601306286, 0.0016052519],
      [0.1601494646, 0.6777992427, 0.2051388389, -0.4332563824],
      [0.1517796214, 0.1214982949, 0.6712370500, 0.4560241549],
      [0.0019595398, 0.4611075513, -0.4280233214, 0.5140738679],
    ],

    # phi = 5.29 rad
    [
      [0.9005665487, 0.2266126532, -0.0447635953, 0.0020660011],
      [0.2234578385, 0.8249030891, -0.0177690167, 0.1559866290],
      [-0.0584543060, -0.1131687701, 0.5484105452, 0.5793841642],
      [0.0019291268, -0.1147144361, -0.5888947093, 0.5403597570],
    ],

    # phi = 5.83 rad
    [
      [0.9067965616, 0.0813690507, -0.2268601038, 0.0023588773],
      [0.0665548219, 0.5608859205, -0.0399624348, 0.5876523683],
      [-0.2316140921, -0.1407792244, 0.8050521444, 0.1634897841],
      [0.0021396372, -0.5737301464, -0.2068737744, 0.5293115444],
    ],
  ],

  # ring 1
  [
    # phi = 0.00 rad
    [
      [0.8877242023, -0.1988130124, -0.2011744990, -0.1232612224],
      [-0.1988130124, 0.3973942024, -0.0014131889, 0.6913403561],
      [-0.2011744990, -0.0014131889, 0.7864971881, 0.0572058382],
      [0.1232612224, -0.6913403561, -0.0572058382, 0.3579036364],
    ],

    # phi = 0.47 rad
    [
      [0.9129381505, -0.1475288592, -0.2443899242, -0.1295988961],
      [-0.1475288592, 0.3933425287, -0.0981424857, 0.7016632567],
      [-0.2443899242, -0.0981424857, 0.7851347390, 0.2139311134],
      [0.1295988961, -0.7016632567, -0.2139311134, 0.3322564880],
    ],

    # phi = 1.02 rad
    [
      [0.9338441205, -0.0955222766, -0.2693990360, -0.1261691547],
      [-0.0955222766, 0.4426690821, -0.1661195509, 0.6782722508],
      [-0.2693990360, -0.1661195509, 0.7596967389, 0.3325990300],
      [0.1261691547, -0.6782722508, -0.3325990300, 0.3402829847],
    ],

    # phi = 1.49 rad
    [
      [0.9471604919, -0.0780473036, -0.2748169703, -0.1155825789],
      [-0.0780473036, 0.4884497949, -0.1668075220, 0.6659030097],
      [-0.2748169703, -0.1668075220, 0.7698153460, 0.3485169833],
      [0.1155825789, -0.6659030097, -0.3485169833, 0.3840555453],
    ],

    # phi = 2.07 rad
    [
      [0.9530004893, -0.0963710601, -0.2647630058, -0.1008398197],
      [-0.0963710601, 0.5011269154, -0.1168726887, 0.6833847122],
      [-0.2647630058, -0.1168726887, 0.8188418793, 0.2623371938],
      [0.1008398197, -0.6833847122, -0.2623371938, 0.4347769422],
    ],

    # phi = 2.61 rad
    [
      [0.9527412686, -0.1412875293, -0.2373050274, -0.0940829776],
      [-0.1412875293, 0.4696369011, -0.0338591194, 0.7244226003],
      [-0.2373050274, -0.0338591194, 0.8577995602, 0.1058602326],
      [0.0940829776, -0.7244226003, -0.1058602326, 0.4338169670],
    ],

    # phi = 3.12 rad
    [
      [0.9493315555, -0.1894499576, -0.1952020845, -0.0954275693],
      [-0.1894499576, 0.4386081308, 0.0746620992, 0.7478383500],
      [-0.1952020845, 0.0746620992, 0.8492633436, -0.0808770909],
      [0.0954275693, -0.7478383500, 0.0808770909, 0.3913081457],
    ],

    # phi = 3.71 rad
    [
      [0.9417230190, -0.2331830483, -0.1380633630, -0.1020241848],
      [-0.2331830483, 0.4613246232, 0.1872386770, 0.7170113403],
      [-0.1380633630, 0.1872386770, 0.7722094078, -0.2774335930],
      [0.1020241848, -0.7170113403, 0.2774335930, 0.3432957188],
    ],

    # phi = 4.18 rad
    [
      [0.9386784516, -0.2564057227, -0.1009787877, -0.1079150290],
      [-0.2564057227, 0.5134463476, 0.2200844552, 0.6725964460],
      [-0.1009787877, 0.2200844552, 0.7162322343, -0.3646174240],
      [0.1079150290, -0.6725964460, 0.3646174240, 0.3456246137],
    ],

    # phi = 4.77 rad
    [
      [0.9350488437, -0.2734292579, -0.0833238153, -0.1127894092],
      [-0.2734292579, 0.5524808109, 0.1986406613, 0.6426516216],
      [-0.0833238153, 0.1986406613, 0.7148295551, -0.3584140674],
      [0.1127894092, -0.6426516216, 0.3584140674, 0.3918269862],
    ],

    # phi = 5.29 rad
    [
      [0.9373644441, -0.2768092225, -0.1059532849, -0.1168682867],
      [-0.2768092225, 0.5384335103, 0.1554937379, 0.6616108908],
      [-0.1059532849, 0.1554937379, 0.7632205385, -0.2641155619],
      [0.1168682867, -0.6616108908, 0.2641155619, 0.4264889165],
    ],

    # phi = 5.83 rad
    [
      [0.9438496723, -0.2558956990, -0.1621220689, -0.1243762293],
      [-0.2558956990, 0.4793413902, 0.0870458763, 0.7099302929],
      [-0.1621220689, 0.0870458763, 0.8196157207, -0.1037595407],
      [0.1243762293, -0.7099302929, 0.1037595407, 0.4188699463],
    ],
  ],

  # ring 2
  [
    # phi = 0.00 rad
    [
      [0.8288417814, -0.1903625457, -0.0954212133, 0.0002455790],
      [-0.1904727936, 0.7153763973, 0.1053401815, 0.2399439196],
      [-0.0952012182, 0.1018428201, 0.5524932868, -0.4743713150],
      [0.0005552446, -0.2413708759, 0.4736525316, 0.5044356433],
    ],

    # phi = 0.47 rad
    [
      [0.8546118141, -0.0265670974, -0.2169646392, 0.0010929518],
      [-0.0258114851, 0.5034994714, 0.0351334105, 0.5711107027],
      [-0.2170542866, 0.0335918377, 0.7739160271, -0.0681796938],
      [0.0012230959, -0.5712372358, 0.0671241263, 0.4948406000],
    ],

    # phi = 1.02 rad
    [
      [0.8768335193, 0.1877499629, -0.1171004578, 0.0000511428],
      [0.1876937892, 0.7310014728, -0.1331105568, 0.3147531980],
      [-0.1171879141, -0.1323133857, 0.5875377335, 0.4927190964],
      [-0.0005929996, -0.3151250799, -0.4924847224, 0.5182580582],
    ],

    # phi = 1.49 rad
    [
      [0.8910425884, 0.2032319615, 0.0787914413, -0.0014897869],
      [0.2035384183, 0.7958176696, 0.0901081643, -0.2094544794],
      [0.0780155647, 0.0911433808, 0.5789382848, 0.5272725922],
      [-0.0009715827, 0.2087125792, -0.5275612375, 0.5613025090],
    ],

    # phi = 2.07 rad
    [
      [0.8975695075, 0.0162253219, 0.2051633516, 0.0002436245],
      [0.0161877099, 0.6048117565, 0.0189086779, -0.5462907266],
      [0.2051661132, 0.0175928342, 0.8153127201, 0.0421640890],
      [0.0003573839, 0.5463362261, -0.0415805598, 0.5959623051],
    ],

    # phi = 2.61 rad
    [
      [0.8979279912, -0.1612948071, 0.1156392492, 0.0015764874],
      [-0.1615510861, 0.7606754272, -0.1122763049, -0.3268597768],
      [0.1152932111, -0.1161066750, 0.6642526710, -0.4422680480],
      [0.0009227428, 0.3253902327, 0.4433321078, 0.5884548216],
    ],

    # phi = 3.12 rad
    [
      [0.8944152277, -0.1823943931, -0.0626039830, -0.0004115063],
      [-0.1826291448, 0.8126465833, 0.0858432344, 0.1909297987],
      [-0.0619261046, 0.0818928278, 0.5797600789, -0.5477826743],
      [-0.0000213445, -0.1924453644, 0.5472591592, 0.5516745503],
    ],

    # phi = 3.71 rad
    [
      [0.8857490511, -0.0209201804, -0.1877847517, -0.0011084215],
      [-0.0220328812, 0.5337927184, 0.0342844784, 0.6060708959],
      [-0.1876593853, 0.0330149941, 0.8124342909, -0.0680984648],
      [-0.0009926354, -0.6061035843, 0.0678083282, 0.5155919131],
    ],

    # phi = 4.18 rad
    [
      [0.8813965393, 0.1400646481, -0.1334349936, 0.0005029760],
      [0.1403388054, 0.6858026346, -0.1464262336, 0.4149409911],
      [-0.1331497652, -0.1455131585, 0.6620908627, 0.4299769098],
      [0.0001044210, -0.4151737227, -0.4297537715, 0.5250101264],
    ],

    # phi = 4.77 rad
    [
      [0.8757265794, 0.1844239980, 0.0830782999, 0.0009500481],
      [0.1840253650, 0.7736858056, 0.0948417942, -0.2310052986],
      [0.0839501810, 0.0956735679, 0.6000804116, 0.5039269342],
      [0.0011577953, 0.2309671344, -0.5039447418, 0.5623454035],
    ],

    # phi = 5.29 rad
    [
      [0.8763850678, 0.0129766572, 0.2110458220, -0.0005239701],
      [0.0136218842, 0.5842939588, 0.0152968880, -0.5282059340],
      [0.2110054253, 0.0135281925, 0.8061288041, 0.0332326356],
      [-0.0004941645, 0.5282387545, -0.0327090313, 0.5817697130],
    ],

    # phi = 5.83 rad
    [
      [0.8823732641, -0.1964656279, 0.1041542648, -0.0008550538],
      [-0.1958064964, 0.7631796066, -0.0997403577, -0.2565027065],
      [0.1053791790, -0.1037927092, 0.6203029982, -0.4735841798],
      [-0.0010995765, 0.2553712127, 0.4741922183, 0.5693787070],
    ],
  ],
])
```

## Physical model

Each measured matrix is divided by its own M[0,0], its real principal matrix logarithm is taken with every imaginary part at most 1e-8, and the physically valid differential decomposition used by the study separates the material generator Lm from the depolarizing remainder. The six intrinsic anisotropies, in radians per unit path, follow this exact entry map.

```text
     [  0    -LD   -LD'    CD  ]
Lm = [ -LD    0     CB     LB' ]
     [ -LD'  -CB    0     -LB  ]
     [  CD   -LB'   LB      0  ]
```

The conventional circular reading illuminates with s = (1, 0, 1, 0) and reports, as a principal value in (-pi, pi], twice the signed rotation from the output azimuth back to the plus 45 degree input, so an unchanged plus 45 degree output reads zero and an output without linear polarization is invalid. At each ring point the material model is the homogeneous nondepolarizing medium of unit path length generated by the complete Lm, with all material anisotropies acting simultaneously, divided by its own M[0,0] and read conventionally.

## Task

Exactly one pair of samples consists of opposite enantiomers of one array as the study reports such pairs, and exactly one ring shows the response in momentum space that the study attributes to topology-induced geometric phase rather than to intrinsic chirality. For the selected samples a and b let C = |CB_a - CB_b|/2, let G be the periodic root mean square over the selected ring of its model reading minus its intrinsic CB, and determine **J = G/C**.

In the reasoning, state the enantiomer relation and the evidence in momentum space that the study reports, state the projection that yields the material generator, and report the selected pair with its two intrinsic CB values, C, the selected ring with the closed winding of its linear diattenuation vector (LD, LD'), G and J, giving C, G and J to seven decimal places.

## Numerical conventions

- A parity relation between two real arrays x and y holds when its defect is at most 0.08, with even defect `||x-y|| / max(||x+y||, 1e-12)` and odd defect `||x+y|| / max(||x-y||, 1e-12)` in the Euclidean norm
- An intrinsic differential parameter is negligible on a ring when its magnitude is at most 0.006 rad per unit path at every sampled azimuth
- The closed winding of a planar vector sampled around a ring is the sum of the principal angle increments in (-pi, pi] between consecutive samples, including the increment from the last sample back to the first, divided by 2*pi, and a vector magnitude at or below 1e-10 is invalid
- Periodic weights use `phi_n = phi_0 + 2*pi`, `Delta_j = phi_(j+1) - phi_j` and `w_j = (Delta_(j-1) + Delta_j)/2` with cyclic indices, and the periodic root mean square of f is `sqrt(sum_j w_j*f_j**2 / (2*pi))`
- J is dimensionless and is graded with absolute tolerance 1e-5

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

normalize_mueller_stack

Goal
----
Intensity normalization of a stack of complete Mueller matrices.

```python
def normalize_mueller_stack(mueller_stack: np.ndarray) -> np.ndarray:
    """Divide every complete Mueller matrix by its own M[0,0] element.

    mueller_stack must be a finite real array of shape (n,4,4) with n>=1.
    Raise ValueError for any other shape, for a nonfinite entry, when any
    M[0,0] is at most 1e-12, or when the normalized stack is not finite.
    Return a float array of shape (n,4,4) whose [0,0] entries equal one.
    """
    return result
```

### Step 2

principal_mueller_logarithms

Goal
----
Principal logarithmic Mueller generators.

```python
def principal_mueller_logarithms(normalized_stack: np.ndarray, imag_tol: float = 1e-8) -> np.ndarray:
    """Return the real principal matrix logarithm of every normalized matrix.

    normalized_stack must be a finite real array of shape (n,4,4) with n>=1
    whose [0,0] entries each lie within 1e-8 of one. imag_tol must be a
    finite positive real number and not a Boolean. Take the principal
    logarithm of each complete 4x4 matrix. Raise ValueError for any contract
    violation or when any imaginary component of a logarithm has magnitude
    above imag_tol. Return the real parts with shape (n,4,4).
    """
    return result
```

### Step 3

lorentz_decomposition

Goal
----
Physical differential decomposition of logarithmic Mueller generators.

```python
def lorentz_decomposition(logarithm_stack: np.ndarray) -> np.ndarray:
    """Separate material and depolarizing differential generators.

    logarithm_stack must be a finite real array of shape (n,4,4) with n>=1.
    Apply the physically valid differential decomposition used by the cited
    study to every generator. Return a finite (2,n,4,4) array in lane order
    (material, depolarization) whose two lanes sum to the input. Raise
    ValueError for any contract violation.
    """
    return result
```

### Step 4

anisotropy_coordinates

Goal
----
Six simultaneous anisotropy coordinates from material generators.

```python
def anisotropy_coordinates(material_stack: np.ndarray) -> np.ndarray:
    """Read all six coordinates with the exact public entry map.

    material_stack must be a finite real array of shape (n,4,4) with n>=1,
    and every matrix must match the public material-generator entry
    structure within absolute tolerance 1e-8. Raise ValueError for any
    contract violation. Read LD=-M[0,1], LD_prime=-M[0,2], LB=M[3,2],
    LB_prime=M[1,3], CD=M[0,3] and CB=M[1,2], and return them as an (n,6)
    array in that order, in radians per unit path.
    """
    return result
```

### Step 5

conventional_circular_reading

Goal
----
Conventional circular-birefringence reading under plus 45 degree illumination.

```python
def conventional_circular_reading(normalized_stack: np.ndarray) -> np.ndarray:
    """Read the conventional circular retardance of every normalized matrix.

    normalized_stack must be a finite real array of shape (n,4,4) with n>=1
    whose [0,0] entries each lie within 1e-8 of one. Illuminate with the
    Stokes vector (1,0,1,0). The reading is twice the signed rotation from
    the output azimuth back to the plus 45 degree input azimuth, equal to
    zero for an unchanged plus 45 degree output and reported as a principal
    value in (-pi,pi]. Raise ValueError for any contract violation or when
    the output Q and U satisfy hypot(Q,U)<=1e-12. Return shape (n,).
    """
    return result
```

### Step 6

material_model_reading

Goal
----
Conventional reading of homogeneous nondepolarizing material media.

```python
def material_model_reading(material_stack: np.ndarray) -> np.ndarray:
    """Read the conventional circular retardance of each material model medium.

    material_stack must be a finite real array of shape (n,4,4) with n>=1.
    For every generator construct the homogeneous unit-path Mueller medium
    it generates, divide that matrix by its own [0,0] entry, and apply the
    step-05 plus 45 degree reading. Raise ValueError for a contract
    violation, a nonfinite model matrix, a model [0,0] entry at most 1e-12,
    or an output with hypot(Q,U)<=1e-12. Return principal values in
    (-pi,pi] with shape (n,).
    """
    return result
```

### Step 7

parity_defects

Goal
----
Even and odd parity defects of two real arrays.

```python
def parity_defects(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the even and odd parity defects of two same-shape real arrays.

    first and second must be finite real numeric arrays (not Boolean or
    complex) with identical shapes and at least one element. With Euclidean
    norms over all elements, the even defect is
    ||first-second|| / max(||first+second||, 1e-12) and the odd defect is
    ||first+second|| / max(||first-second||, 1e-12). Raise ValueError for any
    contract violation. Return np.array([even_defect, odd_defect]).
    """
    return result
```

### Step 8

closed_vector_winding

Goal
----
Closed winding of a planar vector sampled around a ring.

```python
def closed_vector_winding(planar_vectors: np.ndarray, magnitude_floor: float = 1e-10) -> float:
    """Return the closed winding of an ordered planar-vector sequence.

    planar_vectors must be a finite real array of shape (n,2) with n>=3,
    whose rows (x_j, y_j) are consecutive samples around one closed ring.
    magnitude_floor must be a finite positive real number and not a Boolean.
    With z_j = x_j + i*y_j, sum the principal arguments in (-pi,pi] of
    z_(j+1)/z_j for j=0..n-1 with z_n = z_0, and divide the sum by 2*pi.
    Raise ValueError for any contract violation or when any |z_j| is at most
    magnitude_floor. Return the winding as a float in full turns.
    """
    return 0.0
```

### Step 9

periodic_ring_rms

Goal
----
Root mean square of a real response around an irregularly sampled closed ring.

```python
def periodic_ring_rms(values: np.ndarray, azimuths_rad: np.ndarray) -> float:
    """Return the periodic-trapezoid root mean square of one ring response.

    values and azimuths_rad must be finite real one-dimensional arrays of the
    same length n>=3, with 0<=azimuths_rad[0]<...<azimuths_rad[-1]<2*pi.
    With phi_n = phi_0 + 2*pi, Delta_j = phi_(j+1) - phi_j for j=0..n-1 and
    w_j = (Delta_(j-1) + Delta_j)/2 using cyclic indices, return
    sqrt(sum_j w_j*values_j**2 / (2*pi)). Raise ValueError for any contract
    violation.
    """
    return 0.0
```

### Step 10

enantiomer_geometric_ratio

Goal
----
Ratio of topology-induced false circular birefringence to enantiomeric circular birefringence.

```python
def enantiomer_geometric_ratio(sample_mueller: np.ndarray | None = None, ring_mueller: np.ndarray | None = None, azimuths_rad: np.ndarray | None = None) -> float:
    """Return J = G / C for the study-identified sample pair and ring.

    Supply all three arrays, or none of them to use the Problem statement
    instance. sample_mueller must have shape (s,4,4) with s>=2 homogeneous
    samples that share one laboratory mounting. ring_mueller must have shape
    (r,m,4,4) with r>=1 and m>=3, every ring being sampled in order at
    azimuths_rad, which must satisfy the step-09 azimuth contract with length
    m. Compose public steps 01 through 09 on every matrix.

    From the samples' step-04 coordinates, with step-07 parity relations
    holding when their defect is at most 0.08, identify the unique pair of
    samples that are opposite enantiomers of one array as the cited study
    reports such pairs. From the rings' step-04 coordinates and step-08
    windings, with an intrinsic differential parameter counted as negligible
    when its magnitude is at most 0.006 rad per unit path at every sampled
    azimuth, identify the unique ring showing the response in momentum space
    that the study attributes to topology-induced geometric phase rather
    than intrinsic chirality.

    For selected samples a and b let C = |CB_a - CB_b|/2. For the selected
    ring let G be the step-09 root mean square of (step-06 reading of its
    material model medium minus its intrinsic CB) at every azimuth. Raise
    ValueError if only some inputs are supplied, if the pair or the ring is
    not unique, if C<=1e-12, or for an upstream contract violation. Return
    J = G / C as a finite float accurate to 1e-10.
    """
    return 0.0
```
