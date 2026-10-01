# Biology-Ecology-24

## Background

Sea turtles bury their eggs in sand and leave the clutch to incubate at the temperature of the beach. The sex of the hatchlings is not fixed at laying. It is fixed during a developmental window partway through incubation, when the gonads differentiate, and the temperature during that window decides whether an embryo develops as male or female. Warmer nests produce more females, so feminizing beaches are a first-order conservation concern for these populations.

Field studies follow nests with temperature loggers buried in the clutch and record the temperature every few minutes from laying until the hatchlings emerge. Laboratory studies incubate eggs at constant temperatures and time development from laying to pipping, the moment a hatchling breaks the shell, which happens several days before the clutch leaves the nest. The two kinds of study therefore measure incubation between different events, so a duration measured to one event cannot be compared directly with a duration measured to the other.

The method behind this task turns a nest temperature record into an expected fraction of male hatchlings. It couples a temperature-driven growth model of the embryo to the temperature sensitivity of sex determination during the developmental window, so that the window is located by the state of the embryo rather than by the calendar. The same record can then be read with incubation taken to end at different events, and the two readings do not agree.

The analysis to reproduce is deterministic. One set of records, one parameter set, one number.

## Problem

Sea turtle hatchling sex is set by the incubation temperature during a developmental window, and a recent model turns a nest temperature record into an expected fraction of male hatchlings.
The end of incubation can be defined two ways: at pipping, when the eggshell first breaks, or at emergence, when the hatchling leaves the nest. Field records end at emergence, and pipping times are not recorded, so the model expected male fraction depends on which end is used. The nine nests below are the mixed-sex nests where the pipping time has been derived from the record.
Your task is to compute how much the expected number of male hatchlings across those nests changes when emergence is used as the end of incubation instead of pipping, and report it to two decimal places.

Compute the change in the expected number of male hatchlings across those nests when emergence, rather than pipping, is treated as the end of incubation, and report it to two decimal places.

The model combines a four-parameter development rate that drives a Gompertz growth curve of embryo size over the held record, a developmental window placed by size and weighted by two sensitivity curves, and a sigmoid that turns the equivalent window temperature into the male fraction. Both thermal curves are four-parameter reaction norms that keep only the high-temperature inactivation, with a pre-exponential term proportional to absolute temperature.

Rules:

* Each branch sizes its window from the size the growth model reaches at that branch's end of incubation, which is the derived pipping time for one branch and the record end for the other, and the window runs between two fractions of the species stage table applied to that size.
* The temperature of a reading holds until the next reading, so a reading is kept only when the temperature changes, and the supplied records are already rounded to 0.1 degrees Celsius and re-thinned, so the solver performs no rounding.
* The last entry of each nest marks the end of its record at emergence, even when its temperature repeats the one before.
* The window is cut at its two edges and at every reading inside it, and a piece carries the held temperature of the reading at or before its left edge.
* The standardised size runs from zero at the window start to one at the window end, and the weight of a piece is the sexualization curve at its temperature times the probability the fitted beta density assigns to its span of standardised size.
* The equivalent temperature is the weighted mean of the piece temperatures, and the male fraction is the asymmetric sigmoid of that temperature. It is 0.95 at P - SL and 0.05 at P + SH, so SL is the span below the pivot and SH the span above it, and it falls with temperature.
* The expected males of a nest are its number of sexed hatchlings times its male fraction, and the change is the pipping sum minus the emergence sum.

* In your reasoning, report the total expected number of male hatchlings under the pipping and emergence conventions, the derived pipping-to-emergence intervals of cm121.cm700s.2020 and CM7.CM200s.2021, the modelled size of CM7.CM200s.2021 at the end of its record, the window limits and weighted temperatures of CM7.CM200s.2021 under both conventions, the emergence-minus-pipping shift of those two weighted temperatures, the contribution of CM7.CM200s.2021 to the change, the direction of the male-fraction curve, and the change itself.

Data:

The stage table gives the fraction of final size at the start of each stage.

```
stage fraction
21 0.195122
22 0.221729
23 0.250554
24 0.356984
25 0.443681
26 0.539246
27 0.783370
28 0.839024
29 0.886696
30 0.942572
31 1.000000
```

The parameter values below use DHA and DHH as enthalpies in kJ/mol and T12H as a temperature in kelvin, Rho25 as a rate in units of 1e-7 per minute with the development rate itself per minute, temperatures in degrees Celsius converted to kelvin with +273.15, sizes in mm with the asymptotic size as asymptotic_ratio times the hatchling length, and the gas constant in J/(mol K).

```
growth DHA 547.922961
growth DHH 576.474056
growth T12H 300.631447
growth Rho25 88.421611
sex_ratio P 29.349664
sex_ratio SL 2.238029
sex_ratio SH 2.080107
sexualization DHA -719.576256
sexualization DHH 685.203574
sexualization T12H 552.204465
sexualization dbeta_shape1 1.725323
sexualization dbeta_shape2 0.802045
fixed Rho25 100.000000
start_size 0.347089
asymptotic_ratio 1.208968
gas_constant 8.314472
```

The nest records give the minutes since the previous reading within the nest and the temperature in Celsius, with the first entry of every nest at laying so its gap is 0, and the last entry of each nest marks the end of its record at emergence even when its temperature repeats the one before.

```
cm311.cm800s.2019, hatchling 49.13 mm, sexed 10, entries are gap:temperature:
0:30.7 30:30.6 105:30.5 135:30.4 180:30.3 300:30.2 330:30.3 615:30.2 585:30.1 210:30.2 30:30.1 15:30.2 795:30.1 720:30.2 615:30.1 750:30.2 270:30.3 405:30.4 705:30.5 450:30.6
300:30.5 690:30.6 375:30.7 570:30.6 495:30.7 840:30.6 1425:30.5 1635:30.4 255:30.5 300:30.6 360:30.7 345:30.6 615:30.7 300:30.8 1230:30.9 300:31.0 1080:31.1 195:31.2 225:31.3 315:31.4
780:31.5 885:31.4 675:31.5 1125:31.6 240:31.7 225:31.8 390:31.9 2835:31.8 420:31.7 1035:31.6 885:31.7 1845:31.6 1005:31.7 705:31.6 540:31.7 240:31.8 180:31.9 1290:32.0 885:31.9 285:32.0
540:32.1 465:32.0 255:32.1 225:32.2 225:32.3 315:32.4 90:32.5 9750:31.5 15:31.8 15:31.9 30:32.0 30:32.1 30:32.2 60:32.3 90:32.4 150:32.5 330:32.6 210:32.7 150:32.8 165:32.9
240:33.0 510:33.1 15:33.0 45:33.1 300:33.2 150:33.3 180:33.4 240:33.5 795:33.6 210:33.7 330:33.8 555:33.7 450:33.8 750:33.7 420:33.6 990:33.5 1590:33.4 2175:33.5 240:33.6 825:33.7
345:33.8 255:33.9 1275:34.0 1035:33.9 90:34.0 1155:33.9 630:33.8 450:33.7 30:33.8 30:33.7 360:33.6 285:33.7 45:33.8 45:33.9 60:34.0 60:34.1 75:34.2 210:34.3 210:34.4 510:34.3
360:34.2 1125:34.1 240:34.0 825:33.9 120:33.8 90:33.7 60:33.6 45:33.5 60:33.4 75:33.3 90:33.2 135:33.1 195:33.0 300:32.9 240:32.0
cm341.cm.1000s.2019, hatchling 50.80 mm, sexed 10, entries are gap:temperature:
0:31.2 30:31.1 75:31.0 240:30.9 345:30.8 210:30.9 465:31.0 375:30.9 855:31.0 795:30.9 570:31.0 1230:31.1 270:31.2 210:31.3 240:31.4 510:31.5 495:31.6 810:31.5 750:31.6 1200:31.7
270:31.8 375:31.9 3015:31.8 480:31.7 735:31.6 390:31.5 705:31.6 1020:31.5 1530:31.6 690:31.5 360:31.4 600:31.5 315:31.6 1110:31.7 2925:31.8 10335:30.8 15:30.7 15:30.8 45:30.9 75:31.0
480:31.1 480:31.2 270:31.3 300:31.4 495:31.5 360:31.6 240:31.7 255:31.8 375:31.9 510:32.0 330:32.1 270:32.2 825:32.3 1710:32.4 870:32.3 390:32.4 1830:32.5 1125:32.6 285:32.7 345:32.8
555:32.9 285:33.0 210:33.1 390:33.2 840:33.3 510:33.4 1245:33.5 960:33.4 30:33.5 15:33.4 690:33.3 30:33.2 30:33.1 60:33.0 120:32.9 675:32.8 315:32.7 405:32.6 345:32.5 960:32.4
1665:32.3 450:32.4 525:32.5 285:32.4 15:32.5 15:32.4 615:32.5 420:32.6 600:32.5 240:32.6 45:32.5 15:32.6 495:32.7 780:32.6 90:32.7 630:32.8 1575:32.9 15:32.8 15:32.9 15:32.8
30:32.9 375:33.0 165:33.1 135:33.2 75:33.3 75:33.4 15:33.3 30:33.4 75:33.5 135:33.6 45:33.7 210:33.8 150:33.9 15:33.8 15:33.9 15:33.8 15:33.9 195:34.0 135:34.1 45:34.0
15:34.1 30:34.0 15:34.1 60:34.0 420:34.1 105:34.0 15:34.1 75:34.0 180:33.9 105:33.8 450:33.7 390:33.6 405:33.5 285:33.4 90:33.5 15:33.4 30:33.5 30:33.4 330:33.3 135:33.2
195:33.1 225:33.0 165:32.9 300:32.8 75:32.7 45:32.6 165:32.5 45:32.6 30:32.5 615:32.4 390:32.3 540:32.2
30:32.3 90:32.2 15:32.3 15:32.2 615:32.1 30:32.2 30:32.1 570:32.2 75:32.1 15:32.2 45:32.1 345:32.1
CM2.CM100s.2020, hatchling 48.37 mm, sexed 10, entries are gap:temperature:
0:28.3 30:28.2 90:28.1 225:28.0 510:28.1 180:28.2 135:28.3 150:28.4 195:28.5 375:28.6 435:28.7 195:28.8 195:28.9 270:29.0 630:28.9 360:28.8 240:28.7 195:28.6 180:28.5 180:28.4
135:28.3 75:28.2 60:28.1 60:28.0 45:27.9 60:27.8 75:27.7 45:27.6 60:27.5 75:27.4 75:27.3 75:27.2 105:27.1 105:27.0 150:26.9 165:26.8 615:26.9 195:27.0 345:27.1 390:27.0
15:27.1 345:27.2 150:27.3 135:27.4 165:27.5 255:27.6 540:27.7 165:27.8 135:27.9 90:28.0 90:28.1 90:28.2 105:28.3 120:28.4 165:28.5 315:28.6 285:28.7 150:28.8 120:28.9 120:29.0
120:29.1 150:29.2 225:29.3 570:29.4 225:29.5 165:29.6 180:29.7 795:29.8 240:29.9 150:30.0 150:30.1 180:30.2 930:30.3 240:30.4 270:30.5 600:30.4 450:30.5 660:30.4 210:30.3 90:30.2
75:30.1 60:30.0 60:29.9 60:29.8 60:29.7 60:29.6 75:29.5 75:29.4 75:29.3 105:29.2 105:29.1 105:29.0 120:28.9 135:28.8 135:28.7 150:28.6 150:28.5 165:28.4 180:28.3 195:28.2
240:28.1 255:28.0 570:28.1 210:28.2 225:28.3 540:28.4 510:28.5 255:28.6 315:28.7 690:28.8 240:28.9 195:29.0 225:29.1 720:29.2 300:29.3 240:29.4 1005:29.5 180:29.6 120:29.7 135:29.8
135:29.9 180:30.0 540:30.1 270:30.2 150:30.3 135:30.4 150:30.5 1050:30.6 150:30.7 165:30.8 270:30.9 255:30.8 435:30.9 150:31.0 150:31.1 150:31.2 180:31.3 660:31.2 30:31.3 300:31.4
165:31.5 165:31.6 345:31.7 105:31.6 570:31.7 255:31.8 840:31.7 345:31.8 855:31.7 210:31.6 690:31.7 810:31.6 225:31.7 285:31.8 240:31.9 735:31.8 285:31.7 210:31.6 180:31.5 150:31.4
135:31.3 150:31.2 165:31.1 180:31.0 885:30.9 195:30.8 255:30.7 240:30.8 180:30.9 75:31.0 120:31.1 105:31.2 150:31.3 195:31.4 570:31.5 180:31.6 135:31.7 150:31.8 195:31.9 810:32.0
150:32.1 75:32.2 120:32.3 120:32.4 135:32.5 240:32.6 705:32.7 150:32.8 135:32.9 135:33.0 195:33.1 810:33.2 150:33.3 135:33.4 135:33.5 225:33.6 660:33.7 225:33.8 135:33.9 165:34.0
195:34.1 825:34.2 195:34.3 180:34.4 195:34.5 975:34.6 165:34.7 165:34.8 15:34.7 15:34.8 1245:34.9 315:35.0 360:34.9 735:35.0 375:35.1 120:35.0 405:34.9 510:35.0 315:35.1 450:35.0
1290:34.9 210:34.8 225:34.7 330:34.8 285:34.9 675:34.8 495:34.9 240:35.0 1290:35.1 690:35.0 225:34.9 900:34.8 240:34.7 195:34.6 120:34.5 300:34.4 510:34.3 270:34.2 210:34.1 165:34.0
465:34.1 270:34.2 510:34.1 1275:34.0 285:33.9 255:34.0 180:34.1 195:34.2 105:34.1 465:34.0 270:33.9 885:33.8 150:33.7 135:33.6
45:33.5 90:33.4 60:33.3 75:33.2 75:33.1 90:33.0 75:32.9 90:32.8 120:32.7 120:32.6 120:32.5 135:32.4 150:32.3 135:32.2
165:32.1 150:32.0 660:31.9 375:31.8 225:31.7 225:31.6 15:31.7 90:31.6 285:31.7 315:31.8 990:31.9 405:31.8 165:31.8
cm121.cm700s.2020, hatchling 48.11 mm, sexed 8, entries are gap:temperature:
0:31.2 15:31.1 15:31.0 60:30.9 120:30.8 165:30.9 75:31.0 75:31.1 90:31.2 90:31.3 210:31.4 90:31.3 270:31.2 150:31.1 135:31.0 330:31.1 120:31.2 135:31.3 435:31.2 180:31.1
120:31.0 135:30.9 300:31.0 75:31.1 60:31.2 60:31.3 75:31.4 105:31.5 555:31.4 165:31.3 375:31.4 75:31.5 60:31.6 75:31.7 90:31.8 510:31.7 150:31.6 120:31.5 180:31.4 45:31.5
135:31.6 75:31.7 60:31.8 75:31.9 75:32.0 120:32.1 180:32.2 45:32.1 225:32.0 150:31.9 105:31.8 135:31.7 405:31.8 135:31.7 225:31.6 135:31.5 120:31.4 105:31.3 120:31.2 120:31.1
360:31.2 150:31.3 660:31.2 210:31.1 285:31.2 105:31.3 75:31.4 105:31.5 105:31.6 615:31.5 210:31.4 360:31.5 255:31.6 180:31.5 270:31.4 180:31.3 165:31.2 240:31.3 105:31.4 75:31.5
75:31.6 105:31.7 375:31.6 150:31.5 120:31.4 120:31.3 120:31.2 360:31.3 195:31.4 345:31.3 240:31.2 180:31.1 360:31.2 75:31.3 60:31.4 75:31.5 90:31.6 120:31.7 405:31.6 210:31.5
165:31.4 240:31.5 120:31.6 135:31.7 585:31.6 165:31.5 150:31.4 255:31.5 90:31.6 60:31.7 75:31.8 75:31.9 105:32.0 465:31.9 180:31.8 135:31.7 180:31.6 120:31.7 135:31.8 90:31.9
90:32.0 105:32.1 120:32.2 315:32.1 150:32.0 150:31.9 210:31.8 15:31.9 165:32.0 90:32.1 45:32.2 90:32.3 90:32.4 195:32.5 165:32.4 210:32.3 135:32.2 105:32.1 105:32.0 570:31.9
105:31.8 105:31.7 75:31.6 90:31.5 75:31.4 90:31.3 75:31.2 90:31.1 105:31.0 210:31.1 90:31.2 75:31.3 75:31.4 105:31.5 300:31.4 150:31.3 105:31.2 120:31.1 105:31.0 300:31.1
60:31.2 45:31.3 60:31.4 45:31.5 60:31.6 60:31.7 90:31.8 135:31.9 255:31.8 195:31.7 150:31.6 270:31.7 90:31.8 60:31.9 75:32.0 75:32.1 60:32.2 120:32.3 405:32.2 195:32.1
120:32.0 285:32.1 75:32.2 90:32.3 90:32.4 150:32.5 345:32.4 180:32.3 150:32.2 360:32.3 75:32.4 60:32.5 75:32.6 90:32.7 150:32.8 285:32.7 210:32.6 165:32.5 255:32.6 90:32.7
75:32.8 60:32.9 60:33.0 60:33.1 90:33.2 180:33.3 165:33.2 180:33.1 105:33.0 75:32.9 105:32.8 120:32.7 180:32.8 390:32.7 90:32.6 90:32.5 60:32.4 75:32.3 60:32.2 60:32.1
45:32.0 75:31.9 60:31.8 75:31.7 105:31.6 165:31.7 105:31.8 90:31.9 150:32.0 135:31.9 90:31.8 60:31.7 45:31.6 60:31.5 60:31.4 75:31.3 75:31.2 60:31.1 90:31.0 90:30.9
90:30.8 165:30.9 90:31.0 105:31.1 270:31.0 75:30.9 45:30.8 75:30.7 60:30.6 75:30.5 75:30.4 90:30.3 105:30.2 120:30.1 225:30.2 165:30.3 135:30.4 195:30.5 285:30.4 270:30.3
435:30.4 135:30.5 180:30.6 240:30.5 225:30.4 165:30.3 150:30.2 330:30.3 60:30.4 60:30.5 60:30.6 60:30.7 75:30.8 120:30.9 255:30.8 165:30.7 120:30.6 105:30.5 90:30.4 75:30.3
90:30.2 135:30.1 195:30.0 120:29.9 45:29.8 60:29.7 60:29.6 60:29.5 60:29.4 75:29.3 90:29.2 105:29.1 105:29.0 135:28.9 345:29.0 105:29.1 90:29.2 105:29.3 120:29.4 195:29.5
390:29.4 225:29.5 120:29.6 60:29.7 45:29.8 30:29.9 45:30.0 45:30.1 30:30.2 45:30.3 45:30.4 45:30.5 75:30.6 75:30.7 150:30.8 255:30.7 255:30.6 135:30.7 105:30.8 60:30.9
30:31.0 45:31.1 30:31.2 45:31.3 45:31.4 45:31.5 60:31.6 75:31.7 105:31.8 390:31.7 195:31.6 210:31.5 45:31.6 135:31.7 75:31.8 60:31.9 60:32.0 90:32.1 90:32.2 375:32.1
120:32.0 135:31.9 135:31.8 270:31.9 90:32.0 60:32.1 60:32.2 75:32.3 450:32.2 135:32.1 75:32.0 90:31.9 105:31.8 120:31.7 165:31.8 105:31.9 75:32.0 75:32.1 75:32.2 210:32.3
60:32.2 240:32.1 90:32.0 105:31.9 105:31.8 330:31.9 75:32.0 60:32.1 45:32.2 60:32.3 75:32.4 135:32.5 330:32.4 180:32.3 120:32.2 180:32.1 45:32.2 120:32.3 60:32.4 60:32.5
45:32.6 45:32.7 60:32.8 60:32.9 75:33.0 105:33.1 180:33.0 105:32.9 105:32.8 75:32.7 75:32.6 60:32.5 60:32.4 60:32.3 60:32.2 60:32.1 30:32.0 60:31.9 45:31.8 75:31.7
75:31.6 90:31.5 105:31.4 105:31.3 90:31.2 90:31.1 75:31.0 60:30.9 60:30.8 60:30.7 75:30.6 105:30.5 90:30.4 330:30.5 90:30.6 120:30.7 270:30.6 255:30.5 30:30.4 60:30.3
165:30.2 240:30.3 105:30.4 90:30.5 75:30.6 75:30.7 90:30.8 480:30.7 105:30.6 120:30.5 180:30.4 270:30.5 345:30.4 240:30.3 165:30.2 135:30.1 165:30.0 390:30.1 420:30.0 165:29.9
210:29.8 195:29.7 300:29.8 90:29.9 90:30.0 90:30.1 180:30.2 195:30.1 240:30.0 150:29.9 165:29.8 270:29.9 75:30.0 45:30.1 60:30.2
45:30.3 60:30.4 60:30.5 60:30.6 120:30.7 405:30.6 240:30.5 165:30.6 135:30.7 90:30.8 75:30.9 75:31.0 120:31.1 135:31.1
CM3.CM100s.2021, hatchling 49.41 mm, sexed 10, entries are gap:temperature:
0:27.9 45:28.0 75:28.1 75:28.2 105:28.3 240:28.4 420:28.5 210:28.6 210:28.7 300:28.8 750:28.9 210:29.0 210:29.1 855:29.2 300:29.3 240:29.4 465:29.5 75:29.4 30:29.5 15:29.4
330:29.5 345:29.6 300:29.7 1035:29.8 270:29.9 1110:30.0 225:30.1 240:30.2 1065:30.3 315:30.4 1140:30.5 285:30.6 1200:30.7 945:30.6 375:30.7 1695:30.6 390:30.5 240:30.4 240:30.3 600:30.4
1275:30.5 420:30.6 285:30.5 720:30.6 375:30.7 1020:30.8 225:30.9 195:31.0 825:31.1 285:31.2 270:31.3 1050:31.4 375:31.5 1260:31.6 1170:31.7 270:31.8 285:31.9 945:32.0 270:32.1 255:32.2
990:32.3 300:32.4 1275:32.5 300:32.4 315:32.3 240:32.2 330:32.1 255:32.0 255:31.9 180:31.8 165:31.7 165:31.6 165:31.5 180:31.4 180:31.3 195:31.2 180:31.1 165:31.0 165:30.9 120:30.8
210:30.7 360:30.6 270:30.5 210:30.4 195:30.3 180:30.2 225:30.1 435:30.0 360:29.9 255:29.8 255:29.7 1635:29.8 240:29.9 195:30.0 195:30.1 285:30.2 390:30.3 195:30.4 135:30.5 105:30.6
105:30.7 120:30.8 105:30.9 105:31.0 180:31.1 240:31.2 210:31.3 165:31.4 120:31.5 135:31.6 165:31.7 270:31.8 855:31.9 195:32.0 195:32.1 165:32.2 465:32.3 270:32.4 150:32.5 120:32.6
135:32.7 135:32.8 165:32.9 390:33.0 405:33.1 300:33.2 1350:33.3 375:33.4 615:33.3 135:33.4 30:33.3 15:33.4 1575:33.5 165:33.6 135:33.7 180:33.8 690:33.9 255:34.0 150:34.1 105:34.2
165:34.3 240:34.4 930:34.5 780:34.4 1425:34.3 240:34.2 345:34.1 285:34.0 300:33.9 180:33.8 180:33.7 180:33.6 285:33.5 690:33.4 225:33.3 600:33.4 255:33.5 660:33.4 285:33.5 240:33.6
120:33.7 165:33.8 285:33.9 660:34.0 135:34.1 90:34.2 120:34.3 105:34.4 135:34.5 195:34.6 735:34.7 540:34.8 105:34.7 75:34.6 30:34.7 15:34.6 60:34.7 30:34.6 30:34.5 105:34.4
90:34.3 120:34.2 90:34.1 75:34.0 75:33.9 75:33.8 45:33.7 75:33.6 60:33.5 105:33.4 150:33.3 60:33.2 75:33.1 75:33.0 75:32.9 135:32.8
150:32.7 165:32.6 1005:32.5 315:32.4 240:32.5 420:32.6 240:32.5 315:32.4 675:32.5 210:32.6 645:32.5 1365:32.4 480:32.3 240:32.2 285:32.2
CM7.CM200s.2021, hatchling 52.65 mm, sexed 10, entries are gap:temperature:
0:29.8 15:29.4 495:29.3 75:29.4 315:29.5 225:29.6 1050:29.7 210:29.8 255:29.9 690:29.8 150:29.9 360:30.0 570:29.9 210:29.8 240:29.7 825:29.6 225:29.5 180:29.4 180:29.3 165:29.2
180:29.1 195:29.0 195:28.9 180:28.8 180:28.7 165:28.6 225:28.5 450:28.4 240:28.3 210:28.2 180:28.1 180:28.0 270:27.9 540:27.8 330:27.7 240:27.6 1800:27.7 240:27.8 240:27.9 300:28.0
600:28.1 180:28.2 165:28.3 165:28.4 270:28.5 675:28.6 150:28.7 135:28.8 165:28.9 750:28.8 465:28.9 480:29.0 195:28.9 510:29.0 180:29.1 150:29.2 150:29.3 255:29.4 420:29.3 360:29.4
285:29.5 810:29.4 1245:29.3 330:29.2 525:29.1 375:29.0 255:28.9 240:28.8 390:28.9 240:29.0 990:29.1 150:29.2 105:29.3 105:29.4 120:29.5 150:29.6 555:29.5 1110:29.4 240:29.3 255:29.2
315:29.3 645:29.2 255:29.1 255:29.0 390:28.9 330:28.8 255:28.7 180:28.6 225:28.5 1065:28.4 345:28.3 60:28.4 225:28.5 135:28.6 150:28.7 300:28.8 135:28.7 690:28.8 120:28.9 105:29.0
120:29.1 150:29.2 270:29.3 600:29.4 105:29.5 90:29.6 90:29.7 105:29.8 135:29.9 225:30.0 795:30.1 165:30.2 600:30.1 225:30.0 195:29.9 255:29.8 360:29.7 330:29.6 300:29.5 420:29.6
180:29.7 195:29.8 330:29.9 15:29.8 15:29.9 120:29.8 420:29.9 165:30.0 105:30.1 120:30.2 120:30.3 225:30.4 285:30.3 555:30.4 135:30.5 105:30.6 120:30.7 150:30.8 675:30.7 315:30.8
240:30.9 210:31.0 555:30.9 360:31.0 210:31.1 225:31.2 675:31.1 405:31.0 240:30.9 165:30.8 195:30.7 195:30.6 195:30.5 525:30.6 765:30.5 210:30.4 405:30.5 150:30.6 165:30.7 300:30.8
165:30.7 495:30.8 120:30.9 60:31.0 90:31.1 105:31.2 105:31.3 135:31.4 1050:31.5 390:31.4 15:31.5 15:31.4 270:31.3 210:31.2 510:31.3 150:31.4 210:31.5 930:31.6 135:31.7 105:31.8
120:31.9 120:32.0 240:32.1 675:32.2 150:32.3 120:32.4 120:32.5 165:32.6 750:32.5 15:32.6 15:32.5 15:32.6 345:32.7 645:32.6 195:32.5 495:32.6 135:32.7 135:32.8 180:32.9 615:32.8
15:32.9 30:32.8 30:32.9 15:32.8 15:32.9 195:33.0 105:33.1 105:33.2 120:33.3 165:33.4 990:33.5 210:33.6 315:33.7 15:33.6 315:33.5 180:33.4 795:33.3 315:33.2 150:33.1 135:33.0
105:32.9 225:32.8 105:32.9 465:33.0 150:32.9 15:33.0 15:32.9 420:32.8 345:32.9 150:33.0 105:33.1 210:33.2 630:33.1 315:33.2 165:33.3 150:33.4 735:33.3 240:33.2 855:33.1 195:33.0
135:32.9 150:32.8 180:32.7 225:32.8 660:32.7 210:32.6 150:32.5 75:32.4 120:32.3 15:32.4 75:32.3 30:32.4 75:32.5 45:32.6 30:32.7 15:32.6 15:32.7 75:32.8 30:32.9 30:33.0
30:33.1 45:33.2 75:33.3 30:33.2 75:33.3 60:33.4 75:33.5 45:33.6 90:33.5 120:33.6 15:33.5 15:33.6 15:33.7 15:33.6 45:33.7 15:33.6 15:33.7 195:33.8 60:33.9 60:34.0
75:34.1 150:34.2 30:34.3 45:34.4 30:34.3 15:34.4 60:34.5 90:34.4 30:34.5 30:34.4 135:34.5 90:34.6 30:34.7 30:34.6 30:34.5 60:34.6 15:34.5 15:34.6 45:34.5 135:34.4
30:34.5 45:34.4 270:34.5 105:34.6 60:34.7 30:34.6 30:34.7 60:34.8 90:34.7 60:34.8 15:34.7 15:34.8 15:34.7 15:34.8 210:34.7 45:34.8 45:34.7 105:34.6 15:34.7 30:34.6
30:34.5 270:34.4 15:34.5 30:34.4 165:34.5 15:34.4 15:34.5 15:34.4 15:34.5 15:34.4 90:34.5 30:34.4 60:34.3 15:34.4 30:34.3 15:34.4 15:34.3 15:34.4 15:34.3 210:34.2
45:34.3 45:34.2 150:34.1 105:34.0 30:34.1 60:34.0 15:33.9 45:33.8 60:33.7 45:33.6 60:33.5 60:33.4 105:33.3 75:33.2 60:33.1 90:33.0 60:32.9 90:32.8 45:32.7 90:32.6
90:32.5 75:32.4 90:32.3 90:32.2 90:32.1 60:32.0 105:31.9 75:31.8 150:31.7 150:31.6 315:31.5 345:31.4
255:31.3 195:31.2 15:31.3 30:31.2 210:31.3 135:31.2 60:31.1 60:31.0 75:30.9 75:30.8 150:30.7 225:30.6
cm170.cm1500s.2021, hatchling 50.73 mm, sexed 10, entries are gap:temperature:
0:31.5 30:31.6 15:31.5 150:31.4 180:31.3 225:31.2 420:31.3 240:31.4 300:31.5 375:31.4 405:31.5 240:31.6 225:31.7 630:31.6 645:31.5 315:31.4 195:31.3 180:31.2 165:31.1 1005:31.0
240:30.9 150:30.8 555:30.9 165:31.0 300:31.1 375:31.0 285:31.1 240:31.2 165:31.3 210:31.4 1065:31.5 240:31.6 675:31.5 210:31.4 165:31.3 180:31.2 180:31.1 165:31.0 150:30.9 120:30.8
150:30.7 150:30.6 240:30.5 945:30.4 225:30.3 645:30.4 330:30.5 285:30.4 630:30.5 180:30.6 195:30.7 840:30.6 15:30.7 15:30.6 90:30.7 465:30.8 525:30.7 540:30.8 255:30.9 240:31.0
360:30.9 345:30.8 225:30.9 270:31.0 885:30.9 270:31.0 330:31.1 780:31.0 375:31.1 315:31.2 1530:31.3 480:31.2 420:31.1 885:31.0 405:30.9 240:31.0 375:31.1 1335:31.2 2820:31.3 300:31.4
1020:31.5 195:31.6 225:31.7 870:31.8 315:31.9 345:32.0 465:31.9 285:32.0 360:32.1 690:32.0 270:31.9 540:31.8 240:31.7 165:31.6 150:31.5 135:31.4 135:31.3 270:31.2 165:31.3 675:31.2
315:31.1 1050:31.0 270:30.9 435:31.0 180:31.1 165:31.2 300:31.3 210:31.2 810:31.3 1005:31.2 15:31.3 435:31.4 930:31.3 15:31.4 15:31.3 135:31.4 15:31.3 15:31.4 885:31.3 270:31.2
435:31.3 345:31.4 300:31.3 345:31.2 330:31.3 240:31.4 285:31.5 405:31.4 585:31.5 270:31.6 720:31.5 360:31.6 285:31.7 1320:31.8 240:31.9 1125:32.0 195:32.1 135:32.2 735:32.1 105:32.0
165:31.9 165:31.8 180:31.7 180:31.6 150:31.5 150:31.4 120:31.3 105:31.2 105:31.1 135:31.0 255:30.9 480:30.8 360:30.7 1635:30.6 330:30.7 870:30.6 540:30.7 120:30.8 120:30.9 75:31.0
120:31.1 225:31.2 345:31.1 435:31.2 165:31.3 150:31.4 135:31.5 405:31.6 405:31.5 15:31.6 345:31.7 15:31.6 195:31.7 180:31.8
510:31.7 570:31.8 270:31.7 315:31.6 180:31.5 150:31.4 195:31.3 555:31.2 240:31.1 150:31.0 135:30.9 90:30.8 120:30.7 165:30.6
630:30.5 315:30.4 180:30.3 150:30.2 195:30.1 210:30.0 240:29.9 225:29.8 210:29.7 195:29.6 180:29.5 180:29.4 195:29.3 360:29.2
cm1.cm100s.2023, hatchling 50.51 mm, sexed 10, entries are gap:temperature:
0:28.9 120:28.8 165:28.7 270:28.6 810:28.7 1245:28.8 495:28.9 375:28.8 510:28.9 855:28.8 690:28.9 1590:29.0 945:29.1 1065:29.0 975:29.1 555:29.0 420:29.1 375:29.2 900:29.3 390:29.4
360:29.5 1035:29.6 1275:29.7 390:29.8 2145:29.7 1290:29.6 405:29.5 510:29.4 345:29.3 300:29.2 570:29.1 525:29.0 300:28.9 855:29.0 990:29.1 300:29.2 285:29.3 840:29.4 330:29.5 1395:29.6
975:29.5 975:29.4 1020:29.5 1560:29.6 180:29.5 1140:29.6 525:29.5 765:29.6 330:29.7 870:29.8 90:29.9 120:30.0 270:30.1 270:30.0 300:29.9 300:29.8 795:29.7 915:29.8 450:29.9 1095:30.0
900:30.1 345:30.2 810:30.1 525:30.2 195:30.3 165:30.4 195:30.5 315:30.6 480:30.7 210:30.8 180:30.9 150:31.0 240:31.1 675:31.2 1740:31.3 795:31.2 285:31.3 285:31.4 210:31.5 240:31.6
615:31.7 240:31.8 180:31.9 150:32.0 195:32.1 195:32.2 540:32.3 270:32.4 240:32.5 420:32.6 15:32.5 30:32.6 15:32.5 15:32.6 15:32.5 15:32.6 15:32.5 585:32.6 630:32.5 300:32.4
255:32.3 495:32.4 1080:32.5 330:32.6 495:32.5 375:32.4 255:32.5 180:32.6 645:32.5 210:32.4 630:32.5 240:32.6 465:32.7 60:32.6 30:32.7 30:32.6 15:32.7 60:32.6 15:32.7 15:32.6
15:32.7 495:32.8 345:32.9 915:33.0 195:33.1 345:33.2 330:33.1 15:33.2 15:33.1 540:33.2 885:33.1 15:33.2 15:33.1 15:33.2 15:33.1 1320:33.0 300:32.9 375:33.0 270:33.1 1140:33.2
15:33.1 15:33.2 180:33.3 165:33.4 300:33.5 780:33.6 165:33.7 315:33.8 285:33.7 75:33.8 15:33.7 285:33.6 15:33.7 15:33.6
135:33.7 45:33.6 15:33.7 255:33.8 300:33.9 15:33.8 30:33.9 480:33.8 180:33.7 315:33.8 180:33.7 15:33.8 30:33.7 225:33.6
90:33.5 165:33.4 135:33.3 150:33.2 195:33.1 495:33.0 300:32.9 270:32.8 585:32.9 210:33.0 255:33.1 1095:33.2 540:33.3 1275:33.3
cm17.cm300s.2023, hatchling 50.75 mm, sexed 10, entries are gap:temperature:
0:29.5 90:29.4 435:29.3 630:29.4 735:29.3 840:29.4 450:29.3 390:29.2 1185:29.1 645:29.2 465:29.3 405:29.2 570:29.3 720:29.2 1590:29.1 2175:29.2 1020:29.1 915:29.0 1095:29.1 555:29.0
300:29.1 525:29.2 2055:29.1 435:29.2 270:29.3 300:29.4 810:29.5 315:29.6 390:29.7 1740:29.6 510:29.7 495:29.8 120:29.7 1020:29.8 315:29.9 840:30.0 270:30.1 270:30.2 585:30.3 15:30.2
30:30.3 585:30.4 1995:30.3 270:30.2 270:30.1 705:30.0 915:30.1 810:30.0 1290:29.9 210:29.8 240:29.7 405:29.8 390:29.9 1050:30.0 495:30.1 810:30.2 300:30.3 1020:30.4 225:30.5 885:30.4
1260:30.3 870:30.4 990:30.5 240:30.6 195:30.7 195:30.8 300:30.9 465:31.0 270:31.1 180:31.2 225:31.3 1095:31.4 255:31.5 375:31.6 885:31.7 660:31.6 480:31.7 405:31.8 465:31.7 720:31.8
225:31.9 270:32.0 795:32.1 195:32.2 210:32.3 255:32.4 810:32.5 240:32.6 225:32.7 720:32.8 375:32.9 195:33.0 180:33.1 630:33.2 405:33.3 240:33.4 1455:33.3 255:33.2 180:33.1 180:33.0
150:32.9 300:32.8 405:32.7 285:32.6 195:32.5 195:32.4 795:32.5 645:32.4 270:32.5 315:32.6 225:32.7 600:32.6 405:32.7 240:32.8 195:32.9 390:33.0 15:32.9 15:33.0 15:32.9 90:33.0
90:32.9 90:33.0 345:33.1 180:33.2 210:33.3 990:33.4 1005:33.3 705:33.4 420:33.5 15:33.4 60:33.5 315:33.6 15:33.5 30:33.6 60:33.7 150:33.8 15:33.7 90:33.8 120:33.9 150:33.8
210:33.7 345:33.6 120:33.5 300:33.4 360:33.3 645:33.2 210:33.1 240:33.0 195:32.9 255:32.8 360:32.7 210:32.6 210:32.5 1035:32.6 1530:32.7 285:32.6 690:32.5 345:32.4 15:32.5 15:32.9
```

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep the reasoning focused on the calculation and report the values named above inside the <reasoning> tags. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_development_rate

Goal
----
Step 1: the temperature-dependent development rate of the embryo.

```python
def development_rate(
    temperature: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
) -> "numpy.ndarray":
    """Evaluate the development rate curve at each temperature, per minute.

    Parameters
    ----------
    temperature : array-like
        Temperatures in degrees Celsius. Every entry must be finite and above
        -273.15.
    DHA : float, optional
        Enthalpy of activation, in kJ/mol (547.922961 in the task data).
    DHH : float, optional
        Enthalpy of high-temperature inactivation, in kJ/mol (576.474056).
    T12H : float, optional
        Temperature of half high-temperature inactivation, in kelvin (300.631447).
    Rho25 : float, optional
        Rate at the reference temperature of 298 K, in units of 1e-7 per minute
        (88.421611).

    Returns
    -------
    rate : numpy.ndarray
        Development rate per minute, with the shape of `temperature`, with
        temperatures converted to kelvin by adding 273.15, the reference
        temperature taken as exactly 298 K and the gas constant as 8.314472
        J/(mol K).

    Raises
    ------
    ValueError
        If an entry of `temperature` is not finite or is at or below -273.15
        degrees Celsius.
    """
    return rate
```

### Step 2

02_embryo_growth

Goal
----
Step 2: the modelled embryo size over a held temperature record.

```python
def embryo_growth(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    query_times: "numpy.typing.ArrayLike | None" = None,
) -> "numpy.ndarray":
    """Modelled embryo size at the reading times, or at supplied query times.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite. The temperature
        of a reading holds until the next reading.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive. The Gompertz
        asymptote is ``asymptotic_ratio * hatchling_length``.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    start_size : float, optional
        Modelled size at laying in mm (0.347089). Must lie between zero and the
        asymptote.
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    query_times : array-like, optional
        Times in minutes at which the size is returned. A query time inside a
        logging interval is evaluated with that interval's held temperature and
        rate. By default the reading times are used.

    Returns
    -------
    size : numpy.ndarray
        Modelled size in mm. At the reading times it has one entry per reading,
        with `start_size` at the first reading. For `query_times` it has their
        shape. Each entry is the Gompertz trajectory under the held temperature
        of its interval.

    Raises
    ------
    ValueError
        If the two arrays differ in length or hold fewer than two readings, if a
        time is not strictly increasing, if a temperature is not finite or is at
        or below -273.15 degrees Celsius, if a size parameter is not positive or
        does not place the start size below the asymptote, or if a query time is
        not finite.
    """
    return size
```

### Step 3

03_pipping_time

Goal
----
Step 3: the time at which the modelled embryo reaches a size.

```python
def pipping_time(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    target_size: "float | None" = None,
) -> float:
    """Time at which the modelled embryo size reaches a target size.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite. The temperature
        of a reading holds until the next reading.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive. It sets the Gompertz
        asymptote of the growth model.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    target_size : float, optional
        Size in mm whose crossing time is returned. The default is the hatchling
        length, which is the definition of pipping.

    Returns
    -------
    float
        Time in minutes from laying at which the modelled size reaches
        `target_size`. The crossing lies inside the first logging interval whose
        end size exceeds the target. If the modelled size never reaches the
        target inside the record, the last reading time is returned. If the
        modelled size already exceeds the target at the first reading, the first
        reading time is returned.

    Raises
    ------
    ValueError
        If the growth inputs fail the validation of step 2, or if `target_size`
        is not positive.
    """
    return time
```

### Step 4

04_window_limits

Goal
----
Step 4: the developmental window of a nest, given the end of incubation.

```python
def window_limits(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    end_time: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
) -> "numpy.ndarray":
    """Place the developmental window from a supplied end of incubation.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive.
    end_time : float
        Time in minutes from laying at which this branch's incubation ends, from
        the first to the last reading time. The window is sized from the modelled
        size at this time, not from the measured hatchling length.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    frac_begin : float, optional
        Lower stage fraction of the modelled size at `end_time`.
    frac_end : float, optional
        Upper stage fraction of the modelled size at `end_time`.

    Returns
    -------
    numpy.ndarray
        Three floats. Entry 0 is the modelled size in mm at `end_time`, the size
        that scales the window. Entries 1 and 2 are the window start and window
        end in minutes from laying, the crossing times of ``frac_begin`` and
        ``frac_end`` times that size, located as in step 3, so a target below the
        modelled size at laying gives the first reading time.

    Raises
    ------
    ValueError
        If the growth inputs fail the validation of step 2, if `end_time` is not
        finite or lies outside the record, or if the two fractions are not
        positive and strictly increasing.
    """
    return limits
```

### Step 5

05_sexualization_weight

Goal
----
Step 5: the weight of one window piece under the two sensitivity curves.

```python
def sexualization_weight(
    temperature: "numpy.typing.ArrayLike",
    standardised_edges: "numpy.typing.ArrayLike",
    DHA: float = -719.576256,
    DHH: float = 685.203574,
    T12H: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    Rho25: float = 100.0,
) -> "numpy.ndarray":
    """Weight of each window piece from the sexualization curve and the beta mass.

    Parameters
    ----------
    temperature : array-like
        Held temperature in degrees Celsius of each piece, one entry per piece.
    standardised_edges : array-like
        Standardised size at the piece edges, one more entry than there are
        pieces, strictly increasing inside the closed unit interval. The task
        pipeline supplies the full window, from 0 at the window start to 1 at the
        window end, and any increasing subinterval is also valid.
    DHA, DHH, T12H : float, optional
        Parameters of the sexualization reaction norm (-719.576256, 685.203574,
        552.204465).
    shape1, shape2 : float, optional
        Shapes of the fitted beta density over the standardised size (1.725323,
        0.802045).
    Rho25 : float, optional
        Rate of the sexualization norm at the reference temperature of 298 K
        (100.0, fixed by the fitted model).

    Returns
    -------
    weight : numpy.ndarray
        One weight per piece. Each entry is the sexualization norm at the piece
        temperature times the fitted beta sensitivity over the piece's span of
        standardised size.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional and finite, if the edges number
        fewer than two or are not strictly increasing inside the closed unit
        interval, if the edges do not carry one more entry than the
        temperatures, or if a shape or the fixed rate is not positive.
    """
    return weight
```

### Step 6

06_constant_temperature_equivalent

Goal
----
Step 6: the equivalent temperature of the developmental window.

```python
def constant_temperature_equivalent(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    window_start: float,
    window_end: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    Rho25_s: float = 100.0,
) -> float:
    """Weighted mean temperature over a developmental window.

    Parameters
    ----------
    times : array-like
        Reading times in minutes from laying, strictly increasing and finite.
    temperature : array-like
        Temperature in degrees Celsius at each reading, finite.
    hatchling_length : float
        Hatchling straight carapace length in mm, positive.
    window_start, window_end : float
        Window limits in minutes from laying, the crossing times of step 4, with
        ``window_start < window_end`` inside the record.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1, which set the modelled
        sizes at the window edges.
    DHA_s, DHH_s, T12H_s : float, optional
        Parameters of the sexualization reaction norm.
    shape1, shape2 : float, optional
        Shapes of the fitted beta density over the standardised size.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    Rho25_s : float, optional
        Rate of the sexualization norm at the reference temperature (100.0).

    Returns
    -------
    float
        The weighted mean temperature in degrees Celsius. The window is cut at its
        two edges and at every reading strictly inside it. Each piece carries the
        held temperature of the reading at or before its left edge and the span of
        standardised modelled size between its edges. The mean weights each piece
        temperature by its step 5 weight.

    Raises
    ------
    ValueError
        If the growth inputs fail the validation of step 2, if the window limits
        are not finite and strictly ordered, or if the window is not contained in
        the record.
    """
    return temperature_equivalent
```

### Step 7

07_male_fraction

Goal
----
Step 7: the male fraction at a temperature.

```python
def male_fraction(
    temperature: float,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    l: float = 0.05,
) -> float:
    """Male fraction at one temperature, from the asymmetric sigmoid.

    Parameters
    ----------
    temperature : float
        Equivalent temperature of the developmental window, in degrees Celsius.
        It must be finite.
    P : float, optional
        Pivotal temperature, in degrees Celsius (29.349664). The fraction is one
        half at the pivot.
    SL : float, optional
        Temperature span below the pivot over which the fraction rises from one
        half to 0.95 (2.238029). It is the steepness of the curve below the pivot.
    SH : float, optional
        Temperature span above the pivot over which the fraction falls from one
        half to 0.05 (2.080107).
    l : float, optional
        Tail level of the sigmoid (0.05), strictly between zero and one half. It
        sets the logistic scale.

    Returns
    -------
    float
        The male fraction at `temperature`, between zero and one. It is 0.95 at
        ``P - SL``, 0.5 at `P` and 0.05 at ``P + SH``, with the steepness ``SL``
        below the pivot and ``SH`` at and above it.

    Raises
    ------
    ValueError
        If `temperature` is not finite, if `P` is not finite, if `SL` or `SH` is
        not positive, or if `l` is not strictly between zero and one half.
    """
    return fraction
```

### Step 8

08_expected_males

Goal
----
Step 8: the expected number of male hatchlings over a set of nests.

```python
def expected_males(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
    end_times: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
    Rho25_s: float = 100.0,
) -> float:
    """Sum of the expected male hatchlings over the nests, for supplied ends.

    Parameters
    ----------
    records : sequence of array-like
        One record per nest. Each is a 2-D array with two columns, the times in
        minutes from laying and the temperature in degrees Celsius at each
        reading, with strictly increasing times and at least two readings.
    hatchling_lengths : array-like
        One hatchling straight carapace length in mm per nest, positive.
    sexed_counts : array-like
        One number of sexed hatchlings per nest, non-negative.
    end_times : array-like
        One end of incubation in minutes from laying per nest, inside its record.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    P, SL, SH : float, optional
        Parameters of the male fraction curve of step 7.
    DHA_s, DHH_s, T12H_s, shape1, shape2 : float, optional
        Parameters of the sexualization norm and the fitted beta density.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    frac_begin, frac_end : float, optional
        Stage fractions of the modelled size at each nest's end of incubation.
    Rho25_s : float, optional
        Rate of the sexualization norm at the reference temperature (100.0).

    Returns
    -------
    float
        The sum over the nests of the number of sexed hatchlings times the male
        fraction of that nest. Each male fraction is the sex-ratio curve of step 7
        at the equivalent temperature of the window placed by step 4 and weighted
        by step 6 for that nest's supplied end of incubation.

    Raises
    ------
    ValueError
        If the four per-nest inputs do not have the same length, if a record is
        not a 2-D array with two columns and at least two readings, if a
        hatchling length is not positive, if a sexed count is negative or not
        finite, or if any per-nest computation rejects its inputs.
    """
    return total
```

### Step 9

09_emergence_effect

Goal
----
Step 9: orchestrator: the emergence effect on the expected male production.

```python
def emergence_effect(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
    Rho25_s: float = 100.0,
) -> float:
    """Change in the expected male hatchlings from the emergence convention.

    Parameters
    ----------
    records : sequence of array-like
        One record per nest. Each is a 2-D array with two columns, the times in
        minutes from laying and the temperature in degrees Celsius at each
        reading, with strictly increasing times and at least two readings. The
        record ends at emergence.
    hatchling_lengths : array-like
        One hatchling straight carapace length in mm per nest, positive.
    sexed_counts : array-like
        One number of sexed hatchlings per nest, non-negative.
    DHA, DHH, T12H, Rho25 : float, optional
        Parameters of the development rate curve of step 1.
    P, SL, SH : float, optional
        Parameters of the male fraction curve of step 7.
    DHA_s, DHH_s, T12H_s, shape1, shape2 : float, optional
        Parameters of the sexualization norm and the fitted beta density.
    start_size : float, optional
        Modelled size at laying in mm (0.347089).
    asymptotic_ratio : float, optional
        Ratio of the asymptote to the hatchling length (1.208968).
    frac_begin, frac_end : float, optional
        Stage fractions of the modelled size at each nest's end of incubation.
    Rho25_s : float, optional
        Rate of the sexualization norm at the reference temperature (100.0).

    Returns
    -------
    float
        The pipping sum minus the emergence sum of the expected male hatchlings
        over the nests. Pipping is derived per nest as the modelled crossing of
        that nest's hatchling length. Emergence is the last reading of that
        nest's record, including a final reading whose temperature repeats the
        one before. A nest whose modelled embryo never reaches its hatchling
        length has coincident ends and contributes zero, and it does not cancel
        the other nests. A window limit whose target lies below the modelled
        size at laying is the first reading time.

    Raises
    ------
    ValueError
        If the three per-nest inputs do not have the same length, if a record is
        not a 2-D array with two columns and at least two readings, if a hatchling
        length is not positive, if a sexed count is negative or not finite, or if
        any per-nest computation rejects its inputs.
    """
    return effect
```
