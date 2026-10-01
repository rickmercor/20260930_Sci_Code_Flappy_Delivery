# Biology-Ecology-3

## Background

Roe deer are long-lived mammals that reproduce many times over their lives. The monitored population at Chizé in western France has been a reference case in demographic research for decades. Populations like this one are followed by capture and recapture. At each yearly census a fraction of the females present is caught, marked and released, and their later recaptures are recorded. Females first reproduce at age two, and survival stays high through the adult years before falling in the oldest classes. A projection matrix built from such a life table must combine reproduction and survival, which run on different timescales. The same life table yields different matrices depending on when the population is counted relative to the pulse, whether just before it, just after it, or somewhere in the months between. Misreading this timing is a common and costly error in the published literature.

A recent methodological study builds projection matrices from a life table without that ambiguity. It keeps survival and reproduction in two separate matrices and composes them in the arrangement the census timing requires. It then examines what the resulting matrices share and what they do not:

- the construction gives a projection matrix for the census design at hand, including censuses taken between birth pulses,
- the study proves that the growth rate is identical at every census timing, so it cannot separate a correct construction from a wrong one,
- the study works through the Chizé roe deer.

The task supplies constructed capture and recruitment records built on that monitoring design, and the rates are estimated from them by the rules stated in the task.

The analysis to reproduce is deterministic. One set of records, one census design, one reproductive-value vector.

## Problem

A projection matrix predicts the next year population from survival and fecundity. Births come in one short annual pulse, and survival runs through the whole year, so the matrix entries depend on when the population is counted relative to the pulse. A recent study builds these matrices from a life table, applies the method to a long-term roe deer monitoring programme, and compares matrices read at different census times.

Your task is to reproduce one instance of that analysis on constructed monitoring records. Estimate the age-specific survival and fecundity from the records below, using the rules stated below. Then compute the reproductive value of the class aged 8 years relative to the youngest class, and report it to four decimal places. Identify the study that matches the supplied configuration from the recent literature, and apply its procedure.

The records below were constructed for this task on the design of the study monitoring data. They hold the release records of a capture programme at the April censuses and the reproduction records of the May birth pulses.

**Release records.** One row per release group, in any order: the April census year, the cohort birth year, the number of marked females released at that census, the number of those females later found dead, and the number first recaptured at each of the following 6 censuses. The first of the 6 columns is the immediately following census. The programme paused in 1997 and 1998, so no census was conducted in those two years. The census years are the years present in the table plus the consecutive years after the last one. A row columns therefore follow the census sequence, not the calendar years. Survival between two censuses several years apart is the product of the stated yearly curve over each year in between, so a recapture after the pause includes the survival of every year in between. At each conducted census every female present is captured with the same probability, independently of the other females and of the year, and is released at once.

  [
    [1991, 1990, 138, 11, 68, 10, 0, 0, 0, 0],
    [1992, 1990, 91, 4, 55, 9, 2, 0, 0, 0],
    [1992, 1991, 117, 14, 53, 1, 0, 0, 0, 0],
    [1993, 1990, 61, 6, 39, 3, 1, 0, 0, 0],
    [1993, 1991, 73, 4, 45, 6, 0, 0, 0, 0],
    [1993, 1992, 165, 18, 89, 9, 0, 0, 0, 0],
    [1994, 1990, 53, 2, 34, 9, 0, 0, 0, 0],
    [1994, 1991, 55, 5, 36, 3, 1, 0, 0, 0],
    [1994, 1992, 109, 7, 67, 9, 1, 0, 0, 0],
    [1994, 1993, 135, 21, 61, 3, 0, 0, 0, 0],
    [1995, 1990, 44, 2, 33, 0, 1, 0, 0, 0],
    [1995, 1991, 45, 4, 28, 4, 0, 0, 0, 0],
    [1995, 1992, 83, 5, 56, 6, 2, 0, 0, 0],
    [1995, 1993, 77, 5, 45, 5, 0, 0, 0, 0],
    [1995, 1994, 111, 11, 54, 2, 0, 0, 0, 0],
    [1996, 1990, 45, 4, 24, 1, 0, 0, 0, 0],
    [1996, 1991, 45, 4, 26, 3, 0, 0, 0, 0],
    [1996, 1992, 71, 6, 36, 5, 0, 0, 0, 0],
    [1996, 1993, 60, 6, 31, 6, 0, 0, 0, 0],
    [1996, 1994, 64, 11, 31, 1, 0, 0, 0, 0],
    [1996, 1995, 187, 42, 60, 4, 0, 0, 0, 0],
    [1999, 1990, 39, 1, 29, 4, 0, 0, 0, 0],
    [1999, 1991, 35, 2, 28, 1, 0, 0, 0, 0],
    [1999, 1992, 70, 3, 53, 8, 0, 0, 0, 0],
    [1999, 1993, 41, 2, 26, 5, 2, 1, 1, 0],
    [1999, 1994, 50, 3, 32, 7, 0, 0, 0, 0],
    [1999, 1995, 90, 4, 68, 7, 2, 0, 0, 0],
    [1999, 1996, 142, 4, 104, 14, 1, 0, 0, 0],
    [1999, 1997, 161, 10, 89, 15, 3, 0, 0, 0],
    [1999, 1998, 132, 12, 67, 4, 0, 0, 0, 0],
    [2000, 1990, 38, 2, 30, 0, 0, 0, 0, 0],
    [2000, 1991, 32, 2, 20, 4, 0, 0, 0, 0],
    [2000, 1992, 57, 2, 41, 8, 0, 0, 0, 0],
    [2000, 1993, 39, 2, 29, 3, 0, 0, 0, 0],
    [2000, 1994, 46, 3, 28, 3, 1, 0, 0, 0],
    [2000, 1995, 78, 2, 54, 9, 1, 0, 0, 0],
    [2000, 1996, 119, 6, 75, 16, 2, 0, 0, 0],
    [2000, 1997, 119, 7, 85, 8, 2, 1, 0, 0],
    [2000, 1998, 64, 3, 44, 6, 0, 0, 0, 0],
    [2000, 1999, 167, 15, 80, 6, 1, 0, 0, 0],
    [2001, 1990, 34, 2, 22, 1, 0, 0, 0, 0],
    [2001, 1991, 35, 1, 28, 3, 0, 0, 0, 0],
    [2001, 1992, 45, 1, 35, 8, 0, 0, 0, 0],
    [2001, 1993, 37, 2, 26, 4, 0, 0, 0, 0],
    [2001, 1994, 42, 2, 32, 3, 1, 0, 0, 0],
    [2001, 1995, 86, 3, 66, 8, 0, 0, 0, 0],
    [2001, 1996, 108, 3, 80, 9, 2, 0, 0, 0],
    [2001, 1997, 93, 3, 69, 11, 2, 0, 0, 0],
    [2001, 1998, 56, 4, 37, 4, 2, 0, 0, 0],
    [2001, 1999, 95, 5, 65, 8, 0, 0, 0, 0],
    [2001, 2000, 117, 15, 58, 3, 1, 0, 0, 0],
    [2002, 1990, 22, 2, 13, 1, 0, 0, 0, 0],
    [2002, 1991, 28, 3, 20, 0, 0, 0, 0, 0],
    [2002, 1992, 43, 4, 27, 3, 1, 0, 0, 0],
    [2002, 1993, 34, 2, 24, 3, 0, 0, 0, 0],
    [2002, 1994, 38, 2, 29, 1, 1, 1, 0, 0],
    [2002, 1995, 74, 2, 53, 7, 1, 0, 0, 0],
    [2002, 1996, 84, 2, 62, 10, 1, 0, 0, 0],
    [2002, 1997, 85, 2, 59, 15, 0, 0, 0, 0],
    [2002, 1998, 47, 2, 32, 6, 0, 0, 0, 0],
    [2002, 1999, 77, 4, 51, 11, 0, 0, 0, 0],
    [2002, 2000, 76, 2, 54, 4, 2, 0, 0, 0],
    [2002, 2001, 139, 10, 68, 4, 1, 0, 0, 0],
    [2003, 1990, 13, 1, 7, 2, 0, 0, 0, 0],
    [2003, 1991, 23, 2, 9, 1, 0, 0, 0, 0],
    [2003, 1992, 37, 2, 25, 3, 0, 0, 0, 0],
    [2003, 1993, 32, 2, 18, 4, 0, 0, 0, 0],
    [2003, 1994, 31, 1, 27, 2, 0, 0, 0, 0],
    [2003, 1995, 60, 1, 45, 8, 1, 0, 0, 0],
    [2003, 1996, 67, 2, 54, 3, 0, 0, 0, 0],
    [2003, 1997, 71, 3, 49, 8, 5, 0, 0, 0],
    [2003, 1998, 43, 2, 25, 8, 0, 0, 0, 0],
    [2003, 1999, 61, 3, 38, 9, 0, 0, 0, 0],
    [2003, 2000, 59, 2, 43, 5, 0, 0, 0, 0],
    [2003, 2001, 73, 6, 43, 1, 1, 0, 0, 0],
    [2004, 1990, 8, 2, 3, 0, 0, 0, 0, 0],
    [2004, 1991, 17, 2, 6, 1, 1, 0, 0, 0],
    [2004, 1992, 25, 2, 16, 1, 0, 0, 0, 0],
    [2004, 1993, 25, 3, 12, 3, 0, 0, 0, 0],
    [2004, 1994, 29, 2, 17, 3, 1, 0, 0, 0],
    [2004, 1995, 62, 2, 45, 7, 1, 0, 0, 0],
    [2004, 1996, 74, 2, 59, 6, 2, 0, 0, 0],
    [2004, 1997, 66, 2, 45, 11, 1, 0, 0, 0],
    [2004, 1998, 37, 1, 30, 2, 1, 0, 0, 0],
    [2004, 1999, 51, 2, 37, 5, 1, 0, 0, 0],
    [2004, 2000, 48, 1, 40, 4, 0, 0, 0, 0],
    [2004, 2001, 51, 2, 39, 4, 0, 0, 0, 0],
    [2005, 1990, 5, 1, 2, 0, 0, 0, 0, 0],
    [2005, 1991, 10, 1, 4, 0, 0, 0, 0, 0],
    [2005, 1992, 14, 2, 7, 0, 0, 0, 0, 0],
    [2005, 1993, 17, 3, 8, 0, 0, 0, 0, 0],
    [2005, 1994, 26, 2, 16, 2, 0, 0, 0, 0],
    [2005, 1995, 64, 6, 36, 8, 0, 0, 0, 0],
    [2005, 1996, 67, 3, 51, 4, 1, 0, 0, 0],
    [2005, 1997, 62, 2, 45, 9, 0, 0, 0, 0],
    [2005, 1998, 38, 1, 28, 6, 1, 0, 0, 0],
    [2005, 1999, 44, 2, 27, 6, 1, 0, 0, 0],
    [2005, 2000, 41, 1, 32, 2, 1, 0, 0, 0],
    [2005, 2001, 42, 1, 30, 3, 1, 0, 0, 0],
    [2006, 1990, 2, 0, 0, 0, 0, 0, 0, 0],
    [2006, 1992, 8, 1, 4, 0, 0, 0, 0, 0],
    [2006, 1993, 8, 1, 4, 0, 0, 0, 0, 0],
    [2006, 1994, 18, 3, 4, 2, 0, 0, 0, 0],
    [2006, 1995, 44, 4, 24, 5, 0, 0, 0, 0],
    [2006, 1996, 58, 4, 43, 3, 0, 0, 0, 0],
    [2006, 1997, 59, 1, 43, 6, 1, 0, 0, 0],
    [2006, 1998, 34, 1, 23, 8, 0, 0, 0, 0],
    [2006, 1999, 46, 2, 34, 6, 0, 0, 0, 0],
    [2006, 2000, 38, 1, 27, 4, 2, 0, 0, 0],
    [2006, 2001, 27, 1, 23, 0, 0, 0, 0, 0],
    [2007, 1991, 2, 0, 1, 0, 0, 0, 0, 0],
    [2007, 1992, 3, 0, 2, 0, 0, 0, 0, 0],
    [2007, 1993, 6, 1, 3, 0, 0, 0, 0, 0],
    [2007, 1994, 13, 1, 7, 0, 0, 0, 0, 0],
    [2007, 1995, 28, 4, 10, 2, 0, 0, 0, 0],
    [2007, 1996, 40, 3, 30, 2, 0, 0, 0, 0],
    [2007, 1997, 45, 4, 23, 2, 1, 0, 0, 0],
    [2007, 1998, 31, 1, 22, 5, 0, 0, 0, 0],
    [2007, 1999, 38, 2, 24, 6, 1, 0, 0, 0],
    [2007, 2000, 37, 1, 29, 4, 0, 0, 0, 0],
    [2007, 2001, 32, 0, 27, 2, 0, 0, 0, 0],
    [2008, 1991, 3, 0, 1, 0, 0, 0, 0, 0],
    [2008, 1992, 3, 0, 3, 0, 0, 0, 0, 0],
    [2008, 1994, 10, 1, 3, 0, 0, 0, 0, 0],
    [2008, 1995, 24, 3, 9, 1, 0, 0, 0, 0],
    [2008, 1996, 30, 2, 19, 1, 0, 0, 0, 0],
    [2008, 1997, 41, 3, 27, 2, 0, 0, 0, 0],
    [2008, 1998, 26, 2, 15, 3, 1, 0, 0, 0],
    [2008, 1999, 35, 2, 24, 3, 1, 0, 0, 0],
    [2008, 2000, 29, 1, 24, 1, 0, 0, 0, 0],
    [2008, 2001, 26, 1, 19, 4, 0, 0, 0, 0],
    [2009, 1991, 2, 0, 1, 0, 0, 0, 0, 0],
    [2009, 1992, 2, 0, 1, 0, 0, 0, 0, 0],
    [2009, 1993, 2, 1, 0, 0, 0, 0, 0, 0],
    [2009, 1994, 4, 1, 2, 0, 0, 0, 0, 0],
    [2009, 1995, 15, 3, 5, 0, 0, 0, 0, 0],
    [2009, 1996, 24, 3, 12, 0, 0, 0, 0, 0],
    [2009, 1997, 34, 3, 17, 1, 0, 0, 0, 0],
    [2009, 1998, 21, 3, 12, 1, 0, 0, 0, 0],
    [2009, 1999, 33, 2, 22, 3, 0, 0, 0, 0],
    [2009, 2000, 31, 1, 23, 5, 1, 0, 0, 0],
    [2009, 2001, 20, 1, 16, 1, 0, 0, 0, 0],
    [2010, 1992, 2, 0, 1, 1, 0, 0, 0, 0],
    [2010, 1993, 2, 0, 1, 0, 0, 0, 0, 0],
    [2010, 1994, 2, 0, 1, 0, 0, 0, 0, 0],
    [2010, 1995, 7, 1, 5, 0, 0, 0, 0, 0],
    [2010, 1996, 15, 2, 6, 1, 0, 0, 0, 0],
    [2010, 1997, 18, 1, 11, 0, 0, 0, 0, 0],
    [2010, 1998, 22, 3, 11, 0, 0, 0, 0, 0],
    [2010, 1999, 25, 3, 13, 0, 1, 0, 0, 0],
    [2010, 2000, 25, 2, 13, 4, 0, 0, 0, 0],
    [2010, 2001, 18, 1, 13, 3, 0, 0, 0, 0],
  ]

**Reproduction records.** One row per birth cohort and May birth pulse, in the order: the pulse year, the cohort birth year, the females present at the pulse, the females that gave birth, and the fawns born. The records cover maternal ages of two years and older. The female share of the fawns is one half.

  [
    [1992, 1990, 112, 61, 93],
    [1993, 1990, 83, 59, 104],
    [1994, 1990, 74, 58, 108],
    [1995, 1990, 69, 59, 114],
    [1996, 1990, 63, 53, 102],
    [1998, 1990, 54, 44, 78],
    [1999, 1990, 50, 40, 71],
    [2000, 1990, 44, 36, 66],
    [2001, 1990, 39, 28, 49],
    [2002, 1990, 22, 13, 23],
    [2003, 1990, 18, 10, 17],
    [2004, 1990, 17, 9, 15],
    [2005, 1990, 10, 5, 9],
    [2006, 1990, 5, 2, 3],
    [1993, 1991, 92, 50, 77],
    [1994, 1991, 66, 47, 81],
    [1995, 1991, 59, 48, 87],
    [1996, 1991, 54, 46, 88],
    [1998, 1991, 47, 39, 74],
    [1999, 1991, 43, 35, 64],
    [2000, 1991, 41, 33, 58],
    [2001, 1991, 36, 29, 51],
    [2002, 1991, 19, 14, 25],
    [2003, 1991, 18, 11, 20],
    [2004, 1991, 15, 8, 14],
    [2005, 1991, 13, 6, 10],
    [2006, 1991, 8, 4, 7],
    [1994, 1992, 131, 72, 113],
    [1995, 1992, 97, 70, 124],
    [1996, 1992, 85, 67, 124],
    [1998, 1992, 73, 62, 116],
    [1999, 1992, 68, 57, 109],
    [2000, 1992, 63, 49, 89],
    [2001, 1992, 59, 46, 84],
    [2002, 1992, 32, 26, 47],
    [2003, 1992, 28, 20, 35],
    [2004, 1992, 26, 16, 29],
    [2005, 1992, 22, 12, 21],
    [2006, 1992, 20, 10, 17],
    [2008, 1992, 6, 3, 5],
    [1995, 1993, 102, 55, 84],
    [1996, 1993, 72, 51, 88],
    [1998, 1993, 59, 50, 95],
    [1999, 1993, 53, 46, 88],
    [2000, 1993, 50, 42, 81],
    [2001, 1993, 45, 36, 64],
    [2002, 1993, 27, 22, 40],
    [2003, 1993, 25, 20, 37],
    [2004, 1993, 21, 15, 27],
    [2005, 1993, 20, 12, 21],
    [2006, 1993, 16, 9, 16],
    [2008, 1993, 9, 4, 7],
    [2009, 1993, 5, 2, 3],
    [1996, 1994, 85, 46, 72],
    [1998, 1994, 58, 47, 87],
    [1999, 1994, 50, 42, 81],
    [2000, 1994, 46, 38, 72],
    [2001, 1994, 45, 38, 71],
    [2002, 1994, 25, 20, 36],
    [2003, 1994, 24, 19, 34],
    [2004, 1994, 22, 18, 32],
    [2005, 1994, 18, 13, 23],
    [2006, 1994, 16, 9, 16],
    [2008, 1994, 12, 6, 10],
    [2009, 1994, 7, 3, 5],
    [2010, 1994, 4, 2, 3],
    [1998, 1995, 101, 72, 125],
    [1999, 1995, 93, 75, 137],
    [2000, 1995, 81, 70, 132],
    [2001, 1995, 75, 65, 124],
    [2002, 1995, 44, 37, 70],
    [2003, 1995, 41, 33, 60],
    [2004, 1995, 38, 31, 55],
    [2005, 1995, 34, 27, 48],
    [2006, 1995, 31, 22, 40],
    [2008, 1995, 22, 12, 21],
    [2009, 1995, 20, 10, 17],
    [2010, 1995, 12, 5, 9],
    [2011, 1995, 6, 2, 3],
    [1998, 1996, 105, 58, 89],
    [1999, 1996, 79, 56, 99],
    [2000, 1996, 71, 57, 108],
    [2001, 1996, 66, 56, 107],
    [2002, 1996, 39, 33, 64],
    [2003, 1996, 36, 30, 57],
    [2004, 1996, 33, 27, 49],
    [2005, 1996, 30, 24, 43],
    [2006, 1996, 27, 22, 39],
    [2008, 1996, 21, 13, 23],
    [2009, 1996, 18, 10, 18],
    [2010, 1996, 15, 7, 12],
    [2011, 1996, 10, 5, 9],
    [2012, 1996, 5, 2, 3],
    [1999, 1997, 120, 66, 102],
    [2000, 1997, 90, 65, 113],
    [2001, 1997, 79, 64, 120],
    [2002, 1997, 43, 37, 71],
    [2003, 1997, 40, 34, 64],
    [2004, 1997, 37, 32, 61],
    [2005, 1997, 35, 27, 48],
    [2006, 1997, 32, 26, 47],
    [2008, 1997, 25, 18, 32],
    [2009, 1997, 23, 14, 25],
    [2010, 1997, 20, 11, 19],
    [2011, 1997, 17, 8, 14],
    [2012, 1997, 10, 5, 8],
    [2013, 1997, 5, 2, 3],
    [2000, 1998, 95, 53, 84],
    [2001, 1998, 71, 51, 90],
    [2002, 1998, 40, 33, 61],
    [2003, 1998, 36, 30, 57],
    [2004, 1998, 34, 28, 53],
    [2005, 1998, 31, 26, 50],
    [2006, 1998, 30, 24, 43],
    [2008, 1998, 25, 20, 36],
    [2009, 1998, 22, 16, 29],
    [2010, 1998, 19, 12, 22],
    [2011, 1998, 17, 9, 16],
    [2012, 1998, 15, 7, 12],
    [2013, 1998, 9, 4, 7],
    [2014, 1998, 5, 2, 3],
    [2001, 1999, 126, 69, 106],
    [2002, 1999, 57, 41, 70],
    [2003, 1999, 49, 39, 73],
    [2004, 1999, 46, 38, 72],
    [2005, 1999, 43, 37, 70],
    [2006, 1999, 39, 34, 64],
    [2008, 1999, 33, 27, 48],
    [2009, 1999, 30, 24, 43],
    [2010, 1999, 27, 20, 36],
    [2011, 1999, 24, 14, 25],
    [2012, 1999, 20, 11, 19],
    [2013, 1999, 18, 9, 15],
    [2014, 1999, 11, 5, 8],
    [2015, 1999, 6, 2, 3],
    [2002, 2000, 59, 32, 49],
    [2003, 2000, 42, 31, 55],
    [2004, 2000, 39, 32, 58],
    [2005, 2000, 35, 30, 56],
    [2006, 2000, 33, 28, 53],
    [2008, 2000, 28, 22, 39],
    [2009, 2000, 24, 19, 34],
    [2010, 2000, 23, 19, 34],
    [2011, 2000, 20, 14, 25],
    [2012, 2000, 17, 10, 18],
    [2013, 2000, 16, 9, 16],
    [2014, 2000, 14, 7, 12],
    [2015, 2000, 8, 4, 7],
    [2016, 2000, 4, 2, 3],
    [2003, 2001, 110, 61, 95],
    [2004, 2001, 83, 59, 103],
    [2005, 2001, 74, 58, 108],
    [2006, 2001, 69, 58, 111],
    [2008, 2001, 59, 50, 96],
    [2009, 2001, 54, 42, 76],
    [2010, 2001, 50, 40, 72],
    [2011, 2001, 45, 35, 63],
    [2012, 2001, 39, 29, 52],
    [2013, 2001, 34, 20, 35],
    [2014, 2001, 30, 17, 30],
    [2015, 2001, 27, 14, 24],
    [2016, 2001, 16, 7, 12],
  ]

Estimating the rates from the records:

* A female class is her age in completed years. For the release records this is her age at the April census, and for the reproduction records her age at the May pulse. Ages 13 and older are pooled into one terminal class.
* Three rules turn the printed release records into the data the fit uses, in this order.
  * Confirmation: the most recent census at which a release group was captured is not confirmed by a later sighting. That census is the last recapture column of the group holding a positive count. Those females are treated as never recaptured, so their count moves into the group not-recaptured total.
  * Validity: a release group with no confirmed capture does not enter the fit.
  * Recovery removal: a female found dead is not at risk after her death, so a group at-risk total is its released count minus the females found dead.
* Two survival curves are stated in the class age a (in years). The reported rates are those of the model under which the usable records are more likely:
  - late: s(a) = (s0 + (sp - s0) * (1 - exp(-k * a))) * exp(-d * max(0, a - 8))
  - logistic: s(a) = (s0 + (sp - s0) * (1 - exp(-k * a))) / (1 + exp((a - 11) / w))
  Here s0 is the newborn survival, sp the adult plateau, k the juvenile approach rate, and d or w the decline parameter. Both models take the curve value at age 13 for the terminal class.
* Survival estimate: the four parameters of each model and the common capture probability are estimated jointly by maximum likelihood, once per model. The two maximized likelihoods are compared, and the reported rates are those of the model with the larger likelihood. Each fit must be converged so that every estimate is stable to at least four decimal places, and the reported survival probabilities are then rounded to three decimal places.
* Fecundity estimate, for each class: the pooled ratio of female fawns (fawns born times the female share) to females present over that class records, keyed by the mother age at the pulse, rounded to three decimal places. Females younger than two years contribute zero.

Configuration:

* Females only, as in the published study. The records were constructed for this task.
* The population is counted each year 11 months after the May birth pulse, in the following April.

Numerical conventions:

* The result is a single positive decimal number, reported to four decimal places.
* The construction and every convention that turns the estimated rates into the projection matrix follow the matching published study. This covers how survival and reproduction are arranged around the birth pulse, the class structure of the matrix, and the treatment of the pooled terminal class. The construction must use the reported vectors at three decimal places.
* In your reasoning, report the estimated survival of the newborn class and of the pooled terminal class, the estimated fecundities of the age-2 class and the terminal class, the growth rate of the projection matrix, the number of age classes it contains, the reproductive values of the classes aged 5, 7 and 8 years relative to the youngest class (which has value 1), and the recruitment (first-row) terms for females aged 12 years and for the terminal 13+ class.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> clear and precise. Show only the few scalars that determine the final number.
Do not paste the full projection matrix, intermediate matrices, or per-class candidate tables.

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

01_estimate_rates

Goal
----
Step 1: estimate the age-specific rates from the monitoring records.

```python
def estimate_rates(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    terminal_age: int = 13,
) -> tuple:
    """Estimate the age-specific survival and fecundity from the monitoring records.

    Parameters
    ----------
    release_records : array-like of shape (n_records, 4 + K)
        Integer records of the April census programme, one row per release group:
        ``[release year, birth year, females released, females later found dead,
        first recaptures at the 1st, 2nd, ..., K-th following census]``. A female
        released at an April census is counted in the release year. A recapture at the
        j-th following census means the female was seen again for the first time at
        that census, and a female not listed there counts as not yet recaptured. ``K``
        is the same for every row. The census paused in 1997 and 1998, so the rows'
        release years skip those two years. The j-th following census is the j-th
        census that was run after the release, not the j-th calendar year. The census
        years are the years present in the table plus the consecutive years after its
        last release, and a missing year inside the table's span is a paused census.
        At each census every female present is captured with the same probability,
        independently of the other females and of the year, and is released at once.
        Three rules turn these printed counts into the data the fit uses, in this
        order. Confirmation: the most recent census at which a group was captured is
        not confirmed by a later sighting, so the last recapture column of the group
        holding a positive count is emptied and those females move into the group's
        not-recaptured total. Validity: a group left with no confirmed capture does
        not enter the fit. Recovery removal: a female found dead is not at risk after
        her death, so a group's at-risk total is its released count minus its females
        found dead.
    reproduction_records : array-like of shape (n_records, 5)
        Integer records, one row per cohort and May birth pulse: ``[pulse year, birth
        year, females present at the pulse, females that gave birth, fawns born]``.
        The records cover maternal ages of two years and older (females first
        reproduce at age two), and the female share of the fawns is one half.
    terminal_age : int, optional
        Age class at and above which classes are pooled (the records pool ages 13 and
        older). A female's class is her age in completed years: at the April census
        for the release records, and at the May pulse for the reproduction records.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        ``survival`` (length ``terminal_age + 1``): for each class, the survival
        probability given by the better fitting of two stated curves in the class age
        ``a`` in years,

            late:     s(a) = (s0 + (sp - s0) * (1 - exp(-k * a))) * exp(-d * max(0, a - 8))
            logistic: s(a) = (s0 + (sp - s0) * (1 - exp(-k * a))) / (1 + exp((a - 11) / w))

        where ``s0`` is the newborn survival, ``sp`` the adult plateau, ``k`` the
        juvenile approach rate, and ``d`` or ``w`` the decline parameter. Both curves
        take their value at age 13 for the pooled terminal class, which keeps the same
        females in place. The four parameters of each curve and the common capture
        probability are estimated jointly by maximum likelihood, once per curve, and
        the reported rates are those of the curve whose maximized likelihood is larger.
        Each fit must be converged, not merely started, with every estimate stable to
        at least four decimal places. The rates are rounded to three decimal places.
        ``fecundity`` (length ``terminal_age + 1``): for each class, the pooled ratio
        of female fawns (fawns born times one half) to females present over that
        class's records, keyed by the mother's age at the pulse, rounded to three
        decimal places. Classes zero and one contribute zero.

    Raises
    ------
    ValueError
        If the release table is not a non-empty 2-D integer array. The table must have
        non-negative entries, at least six columns, whole-integer years with the
        release year after the birth year, and no more first recaptures in a row than
        females released. ``terminal_age`` must be at least 2. The rules must not
        leave a group with fewer females at risk than its confirmed captures. Every
        survival class from zero through ``terminal_age`` must appear among the usable
        release groups. The reproduction records must be a 2-D integer array of shape
        ``(n, 5)`` with non-negative entries and no more females giving birth than
        were present. Every fecundity class from two through ``terminal_age`` must
        have records.
    """
    return result
```

### Step 2

02_transition_matrix

Goal
----
Step 2: build the transition matrix.

```python
def transition_matrix(
    survival: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Build the age-structured transition matrix.

    Parameters
    ----------
    survival : sequence of float
        Annual survival probabilities in age order, one per class, as estimated by
        step 01. The last entry is the pooled terminal class's survival.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` transition matrix: entry ``[a+1, a] = survival[a]`` for every
        ``a < n-1``, and ``[n-1, n-1] = survival[n-1]`` for the pooled terminal class.
        All other entries are zero.

    Raises
    ------
    ValueError
        If the input is not a finite 1-D sequence of at least two probabilities in
        [0, 1].
    """
    return result
```

### Step 3

03_birth_pulse_operator

Goal
----
Step 3: build the birth-pulse operator.

```python
def birth_pulse_operator(
    fecundity: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Build the birth-pulse operator ``(I + R)``.

    Parameters
    ----------
    fecundity : sequence of float
        Annual female fecundities in age order, one per class, as estimated by step 01.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` birth-pulse matrix: the identity plus the raw fecundity matrix,
        i.e. ones on the diagonal and the fecundities in the first row.

    Raises
    ------
    ValueError
        If the input is not a finite 1-D sequence of at least two non-negative
        fecundities.
    """
    return result
```

### Step 4

04_survival_ageing_decomposition

Goal
----
Step 4: decompose the annual transition into survival and ageing.

```python
def survival_ageing_decomposition(
    transition: "numpy.typing.ArrayLike",
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Split the annual transition matrix into survival and ageing operators.

    Parameters
    ----------
    transition : array-like of shape (n, n)
        The age-structured transition matrix of step 02: the annual survival
        probability of class a at ``transition[a+1, a]`` for every ``a < n-1``, the
        pooled terminal class's survival at ``transition[n-1, n-1]``, and zeros
        everywhere else.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S, T)``: the diagonal survival operator, with ``S[a, a] = transition[a+1, a]``
        for every ``a < n-1`` and ``S[n-1, n-1] = transition[n-1, n-1]``; and the ageing
        operator ``T``, with ``T[a+1, a] = 1`` for every ``a < n-1`` and
        ``T[n-1, n-1] = 1`` so that the pooled terminal class keeps its position. The
        product ``T S`` equals the input matrix.

    Raises
    ------
    ValueError
        If the input is not a square matrix with at least two classes, if an entry is
        not finite or lies outside [0, 1], or if any entry other than the first
        sub-diagonal and the last diagonal entry is non-zero.
    """
    return result
```

### Step 5

05_fractional_survival_powers

Goal
----
Step 5: fractional powers of the survival operator.

```python
def fractional_survival_powers(
    survival: "numpy.typing.ArrayLike",
    months: float,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Fractional powers of the annual survival operator.

    Parameters
    ----------
    survival : sequence of float
        Annual survival probabilities in age order, one per class (see step 01).
    months : float
        Months elapsed since the birth pulse at the census, from 0 to 12 inclusive.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S_a, S_b)``: the diagonal operators ``S^(months/12)`` and
        ``S^(1 - months/12)``, whose product is the annual survival operator ``S``.

    Raises
    ------
    ValueError
        If the input is not a finite 1-D sequence of at least two probabilities in
        [0, 1], or if `months` is not finite or lies outside [0, 12].
    """
    return result
```

### Step 6

06_intermediate_projection

Goal
----
Step 6: assemble the projection matrix for an intermediate census.

```python
def intermediate_projection(
    survival: "numpy.typing.ArrayLike",
    fecundity: "numpy.typing.ArrayLike",
    months: float = 11,
) -> "numpy.ndarray":
    """Projection matrix for a census taken `months` after the birth pulse.

    Parameters
    ----------
    survival : sequence of float
        Annual survival probabilities in age order, one per class (see step 01).
    fecundity : sequence of float
        Annual female fecundities in age order, one per class (see step 01).
    months : float, optional
        Months elapsed since the birth pulse at the census, from 0 to 12 inclusive.
        The task's graded census uses the default, 11 months.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` matrix ``S^(M/12) (I + R) T S^(1 - M/12)``: survival over the
        elapsed months, then the birth pulse and the transition, then survival over the
        remaining months. At M = 0 it equals the post-breeding matrix ``(I + R) U``.

    Raises
    ------
    ValueError
        If the two inputs do not share a length of at least two classes, or if either
        fails the validation of steps 02, 03 and 05.
    """
    return result
```

### Step 7

07_dominant_eigenpair

Goal
----
Step 7: the growth rate and the stable age distribution.

```python
def dominant_eigenpair(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Dominant eigenvalue and stable age distribution of a projection matrix.

    Parameters
    ----------
    matrix : array-like of shape (n, n)
        A projection matrix over the age classes (see step 06) with
        non-negative entries.

    Returns
    -------
    numpy.ndarray
        One array of length ``n + 1``: the dominant eigenvalue first, then the
        corresponding right eigenvector normalized to sum to one, holding the stable
        proportion of each class.

    Raises
    ------
    ValueError
        If the input is not a finite, non-negative square matrix with at least two
        classes, if the dominant eigenvalue is not real and positive, or if the
        stable vector cannot be normalized to a non-negative distribution.
    """
    return result
```

### Step 8

08_reproductive_values

Goal
----
Step 8: reproductive values: the left eigenvector of the projection matrix.

```python
def reproductive_values(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reproductive values of a projection matrix, anchored to the youngest class.

    Parameters
    ----------
    matrix : array-like of shape (n, n)
        A projection matrix over the age classes (see step 06), with non-negative
        entries.

    Returns
    -------
    numpy.ndarray
        The dominant left eigenvector, normalized so that its first entry (the
        youngest class) is 1: ``rv[0] = 1`` and ``rv[i]`` is the reproductive value
        of class ``i`` relative to the youngest class.

    Raises
    ------
    ValueError
        If the input is not a finite, non-negative square matrix with at least two
        classes, if the dominant eigenvalue is not real and positive, or if the
        vector cannot be anchored because its first entry vanishes.
    """
    return result
```

### Step 9

09_relative_reproductive_value

Goal
----
Step 9: orchestrator: the relative reproductive value of one class.

```python
def relative_reproductive_value(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    age: int = 8,
    months: float = 11,
) -> float:
    """Reproductive value of one age class, relative to the youngest class.

    Parameters
    ----------
    release_records : array-like of shape (n_records, 4 + K)
        Release records of step 01: one row per release group of the April census
        programme, ``[release year, birth year, females released, females later found
        dead, first recaptures at the 1st, 2nd, ..., K-th following census]``.
    reproduction_records : array-like of shape (n_records, 5)
        Reproduction records of step 01: one row per cohort and May birth pulse,
        ``[pulse year, birth year, females present at the pulse, females that gave
        birth, fawns born]``. The female share of the fawns is one half.
    age : int, optional
        Age class whose value is returned, from 0 to n-1 (the last class is the pooled
        terminal class). The graded endpoint uses the default, age 8.
    months : float, optional
        Months elapsed since the birth pulse at the census, from 0 to 12 inclusive.
        The task's graded census uses the default, 11 months.

    Returns
    -------
    float
        The reproductive value of that class relative to the youngest class (class 0,
        which has value 1), for a census taken `months` after the birth pulse.

    Raises
    ------
    ValueError
        If the release records or the reproduction records fail the validation of
        step 01, if a later pipeline step rejects its inputs (including `months`
        outside 0 to 12), or if `age` is outside 0 to n-1.
    """
    return result
```
