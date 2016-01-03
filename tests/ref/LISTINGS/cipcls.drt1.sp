
 program cipc      

 print the csf info for mrsdci wave functions

 written by: ron shepard

 version date: 06-jun-96

 This Version of Program cipc is Maintained by:
     Thomas Mueller
     Juelich Supercomputing Centre (JSC)
     Institute of Advanced Simulation (IAS)
     D-52425 Juelich, Germany 
     Email: th.mueller@fz-juelich.de



     ******************************************
     **    PROGRAM:              CIPC        **
     **    PROGRAM VERSION:      5.5         **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************


 workspace allocation parameters: lencor=  65536000 mem1=         0 ifirst=         1

 drt header information:
  cidrt_title DRT#1                                                              
 nmot  =    10 niot  =     2 nfct  =     0 nfvt  =     0
 nrow  =     8 nsym  =     1 ssym  =     1 lenbuf=  1600
 spnorb=     F spnodd=     F lxyzir(1:3)= 0 0 0
 nwalk,xbar:          5        3        2        0        0
 nvalwt,nvalw:        5        3        2        0        0
 ncsft:              19
 map(*)=     9 10  1  2  3  4  5  6  7  8
 mu(*)=      0  0
 syml(*) =   1  1
 rmo(*)=     1  2

 indx01:     5 indices saved in indxv(*)
 test nroots froot                      1                    -1
===================================ROOT # 1===================================

 rdhciv: CI vector file information:
 energy computed by program ciudg.       localhost.localdo 22:20:23.015 02-Jan-16

 lenrec =   32768 lenci =        19 ninfo =  6 nenrgy =  4 ntitle =  1

 Max. overlap with ref vector #        1
 Valid ci vector #        1
 Method:        0       99% overlap
 energy( 1)=  7.137540490910E-01, ietype=   -1,    core energy of type: Nuc.Rep.
 energy( 2)= -1.139858471882E+00, ietype=-1026,   total energy of type: MRSDCI  
 energy( 3)=  2.972357557988E-06, ietype=-2055, cnvginf energy of type: CI-Resid
 energy( 4)=  4.440892098501E-16, ietype=-2056, cnvginf energy of type: CI-D.E. 
==================================================================================
 test nroots froot                      2                    -1
===================================ROOT # 2===================================

 rdhciv: CI vector file information:
 energy computed by program ciudg.       localhost.localdo 22:20:23.016 02-Jan-16

 lenrec =   32768 lenci =        19 ninfo =  6 nenrgy =  5 ntitle =  1

 Max. overlap with ref vector #        2
 Valid ci vector #        2
 Method:        0       99% overlap
 energy( 1)=  7.137540490910E-01, ietype=   -1,    core energy of type: Nuc.Rep.
 energy( 2)= -6.457000374732E-01, ietype=-1026,   total energy of type: MRSDCI  
 energy( 3)=  4.788181807983E-05, ietype=-2055, cnvginf energy of type: CI-Resid
 energy( 4)=  1.347544904551E-07, ietype=-2056, cnvginf energy of type: CI-D.E. 
 energy( 5)=  8.593869637633E-10, ietype=-2057, cnvginf energy of type: CI-ApxDE
==================================================================================
space sufficient for valid walk range           1          5
               respectively csf range           1         19
space sufficient for valid walk range           1          5
               respectively csf range           1         19

 space is available for  21834380 coefficients.

 updated histogram parameters:
 csfmn = 0.0000E+00 csfmx = 1.0000E+00 fhist = 5.0000E-01 nhist =  20

 this program will print the csfs generated from
 the drt according to the following print options :

 1) run in batch mode: all valid roots are automatically
    analysed and csf info is printed by default contribution
    threshold 0.01 
 2) run in interactive mode
 3) generate files for cioverlap

 input menu number [  1]:
 --> input value for csfmin: 

 input CSF min value [ 0.0010]:
================================================================================
===================================VECTOR # 1===================================
================================================================================


 rdcivnew:       5 coefficients were selected.
 workspace: ncsfmx=      19
 ncsfmx=                    19

 histogram parameters:
 csfmn = 1.0000E-02 csfmx = 1.0000E+00 fhist = 5.0000E-01
 nhist =  20 icsfmn =       1 icsfmx =      19 ncsft =      19 ncsf =       5
 nhist =  20 fhist = 0.50000

    cmin                cmax        num  '*'=     1 csfs.
 ----------          ----------   ----- ---------|---------|---------|---------|
 5.0000E-01 <= |c| < 1.0000E+00       1 *
 2.5000E-01 <= |c| < 5.0000E-01       0
 1.2500E-01 <= |c| < 2.5000E-01       0
 6.2500E-02 <= |c| < 1.2500E-01       3 ***
 3.1250E-02 <= |c| < 6.2500E-02       1 *
 1.5625E-02 <= |c| < 3.1250E-02       0
 7.8125E-03 <= |c| < 1.5625E-02       0
 3.9062E-03 <= |c| < 7.8125E-03       0
 1.9531E-03 <= |c| < 3.9062E-03       0
 9.7656E-04 <= |c| < 1.9531E-03       0
 4.8828E-04 <= |c| < 9.7656E-04       0
 2.4414E-04 <= |c| < 4.8828E-04       0
 1.2207E-04 <= |c| < 2.4414E-04       0
 6.1035E-05 <= |c| < 1.2207E-04       0
 3.0518E-05 <= |c| < 6.1035E-05       0
 1.5259E-05 <= |c| < 3.0518E-05       0
 7.6294E-06 <= |c| < 1.5259E-05       0
 3.8147E-06 <= |c| < 7.6294E-06       0
 0.0000E+00 <= |c| < 3.8147E-06       0
                                  ----- ---------|---------|---------|---------|
                  total read =        5 total stored =       5

 from the selected csfs,
 min(|csfvec(:)|) = 3.1751E-02    max(|csfvec(:)|) = 9.9134E-01
 norm=   1.00000000000000     
 csfs will be printed based on coefficient magnitudes.

 current csfvec(*) selection parameters:
 csfmn = 1.0000E-02 csfmx = 1.0000E+00 fhist = 5.0000E-01
 nhist =  20 icsfmn =       1 icsfmx =      19 ncsft =      19 ncsf =       5

 i:slabel(i) =  1: a  
 
 internal level =    1    2
 syml(*)        =    1    1
 label          =  a    a  
 rmo(*)         =    1    2

 printing selected csfs in sorted order from cmin = 0.00000 to cmax = 1.00000

   indcsf     c     c**2   v  lab:rmo  lab:rmo   step(*)
  ------- -------- ------- - ---- --- ---- --- ------------
          1 -0.9913359207555 0.9827469077802 z*                    30
          4 -0.0876450032888 0.0076816466015 y           a  :  3  120
         13  0.0674719054125 0.0045524580200 y           a  :  4  102
          3  0.0627720862629 0.0039403348138 z*                    03
          8  0.0317505017541 0.0010080943616 y           a  :  7  120
            5 csfs were printed in this range.

================================================================================
===================================VECTOR # 2===================================
================================================================================


 rdcivnew:       4 coefficients were selected.
 workspace: ncsfmx=      19
 ncsfmx=                    19

 histogram parameters:
 csfmn = 1.0000E-02 csfmx = 1.0000E+00 fhist = 5.0000E-01
 nhist =  20 icsfmn =       1 icsfmx =      19 ncsft =      19 ncsf =       4
 nhist =  20 fhist = 0.50000

    cmin                cmax        num  '*'=     1 csfs.
 ----------          ----------   ----- ---------|---------|---------|---------|
 5.0000E-01 <= |c| < 1.0000E+00       1 *
 2.5000E-01 <= |c| < 5.0000E-01       0
 1.2500E-01 <= |c| < 2.5000E-01       0
 6.2500E-02 <= |c| < 1.2500E-01       1 *
 3.1250E-02 <= |c| < 6.2500E-02       1 *
 1.5625E-02 <= |c| < 3.1250E-02       1 *
 7.8125E-03 <= |c| < 1.5625E-02       0
 3.9062E-03 <= |c| < 7.8125E-03       0
 1.9531E-03 <= |c| < 3.9062E-03       0
 9.7656E-04 <= |c| < 1.9531E-03       0
 4.8828E-04 <= |c| < 9.7656E-04       0
 2.4414E-04 <= |c| < 4.8828E-04       0
 1.2207E-04 <= |c| < 2.4414E-04       0
 6.1035E-05 <= |c| < 1.2207E-04       0
 3.0518E-05 <= |c| < 6.1035E-05       0
 1.5259E-05 <= |c| < 3.0518E-05       0
 7.6294E-06 <= |c| < 1.5259E-05       0
 3.8147E-06 <= |c| < 7.6294E-06       0
 0.0000E+00 <= |c| < 3.8147E-06       0
                                  ----- ---------|---------|---------|---------|
                  total read =        4 total stored =       4

 from the selected csfs,
 min(|csfvec(:)|) = 2.1035E-02    max(|csfvec(:)|) = 9.9170E-01
 norm=   1.00000000000000     
 csfs will be printed based on coefficient magnitudes.

 current csfvec(*) selection parameters:
 csfmn = 1.0000E-02 csfmx = 1.0000E+00 fhist = 5.0000E-01
 nhist =  20 icsfmn =       1 icsfmx =      19 ncsft =      19 ncsf =       4

 i:slabel(i) =  1: a  
 
 internal level =    1    2
 syml(*)        =    1    1
 label          =  a    a  
 rmo(*)         =    1    2

 printing selected csfs in sorted order from cmin = 0.00000 to cmax = 1.00000

   indcsf     c     c**2   v  lab:rmo  lab:rmo   step(*)
  ------- -------- ------- - ---- --- ---- --- ------------
          2 -0.9917023686360 0.9834735879583 z*                    12
         12  0.1197300051064 0.0143352741228 y           a  :  3  102
         16 -0.0416949708691 0.0017384705958 y           a  :  7  102
          5  0.0210347522085 0.0004424608005 y           a  :  4  120
            4 csfs were printed in this range.
