1
 program ciudg      
 multireference single and double excitation configuration
 interaction based on the graphical unitary group approach.


 references:  h. lischka, r. shepard, f. b. brown, and i. shavitt,
                  int. j. quantum chem. s 15, 91 (1981).
              r. shepard, r. a. bair, r. a. eades, a. f. wagner,
                  m. j. davis, l. b. harding, and t. h. dunning,
                  int j. quantum chem. s 17, 613 (1983).
              r. ahlrichs, h.-j. boehm, c. ehrhardt, p. scharf,
                  h. schiffer, h. lischka, and m. schindler,
                  j. comp. chem. 6, 200 (1985).
              r. shepard, i. shavitt, r. m. pitzer, d. c. comeau, m. pepper
                  h. lischka, p. g. szalay, r. ahlrichs, f. b. brown, and
                  j.-g. zhao, int. j. quantum chem. symp. 22, 149 (1988).

 This Version of Program CIUDG is Maintained by:
     Thomas Mueller
     Juelich Supercomputing Centre (JSC)
     Institute of Advanced Simulation (IAS)
     D-52425 Juelich, Germany 
     Email: th.mueller@fz-juelich.de



     ******************************************
     **    PROGRAM:              CIUDG       **
     **    PROGRAM VERSION:      2009-03.    **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************


================ Computing sorting integral file structure ================

                    -----z----- -----y----- -----x----- -----w----- ---total---

                CSFs         3          16           0           0          19
      internal walks         3           2           0           0           5
valid internal walks         3           2           0           0           5
  MR-CIS calculation - skip 3- and 4-external 2e integrals
 lcore1,lcore2=              65500342              55695944
 lencor,maxblo              65536000                240000
========================================
 current settings:
 minbl3         105
 minbl4         105
 locmaxbl3      564
 locmaxbuf      282
 maxbl3      240000
 maxbl3      240000
 maxbl4      240000
 maxbuf     4800000
========================================
 Orig.  diagonal integrals:  1electron:        10
                             0ext.    :         6
                             2ext.    :        32
                             4ext.    :        72


 Orig. off-diag. integrals:  4ext.    :       882
                             3ext.    :       672
                             2ext.    :       324
                             1ext.    :        96
                             0ext.    :         6
                             2ext. SO :         0
                             1ext. SO :         0
                             0ext. SO :         0
                             1electron:        36


 Sorted integrals            3ext.  w :         0 x :         0
                             4ext.  w :         0 x :         0


Cycle #  1 sortfile size=     98301(       3 records of    32767) #buckets=   2
Cycle #  2 sortfile size=     98301(       3 records of    32767) #buckets=   2

 intermediate da file sorting parameters:
 nbuk=   2 lendar=   32767 nipbk=   21844 nipsg=  55536903
 minimum size of srtscr:     32767 WP (     1 records)
 maximum size of srtscr:     98301 WP (     3 records)
 compressed index vector length=                     1
 echo of the input for program ciudg:
 ------------------------------------------------------------------------
  &input
  NTYPE = 0,
  GSET = 0,
   DAVCOR =10,
  NCOREL = 2
  NROOT = 2
  IVMODE = 3
  NBKITR = 1
  NVBKMN = 2
  RTOLBK = 1e-4,1e-4,
  NITER = 60
  NVCIMN = 4
  RTOLCI = 1e-4,1e-4,
  NVCIMX = 7
  NVRFMX = 7
  NVBKMX = 7
   iden=2
  CSFPRN = 10,
 /&end
 transition
   1  1  1  2
 ------------------------------------------------------------------------
transition densities: resetting nroot to    2
lodens (list->root)=  1  2
invlodens (root->list)=  1  2
 USING SEGMENTS OF EQUAL SIZE

****************  list of control variables  ****************
 lvlprt =    0      nroot  =    2      noldv  =   0      noldhv =   0
 nunitv =    2      nbkitr =    1      niter  =  60      davcor =  10
 csfprn =   10      ivmode =    3      istrt  =   0      vout   =   0
 iortls =    0      nvbkmx =    7      ibktv  =  -1      ibkthv =  -1
 nvcimx =    7      icitv  =   -1      icithv =  -1      frcsub =   0
 nvbkmn =    2      nvcimn =    4      maxseg =   4      nrfitr =  30
 ncorel =    2      nvrfmx =    7      nvrfmn =   4      iden   =   2
 itran  =    0      froot  =    0      rtmode =   0      ncouple=   1
 skipso =    F      dalton2=    0      molcas =   0      finalv =   0
 finalw =    0      cosmocalc=   0    with_tsklst=   0
 nsegwx =    1     1     1     1
 nseg0x =    1     1     1     1
 nseg1x =    1     1     1     1
 nseg2x =    1     1     1     1
 nseg3x =    1     1     1     1
 nseg4x =    1     1     1     1
 no0ex  =      0    no1ex  =      0    no2ex  =     0    no3ex  =     0
 no4ex  =      0    nodiag =      0
 cdg4ex =    1      c3ex1ex=    1      c2ex0ex=   1
 fileloc=    0     0     0     0     0     0     0     1     1     1
 directhd=   1      noaqccshift_zyxw=      0
 critical_crit=-1.00000    critical_delta= 0.05000

 ctol   = 0.010000    lrtshift=1.000000    smalld =0.001000


 convergence tolerances of bk and full diagonalization steps
 root #       rtolbk        rtol
 ------      --------      ------
    1        1.000E-04    1.000E-04
    2        1.000E-04    1.000E-04
 Computing density:                    .drt1.state1
 Computing density:                    .drt1.state2
 Computing transition density:          drt1.state1-> drt1.state2 (.trd1to2)
 using                      1  nodes and                      1  cores.
 szdg/szodg per processor=                  3064                  3064
 Main memory management:
 global                1 DP per process
 vdisk                 0 DP per process
 stack                 0 DP per process
 core           65535999 DP per process

********** Integral sort section *************


 workspace allocation information: lencor=  65535999

 echo of the input for program cisrt:
 ------------------------------------------------------------------------
  &input
  maxbl3=240000
  maxbl4=240000
  maxbuf=4800000
  &end
 ------------------------------------------------------------------------
 
 ( 6) listing file:                    ciudgls             
 ( 5) input file:                      cisrtin   
 (17) cidrt file:                      cidrtfl             
 (11) transformed integrals file:      moints    
 (12) diagonal integral file:          diagint             
 (13) off-diagonal integral file:      ofdgint             
 (31) 4-external w integrals file:     fil4w               
 (32) 4-external x integrals file:     fil4x               
 (33) 3-external w integrals file:     fil3w               
 (34) 3-external x integrals file:     fil3x               
 (21) scratch da sorting file:         srtscr              
 (12) 2-e integral file [fsplit=2]:    moints2   

 input integral file header information:
 Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16
  cidrt_title DRT#1                                                              
 MO-coefficients from mcscf.x                                                    
  total ao core energy =    0.713754049                                          
 SIFS file created by program tran.      localhost.localdo 22:20:22.972 02-Jan-16

 input energy(*) values:
 energy( 1)=  7.137540490910E-01, ietype=   -1,    core energy of type: Nuc.Rep.

 total core energy =   7.137540490910E-01

 nsym = 1 nmot=  10

 symmetry  =    1
 slabel(*) =  A  
 nmpsy(*)  =   10

 info(*) =          1      8192      6552      8192      5460         0

 orbital labels, i:molab(i)=
   1:tout:001   2:tout:002   3:tout:003   4:tout:004   5:tout:005   6:tout:006   7:tout:007   8:tout:008   9:tout:009  10:tout:010

 input parameters:
 prnopt=  0
 ldamin=    4095 ldamax=   32767 ldainc=      64
 maxbuf= 4800000 maxbl3=  240000 maxbl4=  240000 intmxo=     766
  Using 32 bit compression 

 drt information:
  cidrt_title DRT#1                                                              
 nmotd =  10 nfctd =   0 nfvtc =   0 nmot  =  10
 nlevel =  10 niot  =   2 lowinl=   9
 orbital-to-level map(*)
    9  10   1   2   3   4   5   6   7   8
 compressed map(*)
    9  10   1   2   3   4   5   6   7   8
 levsym(*)
    1   1   1   1   1   1   1   1   1   1
 repartitioning mu(*)=
   0.  0.
  ... transition density matrix calculation requested 
  ... setting rmu(*) to zero 

 indxdg: diagonal integral statistics.
 total number of integrals contributing to diagonal matrix elements:       110
 number with all external indices:        72
 number with half external - half internal indices:        32
 number with all internal indices:         6

 indxof: off-diagonal integral statistics.
    4-external integrals: num=        882 strt=          1
    3-external integrals: num=        672 strt=        883
    2-external integrals: num=        324 strt=       1555
    1-external integrals: num=         96 strt=       1879
    0-external integrals: num=          6 strt=       1975

 total number of off-diagonal integrals:        1980


 indxof(2nd)  ittp=   3 numx(ittp)=         324
 indxof(2nd)  ittp=   4 numx(ittp)=          96
 indxof(2nd)  ittp=   5 numx(ittp)=           6

 intermediate da file sorting parameters:
 nbuk=   2 lendar=   32767 nipbk=   21844 nipsg=  55641330
 pro2e        1      56     111     166     221     224     227     282   43970   87658
   120425  128617  134077  155916

 pro2e:       354 integrals read in     1 records.
 pro1e        1      56     111     166     221     224     227     282   43970   87658
   120425  128617  134077  155916
 pro1e: eref =    0.000000000000000E+00
 total size of srtscr:                     3  records of                  32767 
 WP =                786408 Bytes
 putdg        1      56     111     166     932   33699   55544     282   43970   87658
   120425  128617  134077  155916

 putf:       4 buffers of length     766 written to file 12
 diagonal integral file completed.

 putf:       4 buffers of length     766 written to file 13
 off-diagonal files sort completed.
 executing brd_struct for cisrtinfo
cisrtinfo:
bufszi   766
 diagfile 4ext:      72 2ext:      32 0ext:       6
 fil4w,fil4x  :     882 fil3w,fil3x :     672
 ofdgint  2ext:     324 1ext:      96 0ext:       6so0ext:       0so1ext:       0so2ext:       0
buffer minbl4     105 minbl3     105 maxbl2     108nbas:   8   0   0   0   0   0   0   0 maxbuf******
 CIUDG version 5.9.7 ( 5-Oct-2004)

 workspace allocation information: lcore=  65535999

 core energy values from the integral file:
 energy( 1)=  7.137540490910E-01, ietype=   -1,    core energy of type: Nuc.Rep.

 total core repulsion energy =  7.137540490910E-01
 nmot  =    10 niot  =     2 nfct  =     0 nfvt  =     0
 nrow  =     8 nsym  =     1 ssym  =     1 lenbuf=  1600
 nwalk,xbar:          5        3        2        0        0
 nvalwt,nvalw:        5        3        2        0        0
 ncsft:              19
 total number of valid internal walks:       5
 nvalz,nvaly,nvalx,nvalw =        3       2       0       0

 cisrt info file parameters:
 file number  12 blocksize    766
 mxbld    766
 nd4ext,nd2ext,nd0ext    72    32     6
 n4ext,n3ext,n2ext,n1ext,n0ext,n2int,n1int,n0int      882      672      324       96        6        0        0        0
 minbl4,minbl3,maxbl2   105   105   108
 maxbuf******
 number of external orbitals per symmetry block:   8
 nmsym   1 number of internal orbitals   2
 bummer (warning):transition densities: resetting ref occupation number to 0     0
 executing brd_struct for drt
 executing brd_struct for orbinf
 executing brd_struct for momap
 calcthrxt: niot,maxw1=                     2                     1
 block size     0
 pthz,pthy,pthx,pthw:     3     2     0     0 total internal walks:       5
 maxlp3,n2lp,n1lp,n0lp     1     0     0     0
 orbsym(*)= 1 1

 setref:        3 references kept,
                0 references were marked as invalid, out of
                3 total.
 nmb.of records onel     1
 nmb.of records 2-ext     1
 nmb.of records 1-ext     1
 nmb.of records 0-ext     1
 nmb.of records 2-int     0
 nmb.of records 1-int     0
 nmb.of records 0-int     0
 ---------memory usage in DP -----------------
 < n-ex core usage >
     routines:
    fourex          9600866
    threx           9600007
    twoex              1482
    onex                927
    allin               766
    diagon             1295
               =======
   maximum          9600866
 
  __ static summary __ 
   reflst                 3
   hrfspc                 3
               -------
   static->               3
 
  __ core required  __ 
   totstc                 3
   max n-ex         9600866
               -------
   totnec->         9600869
 
  __ core available __ 
   totspc          65535999
   totnec -         9600869
               -------
   totvec->        55935130

 number of external paths / symmetry
 vertex x      28
 vertex w      36
segment: free space=    55935130
 reducing frespc by                   100 to               55935030 
  for index/conft/indsym storage .
 resegmenting ...



                   segmentation summary for type all-internal
 -------------------------------------------------------------------------------
 seg.      no. of|    no. of|  starting|  internal|  starting|  starting|
  no.    internal|        ci|       csf|     walks|      walk|       DRT|
            paths|  elements|    number|     /seg.|    number|    record|
 -------------------------------------------------------------------------------
  Z 1           3|         3|         0|         3|         0|         1|
 -------------------------------------------------------------------------------
  Y 2           2|        16|         3|         2|         3|         2|
 -------------------------------------------------------------------------------
max. additional memory requirements:index=           4DP  conft+indsym=          12DP  drtbuffer=          84 DP

dimension of the ci-matrix ->>>        19

 executing brd_struct for civct
 gentasklist: ntask=                     6
                    TASKLIST
----------------------------------------------------------------------------------------------------
TASK# BRA# KET#  T-TYPE    DESCR.   SEGMENTTYPE    SEGEL              SEGCI          VWALKS   
----------------------------------------------------------------------------------------------------
     1  2   1    11      one-ext yz   1X  2 1       2       3         16          3       2       3
     2  1   1     1      allint zz    OX  1 1       3       3          3          3       3       3
     3  2   2     5      0ex2ex yy    OX  2 2       2       2         16         16       2       2
     4  2   2    42      four-ext y   4X  2 2       2       2         16         16       2       2
     5  1   1    75      dg-024ext z  OX  1 1       3       3          3          3       3       3
     6  2   2    76      dg-024ext y  OX  2 2       2       2         16         16       2       2
----------------------------------------------------------------------------------------------------
REDTASK #   1 TIME=   5.000 N=  1 (task/type/sgbra)=(   1/11/0) (
REDTASK #   2 TIME=   4.000 N=  1 (task/type/sgbra)=(   2/ 1/0) (
REDTASK #   3 TIME=   3.000 N=  1 (task/type/sgbra)=(   3/ 5/0) (
REDTASK #   4 TIME=   2.000 N=  1 (task/type/sgbra)=(   4/42/1) (
REDTASK #   5 TIME=   1.000 N=  1 (task/type/sgbra)=(   5/75/1) (
REDTASK #   6 TIME=   0.000 N=  1 (task/type/sgbra)=(   6/76/1) (
 nvalz=                     3
 civect=                    16                    16                    16
                    16                    16                    16
 nvaly=                    16
 nzy=                    19
 initializing v-file: 1:                    19

    ---------trial vector generation----------

    trial vectors will be created by: 

    (ivmode= 3) diagonalizing h in the reference space.                     

      2 vectors will be written to unit 11 beginning with logical record   1

            2 vectors will be created
 =========== Executing IN-CORE method ==========


====================================================================================================
Diagonal     counts:  0x:           5 2x:           0 4x:           0
All internal counts: zz :           3 yy:           0 xx:           0 ww:           0
One-external counts: yz :           0 yx:           0 yw:           0
Two-external counts: yy :           0 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           0    task #     2:           2    task #     3:           0    task #     4:           0
task #     5:           4    task #     6:           0    task #
 reference space has dimension       3
 dsyevx: computed roots 1 to    3(converged:   3)

    root           eigenvalues
    ----           ------------
       1          -1.1240135870
       2          -0.6297059938
       3           0.0169931349

 strefv generated    2 initial ci vector(s).
    ---------end of vector generation---------

 ufvoutnew: ... writing  recamt=                     3

         vector  1 from unit 11 written to unit 49 filename cirefv              
 ufvoutnew: ... writing  recamt=                     3

         vector  2 from unit 11 written to unit 49 filename cirefv              

 ************************************************************************
 beginning the bk-type iterative procedure (nzcsf=     3)...
 ************************************************************************

               initial diagonalization conditions:

 number of configuration state functions:                19
 number of initial trial vectors:                         2
 number of initial matrix-vector products:                0
 maximum dimension of the subspace vectors:               7
 number of roots to converge:                             2
 number of iterations:                                    1
 residual norm convergence criteria:               0.000100  0.000100

          starting bk iteration   1

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    1.044746
ci vector #   2dasum_wr=    0.000000
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           0 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           0 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           0    task #     4:           0
task #     5:           4    task #     6:           5    task #
 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    1.000000
ci vector #   2dasum_wr=    0.000000
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           0 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           0 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           0    task #     4:           0
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2
 sovl   1     1.00000000
 sovl   2    -0.00000000     1.00000000
 Final subspace hamiltonian 

                ht   1         ht   2
   ht   1    -1.83776764
   ht   2     0.00000000    -1.34346004
Spectrum of overlapmatrix:    1.000000    1.000000

          calcsovref: tciref block   1

              civs   1       civs   2
 refs   1    1.00000      -3.308347E-16
 refs   2    0.00000        1.00000    

          calcsovref: scrb block   1

                ci   1         ci   2
 civs   1   -1.00000       4.463214E-16
 civs   2  -7.771561E-16   -1.00000    

          calcsovref: sovref block   1

              v      1       v      2
 ref    1   -1.00000       7.771561E-16
 ref    2  -7.771561E-16   -1.00000    

          reference overlap matrix  block   1

                ci   1         ci   2
 ref:   1    -1.00000000     0.00000000
 ref:   2    -0.00000000    -1.00000000
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  1  1     -1.1240135870  2.6645E-15  1.5751E-02  1.4577E-01  1.0000E-04
 mr-sdci #  1  2     -0.6297059938  2.8866E-15  0.0000E+00  1.3251E-01  1.0000E-04
 
 root number  1 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0020s 
time spent in multnx:                   0.0020s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0020s 

 mr-sdci  convergence not reached after  1 iterations.

 final mr-sdci  convergence information:

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  1  1     -1.1240135870  2.6645E-15  1.5751E-02  1.4577E-01  1.0000E-04
 mr-sdci #  1  2     -0.6297059938  2.8866E-15  0.0000E+00  1.3251E-01  1.0000E-04
 
diagon:itrnv=   2
 expansion vectors are not transformed.
 matrix-vector products are not transformed.

    3 expansion eigenvectors written to unit nvfile (= 11)
    2 matrix-vector products written to unit nhvfil (= 10)

 ************************************************************************
 beginning the ci iterative diagonalization procedure... 
 ************************************************************************

               initial diagonalization conditions:

 number of configuration state functions:                19
 number of initial trial vectors:                         3
 number of initial matrix-vector products:                2
 maximum dimension of the subspace vectors:               7
 number of roots to converge:                             2
 number of iterations:                                   60
 residual norm convergence criteria:               0.000100  0.000100

          starting ci iteration   1

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.000000
ci vector #   2dasum_wr=    0.195300
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3
 sovl   1     1.00000000
 sovl   2    -0.00000000     1.00000000
 sovl   3    -0.00000000    -0.00000000     0.01400644
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3
   ht   1    -1.83776764
   ht   2     0.00000000    -1.34346004
   ht   3     0.01575102     0.00000000    -0.00991526
Spectrum of overlapmatrix:    0.014006    1.000000    1.000000

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3
 refs   1    1.00000      -3.308347E-16  -9.269631E-14
 refs   2    0.00000        1.00000      -4.017309E-29

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3
 civs   1  -0.993316      -6.683660E-16   0.115426    
 civs   2  -1.221245E-15    1.00000       6.938894E-17
 civs   3   0.975302       6.433628E-16    8.39312    

          calcsovref: sovref block   1

              v      1       v      2       v      3
 ref    1  -0.993316      -9.992007E-16   0.115426    
 ref    2  -1.221245E-15    1.00000       6.938894E-17

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3
 ref:   1    -0.99331610    -0.00000000     0.11542584
 ref:   2    -0.00000000     1.00000000     0.00000000
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  1  1     -1.1394789537  1.5465E-02  3.9419E-04  2.0817E-02  1.0000E-04
 mr-sdci #  1  2     -0.6297059938  4.4409E-16  0.0000E+00  1.3251E-01  1.0000E-04
 mr-sdci #  1  3      0.0213122170  6.9244E-01  0.0000E+00  4.8047E-01  1.0000E-04
 
 root number  1 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0020s 
time spent in multnx:                   0.0020s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0020s 

          starting ci iteration   2

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.065111
ci vector #   2dasum_wr=    0.009652
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4
 sovl   1     1.00000000
 sovl   2    -0.00000000     1.00000000
 sovl   3    -0.00000000    -0.00000000     0.01400644
 sovl   4     0.04787560     0.00000000    -0.00025871     0.00267073
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4
   ht   1    -1.83776764
   ht   2     0.00000000    -1.34346004
   ht   3     0.01575102     0.00000000    -0.00991526
   ht   4    -0.08798022    -0.00000000     0.00083345    -0.00448717
Spectrum of overlapmatrix:    0.000373    0.014011    1.000000    1.002293

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4
 refs   1    1.00000      -3.308347E-16  -9.269631E-14   4.787560E-02
 refs   2    0.00000        1.00000      -4.017309E-29   4.699334E-16

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4
 civs   1   -1.03835       7.767695E-16    2.03777        1.37835    
 civs   2  -9.141064E-16    1.00000       2.098377E-14   1.305034E-14
 civs   3   0.998331      -2.123145E-15   -5.16891        6.67787    
 civs   4   0.945788      -1.983006E-14   -44.1503       -26.9171    

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4
 ref    1  -0.993071      -5.034410E-16  -7.594692E-02   8.967685E-02
 ref    2  -4.696491E-16    1.00000       2.360770E-16   4.010825E-16

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4
 ref:   1    -0.99307106    -0.00000000    -0.07594692     0.08967685
 ref:   2    -0.00000000     1.00000000     0.00000000     0.00000000
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  2  1     -1.1398518344  3.7288E-04  6.4646E-06  2.8920E-03  1.0000E-04
 mr-sdci #  2  2     -0.6297059938 -4.4409E-16  0.0000E+00  1.3251E-01  1.0000E-04
 mr-sdci #  2  3     -0.0496812892  7.0994E-02  0.0000E+00  2.8129E-01  1.0000E-04
 mr-sdci #  2  4      0.0476996440  6.6605E-01  0.0000E+00  4.5022E-01  1.0000E-04
 
 root number  1 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0000s 
time spent in multnx:                   0.0000s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0000s 

          starting ci iteration   3

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.002101
ci vector #   2dasum_wr=    0.003665
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4       sovl   5
 sovl   1     1.00000000
 sovl   2    -0.00000000     1.00000000
 sovl   3    -0.00000000    -0.00000000     0.01400644
 sovl   4     0.04787560     0.00000000    -0.00025871     0.00267073
 sovl   5    -0.00153609    -0.00000000    -0.00010600    -0.00007358     0.00000848
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4         ht   5
   ht   1    -1.83776764
   ht   2     0.00000000    -1.34346004
   ht   3     0.01575102     0.00000000    -0.00991526
   ht   4    -0.08798022    -0.00000000     0.00083345    -0.00448717
   ht   5     0.00282465     0.00000000     0.00017392     0.00012850    -0.00000928
Spectrum of overlapmatrix:    0.000005    0.000373    0.014012    1.000000    1.002295

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4       civs   5
 refs   1    1.00000      -3.308347E-16  -9.269631E-14   4.787560E-02  -1.536086E-03
 refs   2    0.00000        1.00000      -4.017309E-29   4.699334E-16  -3.270954E-16

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4         ci   5
 civs   1    1.03768      -1.158864E-15   6.050346E-02    2.33202       0.959295    
 civs   2   9.593367E-16   -1.00000      -7.827274E-14   1.553494E-14   1.118997E-13
 civs   3  -0.998735       1.459939E-14    4.21269       -2.96133        7.47417    
 civs   4  -0.962532      -4.035450E-14   -8.26515       -50.4925       -7.83189    
 civs   5   -1.02591      -2.641105E-12   -273.114       -27.3712        336.120    

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4       v      5
 ref    1   0.993170       1.296939E-15   8.433085E-02  -4.328884E-02   6.802906E-02
 ref    2   8.425798E-16   -1.00000       7.177467E-15   7.598197E-16  -1.724159E-15

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5
 ref:   1     0.99316989     0.00000000     0.08433085    -0.04328884     0.06802906
 ref:   2     0.00000000    -1.00000000     0.00000000     0.00000000    -0.00000000
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  3  1     -1.1398584665  6.6321E-06  5.7288E-09  1.3722E-04  1.0000E-04
 mr-sdci #  3  2     -0.6297059938 -4.4409E-16  0.0000E+00  1.3251E-01  1.0000E-04
 mr-sdci #  3  3     -0.3430123094  2.9333E-01  0.0000E+00  1.6260E-01  1.0000E-04
 mr-sdci #  3  4     -0.0407785105  8.8478E-02  0.0000E+00  2.3377E-01  1.0000E-04
 mr-sdci #  3  5      0.6143430761  9.9411E-02  0.0000E+00  2.2442E-01  1.0000E-04
 
 root number  1 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0010s 
time spent in multnx:                   0.0010s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0010s 

          starting ci iteration   4

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.000042
ci vector #   2dasum_wr=    0.000076
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4       sovl   5       sovl   6
 sovl   1     1.00000000
 sovl   2    -0.00000000     1.00000000
 sovl   3    -0.00000000    -0.00000000     0.01400644
 sovl   4     0.04787560     0.00000000    -0.00025871     0.00267073
 sovl   5    -0.00153609    -0.00000000    -0.00010600    -0.00007358     0.00000848
 sovl   6     0.00003071    -0.00000000     0.00000042     0.00000163    -0.00000004     0.00000000
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4         ht   5         ht   6
   ht   1    -1.83776764
   ht   2     0.00000000    -1.34346004
   ht   3     0.01575102     0.00000000    -0.00991526
   ht   4    -0.08798022    -0.00000000     0.00083345    -0.00448717
   ht   5     0.00282465     0.00000000     0.00017392     0.00012850    -0.00000928
   ht   6    -0.00005644     0.00000000    -0.00000031    -0.00000299     0.00000007     0.00000000

                v:   1         v:   2         v:   3         v:   4         v:   5         v:   6

   eig(s)   1.931830E-09   5.309724E-06   3.729029E-04   1.401215E-02    1.00000        1.00230    
 
   x:   1  -1.313133E-05  -1.276024E-03  -4.783780E-02   9.072610E-04  -1.337796E-13   0.998854    
   x:   2   6.087500E-17  -4.528878E-16  -4.886026E-16   6.867403E-18   -1.00000      -1.339583E-13
   x:   3  -5.909171E-05  -7.670467E-03   1.890798E-02   0.999792       3.592644E-18  -1.235861E-05
   x:   4  -4.541393E-04  -5.428871E-03   0.998661      -1.892768E-02  -6.915858E-15   4.783880E-02
   x:   5  -2.711444E-03  -0.999951      -5.507052E-03  -7.567721E-03   7.997939E-16  -1.534336E-03
   x:   6   0.999996      -2.714257E-03   4.390899E-04   2.997618E-05   5.771129E-17   3.068088E-05
 bummer (warning):overlap matrix: # small eigenvalues=     1

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4       civs   5       civs   6
 refs   1    1.00000      -3.308347E-16  -9.269631E-14   4.787560E-02  -1.536086E-03   3.070863E-05
 refs   2    0.00000        1.00000      -4.017309E-29   4.699334E-16  -3.270954E-16  -6.183075E-17

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6
 civs   1   -1.03765       3.497533E-14  -6.655009E-02    2.32293        1.01322      -0.156490    
 civs   2  -9.363416E-16   -1.00000       8.169821E-14  -1.223732E-13   2.542954E-13   1.332835E-12
 civs   3   0.998735      -1.098826E-14   -4.20543       -3.01553        7.16528       -2.46298    
 civs   4   0.962544      -3.347586E-13    8.35067       -48.8037       -10.9057       -14.6776    
 civs   5    1.02673      -2.729879E-12    273.176       -29.0680        325.922       -102.104    
 civs   6  -0.932551      -7.562430E-10    71.3522       -2481.51        2457.14        22482.0    

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4       v      5       v      6
 ref    1  -0.993171       2.495439E-16  -8.418700E-02  -4.512676E-02   6.591985E-02  -1.195963E-02
 ref    2  -7.621890E-16   -1.00000      -8.143851E-15   1.763385E-14  -9.364008E-15  -3.074215E-14

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6
 ref:   1    -0.99317051     0.00000000    -0.08418700    -0.04512676     0.06591985    -0.01195963
 ref:   2    -0.00000000    -1.00000000    -0.00000000     0.00000000    -0.00000000    -0.00000000
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  4  1     -1.1398584719  5.3424E-09  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  4  2     -0.6297059938  4.4409E-16  1.5605E-02  1.3251E-01  1.0000E-04
 mr-sdci #  4  3     -0.3430343194  2.2010E-05  0.0000E+00  1.5942E-01  1.0000E-04
 mr-sdci #  4  4     -0.0664345841  2.5656E-02  0.0000E+00  1.0849E-01  1.0000E-04
 mr-sdci #  4  5      0.5954605391  1.8883E-02  0.0000E+00  3.4402E-02  1.0000E-04
 mr-sdci #  4  6      2.1515996132 -1.4378E+00  0.0000E+00  9.9314E-01  1.0000E-04
 
 root number  2 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.001000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0010s 
time spent in multnx:                   0.0010s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0020s 

          starting ci iteration   5

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.000000
ci vector #   2dasum_wr=    0.170281
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4       sovl   5       sovl   6       sovl   7
 sovl   1     1.00000000
 sovl   2    -0.00000000     1.00000000
 sovl   3    -0.00000000    -0.00000000     0.01400644
 sovl   4     0.04787560     0.00000000    -0.00025871     0.00267073
 sovl   5    -0.00153609    -0.00000000    -0.00010600    -0.00007358     0.00000848
 sovl   6     0.00003071    -0.00000000     0.00000042     0.00000163    -0.00000004     0.00000000
 sovl   7    -0.00000000    -0.00000000    -0.00000000     0.00000000     0.00000000    -0.00000000     0.01586909
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4         ht   5         ht   6         ht   7
   ht   1    -1.83776764
   ht   2     0.00000000    -1.34346004
   ht   3     0.01575102     0.00000000    -0.00991526
   ht   4    -0.08798022    -0.00000000     0.00083345    -0.00448717
   ht   5     0.00282465     0.00000000     0.00017392     0.00012850    -0.00000928
   ht   6    -0.00005644     0.00000000    -0.00000031    -0.00000299     0.00000007     0.00000000
   ht   7     0.00000000     0.01560511     0.00000000    -0.00000000    -0.00000000     0.00000000    -0.00624362

                v:   1         v:   2         v:   3         v:   4         v:   5         v:   6         v:   7

   eig(s)   1.931830E-09   5.309724E-06   3.729029E-04   1.401215E-02   1.586909E-02    1.00000        1.00230    
 
   x:   1   1.313133E-05  -1.276024E-03  -4.783780E-02  -9.072610E-04   3.197272E-16   1.337811E-13   0.998854    
   x:   2  -5.659172E-17  -3.301373E-16  -4.870743E-16  -6.816914E-18  -5.131250E-30    1.00000      -1.339583E-13
   x:   3   5.909171E-05  -7.670467E-03   1.890798E-02  -0.999792      -8.414445E-14  -1.644742E-18  -1.235861E-05
   x:   4   4.541393E-04  -5.428871E-03   0.998661       1.892768E-02   1.426829E-14   6.893210E-15   4.783880E-02
   x:   5   2.711444E-03  -0.999951      -5.507052E-03   7.567721E-03   2.104817E-15  -5.182774E-16  -1.534336E-03
   x:   6  -0.999996      -2.714257E-03   4.390899E-04  -2.997618E-05  -1.286986E-16   1.126432E-16   3.068088E-05
   x:   7  -1.359169E-16   1.536807E-15  -1.263124E-14  -8.441264E-14    1.00000      -1.350295E-28  -9.997450E-16
 bummer (warning):overlap matrix: # small eigenvalues=     1

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4       civs   5       civs   6       civs   7
 refs   1    1.00000      -3.308347E-16  -9.269631E-14   4.787560E-02  -1.536086E-03   3.070863E-05  -9.945191E-16
 refs   2    0.00000        1.00000      -4.017309E-29   4.699334E-16  -3.270954E-16  -6.183075E-17  -8.690279E-31

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7
 civs   1   -1.03765       2.870313E-14   6.655009E-02    2.32293       2.563353E-13    1.01322      -0.156490    
 civs   2  -9.632788E-16  -0.991876      -8.910977E-14  -1.407502E-13   0.127208       2.625989E-13   1.361975E-12
 civs   3   0.998735       6.764203E-15    4.20543       -3.01553       4.464697E-14    7.16528       -2.46298    
 civs   4   0.962544      -1.677153E-13   -8.35067       -48.8037      -2.142550E-12   -10.9057       -14.6776    
 civs   5    1.02673      -3.967814E-12   -273.176       -29.0680      -2.746774E-11    325.922       -102.104    
 civs   6  -0.932551      -8.226019E-10   -71.3522       -2481.51      -6.125486E-09    2457.14        22482.0    
 civs   7  -1.474058E-17    1.00981      -1.362221E-13  -6.609949E-13    7.87375       6.490694E-13   5.167277E-12

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4       v      5       v      6       v      7
 ref    1  -0.993171       8.314601E-16   8.418700E-02  -4.512676E-02  -2.572080E-17   6.591985E-02  -1.195963E-02
 ref    2  -7.891261E-16  -0.991876       7.322899E-16  -7.431211E-16   0.127208      -1.060500E-15  -1.602128E-15

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7
 ref:   1    -0.99317051     0.00000000     0.08418700    -0.04512676    -0.00000000     0.06591985    -0.01195963
 ref:   2    -0.00000000    -0.99187604     0.00000000    -0.00000000     0.12720821    -0.00000000    -0.00000000

 trial vector basis is being transformed.  new dimension:   4
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  5  1     -1.1398584719  2.2204E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  5  2     -0.6455932359  1.5887E-02  1.1086E-04  1.1661E-02  1.0000E-04
 mr-sdci #  5  3     -0.3430343194 -1.5543E-15  0.0000E+00  1.5942E-01  1.0000E-04
 mr-sdci #  5  4     -0.0664345841 -3.3307E-16  0.0000E+00  1.0849E-01  1.0000E-04
 mr-sdci #  5  5      0.3361959146  2.5926E-01  0.0000E+00  3.6845E-01  1.0000E-04
 mr-sdci #  5  6      0.5954605391  1.5561E+00  0.0000E+00  3.4402E-02  1.0000E-04
 mr-sdci #  5  7      2.1515996132 -1.4378E+00  0.0000E+00  9.9314E-01  1.0000E-04
 
 root number  2 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0010s 
time spent in multnx:                   0.0000s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0010s 

          starting ci iteration   6

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.000000
ci vector #   2dasum_wr=    0.015838
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4       sovl   5
 sovl   1     1.00000000
 sovl   2     0.00000000     1.00000000
 sovl   3     0.00000000     0.00000000     1.00000000
 sovl   4     0.00000000     0.00000000     0.00000000     1.00000000
 sovl   5    -0.00000000    -0.00002573    -0.00000000     0.00000000     0.00013063
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4         ht   5
   ht   1    -1.85361252
   ht   2     0.00000000    -1.35934728
   ht   3     0.00000000     0.00000000    -1.05678837
   ht   4     0.00000000    -0.00000000     0.00000000    -0.78018863
   ht   5     0.00000000    -0.00007588     0.00000000    -0.00000000    -0.00006101
Spectrum of overlapmatrix:    0.000131    1.000000    1.000000    1.000000    1.000000

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4       civs   5
 refs   1  -0.993171       8.314601E-16   8.418700E-02  -4.512676E-02   4.296983E-17
 refs   2  -7.891261E-16  -0.991876       7.322899E-16  -7.431211E-16  -2.795257E-14

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4         ci   5
 civs   1   -1.00000      -9.359124E-17  -3.403040E-17   9.361338E-17   8.437869E-15
 civs   2   1.874581E-18  -0.999965      -7.274948E-16  -3.637163E-16  -8.617652E-03
 civs   3   3.439631E-17   4.933879E-17   -1.00000       1.670936E-15   6.734344E-14
 civs   4   9.366759E-17   4.273143E-16   1.781958E-15    1.00000      -5.458748E-14
 civs   5  -8.360255E-17  -0.950964       2.090174E-12   3.123758E-12    87.4902    

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4       v      5
 ref    1   0.993171      -7.944717E-16  -8.418700E-02  -4.512676E-02   3.504830E-15
 ref    2   7.872668E-16   0.991842      -1.070526E-17  -3.823597E-16   8.547643E-03

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676     0.00000000
 ref:   2     0.00000000     0.99184172    -0.00000000    -0.00000000     0.00854764
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  6  1     -1.1398584719  2.2204E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  6  2     -0.6456986628  1.0543E-04  2.0045E-06  1.1466E-03  1.0000E-04
 mr-sdci #  6  3     -0.3430343194  1.5543E-15  0.0000E+00  1.5942E-01  1.0000E-04
 mr-sdci #  6  4     -0.0664345841  5.5511E-16  0.0000E+00  1.0849E-01  1.0000E-04
 mr-sdci #  6  5      0.2467711574  8.9425E-02  0.0000E+00  6.0227E-01  1.0000E-04
 
 root number  2 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.001000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0010s 
time spent in multnx:                   0.0010s 
integral transfer time:                 0.0010s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0020s 

          starting ci iteration   7

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.006514
ci vector #   2dasum_wr=    0.001625
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4       sovl   5       sovl   6
 sovl   1     1.00000000
 sovl   2     0.00000000     1.00000000
 sovl   3     0.00000000     0.00000000     1.00000000
 sovl   4     0.00000000     0.00000000     0.00000000     1.00000000
 sovl   5    -0.00000000    -0.00002573    -0.00000000     0.00000000     0.00013063
 sovl   6     0.00000000     0.00632943     0.00000000    -0.00000000    -0.00000032     0.00004388
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4         ht   5         ht   6
   ht   1    -1.85361252
   ht   2     0.00000000    -1.35934728
   ht   3     0.00000000     0.00000000    -1.05678837
   ht   4     0.00000000    -0.00000000     0.00000000    -0.78018863
   ht   5     0.00000000    -0.00007588     0.00000000    -0.00000000    -0.00006101
   ht   6    -0.00000000    -0.00860390    -0.00000000     0.00000000     0.00000184    -0.00005637
Spectrum of overlapmatrix:    0.000004    0.000131    1.000000    1.000000    1.000000    1.000040

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4       civs   5       civs   6
 refs   1  -0.993171       8.314601E-16   8.418700E-02  -4.512676E-02   4.296983E-17  -6.630787E-17
 refs   2  -7.891261E-16  -0.991876       7.322899E-16  -7.431211E-16  -2.795257E-14  -6.514355E-03

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6
 civs   1   -1.00000      -1.465883E-16  -3.458605E-17   9.364986E-17  -3.814323E-14   2.133535E-14
 civs   2   1.357644E-18   -1.00388       3.929778E-14   1.111358E-13   -2.46113        2.10252    
 civs   3   3.442342E-17   6.301396E-17   -1.00000       1.871636E-15  -5.682101E-14  -4.286279E-14
 civs   4   9.363631E-17  -7.831552E-17   1.871636E-15    1.00000       2.213308E-13  -1.018576E-13
 civs   5  -8.451776E-17  -0.962146       2.267296E-12   4.072713E-12   -56.1782       -67.0739    
 civs   6   8.163579E-17   0.618553      -6.333208E-12  -1.769001E-11    389.882       -331.253    

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4       v      5       v      6
 ref    1   0.993171      -7.626168E-16  -8.418700E-02  -4.512676E-02  -7.201314E-15   6.290938E-16
 ref    2   7.872477E-16   0.991694       1.545951E-15   4.262954E-15  -9.869351E-02   7.245894E-02

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676    -0.00000000     0.00000000
 ref:   2     0.00000000     0.99169361     0.00000000     0.00000000    -0.09869351     0.07245894
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  7  1     -1.1398584719 -2.2204E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  7  2     -0.6456999027  1.2399E-06  1.5943E-07  5.2440E-04  1.0000E-04
 mr-sdci #  7  3     -0.3430343194  4.4409E-16  0.0000E+00  1.5942E-01  1.0000E-04
 mr-sdci #  7  4     -0.0664345841  3.3307E-16  0.0000E+00  1.0849E-01  1.0000E-04
 mr-sdci #  7  5      0.1283323632  1.1844E-01  0.0000E+00  4.1607E-01  1.0000E-04
 mr-sdci #  7  6      0.3322678542  2.6319E-01  0.0000E+00  4.2570E-01  1.0000E-04
 
 root number  2 is used to define the new expansion vector.
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0010s 
time spent in multnx:                   0.0010s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0020s 

          starting ci iteration   8

 =========== Executing IN-CORE method ==========
ci vector #   1dasum_wr=    0.001290
ci vector #   2dasum_wr=    0.000313
ci vector #   3dasum_wr=    0.000000
ci vector #   4dasum_wr=    0.000000


====================================================================================================
Diagonal     counts:  0x:           7 2x:           2 4x:           2
All internal counts: zz :           3 yy:           1 xx:           0 ww:           0
One-external counts: yz :           8 yx:           0 yw:           0
Two-external counts: yy :           3 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           5    task #
 Final Overlap matrix sovl

              sovl   1       sovl   2       sovl   3       sovl   4       sovl   5       sovl   6       sovl   7
 sovl   1     1.00000000
 sovl   2     0.00000000     1.00000000
 sovl   3     0.00000000     0.00000000     1.00000000
 sovl   4     0.00000000     0.00000000     0.00000000     1.00000000
 sovl   5    -0.00000000    -0.00002573    -0.00000000     0.00000000     0.00013063
 sovl   6     0.00000000     0.00632943     0.00000000    -0.00000000    -0.00000032     0.00004388
 sovl   7    -0.00000000    -0.00127044     0.00000000     0.00000000    -0.00000039    -0.00000833     0.00000173
 Final subspace hamiltonian 

                ht   1         ht   2         ht   3         ht   4         ht   5         ht   6         ht   7
   ht   1    -1.85361252
   ht   2     0.00000000    -1.35934728
   ht   3     0.00000000     0.00000000    -1.05678837
   ht   4     0.00000000    -0.00000000     0.00000000    -0.78018863
   ht   5     0.00000000    -0.00007588     0.00000000    -0.00000000    -0.00006101
   ht   6    -0.00000000    -0.00860390    -0.00000000     0.00000000     0.00000184    -0.00005637
   ht   7     0.00000000     0.00172696    -0.00000000    -0.00000000     0.00000068     0.00001106    -0.00000214
Spectrum of overlapmatrix:    0.000000    0.000004    0.000131    1.000000    1.000000    1.000000    1.000042

          calcsovref: tciref block   1

              civs   1       civs   2       civs   3       civs   4       civs   5       civs   6       civs   7
 refs   1  -0.993171       8.314601E-16   8.418700E-02  -4.512676E-02   4.296983E-17  -6.630787E-17   2.991919E-17
 refs   2  -7.891261E-16  -0.991876       7.322899E-16  -7.431211E-16  -2.795257E-14  -6.514355E-03   1.289887E-03

          calcsovref: scrb block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7
 civs   1   -1.00000      -1.241878E-16  -3.443697E-17   9.374330E-17   2.921788E-14  -3.222661E-14   9.002314E-14
 civs   2   1.239658E-18   -1.00323       1.778021E-14   8.299337E-14    2.12543       -2.48470        2.57569    
 civs   3   3.450473E-17   5.025736E-17   -1.00000       2.466536E-15   6.892388E-14   5.038412E-14  -8.598754E-14
 civs   4   9.363149E-17  -2.930253E-16   2.799603E-15    1.00000      -1.737595E-13   1.672723E-13  -5.849744E-13
 civs   5  -8.456260E-17  -0.963361       2.385588E-12   4.658247E-12    58.1014        62.4369        22.4109    
 civs   6   7.000187E-17   0.685775      -8.812187E-12  -2.310639E-11   -393.731        326.218        247.392    
 civs   7  -1.508226E-16   0.845223      -2.925594E-11  -4.882140E-11   -283.158       -326.163        3260.41    

          calcsovref: sovref block   1

              v      1       v      2       v      3       v      4       v      5       v      6       v      7
 ref    1   0.993171      -7.549309E-16  -8.418700E-02  -4.512676E-02   6.524784E-15  -2.072609E-15   1.400017E-14
 ref    2   7.872460E-16   0.991702       1.300812E-15   4.486872E-15   9.149769E-02  -8.129806E-02   3.920567E-02

          reference overlap matrix  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676     0.00000000    -0.00000000     0.00000000
 ref:   2     0.00000000     0.99170237     0.00000000     0.00000000     0.09149769    -0.08129806     0.03920567

 trial vector basis is being transformed.  new dimension:   4
 NCSF=                     3                    16                     0
                     0

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  8  1     -1.1398584719  4.4409E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  8  2     -0.6457000375  1.3475E-07  8.5939E-10  4.7882E-05  1.0000E-04
 mr-sdci #  8  3     -0.3430343194  0.0000E+00  0.0000E+00  1.5942E-01  1.0000E-04
 mr-sdci #  8  4     -0.0664345841  2.2204E-16  0.0000E+00  1.0849E-01  1.0000E-04
 mr-sdci #  8  5      0.1190435982  9.2888E-03  0.0000E+00  4.2867E-01  1.0000E-04
 mr-sdci #  8  6      0.3206849192  1.1583E-02  0.0000E+00  3.9555E-01  1.0000E-04
 mr-sdci #  8  7      1.4441747328  7.0742E-01  0.0000E+00  3.3722E-02  1.0000E-04
 
======================== TIMING STATISTICS PER TASK    ========================
task# type node    tmult    tloop     tint    tmnx    MFLIPS     mloop 
    1   11    0     0.00     0.00     0.00     0.00         0.    0.0000
    2    1    0     0.00     0.00     0.00     0.00         0.    0.0000
    3    5    0     0.00     0.00     0.00     0.00         0.    0.0000
    4   42    0     0.00     0.00     0.00     0.00         0.    0.0000
    5   75    0     0.00     0.00     0.00     0.00         0.    0.0000
    6   76    0     0.00     0.00     0.00     0.00         0.    0.0000
================================================================
================ TIMING STATISTICS FOR JOB     ================
time for subspace matrix construction  0.000000
time for cinew                         0.000000
time for eigenvalue solver             0.000000
time for vector access                 0.000000
================================================================
time spent in mult:                     0.0010s 
time spent in multnx:                   0.0010s 
integral transfer time:                 0.0000s 
time spent for loop construction:       0.0000s 
time for vector access in mult:         0.0000s 
total time per CI iteration:            0.0010s 

 mr-sdci  convergence criteria satisfied after  8 iterations.

 final mr-sdci  convergence information:

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  8  1     -1.1398584719  4.4409E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  8  2     -0.6457000375  1.3475E-07  8.5939E-10  4.7882E-05  1.0000E-04
 mr-sdci #  8  3     -0.3430343194  0.0000E+00  0.0000E+00  1.5942E-01  1.0000E-04
 mr-sdci #  8  4     -0.0664345841  2.2204E-16  0.0000E+00  1.0849E-01  1.0000E-04

####################CIUDGINFO####################

   ci vector at position   1 energy=   -1.139858471882
   ci vector at position   2 energy=   -0.645700037473

################END OF CIUDGINFO################

 
diagon:itrnv=   0
    2 of the   5 expansion vectors are transformed.
    2 of the   4 matrix-vector products are transformed.

    2 expansion eigenvectors written to unit nvfile (= 11)
    2 matrix-vector products written to unit nhvfil (= 10)
maximum overlap with reference    1(overlap= 0.99317)

 information on vector: 1 from unit 11 written to unit 48 filename civout              
maximum overlap with reference    2(overlap= 0.99170)

 information on vector: 2 from unit 11 written to unit 48 filename civout              


 --- list of ci coefficients ( ctol =   1.00E-02 )  total energy( 1) =        -1.1398584719

                                                       internal orbitals

                                          level       1    2

                                          orbital     1    2

                                         symmetry   a    a  

 path  s ms    csf#    c(i)    ext. orb.(sym)
 z*  1  1       1 -0.991336                        +-      
 z*  1  1       3  0.062772                             +- 
 y   1  1       4 -0.087645              1( a  )    -      
 y   1  1       8  0.031751              5( a  )    -      
 y   1  1      13  0.067472              2( a  )         - 

 ci coefficient statistics:
           rq > 0.1                1
      0.1> rq > 0.01               4
     0.01> rq > 0.001              1
    0.001> rq > 0.0001             0
   0.0001> rq > 0.00001            0
  0.00001> rq > 0.000001           0
 0.000001> rq                     13
           all                    19
 =========== Executing IN-CORE method ==========


====================================================================================================
Diagonal     counts:  0x:           5 2x:           0 4x:           0
All internal counts: zz :           3 yy:           0 xx:           0 ww:           0
One-external counts: yz :           0 yx:           0 yw:           0
Two-external counts: yy :           0 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           1    task #
  iref  icsf         v(icsf)             hv(icsf)
     1     1     -0.991335920756      1.115179437448
     2     2     -0.000000000000      0.000000000000
     3     3      0.062772086263     -0.050828549839

 number of reference csfs (nref) is     3.  root number (iroot) is  1.
 c0**2 =   0.98668724  c**2 (all zwalks) =   0.98668724

 pople ci energy extrapolation is computed with  2 correlated electrons.

 eref      =     -1.123667156809   "relaxed" cnot**2         =   0.986687242594
 eci       =     -1.139858471882   deltae = eci - eref       =  -0.016191315074
 eci+dv1   =     -1.140074022932   dv1 = (1-cnot**2)*deltae  =  -0.000215551050
 eci+dv2   =     -1.140076931228   dv2 = dv1 / cnot**2       =  -0.000218459346
 eci+dv3   =     -1.140079919078   dv3 = dv1 / (2*cnot**2-1) =  -0.000221447195
 eci+pople =     -1.139858471882   (  2e- scaled deltae )    =  -0.016191315074


 --- list of ci coefficients ( ctol =   1.00E-02 )  total energy( 2) =        -0.6457000375

                                                       internal orbitals

                                          level       1    2

                                          orbital     1    2

                                         symmetry   a    a  

 path  s ms    csf#    c(i)    ext. orb.(sym)
 z*  1  1       2 -0.991702                        +     - 
 y   1  1       5  0.021035              2( a  )    -      
 y   1  1      12  0.119730              1( a  )         - 
 y   1  1      16 -0.041695              5( a  )         - 

 ci coefficient statistics:
           rq > 0.1                2
      0.1> rq > 0.01               2
     0.01> rq > 0.001              1
    0.001> rq > 0.0001             0
   0.0001> rq > 0.00001            0
  0.00001> rq > 0.000001           0
 0.000001> rq                     14
           all                    19
 =========== Executing IN-CORE method ==========


====================================================================================================
Diagonal     counts:  0x:           5 2x:           0 4x:           0
All internal counts: zz :           3 yy:           0 xx:           0 ww:           0
One-external counts: yz :           0 yx:           0 yw:           0
Two-external counts: yy :           0 ww:           0 xx:           0 xz:           0 wz:           0 wx:           0
Three-ext.   counts: yx :           0 yw:           0

SO-0ex       counts: zz :           0 yy:           0 xx:           0 ww:           0
SO-1ex       counts: yz :           0 yx:           0 yw:           0
SO-2ex       counts: yy :           0 xx:           0 wx:           0
====================================================================================================




LOOPCOUNT per task:
task #     1:           7    task #     2:           2    task #     3:           3    task #     4:           2
task #     5:           4    task #     6:           1    task #
  iref  icsf         v(icsf)             hv(icsf)
     1     1     -0.000000000000      0.000000000000
     2     2     -0.991702368636      0.624480925590
     3     3     -0.000000000000     -0.000000000000

 number of reference csfs (nref) is     3.  root number (iroot) is  2.
 c0**2 =   0.98347359  c**2 (all zwalks) =   0.98347359

 pople ci energy extrapolation is computed with  2 correlated electrons.

 eref      =     -0.629705993795   "relaxed" cnot**2         =   0.983473587958
 eci       =     -0.645700037473   deltae = eci - eref       =  -0.015994043679
 eci+dv1   =     -0.645964361629   dv1 = (1-cnot**2)*deltae  =  -0.000264324156
 eci+dv2   =     -0.645968803365   dv2 = dv1 / cnot**2       =  -0.000268765892
 eci+dv3   =     -0.645973396931   dv3 = dv1 / (2*cnot**2-1) =  -0.000273359458
 eci+pople =     -0.645700037473   (  2e- scaled deltae )    =  -0.015994043679
 passed aftci ... 
 readint2: molcas,dalton2=                     0                     0
 files%faoints=aoints              
                       Size (real*8) of d2temp for two-external contributions        244
 
                       Size (real*8) of d2temp for all-internal contributions          6
                       Size (real*8) of d2temp for one-external contributions         96
                       Size (real*8) of d2temp for two-external contributions        244
size_thrext:  lsym   l1    ksym   k1strt   k1       cnt3 
                1    1    1    1    9      336
                1    2    1    1    9      336
                       Size (real*8) of d2temp for three-external contributions        672
                       Size (real*8) of d2temp for four-external contributions        990
 serial operation: forcing vdisk for temporary dd012 matrices
location of d2temp files... fileloc(dd012)=       1
location of d2temp files... fileloc(d3)=       1
location of d2temp files... fileloc(d4)=       1
 files%dd012ext =  unit=  22  vdsk=   1  filestart=       1
 files%d3ext =     unit=  23  vdsk=   1  filestart=    3065
 files%d4ext =     unit=  24  vdsk=   1  filestart=  243065
            0xdiag    0ext      1ext      2ext      3ext      4ext
d2off                   767      1533      2299         1         1
d2rec                     1         1         1         1         1
recsize                 766       766       766    240000    240000
d2bufferlen=         240000
maxbl3=              240000
maxbl4=              240000
  allocated                 483065  DP for d2temp 
sifcfg setup: record length 4096 DP
# d1 elements per record  3272
# d2 elements per record  2730
  The MR-CISD density will be calculated.
 item #                     1 suffix=:.drt1.state1:
================================================================================
  Reading record                      1  of civout
 INFO:ref#  1vector#  1 method:  0 last record  0max overlap with ref# 99% root-following 0
 MR-CISD energy:    -1.13985847    -1.85361252
 residuum:     0.00000297
 deltae:     0.00000000

          sovref  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7         ci   8
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676     0.00000000    -0.00000000     0.00000000     0.00000000
 ref:   2     0.00000000     0.99170237     0.00000000     0.00000000     0.09149769    -0.08129806     0.03920567     0.00000000

                ci   9         ci  10         ci  11         ci  12         ci  13         ci  14         ci  15         ci  16

                ci  17         ci  18         ci  19         ci  20         ci  21         ci  22         ci  23         ci  24

                ci  25         ci  26         ci  27         ci  28         ci  29         ci  30         ci  31         ci  32

                ci  33         ci  34         ci  35         ci  36         ci  37         ci  38         ci  39         ci  40

                ci  41         ci  42         ci  43         ci  44         ci  45         ci  46         ci  47         ci  48

                ci  49         ci  50

          tciref  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7         ci   8
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676     0.00000000     0.00000000     0.00000000     0.00000000
 ref:   2     0.00000000     0.99170237     0.00000000     0.00000000     0.00000000     0.00000000     0.00000000     0.00000000

                ci   9         ci  10         ci  11         ci  12         ci  13         ci  14         ci  15         ci  16

                ci  17         ci  18         ci  19         ci  20         ci  21         ci  22         ci  23         ci  24

                ci  25         ci  26         ci  27         ci  28         ci  29         ci  30         ci  31         ci  32

                ci  33         ci  34         ci  35         ci  36         ci  37         ci  38         ci  39         ci  40

                ci  41         ci  42         ci  43         ci  44         ci  45         ci  46         ci  47         ci  48

                ci  49         ci  50
 computing final density
 =========== Executing IN-CORE method ==========
--------------------------------------------------------------------------------
  1e-density for root #    1
--------------------------------------------------------------------------------
executing partial task #   1  11
skipping partial task #   2  91
executing partial task #   1   1
skipping partial task #   2  81
executing partial task #   1   2
skipping partial task #   2  21
skipping partial task #   3 101
skipping partial task #   4  82
executing partial task #   1  71
executing partial task #   1  52
skipping partial task #   2  62
executing partial task #   3  72
================================================================================
   DYZ=       4  DYX=       0  DYW=       0
   D0Z=       0  D0Y=       0  D0X=       0  D0W=       0
  DDZI=       4 DDYI=       2 DDXI=       0 DDWI=       0
  DDZE=       0 DDYE=       2 DDXE=       0 DDWE=       0
================================================================================
adding (  1) 0.00to den1(    1)=   1.9741836
adding (  2) 0.00to den1(    3)=   0.0125037
--------------------------------------------------------------------------------
  2e-density  for root #    1
--------------------------------------------------------------------------------
================================================================================
   DYZ=       4  DYX=       0  DYW=       0
   D0Z=       2  D0Y=       1  D0X=       0  D0W=       0
  DDZI=       5 DDYI=       2 DDXI=       0 DDWI=       0
  DDZE=       0 DDYE=       0 DDXE=       0 DDWE=       0
================================================================================
d2il(     1)=  1.96549382 rmuval=  0.00000000den1=  1.97418356 fact=  0.00000000
d2il(     5)=  0.00788067 rmuval=  0.00000000den1=  0.01250369 fact=  0.00000000
Trace of MO density:     2.000000
    2  correlated and     0  frozen core electrons

Natural orbital populations,block 1
              MO     1       MO     2       MO     3       MO     4       MO     5       MO     6       MO     7       MO     8
  occ(*)=     1.98283521     0.01577159     0.00135511     0.00003808     0.00000000    -0.00000000    -0.00000000    -0.00000000
              MO     9       MO    10       MO
  occ(*)=    -0.00000000    -0.00000000


 total number of electrons =    2.0000000000

 test slabel:                     1
  a  


          Mulliken population analysis


  NOTE: For HERMIT use spherical harmonics basis sets !!!
 

                        a   partial gross atomic populations
   ao class       1a         2a         3a         4a         5a         6a  
      _ s       0.969988   0.007891   0.000671   0.000012  -0.000000   0.000000
      _ p       0.021430  -0.000006   0.000007   0.000007   0.000000  -0.000000
     1_ s       0.969988   0.007891   0.000671   0.000012   0.000000   0.000000
     1_ p       0.021430  -0.000006   0.000007   0.000007   0.000000  -0.000000
 
   ao class       7a         8a         9a        10a  
      _ s      -0.000000  -0.000000   0.000000  -0.000000
      _ p      -0.000000  -0.000000  -0.000000  -0.000000
     1_ s      -0.000000  -0.000000   0.000000  -0.000000
     1_ p      -0.000000  -0.000000  -0.000000  -0.000000


                        gross atomic populations
     ao             _         1_
      s         0.978562   0.978562
      p         0.021438   0.021438
    total       1.000000   1.000000
 

 Total number of electrons:    2.00000000

 item #                     2 suffix=:.drt1.state2:
================================================================================
  Reading record                      1  of civout
 INFO:ref#  1vector#  1 method:  0 last record  0max overlap with ref# 99% root-following 0
 MR-CISD energy:    -1.13985847    -1.85361252
 residuum:     0.00000297
 deltae:     0.00000000
================================================================================
  Reading record                      2  of civout
 INFO:ref#  2vector#  2 method:  0 last record  1max overlap with ref# 99% root-following 0
 MR-CISD energy:    -0.64570004    -1.35945409
 residuum:     0.00004788
 deltae:     0.00000013
 apxde:     0.00000000

          sovref  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7         ci   8
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676     0.00000000    -0.00000000     0.00000000     0.00000000
 ref:   2     0.00000000     0.99170237     0.00000000     0.00000000     0.09149769    -0.08129806     0.03920567     0.00000000

                ci   9         ci  10         ci  11         ci  12         ci  13         ci  14         ci  15         ci  16

                ci  17         ci  18         ci  19         ci  20         ci  21         ci  22         ci  23         ci  24

                ci  25         ci  26         ci  27         ci  28         ci  29         ci  30         ci  31         ci  32

                ci  33         ci  34         ci  35         ci  36         ci  37         ci  38         ci  39         ci  40

                ci  41         ci  42         ci  43         ci  44         ci  45         ci  46         ci  47         ci  48

                ci  49         ci  50

          tciref  block   1

                ci   1         ci   2         ci   3         ci   4         ci   5         ci   6         ci   7         ci   8
 ref:   1     0.99317051    -0.00000000    -0.08418700    -0.04512676     0.00000000     0.00000000     0.00000000     0.00000000
 ref:   2     0.00000000     0.99170237     0.00000000     0.00000000     0.00000000     0.00000000     0.00000000     0.00000000

                ci   9         ci  10         ci  11         ci  12         ci  13         ci  14         ci  15         ci  16

                ci  17         ci  18         ci  19         ci  20         ci  21         ci  22         ci  23         ci  24

                ci  25         ci  26         ci  27         ci  28         ci  29         ci  30         ci  31         ci  32

                ci  33         ci  34         ci  35         ci  36         ci  37         ci  38         ci  39         ci  40

                ci  41         ci  42         ci  43         ci  44         ci  45         ci  46         ci  47         ci  48

                ci  49         ci  50
 computing final density
 =========== Executing IN-CORE method ==========
--------------------------------------------------------------------------------
  1e-density for root #    2
--------------------------------------------------------------------------------
executing partial task #   1  11
skipping partial task #   2  91
executing partial task #   1   1
skipping partial task #   2  81
executing partial task #   1   2
skipping partial task #   2  21
skipping partial task #   3 101
skipping partial task #   4  82
executing partial task #   1  71
executing partial task #   1  52
skipping partial task #   2  62
executing partial task #   3  72
================================================================================
   DYZ=       4  DYX=       0  DYW=       0
   D0Z=       0  D0Y=       0  D0X=       0  D0W=       0
  DDZI=       4 DDYI=       2 DDXI=       0 DDWI=       0
  DDZE=       0 DDYE=       2 DDXE=       0 DDWE=       0
================================================================================
adding (  1) 0.00to den1(    1)=   0.9839263
adding (  2) 0.00to den1(    3)=   0.9995473
--------------------------------------------------------------------------------
  2e-density  for root #    2
--------------------------------------------------------------------------------
================================================================================
   DYZ=       4  DYX=       0  DYW=       0
   D0Z=       2  D0Y=       1  D0X=       0  D0W=       0
  DDZI=       5 DDYI=       2 DDXI=       0 DDWI=       0
  DDZE=       0 DDYE=       0 DDXE=       0 DDWE=       0
================================================================================
d2il(     1)=  0.00000000 rmuval=  0.00000000den1=  0.98392626 fact=  0.00000000
d2il(     5)=  0.00000000 rmuval=  0.00000000den1=  0.99954733 fact=  0.00000000
Trace of MO density:     2.000000
    2  correlated and     0  frozen core electrons

Natural orbital populations,block 1
              MO     1       MO     2       MO     3       MO     4       MO     5       MO     6       MO     7       MO     8
  occ(*)=     0.99999272     0.99999272     0.00000728     0.00000728     0.00000000     0.00000000     0.00000000    -0.00000000
              MO     9       MO    10       MO
  occ(*)=    -0.00000000    -0.00000000


 total number of electrons =    2.0000000000

 test slabel:                     1
  a  


          Mulliken population analysis


  NOTE: For HERMIT use spherical harmonics basis sets !!!
 

                        a   partial gross atomic populations
   ao class       1a         2a         3a         4a         5a         6a  
      _ s       0.489189   0.500353   0.000004   0.000002  -0.000000  -0.000000
      _ p       0.010808  -0.000357   0.000000   0.000001   0.000000   0.000000
     1_ s       0.489189   0.500353   0.000004   0.000002   0.000000  -0.000000
     1_ p       0.010808  -0.000357   0.000000   0.000001   0.000000   0.000000
 
   ao class       7a         8a         9a        10a  
      _ s       0.000000  -0.000000   0.000000  -0.000000
      _ p       0.000000  -0.000000  -0.000000  -0.000000
     1_ s       0.000000  -0.000000   0.000000  -0.000000
     1_ p       0.000000  -0.000000  -0.000000  -0.000000


                        gross atomic populations
     ao             _         1_
      s         0.989548   0.989548
      p         0.010452   0.010452
    total       1.000000   1.000000
 

 Total number of electrons:    2.00000000

 accstate=                     3
 accpdens=                     3
logrecs(*)=   1   2logvrecs(*)=   1   2
 item #                     3 suffix=:.trd1to2:
 computing final density
 =========== Executing IN-CORE method ==========
1e-transition density (   1,   2)
--------------------------------------------------------------------------------
  1e-transition density (root #    1 -> root #   2)
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
  1e-transition density (root #    1 -> root #   2)
--------------------------------------------------------------------------------
executing partial task #   1  11
skipping partial task #   2  91
executing partial task #   1   1
skipping partial task #   2  81
executing partial task #   1   2
skipping partial task #   2  21
skipping partial task #   3 101
skipping partial task #   4  82
executing partial task #   1  71
executing partial task #   1  52
skipping partial task #   2  62
executing partial task #   3  72
================================================================================
   DYZ=       4  DYX=       0  DYW=       0
   D0Z=       0  D0Y=       0  D0X=       0  D0W=       0
  DDZI=       4 DDYI=       2 DDXI=       0 DDWI=       0
  DDZE=       0 DDYE=       2 DDXE=       0 DDWE=       0
================================================================================
adding (  1) 0.00to den1(    1)=  -0.0000000
adding (  2) 0.00to den1(    3)=  -0.0000000
2e-transition density (   1,   2)
--------------------------------------------------------------------------------
  2e-transition density (root #    1 -> root #   2)
--------------------------------------------------------------------------------
================================================================================
   DYZ=       4  DYX=       0  DYW=       0
   D0Z=       2  D0Y=       1  D0X=       0  D0W=       0
  DDZI=       5 DDYI=       2 DDXI=       0 DDWI=       0
  DDZE=       0 DDYE=       0 DDXE=       0 DDWE=       0
================================================================================
d2il(     1)=  0.00000000 rmuval=  0.00000000den1= -0.00000000 fact=  0.00000000
d2il(     5)= -0.00000000 rmuval=  0.00000000den1= -0.00000000 fact=  0.00000000
 maximum diagonal element=  0.000000000000000E+000
========================GLOBAL TIMING PER NODE========================
   process    vectrn  integral   segment    diagon     dmain    davidc   finalvw   driver  
--------------------------------------------------------------------------------
   #   1         0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
--------------------------------------------------------------------------------
       1         0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
 DA ...
