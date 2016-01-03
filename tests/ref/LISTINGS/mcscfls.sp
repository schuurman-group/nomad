

     ******************************************
     **    PROGRAM:              MCSCF       **
     **    PROGRAM VERSION:      5.5         **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************

 This program allows the csf mixing coefficient and orbital expansion coefficient
 optimization using the graphical unitary group approach and the exponential
 operator mcscf method.
 references:  r. shepard and j. simons, ' int. j. quantum chem. symp. 14, 211 (1980).
              r. shepard, i. shavitt, and j. simons, j. chem. phys. 76, 543 (1982).
              r. shepard in "ab initio methods in quantum chemistry ii" advances in chemical
                  physics 69, edited by k. p. lawley (wiley, new york, 1987) pp. 63-200.
 Original autor: Ron Shepard, ANL
 Later revisions: Michal Dallos, University Vienna

 This Version of Program MCSCF is Maintained by:
     Thomas Mueller
     Juelich Supercomputing Centre (JSC)
     Institute of Advanced Simulation (IAS)
     D-52425 Juelich, Germany 
     Email: th.mueller@fz-juelich.de



     ******************************************
     **    PROGRAM:              MCSCF       **
     **    PROGRAM VERSION:      5.4.0.2     **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************

 Workspace allocation information:
        65536000 of real*8 words (  500.00 MB) of work space has been allocated.

 user input information:

 ======== echo of the mcscf input ========
 ------------------------------------------------------------------------
  &input
   niter=100,
   nmiter=50,
   nciitr=50,
   tol(3)=1.e-5,
   tol(2)=1.e-5,
   tol(1)=1.e-9,
   NSTATE=0,
   npath=1,3,9,10,13,17,19,21,11,-12, 2,
   ncoupl=5,
   tol(9)=1.e-3,
   FCIORB=  1,1,20,1,2,20
   NAVST(1) = 2,
   WAVST(1,1)=1 ,
   WAVST(1,2)=1 ,
  &end
 ------------------------------------------------------------------------


 ***  Integral file informations  ***


 input integral file : /tmp/schuurm/COLSCR/WORK/aoints                          
    

 Integral file header information:
 Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16

 Core type energy values:
 energy( 1)=  7.137540490910E-01, ietype=   -1,    core energy of type: Nuc.Rep.
 total ao core energy =    0.713754049


   ******  Basis set information:  ******

 Number of irreps:                  1
 Total number of basis functions:  10

 irrep no.              1
 irrep label           A  
 no. of bas.fcions.    10


 ***  MCSCF optimization procedure parmeters:  ***


 maximum number of mcscf iterations:        niter=   100

 maximum number of psci micro-iterations:   nmiter=   50
 maximum r,s subspace dimension allowed:    nvrsmx=   30

 tol(1)=  1.0000E-09. . . . delta-emc convergence criterion.
 tol(2)=  1.0000E-05. . . . wnorm convergence criterion.
 tol(3)=  1.0000E-05. . . . knorm convergence criterion.
 tol(4)=  1.0000E-08. . . . apxde convergence criterion.
 tol(5)=  1.0000E-04. . . . small diagonal matrix element tolerance.
 tol(6)=  1.0000E-06. . . . minimum ci-psci residual norm.
 tol(7)=  1.0000E-05. . . . maximum ci-psci residual norm.
 tol(8)=  1.0000E+00. . . . maximum abs(k(xy)) allowed.
 tol(9)=  1.0000E-03. . . . wnorm coupling tolerance.
 tol(10)= 0.0000E+00. . . . maximum psci emergency shift parameter.
 tol(11)= 0.0000E+00. . . . minimum psci emergency shift parameter.
 tol(12)= 0.0000E+00. . . . increment of psci emergency shift parameter.


 *** State averaging informations: ***


 MCSCF calculation performed for  1 DRT.

 DRT  first state   no.of aver.states   weights
  1   ground state          2             0.500 0.500

 The number of hmc(*) eigenvalues and eigenvectors calculated each iteration per DRT:
 DRT.   no.of eigenv.(=ncol)
    1        3

 orbital coefficients are optimized for the ground state (nstate=0).

 Orbitals included in invariant subspaces:
   symmetry   orbital   mask
       1       1(  1)    20
       1       2(  2)    20

 npath(*) options:
  2:  orbital-state coupling terms will be included beginning on iteration ncoupl=  5
  3:  print intermediate timing information.
  9:  suppress the drt listing.
 10:  suppress the hmc(*) eigenvector listing.
 11:  construct the hmc(*) matrix explicitly.
 13:  get initial orbitals from the formatted file, mocoef.
 17:  print the final natural orbitals and occupations.
 19:  transform the virtual orbitals to diagonalize qvv(*).
 21:  write out the one- and two- electron density for further use (files:mcd1fl, mcd2fl).


   ******  DRT info section  ******


 Informations for the DRT no.  1

 DRT file header:
  title                                                                          
 Molecular symmetry group:    a  
 Total number of electrons:    2
 Spin multiplicity:            1
 Number of active orbitals:    2
 Number of active electrons:   2
 Total number of CSFs:         3
 

 faar:   0 active-active rotations allowed out of:   1 possible.


 Number of active-double rotations:         0
 Number of active-active rotations:         0
 Number of double-virtual rotations:        0
 Number of active-virtual rotations:       16

 Size of orbital-Hessian matrix B:                      192
 Size of the orbital-state Hessian matrix C:             96
 Total size of the state Hessian matrix M:               21
 Size of HESSIAN-matrix for quadratic conv.:            309


 Source of the initial MO coeficients:

 Input MO coefficient file: /tmp/schuurm/COLSCR/WORK/mocoef                             
 

               starting mcscf iteration...   1

 orbital-state coupling will not be calculated this iteration.

 *** Starting integral transformation ***

 module tranlib input parameters:

 prnopt    =     1, chkopt    =     0,ortopt    =     0, denopt    =     0
 mapin(1 ) =     1, nsymao    =     1, naopsy(1) =    10, freeze(1) =     1
 mapout(1) =     1, nsymmo    =    -1, nmopsy(1) =    -1, fsplit    =     1
 outlab    =     0, seward    =     0, lumorb    =     0, DALTON2   =     0
 nextint   =     2
 LDAMIN    =   127, LDAMAX    = 16383, LDAINC    =    64
 LRC1MX    =    -1, LRC2MX    =    -1, LRCSCR    = 32768

 THRESH    =  5.0000E-12  [cutoff threshold]

 module tranlib: workspace lcore=  65535509

 inoutp: segmentation information:
 in-core transformation space,   avcinc =  65434181
 address segment size,           sizesg =  65428725
 number of in-core blocks,       nincbk =         1
 number of out-of-core blocks,   noutbk =         0
 number of in-core segments,     incseg =         1
 number of out-of-core segments, outseg =         0
 trmain:         74 transformed 1/r12    array elements were written in       1 records.


 Size of orbital-Hessian matrix B:                      192
 Total size of the state Hessian matrix M:               21
 Total size of HESSIAN-matrix for linear con            213


 mosort: allocated sort2 space, avc2is=    65397359 available sort2 space, avcisx=    65397611

 Eigenvalues of the hmc(*) matrix
             total energy     electronic energy
    1*       -1.1314597606       -1.8452138097
    2        -0.6118835508       -1.3256375999
    3         0.0193804657       -0.6943735833
 
  tol(10)=  0.000000000000000E+000  eshsci=  2.315411918728940E-002
 Total number of micro iterations:    4

 ***  micro: final psci convergence values:  ***
    imxov=  1 z0= 0.99827286 pnorm= 0.0000E+00 rznorm= 2.2596E-14 rpnorm= 0.0000E+00 noldr=  4 nnewr=  4 nolds=  0 nnews=  0
 

 qvv(*) eigenvalues. symmetry block  1
     0.848641    1.700127    2.371201    2.371201    3.703757    3.886470    3.886470    6.966198

 restrt: restart information saved on the restart file (unit= 13).

 not all mcscf convergence criteria are satisfied.
 iter=    1 emc=     -0.8716716557 demc= 8.7167E-01 wnorm= 1.8523E-01 knorm= 5.8748E-02 apxde= 5.1142E-03    *not conv.*     

               starting mcscf iteration...   2

 orbital-state coupling will not be calculated this iteration.

 *** Starting integral transformation ***

 module tranlib input parameters:

 prnopt    =     1, chkopt    =     0,ortopt    =     0, denopt    =     0
 mapin(1 ) =     1, nsymao    =     1, naopsy(1) =    10, freeze(1) =     1
 mapout(1) =     1, nsymmo    =    -1, nmopsy(1) =    -1, fsplit    =     1
 outlab    =     0, seward    =     0, lumorb    =     0, DALTON2   =     0
 nextint   =     2
 LDAMIN    =   127, LDAMAX    = 16383, LDAINC    =    64
 LRC1MX    =    -1, LRC2MX    =    -1, LRCSCR    = 32768

 THRESH    =  5.0000E-12  [cutoff threshold]

 module tranlib: workspace lcore=  65535509

 inoutp: segmentation information:
 in-core transformation space,   avcinc =  65434181
 address segment size,           sizesg =  65428725
 number of in-core blocks,       nincbk =         1
 number of out-of-core blocks,   noutbk =         0
 number of in-core segments,     incseg =         1
 number of out-of-core segments, outseg =         0
 trmain:         80 transformed 1/r12    array elements were written in       1 records.


 Size of orbital-Hessian matrix B:                      192
 Total size of the state Hessian matrix M:               21
 Total size of HESSIAN-matrix for linear con            213


 mosort: allocated sort2 space, avc2is=    65397359 available sort2 space, avcisx=    65397611

 Eigenvalues of the hmc(*) matrix
             total energy     electronic energy
    1*       -1.1242663645       -1.8380204136
    2        -0.6294495415       -1.3432035906
    3         0.0169443123       -0.6968097368
 
  tol(10)=  0.000000000000000E+000  eshsci=  3.839333238084363E-004
 Total number of micro iterations:    4

 ***  micro: final psci convergence values:  ***
    imxov=  1 z0= 0.99999927 pnorm= 0.0000E+00 rznorm= 1.4391E-17 rpnorm= 0.0000E+00 noldr=  4 nnewr=  4 nolds=  0 nnews=  0
 

 qvv(*) eigenvalues. symmetry block  1
     0.860376    1.736265    2.417560    2.417560    3.742877    3.923897    3.923897    7.015536

 restrt: restart information saved on the restart file (unit= 13).

 not all mcscf convergence criteria are satisfied.
 iter=    2 emc=     -0.8768579530 demc= 5.1863E-03 wnorm= 3.0715E-03 knorm= 1.2044E-03 apxde= 1.8349E-06    *not conv.*     

               starting mcscf iteration...   3

 orbital-state coupling will not be calculated this iteration.

 *** Starting integral transformation ***

 module tranlib input parameters:

 prnopt    =     1, chkopt    =     0,ortopt    =     0, denopt    =     0
 mapin(1 ) =     1, nsymao    =     1, naopsy(1) =    10, freeze(1) =     1
 mapout(1) =     1, nsymmo    =    -1, nmopsy(1) =    -1, fsplit    =     1
 outlab    =     0, seward    =     0, lumorb    =     0, DALTON2   =     0
 nextint   =     2
 LDAMIN    =   127, LDAMAX    = 16383, LDAINC    =    64
 LRC1MX    =    -1, LRC2MX    =    -1, LRCSCR    = 32768

 THRESH    =  5.0000E-12  [cutoff threshold]

 module tranlib: workspace lcore=  65535509

 inoutp: segmentation information:
 in-core transformation space,   avcinc =  65434181
 address segment size,           sizesg =  65428725
 number of in-core blocks,       nincbk =         1
 number of out-of-core blocks,   noutbk =         0
 number of in-core segments,     incseg =         1
 number of out-of-core segments, outseg =         0
 trmain:         80 transformed 1/r12    array elements were written in       1 records.


 Size of orbital-Hessian matrix B:                      192
 Total size of the state Hessian matrix M:               21
 Total size of HESSIAN-matrix for linear con            213


 mosort: allocated sort2 space, avc2is=    65397359 available sort2 space, avcisx=    65397611

 Eigenvalues of the hmc(*) matrix
             total energy     electronic energy
    1*       -1.1240143019       -1.8377683510
    2        -0.6297052788       -1.3434593278
    3         0.0169844872       -0.6967695619
 
  tol(10)=  0.000000000000000E+000  eshsci=  1.418241904050070E-006
 Total number of micro iterations:    3

 ***  micro: final psci convergence values:  ***
    imxov=  1 z0= 1.00000000 pnorm= 0.0000E+00 rznorm= 1.1322E-08 rpnorm= 0.0000E+00 noldr=  3 nnewr=  3 nolds=  0 nnews=  0
 

 qvv(*) eigenvalues. symmetry block  1
     0.860445    1.736909    2.418413    2.418413    3.743655    3.924606    3.924606    7.016501

 restrt: restart information saved on the restart file (unit= 13).

 not all mcscf convergence criteria are satisfied.
 iter=    3 emc=     -0.8768597903 demc= 1.8374E-06 wnorm= 1.1346E-05 knorm= 1.4085E-05 apxde= 7.0588E-11    *not conv.*     

               starting mcscf iteration...   4

 orbital-state coupling will not be calculated this iteration.

 *** Starting integral transformation ***

 module tranlib input parameters:

 prnopt    =     1, chkopt    =     0,ortopt    =     0, denopt    =     0
 mapin(1 ) =     1, nsymao    =     1, naopsy(1) =    10, freeze(1) =     1
 mapout(1) =     1, nsymmo    =    -1, nmopsy(1) =    -1, fsplit    =     1
 outlab    =     0, seward    =     0, lumorb    =     0, DALTON2   =     0
 nextint   =     2
 LDAMIN    =   127, LDAMAX    = 16383, LDAINC    =    64
 LRC1MX    =    -1, LRC2MX    =    -1, LRCSCR    = 32768

 THRESH    =  5.0000E-12  [cutoff threshold]

 module tranlib: workspace lcore=  65535509

 inoutp: segmentation information:
 in-core transformation space,   avcinc =  65434181
 address segment size,           sizesg =  65428725
 number of in-core blocks,       nincbk =         1
 number of out-of-core blocks,   noutbk =         0
 number of in-core segments,     incseg =         1
 number of out-of-core segments, outseg =         0
 trmain:         80 transformed 1/r12    array elements were written in       1 records.


 Size of orbital-Hessian matrix B:                      192
 Total size of the state Hessian matrix M:               21
 Total size of HESSIAN-matrix for linear con            213


 mosort: allocated sort2 space, avc2is=    65397359 available sort2 space, avcisx=    65397611

 Eigenvalues of the hmc(*) matrix
             total energy     electronic energy
    1*       -1.1240135870       -1.8377676361
    2        -0.6297059938       -1.3434600429
    3         0.0169931349       -0.6967609141
 
  tol(10)=  0.000000000000000E+000  eshsci=  5.315813591589944E-008
 Total number of micro iterations:    1

 ***  micro: final psci convergence values:  ***
    imxov=  1 z0= 1.00000000 pnorm= 0.0000E+00 rznorm= 4.8371E-07 rpnorm= 0.0000E+00 noldr=  1 nnewr=  1 nolds=  0 nnews=  0
 

 qvv(*) eigenvalues. symmetry block  1
     0.860443    1.736907    2.418411    2.418411    3.743654    3.924605    3.924605    7.016499

 restrt: restart information saved on the restart file (unit= 13).

 all mcscf convergence criteria are satisfied.

 final mcscf convergence values:
 iter=    4 emc=     -0.8768597904 demc= 7.3273E-11 wnorm= 4.2527E-07 knorm= 2.1985E-07 apxde= 4.6729E-14    *converged*     




   ---------Individual total energies for all states:----------
   DRT #1 state # 1 wt 0.500 total energy=       -1.124013587, rel. (eV)=   0.000000
   DRT #1 state # 2 wt 0.500 total energy=       -0.629705994, rel. (eV)=  13.450800
   ------------------------------------------------------------



          mcscf orbitals of the final iteration,  A   block   1

               MO    1        MO    2        MO    3        MO    4        MO    5        MO    6        MO    7        MO    8
   1H1s       0.73948113     0.23690070    -1.14210606     1.35230753    -0.00000000     0.00000000    -0.78168583     0.00000000
   2H1s      -0.23407717     1.82757540     1.30493673    -2.30274990     0.00000000    -0.00000000     0.61907286    -0.00000000
   3H1px     -0.00000000    -0.00000000    -0.00000000     0.00000000     0.04696745     0.57739371     0.00000000     0.99006569
   4H1py      0.00000000    -0.00000000    -0.00000000     0.00000000     0.57739371    -0.04696745     0.00000000    -0.00044169
   5H1pz      0.03608148     0.00090557    -0.01885109    -0.38408439     0.00000000    -0.00000000     0.72531279    -0.00000000
   6H2s       0.73948113    -0.23690070    -1.14210606    -1.35230753    -0.00000000     0.00000000    -0.78168583     0.00000000
   7H2s      -0.23407717    -1.82757540     1.30493673     2.30274990     0.00000000    -0.00000000     0.61907286    -0.00000000
   8H2px      0.00000000     0.00000000     0.00000000    -0.00000000     0.04696745     0.57739371    -0.00000000    -0.99006569
   9H2py     -0.00000000    -0.00000000     0.00000000    -0.00000000     0.57739371    -0.04696745    -0.00000000     0.00044169
  10H2pz     -0.03608148     0.00090557     0.01885109    -0.38408439    -0.00000000     0.00000000    -0.72531279     0.00000000

               MO    9        MO   10
   1H1s       0.00000000     4.53204177
   2H1s      -0.00000000    -2.18748413
   3H1px      0.00044169     0.00000000
   4H1py      0.99006569     0.00000000
   5H1pz     -0.00000000     2.02782579
   6H2s       0.00000000    -4.53204177
   7H2s      -0.00000000     2.18748413
   8H2px     -0.00044169    -0.00000000
   9H2py     -0.99006569     0.00000000
  10H2pz      0.00000000     2.02782579

          natural orbitals of the final iteration,block  1    -  A  

               MO    1        MO    2        MO    3        MO    4        MO    5        MO    6        MO    7        MO    8

  occ(*)=     1.49790284     0.50209716     0.00000000     0.00000000     0.00000000     0.00000000     0.00000000     0.00000000
 
   1H1s       0.73948113     0.23690070    -1.14210595     1.35230806     0.00000000    -0.00000000    -0.78168599     0.00000000
   2H1s      -0.23407717     1.82757540     1.30493665    -2.30275016    -0.00000000     0.00000000     0.61907304    -0.00000000
   3H1px     -0.00000000    -0.00000000     0.00000000     0.00000000    -0.57333444     0.08292805    -0.00000000    -0.62047729
   4H1py      0.00000000    -0.00000000     0.00000000     0.00000000     0.08292805     0.57333444    -0.00000000    -0.77151681
   5H1pz      0.03608148     0.00090557    -0.01885118    -0.38408415    -0.00000000     0.00000000     0.72531279    -0.00000000
   6H2s       0.73948113    -0.23690070    -1.14210595    -1.35230806     0.00000000    -0.00000000    -0.78168599     0.00000000
   7H2s      -0.23407717    -1.82757540     1.30493665     2.30275016    -0.00000000    -0.00000000     0.61907304    -0.00000000
   8H2px      0.00000000     0.00000000    -0.00000000    -0.00000000    -0.57333444     0.08292805     0.00000000     0.62047729
   9H2py     -0.00000000    -0.00000000    -0.00000000    -0.00000000     0.08292805     0.57333444     0.00000000     0.77151681
  10H2pz     -0.03608148     0.00090557     0.01885118    -0.38408415     0.00000000     0.00000000    -0.72531279     0.00000000

               MO    9        MO   10

  occ(*)=     0.00000000     0.00000000
 
   1H1s      -0.00000000     4.53204161
   2H1s       0.00000000    -2.18748386
   3H1px     -0.77151681     0.00000000
   4H1py      0.62047729     0.00000000
   5H1pz      0.00000000     2.02782583
   6H2s      -0.00000000    -4.53204161
   7H2s       0.00000000     2.18748386
   8H2px      0.77151681    -0.00000000
   9H2py     -0.62047729    -0.00000000
  10H2pz     -0.00000000     2.02782583
 d1(*), fmc(*), and qmc(*) written to the 1-particle density matrix file.
          4 d2(*) elements written to the 2-particle density matrix file: mcd2fl                                                      


          Mulliken population analysis


  NOTE: For HERMIT use spherical harmonics basis sets !!!
 

                        A   partial gross atomic populations
   ao class       1A         2A         3A         4A         5A         6A  
      _ s       0.732762   0.251228   0.000000   0.000000   0.000000   0.000000
      _ p       0.016189  -0.000179   0.000000   0.000000   0.000000   0.000000
     1_ s       0.732762   0.251228   0.000000   0.000000   0.000000   0.000000
     1_ p       0.016189  -0.000179   0.000000   0.000000   0.000000   0.000000
 
   ao class       7A         8A         9A        10A  


                        gross atomic populations
     ao             _         1_
      s         0.983990   0.983990
      p         0.016010   0.016010
    total       1.000000   1.000000
 

 Total number of electrons:    2.00000000

 !timer: mcscf                           cpu_time=     0.094 walltime=     0.094
