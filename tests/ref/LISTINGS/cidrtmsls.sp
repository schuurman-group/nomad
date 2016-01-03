 
 program cidrt 5.9  

 distinct row table construction, reference csf selection, and internal
 walk selection for multireference single- and double-excitation
configuration interaction.

 references:  r. shepard, i. shavitt, r. m. pitzer, d. c. comeau, m. pepper
                  h. lischka, p. g. szalay, r. ahlrichs, f. b. brown, and
                  j.-g. zhao, int. j. quantum chem. symp. 22, 149 (1988).
              h. lischka, r. shepard, f. b. brown, and i. shavitt,
                  int. j. quantum chem. symp. 15, 91 (1981).

 programmed by: Thomas Müller (FZ Juelich)
 based on the cidrt code for single DRTs.

 version date: 16-jul-04


 This Version of Program CIDRT-MS is Maintained by:
     Thomas Mueller
     Juelich Supercomputing Centre (JSC)
     Institute of Advanced Simulation (IAS)
     D-52425 Juelich, Germany 
     Email: th.mueller@fz-juelich.de



     ******************************************
     **    PROGRAM:              CIDRT-MS    **
     **    PROGRAM VERSION:      5.5         **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************


 workspace allocation parameters: lencor=  65536000 mem1=         0 ifirst=         1
 expanded "keystrokes" are being written to files:
                       
 cidrtky                                                                        
  
 
 number of DRTs to be constructed [  0]: DRT infos are being written to files:
 cidrtfl.1                                                                      
  
 starting with the input common to all drts ... 
******************** ALL   DRTs    ********************
 Spin-Orbit CI Calculation?(y,[n]) Spin-Free Calculation
 
 input the spin multiplicity [  0]: spin multiplicity, smult            :   1    singlet 
 input the total number of electrons [  0]: total number of electrons, nelt     :     2
 input the number of irreps (1:8) [  0]: point group dimension, nsym         :     1
 enter symmetry labels:(y,[n]) enter 1 labels (a4):
 enter symmetry label, default=   1
 symmetry labels: (symmetry, slabel)
 ( 1,  a  ) 
 input nmpsy(*):
 nmpsy(*)=        10
 
   symmetry block summary
 block(*)=         1
 slabel(*)=      a  
 nmpsy(*)=        10
 
 total molecular orbitals            :    10
 
 input the frozen core orbitals (sym(i),rmo(i),i=1,nfct):
 total frozen core orbitals, nfct    :     0
 no frozen core orbitals entered
 
 number of frozen core orbitals      :     0
 number of frozen core electrons     :     0
 number of internal electrons        :     2
 
 input the frozen virtual orbitals (sym(i),rmo(i),i=1,nfvt):
 total frozen virtual orbitals, nfvt :     0

 no frozen virtual orbitals entered
 
 input the internal orbitals (sym(i),rmo(i),i=1,niot):
 niot                                :     2
 
 modrt(*)=         1   2
 slabel(*)=      a   a  
 
 total number of orbitals            :    10
 number of frozen core orbitals      :     0
 number of frozen virtual orbitals   :     0
 number of internal orbitals         :     2
 number of external orbitals         :     8
 
 orbital-to-level mapping vector
 map(*)=           9  10   1   2   3   4   5   6   7   8
==================== START DRT#   1====================
================================================================================
  Input for DRT number                      1
 input the molecular spatial symmetry (irrep 1:nsym) [  0]: state spatial symmetry label        :  a  
 
 input the number of ref-csf doubly-occupied orbitals [  0]: (ref) doubly-occupied orbitals      :     0
 
 no. of internal orbitals            :     2
 no. of doubly-occ. (ref) orbitals   :     0
 no. active (ref) orbitals           :     2
 no. of active electrons             :     2
 
 input the active-orbital, active-electron occmnr(*):
   1  2
 input the active-orbital, active-electron occmxr(*):
   1  2
 
 actmo(*) =        1   2
 occmnr(*)=        0   2
 occmxr(*)=        2   2
 reference csf cumulative electron occupations:
 modrt(*)=         1   2
 occmnr(*)=        0   2
 occmxr(*)=        2   2
 
 input the active-orbital bminr(*):
   1  2
 input the active-orbital bmaxr(*):
   1  2
 reference csf b-value constraints:
 modrt(*)=         1   2
 bminr(*)=         0   0
 bmaxr(*)=         2   2
 input the active orbital smaskr(*):
   1  2
 modrt:smaskr=
   1:1111   2:1111
 
 input the maximum excitation level from the reference csfs [  2]: maximum excitation from ref. csfs:  :     1
 number of internal electrons:       :     2
 
 input the internal-orbital mrsdci occmin(*):
   1  2
 input the internal-orbital mrsdci occmax(*):
   1  2
 mrsdci csf cumulative electron occupations:
 modrt(*)=         1   2
 occmin(*)=        0   1
 occmax(*)=        2   2
 
 input the internal-orbital mrsdci bmin(*):
   1  2
 input the internal-orbital mrsdci bmax(*):
   1  2
 mrsdci b-value constraints:
 modrt(*)=         1   2
 bmin(*)=          0   0
 bmax(*)=          2   2
 
 input the internal-orbital smask(*):
   1  2
 modrt:smask=
   1:1111   2:1111
 
 internal orbital summary:
 block(*)=         1   1
 slabel(*)=      a   a  
 rmo(*)=           1   2
 modrt(*)=         1   2
 
 reference csf info:
 occmnr(*)=        0   2
 occmxr(*)=        2   2
 
 bminr(*)=         0   0
 bmaxr(*)=         2   2
 
 
 mrsdci csf info:
 occmin(*)=        0   1
 occmax(*)=        2   2
 
 bmin(*)=          0   0
 bmax(*)=          2   2
 
******************** END   DRT#   1********************
******************** ALL   DRTs    ********************
 
 impose generalized interacting space restrictions?(y,[n]) generalized interacting space restrictions will not be imposed.

 a priori removal of distinct rows:

 input the level, a, and b values for the vertices 
 to be removed (-1/ to end).

 input level, a, and b (-1/ to end):
 no vertices marked for removal

 number of rows in the drt :   8

 manual arc removal step:


 input the level, a, b, and step values 
 for the arcs to be removed (-1/ to end).

 input the level, a, b, and step (-1/ to end):
 remarc:   0 arcs removed out of   0 specified.

 xbarz=       3
 xbary=       2
 xbarx=       0
 xbarw=       1
        --------
 nwalk=       6
------------------------------------------------------------------------
Core address array: 
             1             2
------------------------------------------------------------------------
 input the range of drt levels to print (l1,l2):
 levprt(*)        -1   0
==================== START DRT#   1====================
  INPUT FOR DRT #                      1

 reference-csf selection step 1:
 total number of z-walks in the drt, nzwalk=       3

 input the list of allowed reference symmetries:
 allowed reference symmetries:             1
 allowed reference symmetry labels:      a  
 keep all of the z-walks as references?(y,[n]) all z-walks are initially deleted.
 
 generate walks while applying reference drt restrictions?([y],n) reference drt restrictions will be imposed on the z-walks.
 
 impose additional orbital-group occupation restrictions?(y,[n]) 
 apply primary reference occupation restrictions?(y,[n]) 
 manually select individual walks?(y,[n])
 step 1 reference csf selection complete.
        3 csfs initially selected from       3 total walks.

 beginning step-vector based selection.
 enter [internal_orbital_step_vector/disposition] pairs:

 enter internal orbital step vector, (-1/ to end):
   1  2

 step 2 reference csf selection complete.
        3 csfs currently selected from       3 total walks.

 beginning numerical walk based selection.
 enter positive walk numbers to add walks,
 negative walk numbers to delete walks, and zero to end:

 input reference walk number (0 to end) [  0]:
 numerical walk-number based selection complete.
        3 reference csfs selected from       3 total z-walks.
 
 input the reference occupations, mu(*):
 reference occupations:
 mu(*)=            0   0
 
------------------------------------------------------------------------
Core address array: 
             1             2            11            20            23

------------------------------------------------------------------------
      1: 30
      2: 12
      3: 03
 number of step vectors saved:      3

 exlimw: beginning excitation-based walk selection...

  number of valid internal walks of each symmetry:

       a  
      ----
 z       3
 y       2
 x       0
 w       0

 csfs grouped by internal walk symmetry:

       a  
      ----
 z       3
 y      16
 x       0
 w       0

 total csf counts:
 z-vertex:        3
 y-vertex:       16
 x-vertex:        0
 w-vertex:        0
           --------
 total:          19
******************** END   DRT#   1********************
******************** ALL   DRTs    ********************
 Generating 3-DRT at exlimw level ... 

 xbarz=       3
 xbary=       2
 xbarx=       0
 xbarw=       0
        --------
 nwalk=       5
 levprt(*)        -1   0
------------------------------------------------------------------------
Core address array: 
             1             2            11            20            23
            26
------------------------------------------------------------------------
==================== START DRT#   1====================
 Converting reference vector for DRT #                     1

 beginning the reference csf index recomputation...

     iref   iwalk  step-vector
   ------  ------  ------------
        1       1  30
        2       2  12
        3       3  03
 indx01:     3 elements set in vec01(*)
******************** END   DRT#   1********************
******************** ALL   DRTs    ********************
==================== START DRT#   1====================
------------------------------------------------------------------------
Core address array: 
             1             2            11            20            23

------------------------------------------------------------------------
      1: 30
      2: 12
      3: 03
 number of step vectors saved:      3

 exlimw: beginning excitation-based walk selection...

  number of valid internal walks of each symmetry:

       a  
      ----
 z       3
 y       2
 x       0
 w       0

 csfs grouped by internal walk symmetry:

       a  
      ----
 z       3
 y      16
 x       0
 w       0

 total csf counts:
 z-vertex:        3
 y-vertex:       16
 x-vertex:        0
 w-vertex:        0
           --------
 total:          19

 final mrsdci walk selection step:

 nvalw(*)=       3       2       0       0 nvalwt=       5

 enter positive walk numbers to add walks,
 negative walk numbers to delete walks, and zero to end.

 input mrsdci walk number (0 to end) [  0]:
 end of manual mrsdci walk selection.
 number added=   0 number removed=   0

 nvalw(*)=       3       2       0       0 nvalwt=       5


 beginning the final csym(*) computation...

  number of valid internal walks of each symmetry:

       a  
      ----
 z       3
 y       2
 x       0
 w       0

 csfs grouped by internal walk symmetry:

       a  
      ----
 z       3
 y      16
 x       0
 w       0

 total csf counts:
 z-vertex:        3
 y-vertex:       16
 x-vertex:        0
 w-vertex:        0
           --------
 total:          19
 drt file is being written...
------------------------------------------------------------------------
Core address array: 
             1             2            11            20            25
            30          1630
------------------------------------------------------------------------
 
 input a title card, default=cidrt_title
 title card for DRT #                      1
  cidrt_title DRT#1                                                             
  
 drt file is being written... for DRT #                     1
 CI version:                     6
 wrtstr:  a  
nwalk=       5 cpos=       1 maxval=    9 cmprfactor=   80.00 %.
nwalk=       5 cpos=       1 maxval=   99 cmprfactor=   60.00 %.
 compressed with: nwalk=       5 cpos=       1 maxval=    9 cmprfactor=   80.00 %.
initial index vector length:         5
compressed index vector length:         1reduction:  80.00%
nwalk=       3 cpos=       1 maxval=    9 cmprfactor=   66.67 %.
nwalk=       3 cpos=       1 maxval=   99 cmprfactor=   33.33 %.
 compressed with: nwalk=       3 cpos=       1 maxval=    9 cmprfactor=   66.67 %.
initial ref vector length:         3
compressed ref vector length:         1reduction:  66.67%
******************** END   DRT#   1********************
