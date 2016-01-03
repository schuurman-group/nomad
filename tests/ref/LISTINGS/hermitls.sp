
 Work memory size (LMWORK) :    65536000 =         500 megabytes.

 Default basis set library used :
        /sphome/kedziora/dalton/basis/                              


    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    $$$$$$$$$$$  DALTON - An electronic structure program  $$$$$$$$$$$
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

               This is output from DALTON (beta-version 0.9) 

                          Principal authors:

            Trygve Helgaker,     University of Oslo,        Norway 
            Hans Joergen Jensen, University of Odense,      Denmark
            Poul Joergensen,     University of Aarhus,      Denmark
            Henrik Koch,         University of Aarhus,      Denmark
            Jeppe Olsen,         University of Lund,        Sweden 
            Hans Aagren,         University of Linkoeping,  Sweden 

                          Contributors:

            Torgeir Andersen,    University of Oslo,        Norway 
            Keld L. Bak,         University of Copenhagen,  Denmark
            Vebjoern Bakken,     University of Oslo,        Norway 
            Ove Christiansen,    University of Aarhus,      Denmark
            Paal Dahle,          University of Oslo,        Norway 
            Erik K. Dalskov,     University of Odense,      Denmark
            Thomas Enevoldsen,   University of Odense,      Denmark
            Asger Halkier,       University of Aarhus,      Denmark
            Hanne Heiberg,       University of Oslo,        Norway 
            Dan Jonsson,         University of Linkoeping,  Sweden 
            Sheela Kirpekar,     University of Odense,      Denmark
            Rika Kobayashi,      University of Aarhus,      Denmark
            Alfredo S. de Meras, Valencia University,       Spain  
            Kurt Mikkelsen,      University of Aarhus,      Denmark
            Patrick Norman,      University of Linkoeping,  Sweden 
            Martin J. Packer,    University of Sheffield,   UK     
            Kenneth Ruud,        University of Oslo,        Norway 
            Trond Saue,          University of Oslo,        Norway 
            Peter Taylor,        San Diego Superc. Center,  USA    
            Olav Vahtras,        University of Linkoeping,  Sweden

                                             Release Date:  August 1996
------------------------------------------------------------------------


      
     NOTE:
      
     This is an experimental code for the evaluation of molecular
     properties using (MC)SCF/CC wave functions. The authors accept
      no responsibility for the performance of the code or for the
     correctness of the results.
      
     The code (in whole or part) is not to be reproduced for further
     distribution without the written permission of T. Helgaker,
     H. J. Aa. Jensen or P. Taylor.
      
     If results obtained with this code are published, an
     appropriate citation would be:
      
     T. Helgaker, H. J. Aa. Jensen, P.Joergensen, H. Koch,
     J. Olsen, H. Aagren, T. Andersen, K. L. Bak, V. Bakken,
     O. Christiansen, P. Dahle, E. K. Dalskov, T. Enevoldsen,
     A. Halkier, H. Heiberg, D. Jonsson, S. Kirpekar, R. Kobayashi,
     A. S. de Meras, K. V. Mikkelsen, P. Norman, M. J. Packer,
     K. Ruud, T.Saue, P. R. Taylor, and O. Vahtras:
     DALTON, an electronic structure program"



     ******************************************
     **    PROGRAM:              DALTON      **
     **    PROGRAM VERSION:      5.4.0.0     **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************



 <<<<<<<<<< OUTPUT FROM GENERAL INPUT PROCESSING >>>>>>>>>>




 Default print level:        0

    Integral sections will be executed
    Starting in Integral Section -



 *************************************************************************
 ****************** Output from HERMIT input processing ******************
 *************************************************************************



 Default print level:        2


 Calculation of one- and two-electron Hamiltonian integrals.


 The following one-electron property integrals are calculated:

          - overlap integrals
          - Cartesian multipole moment integrals of orders 4 and lower
          - electronic angular momentum around the origin


 Changes of defaults for READIN:
 -------------------------------


 Maximum number of primitives per integral block :    4



 *************************************************************************
 ****************** Output from READIN input processing ******************
 *************************************************************************



  Title Cards
  -----------

                                                                          
                                                                          


                      SYMGRP:Point group information
                      ------------------------------

Point group: C1 

   * Character table

        |  E 
   -----+-----
    A   |   1

   * Direct product table

        | A  
   -----+-----
    A   | A  


  Atoms and basis sets
  --------------------

  Number of atom types:     1
  Total number of atoms:    2

  label    atoms   charge   prim    cont     basis   
  ----------------------------------------------------------------------
  H  1        1       1       7       5      [4s1p|2s1p]                            
  H  2        1       1       7       5      [4s1p|2s1p]                            
  ----------------------------------------------------------------------
  ----------------------------------------------------------------------
  total:      2       2      14      10

  Threshold for integrals:  1.00D-15


  Cartesian Coordinates
  ---------------------

  Total number of coordinates:  6


   1   H  1     x      0.0000000000
   2            y      0.0000000000
   3            z     -0.7005214200

   4   H  2     x      0.0000000000
   5            y      0.0000000000
   6            z      0.7005214200



   Interatomic separations (in Angstroms):
   ---------------------------------------

            H  1        H  2

   H  1    0.000000
   H  2    0.741400    0.000000




  Bond distances (angstroms):
  ---------------------------

                  atom 1     atom 2                           distance
                  ------     ------                           --------


  Nuclear repulsion energy :    0.713754049091


  Orbital exponents and contraction coefficients
  ----------------------------------------------


  H  1   1s    1       13.010000    0.0197  0.0000
   gen. cont.  2        1.962000    0.1380  0.0000
               3        0.444600    0.4781  0.0000
               4        0.122000    0.5012  1.0000

  H  1   2px   5        0.727000    1.0000

  H  1   2py   6        0.727000    1.0000

  H  1   2pz   7        0.727000    1.0000

  H  2   1s    8       13.010000    0.0197  0.0000
   gen. cont.  9        1.962000    0.1380  0.0000
              10        0.444600    0.4781  0.0000
              11        0.122000    0.5012  1.0000

  H  2   2px  12        0.727000    1.0000

  H  2   2py  13        0.727000    1.0000

  H  2   2pz  14        0.727000    1.0000


  Contracted Orbitals
  -------------------

   1  H  1    1s       1     2     3     4
   2  H  1    1s       4
   3  H  1    2px      5
   4  H  1    2py      6
   5  H  1    2pz      7
   6  H  2    1s       8     9    10    11
   7  H  2    1s      11
   8  H  2    2px     12
   9  H  2    2py     13
  10  H  2    2pz     14




  Symmetry Orbitals
  -----------------

  Number of orbitals in each symmetry:        10


  Symmetry  A  ( 1)

    1     H  1     1s         1
    2     H  1     1s         2
    3     H  1     2px        3
    4     H  1     2py        4
    5     H  1     2pz        5
    6     H  2     1s         6
    7     H  2     1s         7
    8     H  2     2px        8
    9     H  2     2py        9
   10     H  2     2pz       10

  Symmetries of electric field:  A  (1)  A  (1)  A  (1)

  Symmetries of magnetic field:  A  (1)  A  (1)  A  (1)


 Copy of input to READIN
 -----------------------

INTGRL                                                                          
                                                                                
                                                                                
s   1    0           0.10D-14                                                   
       1.0    2    2    1    1                                                  
H  1   0.000000000000000   0.000000000000000  -0.700521420000000       *        
H  2   0.000000000000000   0.000000000000000   0.700521420000000       *        
H   4   2                                                                       
         13.01000000         0.01968500         0.00000000                      
          1.96200000         0.13797700         0.00000000                      
          0.44460000         0.47814800         0.00000000                      
          0.12200000         0.50124000         1.00000000                      
H   1   1                                                                       
          0.72700000         1.00000000                                         




 ************************************************************************
 ************************** Output from HERONE **************************
 ************************************************************************

found      23 non-vanashing overlap integrals
found      27 non-vanashing nuclear attraction integrals
found      23 non-vanashing kinetic energy integrals






 found      10 non-vanashing integrals ( typea=  1 typeb=  0)
 found      10 non-vanashing integrals ( typea=  1 typeb=  1)
 found      22 non-vanashing integrals ( typea=  1 typeb=  2)


 found      23 non-vanashing integrals ( typea=  1 typeb=  3)
 found       4 non-vanashing integrals ( typea=  1 typeb=  4)
 found      12 non-vanashing integrals ( typea=  1 typeb=  5)
 found      23 non-vanashing integrals ( typea=  1 typeb=  6)
 found      12 non-vanashing integrals ( typea=  1 typeb=  7)
 found      27 non-vanashing integrals ( typea=  1 typeb=  8)


 found      10 non-vanashing integrals ( typea=  1 typeb=  9)
 found      10 non-vanashing integrals ( typea=  1 typeb= 10)
 found      22 non-vanashing integrals ( typea=  1 typeb= 11)
 found      10 non-vanashing integrals ( typea=  1 typeb= 12)
 found       2 non-vanashing integrals ( typea=  1 typeb= 13)
 found      12 non-vanashing integrals ( typea=  1 typeb= 14)
 found      10 non-vanashing integrals ( typea=  1 typeb= 15)
 found      22 non-vanashing integrals ( typea=  1 typeb= 16)
 found      12 non-vanashing integrals ( typea=  1 typeb= 17)
 found      22 non-vanashing integrals ( typea=  1 typeb= 18)


 found      23 non-vanashing integrals ( typea=  1 typeb= 19)
 found       4 non-vanashing integrals ( typea=  1 typeb= 20)
 found      12 non-vanashing integrals ( typea=  1 typeb= 21)
 found      23 non-vanashing integrals ( typea=  1 typeb= 22)
 found      12 non-vanashing integrals ( typea=  1 typeb= 23)
 found      27 non-vanashing integrals ( typea=  1 typeb= 24)
 found       4 non-vanashing integrals ( typea=  1 typeb= 25)
 found      12 non-vanashing integrals ( typea=  1 typeb= 26)
 found       4 non-vanashing integrals ( typea=  1 typeb= 27)
 found      12 non-vanashing integrals ( typea=  1 typeb= 28)
 found      23 non-vanashing integrals ( typea=  1 typeb= 29)
 found      12 non-vanashing integrals ( typea=  1 typeb= 30)
 found      27 non-vanashing integrals ( typea=  1 typeb= 31)
 found      12 non-vanashing integrals ( typea=  1 typeb= 32)
 found      27 non-vanashing integrals ( typea=  1 typeb= 33)


 found      12 non-vanashing integrals ( typea=  2 typeb=  6)
 found      12 non-vanashing integrals ( typea=  2 typeb=  7)
 found       4 non-vanashing integrals ( typea=  2 typeb=  8)




 ************************************************************************
 ************************** Output from TWOINT **************************
 ************************************************************************


 Number of two-electron integrals written:       512 (33.2%)
 Kilobytes written:                               44




 >>>> Total CPU  time used in HERMIT:**** hours 52 minutes 16 seconds
 >>>> Total wall time used in HERMIT:   0.00 seconds

- End of Integral Section
