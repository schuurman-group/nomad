                                program "scfpq"
                            columbus program system
             restricted hartree-fock scf and two-configuration mcscf

                   programmed (in part) by russell m. pitzer
                            version date: 30-sep-00

 This Version of Program SCFPQ is Maintained by:
     Thomas Mueller
     Juelich Supercomputing Centre (JSC)
     Institute of Advanced Simulation (IAS)
     D-52425 Juelich, Germany 
     Email: th.mueller@fz-juelich.de



     ******************************************
     **    PROGRAM:              SCFPQ       **
     **    PROGRAM VERSION:      5.5         **
     **    DISTRIBUTION VERSION: 5.9.a       **
     ******************************************

 echo of the input file:
 ------------------------------------------------------------------------
 Default SCF Title
  -15   9   0  -1   0   8   0   0   0  40  20   1   1   0   0   0  -1   0  11   0
    1
    2
    0.90000   0.10000   0.05000   0.00000    0    0
  (4f18.12)
    1
 ------------------------------------------------------------------------

workspace parameters: lcore=  65536000
 mem1=          0 ifirst=          1

Default SCF Title                                                               

scf flags
 -15   9   0  -1   0   8   0   0   0  40  20   1   1   0   0   0  -1   0  11   0
   1
input orbital labels, i:bfnlab(i)=
   1:  1H1s     2:  2H1s     3:  3H1px    4:  4H1py    5:  5H1pz    6:  6H2s  
   7:  7H2s     8:  8H2px    9:  9H2py   10: 10H2pz 
bfn_to_center map(*), i:map(i)
   1:  1   2:  1   3:  1   4:  1   5:  1   6:  2   7:  2   8:  2   9:  2  10:  2
bfn_to_orbital_type map(*), i:map(i)
   1:  1   2:  1   3:  2   4:  2   5:  2   6:  1   7:  1   8:  2   9:  2  10:  2

Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16

core potentials used

normalization threshold = 10**(-20)
one- and two-electron energy convergence criterion = 10**(- 9)

nuclear repulsion energy =     0.7137540490910
 
 fock matrix built from AO INTEGRALS
                        ^^^^^^^^^^^^
  ao integrals in SIFS format 
 in-core processing switched off

DIIS SWITCHED ON (error vector is FDS-SDF)

total number of SCF basis functions:  10

Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16
Default SCF Title                                                               

                 55 s integrals
                 55 t integrals
                 55 v integrals
                 55 c integrals
 check on positive semi-definiteness of S passed .
                 55 h integrals

           starting vectors from diagonalization of one-electron terms

     occupied and virtual orbitals
Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16
Default SCF Title                                                               

                         A   molecular orbitals
      sym. orb.       1A          2A          3A          4A          5A  
   1A  ,  1H1s      0.852918   -0.883178   -0.997717    1.382424    0.000000
   2A  ,  2H1s     -0.371523   -0.713981    1.227165   -2.944649    0.000000
   3A  ,  3H1px     0.000000    0.000000    0.000000    0.000000    0.000000
   4A  ,  4H1py     0.000000    0.000000    0.000000    0.000000    0.579301
   5A  ,  5H1pz     0.048173    0.088629   -0.061336   -0.248573    0.000000
   6A  ,  6H2s      0.852919    0.883178   -0.997720   -1.382421    0.000000
   7A  ,  7H2s     -0.371524    0.713981    1.227171    2.944646    0.000000
   8A  ,  8H2px     0.000000    0.000000    0.000000    0.000000    0.000000
   9A  ,  9H2py     0.000000    0.000000    0.000000    0.000000    0.579301
  10A  , 10H2pz    -0.048174    0.088629    0.061335   -0.248573    0.000000

      orb. en.     -1.279377   -0.607164   -0.321162   -0.026697    0.117561

      occ. no.      2.000000    0.000000    0.000000    0.000000    0.000000

      sym. orb.       6A          7A          8A          9A         10A  
   1A  ,  1H1s      0.000000   -0.859860    0.000000    0.000000    4.442203
   2A  ,  2H1s      0.000000    0.704934    0.000000    0.000000   -2.060840
   3A  ,  3H1px     0.579301    0.000000    0.000000   -0.990066    0.000000
   4A  ,  4H1py     0.000000    0.000000   -0.990066    0.000000    0.000000
   5A  ,  5H1pz     0.000000    0.722256    0.000000    0.000000    2.046938
   6A  ,  6H2s      0.000000   -0.859860    0.000000    0.000000   -4.442202
   7A  ,  7H2s      0.000000    0.704935    0.000000    0.000000    2.060840
   8A  ,  8H2px     0.579301    0.000000    0.000000    0.990066    0.000000
   9A  ,  9H2py     0.000000    0.000000    0.990066    0.000000    0.000000
  10A  , 10H2pz     0.000000   -0.722256    0.000000    0.000000    2.046938

      orb. en.      0.117561    0.792645    0.903135    0.903135    2.341719

      occ. no.      0.000000    0.000000    0.000000    0.000000    0.000000


mo coefficients will be saved after each iteration

iteration       energy           one-electron energy       two-electron energy
         ttr          tmr           tnr

 total 2e-integrals processed:                   512
    1     -1.0748295125704         -2.5587538174523         0.77017025579087    
       0.9000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    2     -1.1122513287898         -2.5468802121302         0.72087483424934    
       0.9000       0.1000       0.5000E-01
diis info stored; extrapolation off
 total 2e-integrals processed:                   512
    3     -1.1236292004626         -2.5305937323530         0.69321048279940    
       0.9000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    4     -1.1232021279225         -2.5316052391688         0.69464906215521    
       0.8000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    5     -1.1248526227831         -2.5273587320610         0.68875206018692    
       0.8500       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    6     -1.1279908767442         -2.5133284623539         0.67158353651867    
       0.8000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    7     -1.1285953378795         -2.5061590741864         0.66380968721587    
       0.7000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    8     -1.1286983303270         -2.5029267840716         0.66047440465350    
       0.6000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
    9     -1.1287131190384         -2.5016189692424         0.65915180111299    
       0.5000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
   10     -1.1287148085612         -2.5011498117338         0.65868095408161    
       0.4000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
   11     -1.1287149505029         -2.5010052301440         0.65853623055000    
       0.3000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
   12     -1.1287149582573         -2.5009690806133         0.65850007326501    
       0.2000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
   13     -1.1287149584769         -2.5009625073266         0.65849349975859    
       0.1000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
   14     -1.1287149584788         -2.5009619095793         0.65849290200949    
       0.1000       0.1000       0.5000E-01
 total 2e-integrals processed:                   512
   15     -1.1287149584788         -2.5009618552021         0.65849284763231    
       0.1000       0.1000       0.5000E-01

calculation has *converged*
scf gradient information written to file vectgrd

     total energy =         -1.1287149585
     kinetic energy =        1.0959193161
     potential energy =     -2.2246342745
     virial theorem =        1.9709442653
     wavefunction norm =     1.0000000000

     occupied and virtual orbitals
Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16
Default SCF Title                                                               

                         A   molecular orbitals
      sym. orb.       1A          2A          3A          4A          5A  
   1A  ,  1H1s      0.686655    0.256117   -1.192142    1.297493    0.000000
   2A  ,  2H1s     -0.170088    1.826075    1.323969   -2.266801    0.000000
   3A  ,  3H1px     0.000000    0.000000    0.000000    0.000000    0.000000
   4A  ,  4H1py     0.000000    0.000000    0.000000    0.000000    0.579301
   5A  ,  5H1pz     0.022477    0.014199   -0.011521   -0.409030    0.000000
   6A  ,  6H2s      0.686655   -0.256117   -1.192142   -1.297493    0.000000
   7A  ,  7H2s     -0.170088   -1.826075    1.323969    2.266801    0.000000
   8A  ,  8H2px     0.000000    0.000000    0.000000    0.000000    0.000000
   9A  ,  9H2py     0.000000    0.000000    0.000000    0.000000    0.579301
  10A  , 10H2pz    -0.022477    0.014199    0.011521   -0.409030    0.000000

      orb. en.     -0.591988    0.197171    0.479658    0.936316    1.292748

      occ. no.      2.000000    0.000000    0.000000    0.000000    0.000000

      sym. orb.       6A          7A          8A          9A         10A  
   1A  ,  1H1s      0.000000   -0.754700    0.000000    0.000000    4.546996
   2A  ,  2H1s      0.000000    0.599233    0.000000    0.000000   -2.225947
   3A  ,  3H1px     0.579301    0.000000    0.000000   -0.990066    0.000000
   4A  ,  4H1py     0.000000    0.000000   -0.990066    0.000000    0.000000
   5A  ,  5H1pz     0.000000    0.726015    0.000000    0.000000    2.022892
   6A  ,  6H2s      0.000000   -0.754700    0.000000    0.000000   -4.546996
   7A  ,  7H2s      0.000000    0.599233    0.000000    0.000000    2.225947
   8A  ,  8H2px     0.579301    0.000000    0.000000    0.990066    0.000000
   9A  ,  9H2py     0.000000    0.000000    0.990066    0.000000    0.000000
  10A  , 10H2pz     0.000000   -0.726015    0.000000    0.000000    2.022892

      orb. en.      1.292748    1.955670    2.042569    2.042569    3.601889

      occ. no.      0.000000    0.000000    0.000000    0.000000    0.000000


     population analysis
Hermit Integral Program : SIFS version  localhost.localdo 22:20:22.805 02-Jan-16
Default SCF Title                                                               
  NOTE: For HERMIT use spherical harmonics basis sets !!!
 

                        A   partial gross atomic populations
   ao class       1A  
      5 s       0.987747
      5 p       0.012253
     10 s       0.987747
     10 p       0.012253


                        gross atomic populations
     ao             5         10
      s         0.987747   0.987747
      p         0.012253   0.012253
    total       1.000000   1.000000

 Total number of electrons:    2.00000000

