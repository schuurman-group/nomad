1
 program ciudg      
 multireference single and double excitation configuration
 interaction based on the graphical unitary group approach.


 ************************************************************************
 beginning the bk-type iterative procedure (nzcsf=     3)...
 ************************************************************************

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  1  1     -1.1240135870  2.6645E-15  1.5751E-02  1.4577E-01  1.0000E-04
 mr-sdci #  1  2     -0.6297059938  2.8866E-15  0.0000E+00  1.3251E-01  1.0000E-04

 mr-sdci  convergence not reached after  1 iterations.

 final mr-sdci  convergence information:
 mr-sdci #  1  1     -1.1240135870  2.6645E-15  1.5751E-02  1.4577E-01  1.0000E-04
 mr-sdci #  1  2     -0.6297059938  2.8866E-15  0.0000E+00  1.3251E-01  1.0000E-04

 from bk iterations: iconv=   1

 ************************************************************************
 beginning the ci iterative diagonalization procedure... 
 ************************************************************************

   iter      root         energy      deltae       apxde    residual       rtol
        ---- ----   --------------  ----------  ----------  ----------  ----------
 mr-sdci #  1  1     -1.1394789537  1.5465E-02  3.9419E-04  2.0817E-02  1.0000E-04
 mr-sdci #  1  2     -0.6297059938  4.4409E-16  0.0000E+00  1.3251E-01  1.0000E-04
 mr-sdci #  2  1     -1.1398518344  3.7288E-04  6.4646E-06  2.8920E-03  1.0000E-04
 mr-sdci #  2  2     -0.6297059938 -4.4409E-16  0.0000E+00  1.3251E-01  1.0000E-04
 mr-sdci #  3  1     -1.1398584665  6.6321E-06  5.7288E-09  1.3722E-04  1.0000E-04
 mr-sdci #  3  2     -0.6297059938 -4.4409E-16  0.0000E+00  1.3251E-01  1.0000E-04
 mr-sdci #  4  1     -1.1398584719  5.3424E-09  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  4  2     -0.6297059938  4.4409E-16  1.5605E-02  1.3251E-01  1.0000E-04
 mr-sdci #  5  1     -1.1398584719  2.2204E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  5  2     -0.6455932359  1.5887E-02  1.1086E-04  1.1661E-02  1.0000E-04
 mr-sdci #  6  1     -1.1398584719  2.2204E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  6  2     -0.6456986628  1.0543E-04  2.0045E-06  1.1466E-03  1.0000E-04
 mr-sdci #  7  1     -1.1398584719 -2.2204E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  7  2     -0.6456999027  1.2399E-06  1.5943E-07  5.2440E-04  1.0000E-04
 mr-sdci #  8  1     -1.1398584719  4.4409E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  8  2     -0.6457000375  1.3475E-07  8.5939E-10  4.7882E-05  1.0000E-04

 mr-sdci  convergence criteria satisfied after  8 iterations.

 final mr-sdci  convergence information:
 mr-sdci #  8  1     -1.1398584719  4.4409E-16  0.0000E+00  2.9724E-06  1.0000E-04
 mr-sdci #  8  2     -0.6457000375  1.3475E-07  8.5939E-10  4.7882E-05  1.0000E-04

 number of reference csfs (nref) is     3.  root number (iroot) is  1.
 c0**2 =   0.98668724  c**2 (all zwalks) =   0.98668724

 eref      =     -1.123667156809   "relaxed" cnot**2         =   0.986687242594
 eci       =     -1.139858471882   deltae = eci - eref       =  -0.016191315074
 eci+dv1   =     -1.140074022932   dv1 = (1-cnot**2)*deltae  =  -0.000215551050
 eci+dv2   =     -1.140076931228   dv2 = dv1 / cnot**2       =  -0.000218459346
 eci+dv3   =     -1.140079919078   dv3 = dv1 / (2*cnot**2-1) =  -0.000221447195
 eci+pople =     -1.139858471882   (  2e- scaled deltae )    =  -0.016191315074

 number of reference csfs (nref) is     3.  root number (iroot) is  2.
 c0**2 =   0.98347359  c**2 (all zwalks) =   0.98347359

 eref      =     -0.629705993795   "relaxed" cnot**2         =   0.983473587958
 eci       =     -0.645700037473   deltae = eci - eref       =  -0.015994043679
 eci+dv1   =     -0.645964361629   dv1 = (1-cnot**2)*deltae  =  -0.000264324156
 eci+dv2   =     -0.645968803365   dv2 = dv1 / cnot**2       =  -0.000268765892
 eci+dv3   =     -0.645973396931   dv3 = dv1 / (2*cnot**2-1) =  -0.000273359458
 eci+pople =     -0.645700037473   (  2e- scaled deltae )    =  -0.015994043679
