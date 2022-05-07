module quadpack 
 use accuracy
 implicit none

 public dqagse
 public dqagie

 contains

      subroutine dqagse(f,a,b,epsabs,epsrel,limit,result,abserr,neval,ier,alist,blist,rlist,elist,iord,last)
!*********************************************************************72
!
!c DQAGSE estimates the integral of a function.
!
!***begin prologue  dqagse
!***date written   800101   (yymmdd)
!***revision date  830518   (yymmdd)
!***category no.  h2a1a1
!***keywords  automatic integrator, general-purpose,
!             (end point) singularities, extrapolation,
!             globally adaptive
!***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
!           de doncker,elise,appl. math. & progr. div. - k.u.leuven
!***purpose  the routine calculates an approximation result to a given
!            definite integral i = integral of f over (a,b),
!            hopefully satisfying following claim for accuracy
!            abs(i-result).le.max(epsabs,epsrel*abs(i)).
!***description
!
!        computation of a definite integral
!        standard fortran subroutine
!        double precision version
!
!        parameters
!         on entry
!            f      - double precision
!                     function subprogram defining the integrand
!                     function f(x). the actual name for f needs to be
!                     declared e x t e r n a l in the driver program.
!
!            a      - double precision
!                     lower limit of integration
!
!            b      - double precision
!                     upper limit of integration
!
!            epsabs - double precision
!                     absolute accuracy requested
!            epsrel - double precision
!                     relative accuracy requested
!                     if  epsabs.le.0
!                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
!                     the routine will end with ier = 6.
!
!            limit  - integer
!                     gives an upperbound on the number of subintervals
!                     in the partition of (a,b)
!
!         on return
!            result - double precision
!                     approximation to the integral
!
!            abserr - double precision
!                     estimate of the modulus of the absolute error,
!                     which should equal or exceed abs(i-result)
!
!            neval  - integer
!                     number of integrand evaluations
!
!            ier    - integer
!                     ier = 0 normal and reliable termination of the
!                             routine. it is assumed that the requested
!                             accuracy has been achieved.
!                     ier.gt.0 abnormal termination of the routine
!                             the estimates for integral and error are
!                             less reliable. it is assumed that the
!                             requested accuracy has not been achieved.
!            error messages
!                         = 1 maximum number of subdivisions allowed
!                             has been achieved. one can allow more sub-
!                             divisions by increasing the value of limit
!                             (and taking the according dimension
!                             adjustments into account). however, if
!                             this yields no improvement it is advised
!                             to analyze the integrand in order to
!                             determine the integration difficulties. if
!                             the position of a local difficulty can be
!                             determined (e.g. singularity,
!                             discontinuity within the interval) one
!                             will probably gain from splitting up the
!                             interval at this point and calling the
!                             integrator on the subranges. if possible,
!                             an appropriate special-purpose integrator
!                             should be used, which is designed for
!                             handling the type of difficulty involved.
!                         = 2 the occurrence of roundoff error is detec-
!                             ted, which prevents the requested
!                             tolerance from being achieved.
!                             the error may be under-estimated.
!                         = 3 extremely bad integrand behaviour
!                             occurs at some points of the integration
!                             interval.
!                         = 4 the algorithm does not converge.
!                             roundoff error is detected in the
!                             extrapolation table.
!                             it is presumed that the requested
!                             tolerance cannot be achieved, and that the
!                             returned result is the best which can be
!                             obtained.
!                         = 5 the integral is probably divergent, or
!                             slowly convergent. it must be noted that
!                             divergence can occur with any other value
!                             of ier.
!                         = 6 the input is invalid, because
!                             epsabs.le.0 and
!                             epsrel.lt.max(50*rel.mach.acc.,0.5d-28).
!                             result, abserr, neval, last, rlist(1),
!                             iord(1) and elist(1) are set to zero.
!                             alist(1) and blist(1) are set to a and b
!                             respectively.
!
!            alist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the left end points
!                     of the subintervals in the partition of the
!                     given integration range (a,b)
!
!            blist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the right end points
!                     of the subintervals in the partition of the given
!                     integration range (a,b)
!
!            rlist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the integral
!                     approximations on the subintervals
!
!            elist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the moduli of the
!                     absolute error estimates on the subintervals
!
!            iord   - integer
!                     vector of dimension at least limit, the first k
!                     elements of which are pointers to the
!                     error estimates over the subintervals,
!                     such that elist(iord(1)), ..., elist(iord(k))
!                     form a decreasing sequence, with k = last
!                     if last.le.(limit/2+2), and k = limit+1-last
!                     otherwise
!
!            last   - integer
!                     number of subintervals actually produced in the
!                     subdivision process
!
!***references  (none)
!***routines called  d1mach,dqelg,dqk21,dqpsrt
!***end prologue  dqagse
!
      double precision a,abseps,abserr,alist,area,area1,area12,area2,a1, &
       a2,b,blist,b1,b2,correc,dabs,defabs,defab1,defab2,dmax1,  &
       dres,elist,epmach,epsabs,epsrel,erlarg,erlast,errbnd,errmax,     &
       error1,error2,erro12,errsum,ertest,f,oflow,resabs,reseps,result, &
       res3la,rlist,rlist2,small,uflow                 
      integer id,ier,ierro,iord,iroff1,iroff2,iroff3,jupbnd,k,ksgn,     &
       ktmin,last,limit,maxerr,neval,nres,nrmax,numrl2
      logical extrap,noext
!
      dimension alist(limit),blist(limit),elist(limit),iord(limit),    &
       res3la(3),rlist(limit),rlist2(52)
!
      external f
!
!            the dimension of rlist2 is determined by the value of
!            limexp in subroutine dqelg (rlist2 should be of dimension
!            (limexp+2) at least).
!
!            list of major variables
!            -----------------------
!
!           alist     - list of left end points of all subintervals
!                       considered up to now
!           blist     - list of right end points of all subintervals
!                       considered up to now
!           rlist(i)  - approximation to the integral over
!                       (alist(i),blist(i))
!           rlist2    - array of dimension at least limexp+2 containing
!                       the part of the epsilon table which is still
!                       needed for further computations
!           elist(i)  - error estimate applying to rlist(i)
!           maxerr    - pointer to the interval with largest error
!                       estimate
!           errmax    - elist(maxerr)
!           erlast    - error on the interval currently subdivided
!                       (before that subdivision has taken place)
!           area      - sum of the integrals over the subintervals
!           errsum    - sum of the errors over the subintervals
!           errbnd    - requested accuracy max(epsabs,epsrel*
!                       abs(result))
!           *****1    - variable for the left interval
!           *****2    - variable for the right interval
!           last      - index for subdivision
!           nres      - number of calls to the extrapolation routine
!           numrl2    - number of elements currently in rlist2. if an
!                       appropriate approximation to the compounded
!                       integral has been obtained it is put in
!                       rlist2(numrl2) after numrl2 has been increased
!                       by one.
!           small     - length of the smallest interval considered up
!                       to now, multiplied by 1.5
!           erlarg    - sum of the errors over the intervals larger
!                       than the smallest interval considered up to now
!           extrap    - logical variable denoting that the routine is
!                       attempting to perform extrapolation i.e. before
!                       subdividing the smallest interval we try to
!                       decrease the value of erlarg.
!           noext     - logical variable denoting that extrapolation
!                       is no longer allowed (true value)
!
!            machine dependent constants
!            ---------------------------
!
!           epmach is the largest relative spacing.
!           uflow is the smallest positive magnitude.
!           oflow is the largest positive magnitude.
!
!***first executable statement  dqagse
      epmach = d1mach(4)
!
!            test on validity of parameters
!            ------------------------------
      ier = 0
      neval = 0
      last = 0
      result = 0.0d+00
      abserr = 0.0d+00
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0d+00
      elist(1) = 0.0d+00
      if(epsabs.le.0.0d+00.and.epsrel.lt.dmax1(0.5d+02*epmach,0.5d-28))ier = 6
      if(ier.eq.6) go to 999
!
!           first approximation to the integral
!           -----------------------------------
!
      uflow = d1mach(1)
      oflow = d1mach(2)
      ierro = 0
      call dqk21(f,a,b,result,abserr,defabs,resabs)
!
!           test on accuracy.
!
      dres = dabs(result)
      errbnd = dmax1(epsabs,epsrel*dres)
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      if(abserr.le.1.0d+02*epmach*defabs.and.abserr.gt.errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs).or.abserr.eq.0.0d+00) go to 140
!
!           initialization
!           --------------
!
      rlist2(1) = result
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      abserr = oflow
      nrmax = 1
      nres = 0
      numrl2 = 2
      ktmin = 0
      extrap = .false.
      noext = .false.
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ksgn = -1
      if(dres.ge.(0.1d+01-0.5d+02*epmach)*defabs) ksgn = 1
!
!           main do-loop
!           ------------
!
      do 90 last = 2,limit
!
!           bisect the subinterval with the nrmax-th largest error
!           estimate.
!
        a1 = alist(maxerr)
        b1 = 0.5d+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call dqk21(f,a1,b1,area1,error1,resabs,defab1)
        call dqk21(f,a2,b2,area2,error2,resabs,defab2)
!
!           improve previous approximations to integral
!           and error and test for accuracy.
!
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2) go to 15
        if(dabs(rlist(maxerr)-area12).gt.0.1d-04*dabs(area12).or.erro12.lt.0.99d+00*errmax) go to 10
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   10   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   15   rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = dmax1(epsabs,epsrel*dabs(area))
!
!           test for roundoff error and eventually set error flag.
!
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
!
!           set error flag in the case that the number of subintervals
!           equals limit.
!
        if(last.eq.limit) ier = 1
!
!           set error flag in the case of bad integrand behaviour
!           at a point of the integration range.
!
        if(dmax1(dabs(a1),dabs(b2)).le.(0.1d+01+0.1d+03*epmach)*(dabs(a2)+0.1d+04*uflow)) ier = 4
!
!           append the newly-created intervals to the list.
!
        if(error2.gt.error1) go to 20
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 30
   20   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
!
!           call subroutine dqpsrt to maintain the descending ordering
!           in the list of error estimates and select the subinterval
!           with nrmax-th largest error estimate (to be bisected next).
!
   30   call dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
! ***jump out of do-loop
        if(errsum.le.errbnd) go to 115
! ***jump out of do-loop
        if(ier.ne.0) go to 100
        if(last.eq.2) go to 80
        if(noext) go to 90
        erlarg = erlarg-erlast
        if(dabs(b1-a1).gt.small) erlarg = erlarg+erro12
        if(extrap) go to 40
!
!           test whether the interval to be bisected next is the
!           smallest interval.
!
        if(dabs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
        extrap = .true.
        nrmax = 2
   40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
!
!           the smallest interval has the largest error.
!           before bisecting decrease the sum of the errors over the
!           larger intervals (erlarg) and perform extrapolation.
!
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 50 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
! ***jump out of do-loop
          if(dabs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
          nrmax = nrmax+1
   50   continue
!
!           perform extrapolation.
!
   60   numrl2 = numrl2+1
        rlist2(numrl2) = area
        call dqelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1d-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 70
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = dmax1(epsabs,epsrel*dabs(reseps))
! ***jump out of do-loop
        if(abserr.le.ertest) go to 100
!
!           prepare bisection of the smallest interval.
!
   70   if(numrl2.eq.1) noext = .true.
        if(ier.eq.5) go to 100
        maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        small = small*0.5d+00
        erlarg = errsum
        go to 90
   80   small = dabs(b-a)*0.375d+00
        erlarg = errsum
        ertest = errbnd
        rlist2(2) = area
   90 continue
!
!           set final result and error estimate.
!           ------------------------------------
!
  100 if(abserr.eq.oflow) go to 115
      if(ier+ierro.eq.0) go to 110
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0d+00.and.area.ne.0.0d+00) go to 105
      if(abserr.gt.errsum) go to 115
      if(area.eq.0.0d+00) go to 130
      go to 110
  105 if(abserr/dabs(result).gt.errsum/dabs(area)) go to 115
!
!           test on divergence.
!
  110 if(ksgn.eq.(-1).and.dmax1(dabs(result),dabs(area)).le.defabs*0.1d-01) go to 130
      if(0.1d-01.gt.(result/area).or.(result/area).gt.0.1d+03.or.errsum.gt.dabs(area)) ier = 6
      go to 130
!
!           compute global integral sum.
!
  115 result = 0.0d+00
      do 120 k = 1,last
         result = result+rlist(k)
  120 continue
      abserr = errsum
  130 if(ier.gt.2) ier = ier-1
  140 neval = 42*last-21
  999 return
      end

      subroutine dqagie(f,bound,inf,epsabs,epsrel,limit,result,abserr,neval,ier,alist,blist,rlist,elist,iord,last)

!*********************************************************************72
!
!c DQAGIE estimates an integral over a semi-infinite or infinite interval.
!
!***begin prologue  dqagie
!***date written   800101   (yymmdd)
!***revision date  830518   (yymmdd)
!***category no.  h2a3a1,h2a4a1
!***keywords  automatic integrator, infinite intervals,
!             general-purpose, transformation, extrapolation,
!             globally adaptive
!***author  piessens,robert,appl. math & progr. div - k.u.leuven
!           de doncker,elise,appl. math & progr. div - k.u.leuven
!***purpose  the routine calculates an approximation result to a given
!            integral   i = integral of f over (bound,+infinity)
!            or i = integral of f over (-infinity,bound)
!            or i = integral of f over (-infinity,+infinity),
!            hopefully satisfying following claim for accuracy
!            abs(i-result).le.max(epsabs,epsrel*abs(i))
!***description
!
! integration over infinite intervals
! standard fortran subroutine
!
!            f      - double precision
!                     function subprogram defining the integrand
!                     function f(x). the actual name for f needs to be
!                     declared e x t e r n a l in the driver program.
!
!            bound  - double precision
!                     finite bound of integration range
!                     (has no meaning if interval is doubly-infinite)
!
!            inf    - double precision
!                     indicating the kind of integration range involved
!                     inf = 1 corresponds to  (bound,+infinity),
!                     inf = -1            to  (-infinity,bound),
!                     inf = 2             to (-infinity,+infinity).
!
!            epsabs - double precision
!                     absolute accuracy requested
!            epsrel - double precision
!                     relative accuracy requested
!                     if  epsabs.le.0
!                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
!                     the routine will end with ier = 6.
!
!            limit  - integer
!                     gives an upper bound on the number of subintervals
!                     in the partition of (a,b), limit.ge.1
!
!         on return
!            result - double precision
!                     approximation to the integral
!
!            abserr - double precision
!                     estimate of the modulus of the absolute error,
!                     which should equal or exceed abs(i-result)
!
!            neval  - integer
!                     number of integrand evaluations
!
!            ier    - integer
!                     ier = 0 normal and reliable termination of the
!                             routine. it is assumed that the requested
!                             accuracy has been achieved.
!                   - ier.gt.0 abnormal termination of the routine. the
!                             estimates for result and error are less
!                             reliable. it is assumed that the requested
!                             accuracy has not been achieved.
!            error messages
!                     ier = 1 maximum number of subdivisions allowed
!                             has been achieved. one can allow more
!                             subdivisions by increasing the value of
!                             limit (and taking the according dimension
!                             adjustments into account). however,if
!                             this yields no improvement it is advised
!                             to analyze the integrand in order to
!                             determine the integration difficulties.
!                             if the position of a local difficulty can
!                             be determined (e.g. singularity,
!                             discontinuity within the interval) one
!                             will probably gain from splitting up the
!                             interval at this point and calling the
!                             integrator on the subranges. if possible,
!                             an appropriate special-purpose integrator
!                             should be used, which is designed for
!                             handling the type of difficulty involved.
!                         = 2 the occurrence of roundoff error is
!                             detected, which prevents the requested
!                             tolerance from being achieved.
!                             the error may be under-estimated.
!                         = 3 extremely bad integrand behaviour occurs
!                             at some points of the integration
!                             interval.
!                         = 4 the algorithm does not converge.
!                             roundoff error is detected in the
!                             extrapolation table.
!                             it is assumed that the requested tolerance
!                             cannot be achieved, and that the returned
!                             result is the best which can be obtained.
!                         = 5 the integral is probably divergent, or
!                             slowly convergent. it must be noted that
!                             divergence can occur with any other value
!                             of ier.
!                         = 6 the input is invalid, because
!                             (epsabs.le.0 and
!                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
!                             result, abserr, neval, last, rlist(1),
!                             elist(1) and iord(1) are set to zero.
!                             alist(1) and blist(1) are set to 0
!                             and 1 respectively.
!
!            alist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the left
!                     end points of the subintervals in the partition
!                     of the transformed integration range (0,1).
!
!            blist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the right
!                     end points of the subintervals in the partition
!                     of the transformed integration range (0,1).
!
!            rlist  - double precision
!                     vector of dimension at least limit, the first
!                      last  elements of which are the integral
!                     approximations on the subintervals
!
!            elist  - double precision
!                     vector of dimension at least limit,  the first
!                     last elements of which are the moduli of the
!                     absolute error estimates on the subintervals
!
!            iord   - integer
!                     vector of dimension limit, the first k
!                     elements of which are pointers to the
!                     error estimates over the subintervals,
!                     such that elist(iord(1)), ..., elist(iord(k))
!                     form a decreasing sequence, with k = last
!                     if last.le.(limit/2+2), and k = limit+1-last
!                     otherwise
!
!            last   - integer
!                     number of subintervals actually produced
!                     in the subdivision process
!
!***references  (none)
!***routines called  d1mach,dqelg,dqk15i,dqpsrt
!***end prologue  dqagie
      double precision abseps,abserr,alist,area,area1,area12,area2,a1,   &
       a2,blist,boun,bound,b1,b2,correc,dabs,defabs,defab1,defab2,      &
       dmax1,dres,elist,epmach,epsabs,epsrel,erlarg,erlast,      &
       errbnd,errmax,error1,error2,erro12,errsum,ertest,f,oflow,resabs, &
       reseps,result,res3la,rlist,rlist2,small,uflow                    
      integer id,ier,ierro,inf,iord,iroff1,iroff2,iroff3,jupbnd,k,ksgn,  &
       ktmin,last,limit,maxerr,neval,nres,nrmax,numrl2 
      logical extrap,noext
!
      dimension alist(limit),blist(limit),elist(limit),iord(limit),     &
       res3la(3),rlist(limit),rlist2(52)
!
      external f
!
!            the dimension of rlist2 is determined by the value of
!            limexp in subroutine dqelg.
!
!
!            list of major variables
!            -----------------------
!
!           alist     - list of left end points of all subintervals
!                       considered up to now
!           blist     - list of right end points of all subintervals
!                       considered up to now
!           rlist(i)  - approximation to the integral over
!                       (alist(i),blist(i))
!           rlist2    - array of dimension at least (limexp+2),
!                       containing the part of the epsilon table
!                       wich is still needed for further computations
!           elist(i)  - error estimate applying to rlist(i)
!           maxerr    - pointer to the interval with largest error
!                       estimate
!           errmax    - elist(maxerr)
!           erlast    - error on the interval currently subdivided
!                       (before that subdivision has taken place)
!           area      - sum of the integrals over the subintervals
!           errsum    - sum of the errors over the subintervals
!           errbnd    - requested accuracy max(epsabs,epsrel*
!                       abs(result))
!           *****1    - variable for the left subinterval
!           *****2    - variable for the right subinterval
!           last      - index for subdivision
!           nres      - number of calls to the extrapolation routine
!           numrl2    - number of elements currently in rlist2. if an
!                       appropriate approximation to the compounded
!                       integral has been obtained, it is put in
!                       rlist2(numrl2) after numrl2 has been increased
!                       by one.
!           small     - length of the smallest interval considered up
!                       to now, multiplied by 1.5
!           erlarg    - sum of the errors over the intervals larger
!                       than the smallest interval considered up to now
!           extrap    - logical variable denoting that the routine
!                       is attempting to perform extrapolation. i.e.
!                       before subdividing the smallest interval we
!                       try to decrease the value of erlarg.
!           noext     - logical variable denoting that extrapolation
!                       is no longer allowed (true-value)
!
!            machine dependent constants
!            ---------------------------
!
!           epmach is the largest relative spacing.
!           uflow is the smallest positive magnitude.
!           oflow is the largest positive magnitude.
!
!***first executable statement  dqagie
       epmach = d1mach(4)
!
!           test on validity of parameters
!           -----------------------------
!
      ier = 0
      neval = 0
      last = 0
      result = 0.0d+00
      abserr = 0.0d+00
      alist(1) = 0.0d+00
      blist(1) = 0.1d+01
      rlist(1) = 0.0d+00
      elist(1) = 0.0d+00
      iord(1) = 0
      if(epsabs.le.0.0d+00.and.epsrel.lt.dmax1(0.5d+02*epmach,0.5d-28))ier = 6
       if(ier.eq.6) go to 999
!
!
!           first approximation to the integral
!           -----------------------------------
!
!           determine the interval to be mapped onto (0,1).
!           if inf = 2 the integral is computed as i = i1+i2, where
!           i1 = integral of f over (-infinity,0),
!           i2 = integral of f over (0,+infinity).
!
      boun = bound
      if(inf.eq.2) boun = 0.0d+00
      call dqk15i(f,boun,inf,0.0d+00,0.1d+01,result,abserr,defabs,resabs)
!
!           test on accuracy
!
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      dres = dabs(result)
      errbnd = dmax1(epsabs,epsrel*dres)
      if(abserr.le.1.0d+02*epmach*defabs.and.abserr.gt.errbnd) ier = 2
      if(limit.eq.1) ier = 1
      if(ier.ne.0.or.(abserr.le.errbnd.and.abserr.ne.resabs).or.abserr.eq.0.0d+00) go to 130
!
!           initialization
!           --------------
!
      uflow = d1mach(1)
      oflow = d1mach(2)
      rlist2(1) = result
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      abserr = oflow
      nrmax = 1
      nres = 0
      ktmin = 0
      numrl2 = 2
      extrap = .false.
      noext = .false.
      ierro = 0
      iroff1 = 0
      iroff2 = 0
      iroff3 = 0
      ksgn = -1
      if(dres.ge.(0.1d+01-0.5d+02*epmach)*defabs) ksgn = 1
!
!           main do-loop
!           ------------
!
      do 90 last = 2,limit
!
!           bisect the subinterval with nrmax-th largest error estimate.
!
        a1 = alist(maxerr)
        b1 = 0.5d+00*(alist(maxerr)+blist(maxerr))
        a2 = b1
        b2 = blist(maxerr)
        erlast = errmax
        call dqk15i(f,boun,inf,a1,b1,area1,error1,resabs,defab1)
        call dqk15i(f,boun,inf,a2,b2,area2,error2,resabs,defab2)
!
!           improve previous approximations to integral
!           and error and test for accuracy.
!
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(defab1.eq.error1.or.defab2.eq.error2)go to 15
        if(dabs(rlist(maxerr)-area12).gt.0.1d-04*dabs(area12).or.erro12.lt.0.99d+00*errmax) go to 10
        if(extrap) iroff2 = iroff2+1
        if(.not.extrap) iroff1 = iroff1+1
   10   if(last.gt.10.and.erro12.gt.errmax) iroff3 = iroff3+1
   15   rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = dmax1(epsabs,epsrel*dabs(area))
!
!           test for roundoff error and eventually set error flag.
!
        if(iroff1+iroff2.ge.10.or.iroff3.ge.20) ier = 2
        if(iroff2.ge.5) ierro = 3
!
!           set error flag in the case that the number of
!           subintervals equals limit.
!
        if(last.eq.limit) ier = 1
!
!           set error flag in the case of bad integrand behaviour
!           at some points of the integration range.
!
        if(dmax1(dabs(a1),dabs(b2)).le.(0.1d+01+0.1d+03*epmach)*(dabs(a2)+0.1d+04*uflow)) ier = 4
!
!           append the newly-created intervals to the list.
!
        if(error2.gt.error1) go to 20
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 30
   20   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1
!
!           call subroutine dqpsrt to maintain the descending ordering
!           in the list of error estimates and select the subinterval
!           with nrmax-th largest error estimate (to be bisected next).
!
   30   call dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
        if(errsum.le.errbnd) go to 115
        if(ier.ne.0) go to 100
        if(last.eq.2) go to 80
        if(noext) go to 90
        erlarg = erlarg-erlast
        if(dabs(b1-a1).gt.small) erlarg = erlarg+erro12
        if(extrap) go to 40
!
!           test whether the interval to be bisected next is the
!           smallest interval.
!
        if(dabs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
        extrap = .true.
        nrmax = 2
   40   if(ierro.eq.3.or.erlarg.le.ertest) go to 60
!
!           the smallest interval has the largest error.
!           before bisecting decrease the sum of the errors over the
!           larger intervals (erlarg) and perform extrapolation.
!
        id = nrmax
        jupbnd = last
        if(last.gt.(2+limit/2)) jupbnd = limit+3-last
        do 50 k = id,jupbnd
          maxerr = iord(nrmax)
          errmax = elist(maxerr)
          if(dabs(blist(maxerr)-alist(maxerr)).gt.small) go to 90
          nrmax = nrmax+1
   50   continue
!
!           perform extrapolation.
!
   60   numrl2 = numrl2+1
        rlist2(numrl2) = area
        call dqelg(numrl2,rlist2,reseps,abseps,res3la,nres)
        ktmin = ktmin+1
        if(ktmin.gt.5.and.abserr.lt.0.1d-02*errsum) ier = 5
        if(abseps.ge.abserr) go to 70
        ktmin = 0
        abserr = abseps
        result = reseps
        correc = erlarg
        ertest = dmax1(epsabs,epsrel*dabs(reseps))
        if(abserr.le.ertest) go to 100
!
!            prepare bisection of the smallest interval.
!
   70   if(numrl2.eq.1) noext = .true.
        if(ier.eq.5) go to 100
        maxerr = iord(1)
        errmax = elist(maxerr)
        nrmax = 1
        extrap = .false.
        small = small*0.5d+00
        erlarg = errsum
        go to 90
   80   small = 0.375d+00
        erlarg = errsum
        ertest = errbnd
        rlist2(2) = area
   90 continue
!
!           set final result and error estimate.
!           ------------------------------------
!
  100 if(abserr.eq.oflow) go to 115
      if((ier+ierro).eq.0) go to 110
      if(ierro.eq.3) abserr = abserr+correc
      if(ier.eq.0) ier = 3
      if(result.ne.0.0d+00.and.area.ne.0.0d+00)go to 105
      if(abserr.gt.errsum)go to 115
      if(area.eq.0.0d+00) go to 130
      go to 110
  105 if(abserr/dabs(result).gt.errsum/dabs(area))go to 115
!
!           test on divergence
!
  110 if(ksgn.eq.(-1).and.dmax1(dabs(result),dabs(area)).le.defabs*0.1d-01) go to 130
      if(0.1d-01.gt.(result/area).or.(result/area).gt.0.1d+03.or.errsum.gt.dabs(area)) ier = 6
      go to 130
!
!           compute global integral sum.
!
  115 result = 0.0d+00
      do 120 k = 1,last
        result = result+rlist(k)
  120 continue
      abserr = errsum
  130 neval = 30*last-15
      if(inf.eq.2) neval = 2*neval
      if(ier.gt.2) ier=ier-1
  999 return
      end



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      function d1mach ( i )

!*********************************************************************72
!
!c D1MACH returns double precision machine-dependent constants.
!
!  Discussion:
!
!    D1MACH can be used to obtain machine-dependent parameters
!    for the local machine environment.  It is a function
!    with one input argument, and can be called as follows:
!
!      D = D1MACH ( I )
!
!    where I=1,...,5.  The output value of D above is
!    determined by the input value of I:.
!
!    D1MACH ( 1) = B^(EMIN-1), the smallest positive magnitude.
!    D1MACH ( 2) = B^EMAX*(1 - B^(-T)), the largest magnitude.
!    D1MACH ( 3) = B^(-T), the smallest relative spacing.
!    D1MACH ( 4) = B^(1-T), the largest relative spacing.
!    D1MACH ( 5) = LOG10(B)
!
!  Modified:
!
!    06 December 2006
!
!  Author:
!
!    Phyllis Fox, Andrew Hall, Norman Schryer
!
!  Reference:
!
!    Phyllis Fox, Andrew Hall, Norman Schryer,
!    Algorithm 528:
!    Framework for a Portable Library,
!    ACM Transactions on Mathematical Software,
!    Volume 4, Number 2, June 1978, page 176-188.
!
!  Parameters:
!
!    Input, integer I, the index of the desired constant.
!
!    Output, double precision D1MACH, the value of the constant.
!
      implicit none

      double precision d1mach
      integer diver(4)
      double precision dmach(5)
      integer i
      integer large(4)
      integer log10(4)
      integer right(4)
      integer small(4)

      equivalence ( dmach(1), small(1) )
      equivalence ( dmach(2), large(1) )
      equivalence ( dmach(3), right(1) )
      equivalence ( dmach(4), diver(1) )
      equivalence ( dmach(5), log10(1) )
!
!     MACHINE CONSTANTS FOR IEEE ARITHMETIC MACHINES, SUCH AS THE AT&T
!     3B SERIES AND MOTOROLA 68000 BASED MACHINES (E.G. SUN 3 AND AT&T
!     PC 7300), IN WHICH THE MOST SIGNIFICANT BYTE IS STORED FIRST.
!
! === MACHINE = IEEE.MOST-SIG-BYTE-FIRST
! === MACHINE = SUN
! === MACHINE = 68000
! === MACHINE = ATT.3B
! === MACHINE = ATT.7300
!     DATA SMALL(1),SMALL(2) /    1048576,          0 /
!     DATA LARGE(1),LARGE(2) / 2146435071,         -1 /
!     DATA RIGHT(1),RIGHT(2) / 1017118720,          0 /
!     DATA DIVER(1),DIVER(2) / 1018167296,          0 /
!     DATA LOG10(1),LOG10(2) / 1070810131, 1352628735 /
!
!     MACHINE CONSTANTS FOR IEEE ARITHMETIC MACHINES AND 8087-BASED
!     MICROS, SUCH AS THE IBM PC AND AT&T 6300, IN WHICH THE LEAST
!     SIGNIFICANT BYTE IS STORED FIRST.
!
! === MACHINE = IEEE.LEAST-SIG-BYTE-FIRST
! === MACHINE = 8087
! === MACHINE = IBM.PC
! === MACHINE = ATT.6300
!
       data small(1),small(2) /          0,    1048576 /
       data large(1),large(2) /         -1, 2146435071 /
       data right(1),right(2) /          0, 1017118720 /
       data diver(1),diver(2) /          0, 1018167296 /
       data log10(1),log10(2) / 1352628735, 1070810131 /
!
!     MACHINE CONSTANTS FOR AMDAHL MACHINES.
!
! === MACHINE = AMDAHL
!      DATA SMALL(1),SMALL(2) /    1048576,          0 /
!      DATA LARGE(1),LARGE(2) / 2147483647,         -1 /
!      DATA RIGHT(1),RIGHT(2) /  856686592,          0 /
!      DATA DIVER(1),DIVER(2) /  873463808,          0 /
!      DATA LOG10(1),LOG10(2) / 1091781651, 1352628735 /
!
!     MACHINE CONSTANTS FOR THE BURROUGHS 1700 SYSTEM.
!
! === MACHINE = BURROUGHS.1700
!      DATA SMALL(1) / ZC00800000 /
!      DATA SMALL(2) / Z000000000 /
!      DATA LARGE(1) / ZDFFFFFFFF /
!      DATA LARGE(2) / ZFFFFFFFFF /
!      DATA RIGHT(1) / ZCC5800000 /
!      DATA RIGHT(2) / Z000000000 /
!      DATA DIVER(1) / ZCC6800000 /
!      DATA DIVER(2) / Z000000000 /
!      DATA LOG10(1) / ZD00E730E7 /
!      DATA LOG10(2) / ZC77800DC0 /
!
!     MACHINE CONSTANTS FOR THE BURROUGHS 5700 SYSTEM.
!
! === MACHINE = BURROUGHS.5700
!      DATA SMALL(1) / O1771000000000000 /
!      DATA SMALL(2) / O0000000000000000 /
!      DATA LARGE(1) / O0777777777777777 /
!      DATA LARGE(2) / O0007777777777777 /
!      DATA RIGHT(1) / O1461000000000000 /
!      DATA RIGHT(2) / O0000000000000000 /
!      DATA DIVER(1) / O1451000000000000 /
!      DATA DIVER(2) / O0000000000000000 /
!      DATA LOG10(1) / O1157163034761674 /
!      DATA LOG10(2) / O0006677466732724 /
!
!     MACHINE CONSTANTS FOR THE BURROUGHS 6700/7700 SYSTEMS.
!
! === MACHINE = BURROUGHS.6700
! === MACHINE = BURROUGHS.7700
!      DATA SMALL(1) / O1771000000000000 /
!      DATA SMALL(2) / O7770000000000000 /
!      DATA LARGE(1) / O0777777777777777 /
!      DATA LARGE(2) / O7777777777777777 /
!      DATA RIGHT(1) / O1461000000000000 /
!      DATA RIGHT(2) / O0000000000000000 /
!      DATA DIVER(1) / O1451000000000000 /
!      DATA DIVER(2) / O0000000000000000 /
!      DATA LOG10(1) / O1157163034761674 /
!      DATA LOG10(2) / O0006677466732724 /
!
!     MACHINE CONSTANTS FOR THE CONVEX C-120 (NATIVE MODE)
!     WITH OR WITHOUT -R8 OPTION
!
! === MACHINE = CONVEX.C1
! === MACHINE = CONVEX.C1.R8
!      DATA DMACH(1) / 5.562684646268007D-309 /
!      DATA DMACH(2) / 8.988465674311577D+307 /
!      DATA DMACH(3) / 1.110223024625157D-016 /
!      DATA DMACH(4) / 2.220446049250313D-016 /
!      DATA DMACH(5) / 3.010299956639812D-001 /
!
!     MACHINE CONSTANTS FOR THE CONVEX C-120 (IEEE MODE)
!     WITH OR WITHOUT -R8 OPTION
!
! === MACHINE = CONVEX.C1.IEEE
! === MACHINE = CONVEX.C1.IEEE.R8
!      DATA DMACH(1) / 2.225073858507202D-308 /
!      DATA DMACH(2) / 1.797693134862315D+308 /
!      DATA DMACH(3) / 1.110223024625157D-016 /
!      DATA DMACH(4) / 2.220446049250313D-016 /
!      DATA DMACH(5) / 3.010299956639812D-001 /
!
!     MACHINE CONSTANTS FOR THE CYBER 170/180 SERIES USING NOS (FTN5).
!
! === MACHINE = CYBER.170.NOS
! === MACHINE = CYBER.180.NOS
!      DATA SMALL(1) / O"00604000000000000000" /
!      DATA SMALL(2) / O"00000000000000000000" /
!      DATA LARGE(1) / O"37767777777777777777" /
!      DATA LARGE(2) / O"37167777777777777777" /
!      DATA RIGHT(1) / O"15604000000000000000" /
!      DATA RIGHT(2) / O"15000000000000000000" /
!      DATA DIVER(1) / O"15614000000000000000" /
!      DATA DIVER(2) / O"15010000000000000000" /
!      DATA LOG10(1) / O"17164642023241175717" /
!      DATA LOG10(2) / O"16367571421742254654" /
!
!     MACHINE CONSTANTS FOR THE CDC 180 SERIES USING NOS/VE
!
! === MACHINE = CYBER.180.NOS/VE
!      DATA SMALL(1) / Z"3001800000000000" /
!      DATA SMALL(2) / Z"3001000000000000" /
!      DATA LARGE(1) / Z"4FFEFFFFFFFFFFFE" /
!      DATA LARGE(2) / Z"4FFE000000000000" /
!      DATA RIGHT(1) / Z"3FD2800000000000" /
!      DATA RIGHT(2) / Z"3FD2000000000000" /
!      DATA DIVER(1) / Z"3FD3800000000000" /
!      DATA DIVER(2) / Z"3FD3000000000000" /
!      DATA LOG10(1) / Z"3FFF9A209A84FBCF" /
!      DATA LOG10(2) / Z"3FFFF7988F8959AC" /
!
!     MACHINE CONSTANTS FOR THE CYBER 205
!
! === MACHINE = CYBER.205
!      DATA SMALL(1) / X'9000400000000000' /
!      DATA SMALL(2) / X'8FD1000000000000' /
!      DATA LARGE(1) / X'6FFF7FFFFFFFFFFF' /
!      DATA LARGE(2) / X'6FD07FFFFFFFFFFF' /
!      DATA RIGHT(1) / X'FF74400000000000' /
!      DATA RIGHT(2) / X'FF45000000000000' /
!      DATA DIVER(1) / X'FF75400000000000' /
!      DATA DIVER(2) / X'FF46000000000000' /
!      DATA LOG10(1) / X'FFD04D104D427DE7' /
!      DATA LOG10(2) / X'FFA17DE623E2566A' /
!
!     MACHINE CONSTANTS FOR THE CDC 6000/7000 SERIES.
!
! === MACHINE = CDC.6000
! === MACHINE = CDC.7000
!      DATA SMALL(1) / 00604000000000000000B /
!      DATA SMALL(2) / 00000000000000000000B /
!      DATA LARGE(1) / 37767777777777777777B /
!      DATA LARGE(2) / 37167777777777777777B /
!      DATA RIGHT(1) / 15604000000000000000B /
!      DATA RIGHT(2) / 15000000000000000000B /
!      DATA DIVER(1) / 15614000000000000000B /
!      DATA DIVER(2) / 15010000000000000000B /
!      DATA LOG10(1) / 17164642023241175717B /
!      DATA LOG10(2) / 16367571421742254654B /
!
!     MACHINE CONSTANTS FOR THE CRAY 1, XMP, 2, AND 3.
!
! === MACHINE = CRAY
!      DATA SMALL(1) / 201354000000000000000B /
!      DATA SMALL(2) / 000000000000000000000B /
!      DATA LARGE(1) / 577767777777777777777B /
!      DATA LARGE(2) / 000007777777777777776B /
!      DATA RIGHT(1) / 376434000000000000000B /
!      DATA RIGHT(2) / 000000000000000000000B /
!      DATA DIVER(1) / 376444000000000000000B /
!      DATA DIVER(2) / 000000000000000000000B /
!      DATA LOG10(1) / 377774642023241175717B /
!      DATA LOG10(2) / 000007571421742254654B /
!
!     MACHINE CONSTANTS FOR THE DATA GENERAL ECLIPSE S/200
!
!     NOTE - IT MAY BE APPROPRIATE TO INCLUDE THE FOLLOWING LINE -
!     STATIC DMACH(5)
!
! === MACHINE = DATA_GENERAL.ECLIPSE.S/200
!      DATA SMALL/20K,3*0/,LARGE/77777K,3*177777K/
!      DATA RIGHT/31420K,3*0/,DIVER/32020K,3*0/
!      DATA LOG10/40423K,42023K,50237K,74776K/
!
!     ELXSI 6400
!
! === MACHINE = ELSXI.6400
!      DATA SMALL(1), SMALL(2) / '00100000'X,'00000000'X /
!      DATA LARGE(1), LARGE(2) / '7FEFFFFF'X,'FFFFFFFF'X /
!      DATA RIGHT(1), RIGHT(2) / '3CB00000'X,'00000000'X /
!      DATA DIVER(1), DIVER(2) / '3CC00000'X,'00000000'X /
!      DATA LOG10(1), DIVER(2) / '3FD34413'X,'509F79FF'X /
!
!     MACHINE CONSTANTS FOR THE HARRIS 220
!     MACHINE CONSTANTS FOR THE HARRIS SLASH 6 AND SLASH 7
!
! === MACHINE = HARRIS.220
! === MACHINE = HARRIS.SLASH6
! === MACHINE = HARRIS.SLASH7
!      DATA SMALL(1),SMALL(2) / '20000000, '00000201 /
!      DATA LARGE(1),LARGE(2) / '37777777, '37777577 /
!      DATA RIGHT(1),RIGHT(2) / '20000000, '00000333 /
!      DATA DIVER(1),DIVER(2) / '20000000, '00000334 /
!      DATA LOG10(1),LOG10(2) / '23210115, '10237777 /
!
!     MACHINE CONSTANTS FOR THE HONEYWELL 600/6000 SERIES.
!     MACHINE CONSTANTS FOR THE HONEYWELL DPS 8/70 SERIES.
!
! === MACHINE = HONEYWELL.600/6000
! === MACHINE = HONEYWELL.DPS.8/70
!      DATA SMALL(1),SMALL(2) / O402400000000, O000000000000 /
!      DATA LARGE(1),LARGE(2) / O376777777777, O777777777777 /
!      DATA RIGHT(1),RIGHT(2) / O604400000000, O000000000000 /
!      DATA DIVER(1),DIVER(2) / O606400000000, O000000000000 /
!      DATA LOG10(1),LOG10(2) / O776464202324, O117571775714 /
!
!      MACHINE CONSTANTS FOR THE HP 2100
!      3 WORD DOUBLE PRECISION OPTION WITH FTN4
!
! === MACHINE = HP.2100.3_WORD_DP
!      DATA SMALL(1), SMALL(2), SMALL(3) / 40000B,       0,       1 /
!      DATA LARGE(1), LARGE(2), LARGE(3) / 77777B, 177777B, 177776B /
!      DATA RIGHT(1), RIGHT(2), RIGHT(3) / 40000B,       0,    265B /
!      DATA DIVER(1), DIVER(2), DIVER(3) / 40000B,       0,    276B /
!      DATA LOG10(1), LOG10(2), LOG10(3) / 46420B,  46502B,  77777B /
!
!      MACHINE CONSTANTS FOR THE HP 2100
!      4 WORD DOUBLE PRECISION OPTION WITH FTN4
!
! === MACHINE = HP.2100.4_WORD_DP
!      DATA SMALL(1), SMALL(2) /  40000B,       0 /
!      DATA SMALL(3), SMALL(4) /       0,       1 /
!      DATA LARGE(1), LARGE(2) /  77777B, 177777B /
!      DATA LARGE(3), LARGE(4) / 177777B, 177776B /
!      DATA RIGHT(1), RIGHT(2) /  40000B,       0 /
!      DATA RIGHT(3), RIGHT(4) /       0,    225B /
!      DATA DIVER(1), DIVER(2) /  40000B,       0 /
!      DATA DIVER(3), DIVER(4) /       0,    227B /
!      DATA LOG10(1), LOG10(2) /  46420B,  46502B /
!      DATA LOG10(3), LOG10(4) /  76747B, 176377B /
!
!     HP 9000
!
!      D1MACH(1) = 2.8480954D-306
!      D1MACH(2) = 1.40444776D+306
!      D1MACH(3) = 2.22044605D-16
!      D1MACH(4) = 4.44089210D-16
!      D1MACH(5) = 3.01029996D-1
!
! === MACHINE = HP.9000
!      DATA SMALL(1), SMALL(2) / 00040000000B, 00000000000B /
!      DATA LARGE(1), LARGE(2) / 17737777777B, 37777777777B /
!      DATA RIGHT(1), RIGHT(2) / 07454000000B, 00000000000B /
!      DATA DIVER(1), DIVER(2) / 07460000000B, 00000000000B /
!      DATA LOG10(1), LOG10(2) / 07764642023B, 12047674777B /
!
!     MACHINE CONSTANTS FOR THE IBM 360/370 SERIES,
!     THE XEROX SIGMA 5/7/9, THE SEL SYSTEMS 85/86, AND
!     THE INTERDATA 3230 AND INTERDATA 7/32.
!
! === MACHINE = IBM.360
! === MACHINE = IBM.370
! === MACHINE = XEROX.SIGMA.5
! === MACHINE = XEROX.SIGMA.7
! === MACHINE = XEROX.SIGMA.9
! === MACHINE = SEL.85
! === MACHINE = SEL.86
! === MACHINE = INTERDATA.3230
! === MACHINE = INTERDATA.7/32
!      DATA SMALL(1),SMALL(2) / Z00100000, Z00000000 /
!      DATA LARGE(1),LARGE(2) / Z7FFFFFFF, ZFFFFFFFF /
!      DATA RIGHT(1),RIGHT(2) / Z33100000, Z00000000 /
!      DATA DIVER(1),DIVER(2) / Z34100000, Z00000000 /
!      DATA LOG10(1),LOG10(2) / Z41134413, Z509F79FF /
!
!     MACHINE CONSTANTS FOR THE INTERDATA 8/32
!     WITH THE UNIX SYSTEM FORTRAN 77 COMPILER.
!
!     FOR THE INTERDATA FORTRAN VII COMPILER REPLACE
!     THE Z'S SPECIFYING HEX CONSTANTS WITH Y'S.
!
! === MACHINE = INTERDATA.8/32.UNIX
!      DATA SMALL(1),SMALL(2) / Z'00100000', Z'00000000' /
!      DATA LARGE(1),LARGE(2) / Z'7EFFFFFF', Z'FFFFFFFF' /
!      DATA RIGHT(1),RIGHT(2) / Z'33100000', Z'00000000' /
!      DATA DIVER(1),DIVER(2) / Z'34100000', Z'00000000' /
!      DATA LOG10(1),LOG10(2) / Z'41134413', Z'509F79FF' /
!
!     MACHINE CONSTANTS FOR THE PDP-10 (KA PROCESSOR).
!
! === MACHINE = PDP-10.KA
!      DATA SMALL(1),SMALL(2) / "033400000000, "000000000000 /
!      DATA LARGE(1),LARGE(2) / "377777777777, "344777777777 /
!      DATA RIGHT(1),RIGHT(2) / "113400000000, "000000000000 /
!      DATA DIVER(1),DIVER(2) / "114400000000, "000000000000 /
!      DATA LOG10(1),LOG10(2) / "177464202324, "144117571776 /
!
!     MACHINE CONSTANTS FOR THE PDP-10 (KI PROCESSOR).
!
! === MACHINE = PDP-10.KI
!      DATA SMALL(1),SMALL(2) / "000400000000, "000000000000 /
!      DATA LARGE(1),LARGE(2) / "377777777777, "377777777777 /
!      DATA RIGHT(1),RIGHT(2) / "103400000000, "000000000000 /
!      DATA DIVER(1),DIVER(2) / "104400000000, "000000000000 /
!      DATA LOG10(1),LOG10(2) / "177464202324, "047674776746 /
!
!     MACHINE CONSTANTS FOR PDP-11 FORTRAN SUPPORTING
!     32-BIT INTEGERS (EXPRESSED IN INTEGER AND OCTAL).
!
! === MACHINE = PDP-11.32-BIT
!      DATA SMALL(1),SMALL(2) /    8388608,           0 /
!      DATA LARGE(1),LARGE(2) / 2147483647,          -1 /
!      DATA RIGHT(1),RIGHT(2) /  612368384,           0 /
!      DATA DIVER(1),DIVER(2) /  620756992,           0 /
!      DATA LOG10(1),LOG10(2) / 1067065498, -2063872008 /
!
!      DATA SMALL(1),SMALL(2) / O00040000000, O00000000000 /
!      DATA LARGE(1),LARGE(2) / O17777777777, O37777777777 /
!      DATA RIGHT(1),RIGHT(2) / O04440000000, O00000000000 /
!      DATA DIVER(1),DIVER(2) / O04500000000, O00000000000 /
!      DATA LOG10(1),LOG10(2) / O07746420232, O20476747770 /
!
!     MACHINE CONSTANTS FOR PDP-11 FORTRAN SUPPORTING
!     16-BIT INTEGERS (EXPRESSED IN INTEGER AND OCTAL).
!
! === MACHINE = PDP-11.16-BIT
!      DATA SMALL(1),SMALL(2) /    128,      0 /
!      DATA SMALL(3),SMALL(4) /      0,      0 /
!      DATA LARGE(1),LARGE(2) /  32767,     -1 /
!      DATA LARGE(3),LARGE(4) /     -1,     -1 /
!      DATA RIGHT(1),RIGHT(2) /   9344,      0 /
!      DATA RIGHT(3),RIGHT(4) /      0,      0 /
!      DATA DIVER(1),DIVER(2) /   9472,      0 /
!      DATA DIVER(3),DIVER(4) /      0,      0 /
!      DATA LOG10(1),LOG10(2) /  16282,   8346 /
!      DATA LOG10(3),LOG10(4) / -31493, -12296 /
!
!      DATA SMALL(1),SMALL(2) / O000200, O000000 /
!      DATA SMALL(3),SMALL(4) / O000000, O000000 /
!      DATA LARGE(1),LARGE(2) / O077777, O177777 /
!      DATA LARGE(3),LARGE(4) / O177777, O177777 /
!      DATA RIGHT(1),RIGHT(2) / O022200, O000000 /
!      DATA RIGHT(3),RIGHT(4) / O000000, O000000 /
!      DATA DIVER(1),DIVER(2) / O022400, O000000 /
!      DATA DIVER(3),DIVER(4) / O000000, O000000 /
!      DATA LOG10(1),LOG10(2) / O037632, O020232 /
!      DATA LOG10(3),LOG10(4) / O102373, O147770 /
!
!     MACHINE CONSTANTS FOR THE SEQUENT BALANCE 8000
!
! === MACHINE = SEQUENT.BALANCE.8000
!      DATA SMALL(1),SMALL(2) / $00000000,  $00100000 /
!      DATA LARGE(1),LARGE(2) / $FFFFFFFF,  $7FEFFFFF /
!      DATA RIGHT(1),RIGHT(2) / $00000000,  $3CA00000 /
!      DATA DIVER(1),DIVER(2) / $00000000,  $3CB00000 /
!      DATA LOG10(1),LOG10(2) / $509F79FF,  $3FD34413 /
!
!     MACHINE CONSTANTS FOR THE UNIVAC 1100 SERIES. FTN COMPILER
!
! === MACHINE = UNIVAC.1100
!      DATA SMALL(1),SMALL(2) / O000040000000, O000000000000 /
!      DATA LARGE(1),LARGE(2) / O377777777777, O777777777777 /
!      DATA RIGHT(1),RIGHT(2) / O170540000000, O000000000000 /
!      DATA DIVER(1),DIVER(2) / O170640000000, O000000000000 /
!      DATA LOG10(1),LOG10(2) / O177746420232, O411757177572 /
!
!     MACHINE CONSTANTS FOR VAX 11/780
!     (EXPRESSED IN INTEGER AND HEXADECIMAL)
!    *** THE INTEGER FORMAT SHOULD BE OK FOR UNIX SYSTEMS***
!
! === MACHINE = VAX.11/780
!      DATA SMALL(1), SMALL(2) /        128,           0 /
!      DATA LARGE(1), LARGE(2) /     -32769,          -1 /
!      DATA RIGHT(1), RIGHT(2) /       9344,           0 /
!      DATA DIVER(1), DIVER(2) /       9472,           0 /
!      DATA LOG10(1), LOG10(2) /  546979738,  -805796613 /
!
!    ***THE HEX FORMAT BELOW MAY NOT BE SUITABLE FOR UNIX SYSYEMS***
!      DATA SMALL(1), SMALL(2) / Z00000080, Z00000000 /
!      DATA LARGE(1), LARGE(2) / ZFFFF7FFF, ZFFFFFFFF /
!      DATA RIGHT(1), RIGHT(2) / Z00002480, Z00000000 /
!      DATA DIVER(1), DIVER(2) / Z00002500, Z00000000 /
!      DATA LOG10(1), LOG10(2) / Z209A3F9A, ZCFF884FB /
!
!   MACHINE CONSTANTS FOR VAX 11/780 (G-FLOATING)
!     (EXPRESSED IN INTEGER AND HEXADECIMAL)
!    *** THE INTEGER FORMAT SHOULD BE OK FOR UNIX SYSTEMS***
!
!      DATA SMALL(1), SMALL(2) /         16,           0 /
!      DATA LARGE(1), LARGE(2) /     -32769,          -1 /
!      DATA RIGHT(1), RIGHT(2) /      15552,           0 /
!      DATA DIVER(1), DIVER(2) /      15568,           0 /
!      DATA LOG10(1), LOG10(2) /  1142112243, 2046775455 /
!
!    ***THE HEX FORMAT BELOW MAY NOT BE SUITABLE FOR UNIX SYSYEMS***
!      DATA SMALL(1), SMALL(2) / Z00000010, Z00000000 /
!      DATA LARGE(1), LARGE(2) / ZFFFF7FFF, ZFFFFFFFF /
!      DATA RIGHT(1), RIGHT(2) / Z00003CC0, Z00000000 /
!      DATA DIVER(1), DIVER(2) / Z00003CD0, Z00000000 /
!      DATA LOG10(1), LOG10(2) / Z44133FF3, Z79FF509F /
!
      if ( i .lt. 1  .or.  5 .lt. i ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'D1MACH - Fatal error!'
        write ( *, '(a)' ) '  I out of bounds.'
        stop
      end if

      d1mach = dmach(i)

      return
      end


      subroutine dqelg(n,epstab,result,abserr,res3la,nres)

!*********************************************************************72
!
!c DQELG carries out the Epsilon extrapolation algorithm.
!
!***begin prologue  dqelg
!***refer to  dqagie,dqagoe,dqagpe,dqagse
!***routines called  d1mach
!***revision date  830518   (yymmdd)
!***keywords  epsilon algorithm, convergence acceleration,
!             extrapolation
!***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
!           de doncker,elise,appl. math & progr. div. - k.u.leuven
!***purpose  the routine determines the limit of a given sequence of
!            approximations, by means of the epsilon algorithm of
!            p.wynn. an estimate of the absolute error is also given.
!            the condensed epsilon table is computed. only those
!            elements needed for the computation of the next diagonal
!            are preserved.
!***description
!
!           epsilon algorithm
!           standard fortran subroutine
!           double precision version
!
!           parameters
!              n      - integer
!                       epstab(n) contains the new element in the
!                       first column of the epsilon table.
!
!              epstab - double precision
!                       vector of dimension 52 containing the elements
!                       of the two lower diagonals of the triangular
!                       epsilon table. the elements are numbered
!                       starting at the right-hand corner of the
!                       triangle.
!
!              result - double precision
!                       resulting approximation to the integral
!
!              abserr - double precision
!                       estimate of the absolute error computed from
!                       result and the 3 previous results
!
!              res3la - double precision
!                       vector of dimension 3 containing the last 3
!                       results
!
!              nres   - integer
!                       number of calls to the routine
!                       (should be zero at first call)
!
!***end prologue  dqelg
!
      double precision abserr,dabs,delta1,delta2,delta3,dmax1, &
       epmach,epsinf,epstab,error,err1,err2,err3,e0,e1,e1abs,e2,e3,  &
       oflow,res,result,res3la,ss,tol1,tol2,tol3
      integer i,ib,ib2,ie,indx,k1,k2,k3,limexp,n,newelm,nres,num
      dimension epstab(52),res3la(3)
!
!           list of major variables
!           -----------------------
!
!           e0     - the 4 elements on which the computation of a new
!           e1       element in the epsilon table is based
!           e2
!           e3                 e0
!                        e3    e1    new
!                              e2
!           newelm - number of elements to be computed in the new
!                    diagonal
!           error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
!           result - the element in the new diagonal with least value
!                    of error
!
!           machine dependent constants
!           ---------------------------
!
!           epmach is the largest relative spacing.
!           oflow is the largest positive magnitude.
!           limexp is the maximum number of elements the epsilon
!           table can contain. if this number is reached, the upper
!           diagonal of the epsilon table is deleted.
!
!***first executable statement  dqelg
      epmach = d1mach(4)
      oflow = d1mach(2)
      nres = nres+1
      abserr = oflow
      result = epstab(n)
      if(n.lt.3) go to 100
      limexp = 50
      epstab(n+2) = epstab(n)
      newelm = (n-1)/2
      epstab(n) = oflow
      num = n
      k1 = n
      do 40 i = 1,newelm
        k2 = k1-1
        k3 = k1-2
        res = epstab(k1+2)
        e0 = epstab(k3)
        e1 = epstab(k2)
        e2 = res
        e1abs = dabs(e1)
        delta2 = e2-e1
        err2 = dabs(delta2)
        tol2 = dmax1(dabs(e2),e1abs)*epmach
        delta3 = e1-e0
        err3 = dabs(delta3)
        tol3 = dmax1(e1abs,dabs(e0))*epmach
        if(err2.gt.tol2.or.err3.gt.tol3) go to 10
!
!           if e0, e1 and e2 are equal to within machine
!           accuracy, convergence is assumed.
!           result = e2
!           abserr = abs(e1-e0)+abs(e2-e1)
!
        result = res
        abserr = err2+err3
! ***jump out of do-loop
        go to 100
   10   e3 = epstab(k1)
        epstab(k1) = e1
        delta1 = e1-e3
        err1 = dabs(delta1)
        tol1 = dmax1(e1abs,dabs(e3))*epmach
!
!           if two elements are very close to each other, omit
!           a part of the table by adjusting the value of n
!
        if(err1.le.tol1.or.err2.le.tol2.or.err3.le.tol3) go to 20
        ss = 0.1d+01/delta1+0.1d+01/delta2-0.1d+01/delta3
        epsinf = dabs(ss*e1)
!
!           test to detect irregular behaviour in the table, and
!           eventually omit a part of the table adjusting the value
!           of n.
!
        if(epsinf.gt.0.1d-03) go to 30
   20   n = i+i-1
! ***jump out of do-loop
        go to 50
!
!           compute a new element and eventually adjust
!           the value of result.
!
   30   res = e1+0.1d+01/ss
        epstab(k1) = res
        k1 = k1-2
        error = err2+dabs(res-e2)+err3
        if(error.gt.abserr) go to 40
        abserr = error
        result = res
   40 continue
!
!           shift the table.
!
   50 if(n.eq.limexp) n = 2*(limexp/2)-1
      ib = 1
      if((num/2)*2.eq.num) ib = 2
      ie = newelm+1
      do 60 i=1,ie
        ib2 = ib+2
        epstab(ib) = epstab(ib2)
        ib = ib2
   60 continue
      if(num.eq.n) go to 80
      indx = num-n+1
      do 70 i = 1,n
        epstab(i)= epstab(indx)
        indx = indx+1
   70 continue
   80 if(nres.ge.4) go to 90
      res3la(nres) = result
      abserr = oflow
      go to 100
!
!           compute error estimate
!
   90 abserr = dabs(result-res3la(3))+dabs(result-res3la(2))+dabs(result-res3la(1))
      res3la(1) = res3la(2)
      res3la(2) = res3la(3)
      res3la(3) = result
  100 abserr = dmax1(abserr,0.5d+01*epmach*dabs(result))
      return
      end

      subroutine dqk21(f,a,b,result,abserr,resabs,resasc)

!*********************************************************************72
!
!c DQK21 carries out a 21 point Gauss-Kronrod quadrature rule.
!
!***begin prologue  dqk21
!***date written   800101   (yymmdd)
!***revision date  830518   (yymmdd)
!***category no.  h2a1a2
!***keywords  21-point gauss-kronrod rules
!***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
!           de doncker,elise,appl. math. & progr. div. - k.u.leuven
!***purpose  to compute i = integral of f over (a,b), with error
!                           estimate
!                       j = integral of abs(f) over (a,b)
!***description
!
!           integration rules
!           standard fortran subroutine
!           double precision version
!
!           parameters
!            on entry
!              f      - double precision
!                       function subprogram defining the integrand
!                       function f(x). the actual name for f needs to be
!                       declared e x t e r n a l in the driver program.
!
!              a      - double precision
!                       lower limit of integration
!
!              b      - double precision
!                       upper limit of integration
!
!            on return
!              result - double precision
!                       approximation to the integral i
!                       result is computed by applying the 21-point
!                       kronrod rule (resk) obtained by optimal addition
!                       of abscissae to the 10-point gauss rule (resg).
!
!              abserr - double precision
!                       estimate of the modulus of the absolute error,
!                       which should not exceed abs(i-result)
!
!              resabs - double precision
!                       approximation to the integral j
!
!              resasc - double precision
!                       approximation to the integral of abs(f-i/(b-a))
!                       over (a,b)
!
!***references  (none)
!***routines called  d1mach
!***end prologue  dqk21
!
      double precision a,absc,abserr,b,centr,dabs,dhlgth,dmax1,dmin1,     &
       epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,resabs,resasc,  &
       resg,resk,reskh,result,uflow,wg,wgk,xgk
      integer j,jtw,jtwm1
      external f
!
      dimension fv1(10),fv2(10),wg(5),wgk(11),xgk(11)
!
!           the abscissae and weights are given for the interval (-1,1).
!           because of symmetry only the positive abscissae and their
!           corresponding weights are given.
!
!           xgk    - abscissae of the 21-point kronrod rule
!                    xgk(2), xgk(4), ...  abscissae of the 10-point
!                    gauss rule
!                    xgk(1), xgk(3), ...  abscissae which are optimally
!                    added to the 10-point gauss rule
!
!           wgk    - weights of the 21-point kronrod rule
!
!           wg     - weights of the 10-point gauss rule
!
!
! gauss quadrature weights and kronron quadrature abscissae and weights
! as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
! bell labs, nov. 1981.
!
      data wg  (  1) / 0.066671344308688137593568809893332d0 /
      data wg  (  2) / 0.149451349150580593145776339657697d0 /
      data wg  (  3) / 0.219086362515982043995534934228163d0 /
      data wg  (  4) / 0.269266719309996355091226921569469d0 /
      data wg  (  5) / 0.295524224714752870173892994651338d0 /
!
      data xgk (  1) / 0.995657163025808080735527280689003d0 /
      data xgk (  2) / 0.973906528517171720077964012084452d0 /
      data xgk (  3) / 0.930157491355708226001207180059508d0 /
      data xgk (  4) / 0.865063366688984510732096688423493d0 /
      data xgk (  5) / 0.780817726586416897063717578345042d0 /
      data xgk (  6) / 0.679409568299024406234327365114874d0 /
      data xgk (  7) / 0.562757134668604683339000099272694d0 /
      data xgk (  8) / 0.433395394129247190799265943165784d0 /
      data xgk (  9) / 0.294392862701460198131126603103866d0 /
      data xgk ( 10) / 0.148874338981631210884826001129720d0 /
      data xgk ( 11) / 0.000000000000000000000000000000000d0 /
!
      data wgk (  1) / 0.011694638867371874278064396062192d0 /
      data wgk (  2) / 0.032558162307964727478818972459390d0 /
      data wgk (  3) / 0.054755896574351996031381300244580d0 /
      data wgk (  4) / 0.075039674810919952767043140916190d0 /
      data wgk (  5) / 0.093125454583697605535065465083366d0 /
      data wgk (  6) / 0.109387158802297641899210590325805d0 /
      data wgk (  7) / 0.123491976262065851077958109831074d0 /
      data wgk (  8) / 0.134709217311473325928054001771707d0 /
      data wgk (  9) / 0.142775938577060080797094273138717d0 /
      data wgk ( 10) / 0.147739104901338491374841515972068d0 /
      data wgk ( 11) / 0.149445554002916905664936468389821d0 /
!
!
!           list of major variables
!           -----------------------
!
!           centr  - mid point of the interval
!           hlgth  - half-length of the interval
!           absc   - abscissa
!           fval*  - function value
!           resg   - result of the 10-point gauss formula
!           resk   - result of the 21-point kronrod formula
!           reskh  - approximation to the mean value of f over (a,b),
!                    i.e. to i/(b-a)
!
!
!           machine dependent constants
!           ---------------------------
!
!           epmach is the largest relative spacing.
!           uflow is the smallest positive magnitude.
!
!***first executable statement  dqk21
      epmach = d1mach(4)
      uflow = d1mach(1)
!
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      dhlgth = dabs(hlgth)
!
!           compute the 21-point kronrod approximation to
!           the integral, and estimate the absolute error.
!
      resg = 0.0d+00
      fc = f(centr)
      resk = wgk(11)*fc
      resabs = dabs(resk)
      do 10 j=1,5
        jtw = 2*j
        absc = hlgth*xgk(jtw)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtw) = fval1
        fv2(jtw) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(jtw)*fsum
        resabs = resabs+wgk(jtw)*(dabs(fval1)+dabs(fval2))
   10 continue
      do 15 j = 1,5
        jtwm1 = 2*j-1
        absc = hlgth*xgk(jtwm1)
        fval1 = f(centr-absc)
        fval2 = f(centr+absc)
        fv1(jtwm1) = fval1
        fv2(jtwm1) = fval2
        fsum = fval1+fval2
        resk = resk+wgk(jtwm1)*fsum
        resabs = resabs+wgk(jtwm1)*(dabs(fval1)+dabs(fval2))
   15 continue
      reskh = resk*0.5d+00
      resasc = wgk(11)*dabs(fc-reskh)
      do 20 j=1,10
        resasc = resasc+wgk(j)*(dabs(fv1(j)-reskh)+dabs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resabs = resabs*dhlgth
      resasc = resasc*dhlgth
      abserr = dabs((resk-resg)*hlgth)
      if(resasc.ne.0.0d+00.and.abserr.ne.0.0d+00)abserr = resasc*dmin1(0.1d+01,(0.2d+03*abserr/resasc)**1.5d+00)
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1((epmach*0.5d+02)*resabs,abserr)
      return
      end

      subroutine dqk15i(f,boun,inf,a,b,result,abserr,resabs,resasc)

!*********************************************************************72
!
!c DQK15I applies a 15 point Gauss-Kronrod quadrature on an infinite interval.
!
!***begin prologue  dqk15i
!***date written   800101   (yymmdd)
!***revision date  830518   (yymmdd)
!***category no.  h2a3a2,h2a4a2
!***keywords  15-point transformed gauss-kronrod rules
!***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
!           de doncker,elise,appl. math. & progr. div. - k.u.leuven
!***purpose  the original (infinite integration range is mapped
!            onto the interval (0,1) and (a,b) is a part of (0,1).
!            it is the purpose to compute
!            i = integral of transformed integrand over (a,b),
!            j = integral of abs(transformed integrand) over (a,b).
!***description
!
!           integration rule
!           standard fortran subroutine
!           double precision version
!
!           parameters
!            on entry
!              f      - double precision
!                       fuction subprogram defining the integrand
!                       function f(x). the actual name for f needs to be
!                       declared e x t e r n a l in the calling program.
!
!              boun   - double precision
!                       finite bound of original integration
!                       range (set to zero if inf = +2)
!
!              inf    - integer
!                       if inf = -1, the original interval is
!                                   (-infinity,bound),
!                       if inf = +1, the original interval is
!                                   (bound,+infinity),
!                       if inf = +2, the original interval is
!                                   (-infinity,+infinity) and
!                       the integral is computed as the sum of two
!                       integrals, one over (-infinity,0) and one over
!                       (0,+infinity).
!
!              a      - double precision
!                       lower limit for integration over subrange
!                       of (0,1)
!
!              b      - double precision
!                       upper limit for integration over subrange
!                       of (0,1)
!
!            on return
!              result - double precision
!                       approximation to the integral i
!                       result is computed by applying the 15-point
!                       kronrod rule(resk) obtained by optimal addition
!                       of abscissae to the 7-point gauss rule(resg).
!
!              abserr - double precision
!                       estimate of the modulus of the absolute error,
!                       which should equal or exceed abs(i-result)
!
!              resabs - double precision
!                       approximation to the integral j
!
!              resasc - double precision
!                       approximation to the integral of
!                       abs((transformed integrand)-i/(b-a)) over (a,b)
!
!***references  (none)
!***routines called  d1mach
!***end prologue  dqk15i
!
      double precision a,absc,absc1,absc2,abserr,b,boun,centr,dabs,dinf, &
       dmax1,dmin1,epmach,f,fc,fsum,fval1,fval2,fv1,fv2,hlgth,   &
       resabs,resasc,resg,resk,reskh,result,tabsc1,tabsc2,uflow,wg,wgk, &
       xgk
      integer inf,j
      external f
!
      dimension fv1(7),fv2(7),xgk(8),wgk(8),wg(8)
!
!           the abscissae and weights are supplied for the interval
!           (-1,1).  because of symmetry only the positive abscissae and
!           their corresponding weights are given.
!
!           xgk    - abscissae of the 15-point kronrod rule
!                    xgk(2), xgk(4), ... abscissae of the 7-point
!                    gauss rule
!                    xgk(1), xgk(3), ...  abscissae which are optimally
!                    added to the 7-point gauss rule
!
!           wgk    - weights of the 15-point kronrod rule
!
!           wg     - weights of the 7-point gauss rule, corresponding
!                    to the abscissae xgk(2), xgk(4), ...
!                    wg(1), wg(3), ... are set to zero.
!
      data wg(1) / 0.0d0 /
      data wg(2) / 0.129484966168869693270611432679082d0 /
      data wg(3) / 0.0d0 /
      data wg(4) / 0.279705391489276667901467771423780d0 /
      data wg(5) / 0.0d0 /
      data wg(6) / 0.381830050505118944950369775488975d0 /
      data wg(7) / 0.0d0 /
      data wg(8) / 0.417959183673469387755102040816327d0 /
!
      data xgk(1) / 0.991455371120812639206854697526329d0 /
      data xgk(2) / 0.949107912342758524526189684047851d0 /
      data xgk(3) / 0.864864423359769072789712788640926d0 /
      data xgk(4) / 0.741531185599394439863864773280788d0 /
      data xgk(5) / 0.586087235467691130294144838258730d0 /
      data xgk(6) / 0.405845151377397166906606412076961d0 /
      data xgk(7) / 0.207784955007898467600689403773245d0 /
      data xgk(8) / 0.000000000000000000000000000000000d0 /
!
      data wgk(1) / 0.022935322010529224963732008058970d0 /
      data wgk(2) / 0.063092092629978553290700663189204d0 /
      data wgk(3) / 0.104790010322250183839876322541518d0 /
      data wgk(4) / 0.140653259715525918745189590510238d0 /
      data wgk(5) / 0.169004726639267902826583426598550d0 /
      data wgk(6) / 0.190350578064785409913256402421014d0 /
      data wgk(7) / 0.204432940075298892414161999234649d0 /
      data wgk(8) / 0.209482141084727828012999174891714d0 /
!
!
!           list of major variables
!           -----------------------
!
!           centr  - mid point of the interval
!           hlgth  - half-length of the interval
!           absc*  - abscissa
!           tabsc* - transformed abscissa
!           fval*  - function value
!           resg   - result of the 7-point gauss formula
!           resk   - result of the 15-point kronrod formula
!           reskh  - approximation to the mean value of the transformed
!                    integrand over (a,b), i.e. to i/(b-a)
!
!           machine dependent constants
!           ---------------------------
!
!           epmach is the largest relative spacing.
!           uflow is the smallest positive magnitude.
!
!***first executable statement  dqk15i
      epmach = d1mach(4)
      uflow = d1mach(1)
      dinf = min0(1,inf)
!
      centr = 0.5d+00*(a+b)
      hlgth = 0.5d+00*(b-a)
      tabsc1 = boun+dinf*(0.1d+01-centr)/centr
      fval1 = f(tabsc1)
      if(inf.eq.2) fval1 = fval1+f(-tabsc1)
      fc = (fval1/centr)/centr
!
!           compute the 15-point kronrod approximation to
!           the integral, and estimate the error.
!
      resg = wg(8)*fc
      resk = wgk(8)*fc
      resabs = dabs(resk)
      do 10 j=1,7
        absc = hlgth*xgk(j)
        absc1 = centr-absc
        absc2 = centr+absc
        tabsc1 = boun+dinf*(0.1d+01-absc1)/absc1
        tabsc2 = boun+dinf*(0.1d+01-absc2)/absc2
        fval1 = f(tabsc1)
        fval2 = f(tabsc2)
        if(inf.eq.2) fval1 = fval1+f(-tabsc1)
        if(inf.eq.2) fval2 = fval2+f(-tabsc2)
        fval1 = (fval1/absc1)/absc1
        fval2 = (fval2/absc2)/absc2
        fv1(j) = fval1
        fv2(j) = fval2
        fsum = fval1+fval2
        resg = resg+wg(j)*fsum
        resk = resk+wgk(j)*fsum
        resabs = resabs+wgk(j)*(dabs(fval1)+dabs(fval2))
   10 continue
      reskh = resk*0.5d+00
      resasc = wgk(8)*dabs(fc-reskh)
      do 20 j=1,7
        resasc = resasc+wgk(j)*(dabs(fv1(j)-reskh)+dabs(fv2(j)-reskh))
   20 continue
      result = resk*hlgth
      resasc = resasc*hlgth
      resabs = resabs*hlgth
      abserr = dabs((resk-resg)*hlgth)
      if(resasc.ne.0.0d+00.and.abserr.ne.0.d0) abserr = resasc*dmin1(0.1d+01,(0.2d+03*abserr/resasc)**1.5d+00)
      if(resabs.gt.uflow/(0.5d+02*epmach)) abserr = dmax1((epmach*0.5d+02)*resabs,abserr)
      return
      end


      subroutine dqpsrt(limit,last,maxerr,ermax,elist,iord,nrmax)

!*********************************************************************72
!
!c DQPSRT maintains the order of a list of local error estimates.
!
!***begin prologue  dqpsrt
!***refer to  dqage,dqagie,dqagpe,dqawse
!***routines called  (none)
!***revision date  810101   (yymmdd)
!***keywords  sequential sorting
!***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
!           de doncker,elise,appl. math. & progr. div. - k.u.leuven
!***purpose  this routine maintains the descending ordering in the
!            list of the local error estimated resulting from the
!            interval subdivision process. at each call two error
!            estimates are inserted using the sequential search
!            method, top-down for the largest error estimate and
!            bottom-up for the smallest error estimate.
!***description
!
!           ordering routine
!           standard fortran subroutine
!           double precision version
!
!           parameters (meaning at output)
!              limit  - integer
!                       maximum number of error estimates the list
!                       can contain
!
!              last   - integer
!                       number of error estimates currently in the list
!
!              maxerr - integer
!                       maxerr points to the nrmax-th largest error
!                       estimate currently in the list
!
!              ermax  - double precision
!                       nrmax-th largest error estimate
!                       ermax = elist(maxerr)
!
!              elist  - double precision
!                       vector of dimension last containing
!                       the error estimates
!
!              iord   - integer
!                       vector of dimension last, the first k elements
!                       of which contain pointers to the error
!                       estimates, such that
!                       elist(iord(1)),...,  elist(iord(k))
!                       form a decreasing sequence, with
!                       k = last if last.le.(limit/2+2), and
!                       k = limit+1-last otherwise
!
!              nrmax  - integer
!                       maxerr = iord(nrmax)
!
!***end prologue  dqpsrt
!
      double precision elist,ermax,errmax,errmin
      integer i,ibeg,ido,iord,isucc,j,jbnd,jupbn,k,last,limit,maxerr,nrmax
      dimension elist(last),iord(last)
!
!           check whether the list contains more than
!           two error estimates.
!
!***first executable statement  dqpsrt
      if(last.gt.2) go to 10
      iord(1) = 1
      iord(2) = 2
      go to 90
!
!           this part of the routine is only executed if, due to a
!           difficult integrand, subdivision increased the error
!           estimate. in the normal case the insert procedure should
!           start after the nrmax-th largest error estimate.
!
   10 errmax = elist(maxerr)
      if(nrmax.eq.1) go to 30
      ido = nrmax-1
      do 20 i = 1,ido
        isucc = iord(nrmax-1)
! ***jump out of do-loop
        if(errmax.le.elist(isucc)) go to 30
        iord(nrmax) = isucc
        nrmax = nrmax-1
   20    continue
!
!           compute the number of elements in the list to be maintained
!           in descending order. this number depends on the number of
!           subdivisions still allowed.
!
   30 jupbn = last
      if(last.gt.(limit/2+2)) jupbn = limit+3-last
      errmin = elist(last)
!
!           insert errmax by traversing the list top-down,
!           starting comparison from the element elist(iord(nrmax+1)).
!
      jbnd = jupbn-1
      ibeg = nrmax+1
      if(ibeg.gt.jbnd) go to 50
      do 40 i=ibeg,jbnd
        isucc = iord(i)
! ***jump out of do-loop
        if(errmax.ge.elist(isucc)) go to 60
        iord(i-1) = isucc
   40 continue
   50 iord(jbnd) = maxerr
      iord(jupbn) = last
      go to 90
!
!           insert errmin by traversing the list bottom-up.
!
   60 iord(i-1) = maxerr
      k = jbnd
      do 70 j=i,jbnd
        isucc = iord(k)
! ***jump out of do-loop
        if(errmin.lt.elist(isucc)) go to 80
        iord(k+1) = isucc
        k = k-1
   70 continue
      iord(i) = last
      go to 90
   80 iord(k+1) = last
!
!           set maxerr and ermax.
!
   90 maxerr = iord(nrmax)
      ermax = elist(maxerr)
      return
      end



end module quadpack
