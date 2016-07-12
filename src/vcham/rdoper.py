import sys
import src.vcham.parsemod as parse
import src.vcham.hampar as ham

#########################################################################

def convfac(string):

    if (string=='ev'):
        factor=1.0/27.21141
    else:
        print("Unknown conversion factor:",keyword)
        sys.exit()

    return factor

#########################################################################

def isnum(string):

    try:
        float(string)
        flag=True
        return flag
    except:
        try:
            int(string)
            flag=True
            return flag
        except:
            flag=False
            return flag

#########################################################################

def getcoe(string):

    coeff=0.0

    # Split the string into parameters and operators
    k=len(string)
    nfac=1
    ilbl=0
    atmp=["" for i in range(20)]
    aop=["" for i in range(20)]

    for i in range(k):
        if (string[i:i+1]=="*" or string[i:i+1]=="/"):
            nfac+=1
            aop[nfac-1]=string[i:i+1]
            ilbl=0
        else:
            ilbl=ilbl+1
            atmp[nfac-1]+=string[i:i+1]

    # Get the numerical values of the parameters
    val=[0.0 for i in range(20)]
    for i in range(nfac):
        # Is the parameter a number?
        lnum=isnum(atmp[i])

        if (lnum):
            val[i]=float(atmp[i])
            lfound=True
        else:
            lfound=False
            for k in range(ham.npar):
                if (ham.apar[k]==atmp[i]):
                    val[i]=ham.par[k]
                    lfound=True

        if (not lfound):
            print('Parameter ',atmp[i],' not recognised')
            sys.exit()

    # Construct the coefficient from the parameter values and operator
    # strings
    coeff=val[0]
    if (nfac > 1):
        for i in range(nfac):
            if (aop[i]=="*"):
                coeff=coeff*val[i]
            elif (aop[i]=="/"):
                coeff=coeff/val[i]

    return coeff

#########################################################################

def rdoperfile(infile):

    #########################################################################
    # Read to the labels section
    #########################################################################
    found=False
    parse.leof=False
    while (not found and not parse.leof):
        parse.rd1line(infile)
        if (parse.keyword[1]=="labels_section"):
            found=True

    # Exit if a labels section has not been found
    if (not found):
        print("No labels section has been found")
        sys.exit()

    # Save the position of the start of the labels section
    labstart=infile.tell()

    #########################################################################
    # Determine the number of labels
    #########################################################################
    found=False
    parse.leof=False
    ham.npar=0
    while (not found and not parse.leof):
        parse.rd1line(infile)
        if (parse.keyword[1]=="end-labels_section"):
            found=True
        else:
            ham.npar+=1

    #########################################################################
    # Read the label names and values
    #########################################################################
    ham.apar=["" for i in range(ham.npar)]
    ham.par=[0.0 for i in range(ham.npar)]

    # Rewind to the start of the labels section
    infile.seek(labstart)

    # Read the parameter names and values
    for i in range(ham.npar):
        parse.rd1line(infile)
        ham.apar[i]=parse.keyword[1]
        if (parse.keyword[2]=="="):
            ham.par[i]=float(parse.keyword[3])
        else:
            print("No argument has been given with the keyword:",parse.keyword[1])
            sys.exit()
        if (parse.keyword[4]==","):
            fac=convfac(parse.keyword[5])
            ham.par[i]=ham.par[i]*fac

    #########################################################################
    # Read to the Hamiltonian section
    #########################################################################
    found=False
    parse.leof=False
    while (not found and not parse.leof):
        parse.rd1line(infile)
        if (parse.keyword[1]=="hamiltonian_section"):
            found=True

    # Exit if a Hamiltonian section has not been found
    if (not found):
        print("No Hamiltonian section has been found")
        sys.exit()

    # Save the position of the start of the labels section
    hamstart=infile.tell()

    #########################################################################
    # Determine the number of Hamiltonian terms
    #########################################################################
    found=False
    ham.nterms=0
    parse.leof=False
    while (not found and not parse.leof):
        parse.rd1line(infile)
        if (parse.keyword[1]=="end-hamiltonian_section"):
            found=True
        else:
            if (parse.keyword[1][0:3]!='---' \
                and parse.keyword[1][0:5]!='modes'):
                ham.nterms+=1

    #########################################################################
    # Read the Hamiltonian terms
    #########################################################################
    ham.coe=[0.0 for i in range(ham.nterms)]
    ham.stalbl=[[0 for i in range(2)] for j in range(ham.nterms)]
    ham.order=[[0 for i in range(ham.maxdim)] for j in range(ham.nterms)]

    # Rewind to the start of the Hamiltonian section
    infile.seek(hamstart)

    # Read the mode labels
    ham.mlbl_total=['' for i in range(ham.maxdim)]
    parse.rd1line(infile)
    if (parse.keyword[1][0:5]!='modes'):
        print("The Hamiltonian section must start with the mode specification!")
        sys.exit()
    infile.seek(hamstart)
    ismodes=True
    nmdline=0
    ham.nmode_total=0
    while(ismodes):
        parse.rd1line(infile,up2low=False)
        if (parse.keyword[1][0:5].lower()!='modes'):
            ismodes=False
        else:
            nmdline+=1
            i=2
            while(i<=parse.inkw):
                i+=1
                ham.nmode_total+=1
                ham.mlbl_total[ham.nmode_total-1]=parse.keyword[i]
                i+=1

    # Read the Hamiltonian terms: coefficients, monomials and states
    infile.seek(hamstart)
    for i in range(nmdline):
        parse.rd1line(infile)

    for i in range(ham.nterms):
        parse.rd1line(infile)
        # (1) 1st keyword: coefficient
        lnum=isnum(parse.keyword[1])
        if (lnum):
            ham.coe[i]=float(parse.keyword[1])
        else:
            ham.coe[i]=getcoe(parse.keyword[1])

        # (2) Keywords 2 to inkw-1: the monomial for the current term
        for j in range(2,parse.inkw):
            k=parse.keyword[j].index("^")
            m=int(parse.keyword[j][0:k])
            p=int(parse.keyword[j][k+1:])
            ham.order[i][m-1]=p

        # (3) Keyword inkw: state indices
        k=parse.keyword[parse.inkw].index("&")
        s1=parse.keyword[parse.inkw][1:k]
        s2=parse.keyword[parse.inkw][k+1:]

        # Ensure that we are filling in the lower triangle of
        # the Hamiltonian matrix only
        if (s1 > s2):
            k=s1
            s1=s2
            s2=k

        # Set the state indices for the current term
        ham.stalbl[i][0]=int(s1)
        ham.stalbl[i][1]=int(s2)

    #########################################################################
    # Eliminate the non-active Hamiltonian terms and rearrange the order
    # list to correspond to the indexing of the sub-set of active modes
    #########################################################################
    lblmap={}
    for i in range(ham.nmode_active):
        lblmap.update({ham.mlbl_active[i] : i})

    nterms_new=0
    coe_new=[0.0 for i in range(ham.nterms)]
    stalbl_new=[[0 for i in range(2)] for j in range(ham.nterms)]
    order_new=[[0 for i in range(ham.maxdim)] for j in range(ham.nterms)]


    # Loop over all Hamiltonian terms
    for k in range(ham.nterms):
        # Detemine whether the current term is active
        active=True
        for i in range(ham.nmode_total):
            if ham.order[k][i] > 0:
                if ham.mlbl_total[i] not in ham.mlbl_active:
                    active=False

        # If the current term is active, then add it to the list
        # of active terms
        if active:
            nterms_new+=1
            coe_new[nterms_new-1]=ham.coe[k]
            stalbl_new[nterms_new-1][0]=ham.stalbl[k][0]
            stalbl_new[nterms_new-1][1]=ham.stalbl[k][1]
            for i in range(ham.nmode_total):
                if ham.order[k][i] > 0:
                    if ham.mlbl_total[i] in ham.mlbl_active:
                        n=lblmap[ham.mlbl_total[i]]
                        order_new[k][n]=ham.order[k][i]

    # Save the active terms
    ham.nterms=nterms_new
    ham.coe=coe_new
    ham.stalbl=stalbl_new
    ham.order=order_new

