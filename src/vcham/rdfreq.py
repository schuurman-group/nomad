import sys
import src.vcham.hampar   as ham
import src.vcham.parsemod as parse
import src.vcham.rdoper   as rdoper
import src.fmsio.fileio   as fileio
import src.fmsio.glbl     as glbl

def rdfreqfile():

    ham.freq=[0.0 for i in range(ham.maxdim)]
    ham.mlbl_active=['' for i in range(ham.maxdim)]

    # Open the freq.dat file
    freqfile=open(fileio.home_path+'/freq.dat','r')

    # Read the freq.dat file
    n=-1
    parse.leof=False
    while (True):
        parse.rd1line(freqfile,up2low=False)
        if (parse.leof==False):
            mdlbl=parse.keyword[1]

            freq=float(parse.keyword[2])

            if (parse.keyword[3]==','):
                fac=rdoper.convfac(parse.keyword[4])
                freq=freq*fac

            n+=1
            ham.freq[n]=freq
            ham.mlbl_active[n]=mdlbl

        else:
            break

    # Close the freq.dat file
    freqfile.close()

    # Set the no. of active modes
    ham.nmode_active=n+1

    # Set up the mode label-to-frequency map
    ham.freqmap={}
    for i in range(ham.nmode_active):
        ham.freqmap.update({ham.mlbl_active[i] : ham.freq[i] })

    return
