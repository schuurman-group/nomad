"""
Frequency file reading routines.
"""
import numpy as np
import src.vcham.hampar as ham
import src.vcham.parsemod as parse
import src.vcham.rdoper as rdoper
import src.fmsio.fileio as fileio

def rdfreqfile():
    """Reads and interprets a freq.dat file."""
    ham.freq = np.zeros(ham.maxdim)
    ham.mlbl_active = ['' for i in range(ham.maxdim)]

    # Open the freq.dat file
    freqfile = open(fileio.home_path + '/freq.dat', 'r')

    # Read the freq.dat file
    n = -1
    parse.leof = False
    while True:
        parse.rd1line(freqfile, up2low=False)
        if not parse.leof:
            mdlbl = parse.keyword[1]
            freq = float(parse.keyword[2])

            if parse.keyword[3] == ',':
                freq *= rdoper.convfac(parse.keyword[4])

            n += 1
            ham.freq[n] = freq
            ham.mlbl_active[n] = mdlbl
        else:
            break

    # Close the freq.dat file
    freqfile.close()

    # Set the no. of active modes
    ham.nmode_active = n + 1

    # Set up the mode label-to-frequency map
    ham.freqmap = dict()
    for i in range(ham.nmode_active):
        ham.freqmap[ham.mlbl_active[i]] = ham.freq[i]
