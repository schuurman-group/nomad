#
# Read the operator file. Must be named 'fms.op'
#
from ..fmsio import fileio
def load_operator():
    fileio.read_operator('fms.op')

