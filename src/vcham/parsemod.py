
leof=False
maxkw=120
inkw=0
keyword=["" for i in range(maxkw)]

########################################################################
# rd1line: Reads the next non-empty, non-comment line of a file,
#          converts to lowercase and separates into keywords
########################################################################
def rd1line(filename,up2low=True):

    global keyword
    global maxkw
    global inkw
    global leof

    #-----------------------------------------------------------------------
    # Initialisation
    #-----------------------------------------------------------------------
    leof=False
    keyword=["" for i in range(maxkw)]
    inkw=0

    #-----------------------------------------------------------------------
    # Read to the next non-blank, non-comment line
    #-----------------------------------------------------------------------
    string=filename.readline()

    # If we have reached the end of the file, set leof=True and
    # return
    if (not string):
        leof=True
        return

    # Check whether the current line is either empty or a comment
    p1=len(string)-len(string.lstrip())
    if ((len(string.split()))==0):
        ok=False
    elif (string[p1:p1+1]=="#"):
        ok=False
    else:
        ok=True

    # If we are a comment or an empty line, read until we reach
    # the next non-empty and non-comment line
    while (ok==False):
        string=filename.readline()
        p1=len(string)-len(string.lstrip())
        if ((len(string.split()))!=0 and string[p1:p1+1]!="#"):
            ok=True
        if (ok==True):
            break

    #-----------------------------------------------------------------------
    # Convert to lower case
    #-----------------------------------------------------------------------
    if up2low==True:
        string=string.lower()

    #-----------------------------------------------------------------------
    # Extract the individual keywords
    #-----------------------------------------------------------------------
    k=0
    for i in range(0,len(string)-1):
        # Break if we have reached a comment
        if (string[i]=='#'):
            break

        # If the current character is a delimiter other than a space or
        # a tab (=, ',', ], etc.) then read in as a separate keyword and
        # then go to the next keyword...
        if (string[i] == '=' \
            or string[i] == ',' \
            or string[i] == '(' \
            or string[i] == ')' \
            or string[i] == '[' \
            or string[i] == ']' \
            or string[i] == '{' \
            or string[i] == '}'):
            inkw+=1
            keyword[inkw]+=string[i]
            k=0

        # ... otherwise keep adding to the current keyword...
        elif (string[i]!=' ' and string[i]!='\t'):
            k+=1
            if (k==1):
                inkw+=1
            # read(string(i:i),'(a1)') keyword(inkw)(k:k)
            keyword[inkw]+=string[i]

        # ... until a blank space or tab is reached, at which point
        # we move to the next keyword
        else:
            k=0
