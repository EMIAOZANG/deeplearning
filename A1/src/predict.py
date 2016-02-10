import re
import os
import sys

dirs = os.listdir('.')
p = re.compile(r'^results')
model_paths = [dir for dir in dirs if re.search(p,dir)]
print "Directories to be imported:"
print model_paths

for path in model_paths:
    try:
        os.system("th results.lua -size full -modelPath {}".format(path))
    except:
        print >> sys.stderr, 'Error openning model directory'
        sys.exit()
print 'Prediction completed'

