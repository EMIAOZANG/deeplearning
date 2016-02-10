import os

opt_weightDecay = [0,0.1,1,10]

for val in opt_weightDecay:
    print val
    os.system("th -i doall.lua -weightDecay {} -size small -save {} -nepochs 1".format(val,"results_weightDecay="+str(val)))


