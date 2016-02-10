'''
Wrapper script for hyperparam tuning
Author: Jiayi Lu
Email: jiayi.lu@nyu.edu
'''
from __builtin__ import *
import os
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--poolSize',type=int, nargs='*',default=[2])
arg_parser.add_argument('--filtSize',type=int, nargs='*',default=[5])
args = arg_parser.parse_args()
args_dict = vars(args)
print args
print args_dict

#hyperparams


for key, val_list in args_dict.iteritems():
    print "Hyper parameter name: ", key
    for val in val_list:
        print "Hyper parameter value = ", val
        print "Now executing command:\n", "th doall.lua -size small -save {} -nepochs 1 -{} {}".format(key+"="+str(val),key,val)
#Note that nepochs are hard coded as 1 here!!! We might want to use it as another cmd arg in the future
        os.system("th doall.lua -size small -save {} -nepochs 1 -{} {}".format("results_"+key+"="+str(val),key,val))
