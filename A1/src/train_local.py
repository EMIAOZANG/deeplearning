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
<<<<<<< HEAD
arg_parser.add_argument('--dropoutProb', type=float, nargs='*',default=[0,0.5])

=======
arg_parser.add_argument('--dropoutProb',type=float, nargs='*', default=[0,0.5])
>>>>>>> c939c942328b6f0a21488b067fe2473191a184a9
args = arg_parser.parse_args()
args_dict = vars(args)
print args
print args_dict

#hyperparams

for key, val_list in args_dict.iteritems():
    print "Hyper parameter name: ", key
    for val in val_list:
        print "Hyper parameter value = ", val
        print "Now executing command:\n", "th doall.lua -size small -save {} -nepochs 1 -{}{}".format(key+"="+str(val),key,val)
#Note that nepochs are hard coded as 1 here!!! We might want to use it as another cmd arg in the future
<<<<<<< HEAD
        os.system("th doall.lua -size full -save {} -nepochs 10 -{} {}".format("results_"+key+"="+str(val),key,val))
        
=======
        #os.system("th doall.lua -size small -augment -save {} -nepochs 10 -{} {} -sub".format("results_aug"+key+"="+str(val),key,val))
        os.system("th doall.lua -size small -save {} -nepochs 10 -{} {} -sub".format("results_"+key+"="+str(val),key,val))
>>>>>>> c939c942328b6f0a21488b067fe2473191a184a9
