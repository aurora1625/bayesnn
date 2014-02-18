"""
  wrapper code for Merck experiments, modified from mnist.py
  Tianqi Chen
"""
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import sys
import random
import nncfg
import numpy as np
import nnet


#This is based on the mnist.py: the load function is changed, and returns a 2d numpy array of features, and 1d numpy array of labels
#we need to make sure the param.input_size matches the number of features *in the given file*
#each of the 15 files may have different number of features
#I think we will need to make a "test set" in a similar way to the validation set, or simply not have a test set: use training + validation
#for now, commented out test set parts
#changed param input size, param num class

# load Merck dataset
def load(filePath):
    f = open(filePath)
    f.readline() #ignores the header which has column names
    dataList = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line[1:]] #we need to get rid of the index at beginning of line
        dataList.append(sample)
    dataList = np.array(dataList)
    labels = np.array([x[0] for x in dataList])  #pull out true labels from first column
    #labels = dataList[:,0]
    features = dataList[:,1:]
    return features, labels     #each row corresponds to a molecule id/observation, each column a is molecule descriptor/feature


# default parameter
def cfg_param():
    param = nnet.NNParam()
    param.init_sigma = 0.01
    param.input_size = 4306     #the number of features: make sure this matches the given file
    param.num_class = 1
    param.out_type = 'linear'
    param.eta = 0.1
    param.mdecay = 0.1
    param.wd = 0.0
    param.batch_size = 50   #why do we need to set this again, if it was set in demo?
    return param

def run_exp( param ):
    np.random.seed( param.seed )
    net = nncfg.create_net( param )
    print 'network configure end, start loading data ...'

    # load in data 
    train_xdata, train_ylabels = load( param.path_data1 ) #'/Users/jasonxu/bayesnn/MerckTrainSet/ACT4_competition_training.csv'
    #test_images , test_labels  = load( param.path_data2 ) #'/Users/jasonxu/bayesnn/MerckTrainSet/ACT7_competition_training.csv'
    train_xdata, train_ylabel  = nncfg.create_batch( train_xdata, train_ylabels, param.batch_size, True )
    #test_xdata , test_ylabel   = nncfg.create_batch( test_images , test_labels, param.batch_size, True, 1.0/256.0 )
    
    # split validation set
    ntrain = train_xdata.shape[0]    
    nvalid = 500
    assert nvalid % param.batch_size == 0
    nvalid = nvalid / param.batch_size
    valid_xdata, valid_ylabel = train_xdata[0:nvalid], train_ylabel[0:nvalid]
    train_xdata, train_ylabel = train_xdata[nvalid:ntrain], train_ylabel[nvalid:ntrain]
    
    # setup evaluator
    evals = []
    evals.append( nnet.NNEvaluatorMerck( net, train_xdata, train_ylabel, param, 'train' ))
    evals.append( nnet.NNEvaluatorMerck( net, valid_xdata, valid_ylabel, param, 'valid' ))
    
    # set parameters
    param.num_train = train_ylabel.size
    print 'loading end,%d train,%d valid, start update ...' % ( train_ylabel.size, valid_ylabel.size )
        
    for it in xrange( param.num_round ):
        param.set_round( it )
        net.update_all( train_xdata, train_ylabel )  
        sys.stderr.write( '[%d]' % it )
        for ev in evals:
            ev.eval( it, sys.stderr )
        sys.stderr.write('\n')            
    print 'all update end'

