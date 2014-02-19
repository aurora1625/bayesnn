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

class NNEvaluatorMerck:
    def __init__( self, nnet, xdatas, ylabels, param, prefix='' ):
        self.nnet = nnet
        self.xdatas  = xdatas
        self.ylabels = ylabels
        self.param = param
        self.prefix = prefix
        nbatch, nclass = nnet.o_node.shape
        assert xdatas.shape[0] == ylabels.shape[0]
        assert nbatch == xdatas.shape[1]
        assert nbatch == ylabels.shape[1]
        self.o_pred  = np.zeros( ( xdatas.shape[0], nbatch, param.num_class ), 'float32'  )
        self.rcounter = 0
        self.sum_wsample = 0.0

    def __get_alpha( self ):
        if self.rcounter < self.param.num_burn:
            return 1.0
        else:
            self.sum_wsample += self.param.wsample
            return self.param.wsample / self.sum_wsample
        
    def eval( self, rcounter, fo ):
        self.rcounter = rcounter
        alpha = self.__get_alpha()        
        self.o_pred[:] *= ( 1.0 - alpha )
        nbatch = self.xdatas.shape[1]
        sum_bad  = 0.0        

        y_predFull = np.array([])
        y_trueFull = np.array([])
        # need to fix functions for prediction:
        for i in xrange( self.xdatas.shape[0] ):
            self.o_pred[i,:] += alpha * self.nnet.predict( self.xdatas[i] )
            y_pred = self.o_pred[i,:].reshape( (nbatch) )   
            y_predFull = np.append(y_predFull, y_pred )     
            y_trueFull = np.append(y_trueFull, self.ylabels[i,:].reshape( (nbatch) ) )    

        ninst = self.ylabels.size

        avgTrue = np.mean(y_trueFull)
        avgPred = np.mean(y_predFull)
        numerator = sum( (y_trueFull - avgTrue )*(y_predFull-avgPred) )**2
        denom = sum( (y_trueFull - avgTrue)**2) * sum( (y_predFull- avgPred)**2 )
        rst = numerator/denom     

        fo.write( ' %s-r2:%f' % ( self.prefix, rst ) )

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
    features = dataList[:,1:]
    fmax = np.max( features, 0 )  # standardization
    fmax = np.maximum( fmax, 1.0 )
    assert np.min( np.min(features[:]) ) > -1e-6
    features = features / fmax
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

    print 'start loading data ...'
    # load in data 
    train_xdata, train_ylabel = load( param.path_data ) #'/Users/jasonxu/bayesnn/MerckTrainSet/ACT4_competition_training.csv'
    
    # split validation set
    ntrain = train_xdata.shape[0]    
    nvalid = ntrain / 5

    valid_xdata, valid_ylabel = train_xdata[0:nvalid], train_ylabel[0:nvalid]
    train_xdata, train_ylabel = train_xdata[nvalid:ntrain], train_ylabel[nvalid:ntrain]
    
    train_xdata, train_ylabel  = nncfg.create_batch( train_xdata, train_ylabel, param.batch_size, True )
    valid_xdata, valid_ylabel  = nncfg.create_batch( valid_xdata, valid_ylabel, param.batch_size, True )

    # set parameters
    param.input_size = train_xdata.shape[2]
    param.num_train = train_ylabel.size
    param.min_label = np.min(train_ylabel)
    param.max_label = np.max(train_ylabel)    
    param.avg_label = np.mean(train_ylabel)

    print 'loading end, create nnet input=%d...' % param.input_size
    net = nncfg.create_net( param )

    # setup evaluator
    evals = []
    evals.append( NNEvaluatorMerck( net, train_xdata, train_ylabel, param, 'train' ))
    evals.append( NNEvaluatorMerck( net, valid_xdata, valid_ylabel, param, 'valid' ))
    

    print 'loading end,%d train,%d valid, start update ...' % ( train_ylabel.size, valid_ylabel.size )
        
    for it in xrange( param.num_round ):
        param.set_round( it )
        net.update_all( train_xdata, train_ylabel )  
        sys.stderr.write( '[%d]' % it )
        for ev in evals:
            ev.eval( it, sys.stderr )
        sys.stderr.write('\n')            
    print 'all update end'

