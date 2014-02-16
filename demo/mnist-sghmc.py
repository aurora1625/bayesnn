# demo for mnist experiment
import sys
sys.path.append('..')
import mnist

# get default prameters
param = mnist.cfg_param()

# set necessary parameters we want
param.batch_size = 500
param.num_round = 800
param.num_hidden = 100

# change the following line to PATH/TO/MNIST dataset
param.path_data = '/Users/jasonxu/bayesnn/'

param.net_type = 'mlp2'
param.updater  = 'hmc'
param.hyperupdater = 'gibbs-sep'
param.num_burn = 50

param.eta=0.1
param.mdecay=0.01

# run the experiment
mnist.run_exp( param )
