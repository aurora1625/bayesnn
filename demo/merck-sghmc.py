# demo for merck experiment
import sys
sys.path.append('..')
import merck

# get default prameters
param = merck.cfg_param()

# set necessary parameters we want
param.batch_size = 50
param.num_round = 800
param.num_hidden = 100

param.path_data1 = '/Users/jasonxu/bayesnn/MerckTrainSet/ACT4_competition_training.csv'
param.path_data2 = '/Users/jasonxu/bayesnn/MerckTrainSet/ACT4_competition_training.csv'		#we need to make different test set

param.net_type = 'mlp2'
param.updater  = 'hmc'
param.hyperupdater = 'gibbs-sep'
param.num_burn     = 50

param.eta=0.1
param.mdecay=0.01

# run the experiment
merck.run_exp( param )
