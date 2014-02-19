# this is incomplete
# demo for merck experiment
import sys
sys.path.append('..')
import merck

# get default prameters
param = merck.cfg_param()

# set necessary parameters we want
param.batch_size = 100
param.num_round = 800
param.num_hidden = 100

param.path_data = '/projects/grail/tqchen/data/kaggle/Meck/TrainingSet/ACT12_competition_training.csv'

param.net_type = 'mlp2'
param.updater  = 'sgd'
param.out_type = 'linear'
param.hyperupdater = 'none'
param.num_burn     = 50

param.eta=0.01
param.mdecay=0.1

# run the experiment
merck.run_exp( param )
