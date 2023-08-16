#!/bin/sh

#SBATCH --job-name=mbhb_SGD
#SBATCH --output=logs/mbhb_sgd_%j.log  
#SBATCH --error=logs/mbhb_sgd_%j.err  

#SBATCH --partition=gpu                        # Partition choice
#SBATCH --ntasks=1                  
#SBATCH --time=5-00:00:00    
#SBATCH --mem=32000       

#SBATCH --mail-user=<korsakova@apc.in2p3.fr>   # Where to send mail
#SBATCH --mail-type=ALL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)

export PYTHONPATH=$PYTHONPATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow
export PYTHONPATH=$PYTHONPATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments

export PATH=$PATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow
export PATH=$PATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments

source /sps/lisaf/natalia/anaconda3/bin/activate 
conda activate BBH_GBs_rapids_torch

$(pwd)


date

# Copy config files or write full path to this files in the way the code is being called
python train_mbhb_as.py #--config_data 'configs/mbhbs/mbhb_data_radler_no_time_dist.yaml' --config 'configs/mbhbs/mbhb_resample_play.yaml'

date

