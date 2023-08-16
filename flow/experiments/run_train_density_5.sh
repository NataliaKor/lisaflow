#!/bin/sh

#SBATCH --job-name=flow_train_5
#SBATCH --output=logs/gb_freq_%j.out  
#SBATCH --error=logs/gb_freq_%j.err  

#SBATCH --partition=gpu                        # Partition choice
#SBATCH --ntasks=1                  
#SBATCH --mem=32000                            #MiB
#SBATCH --time=6-00:00:00           

#SBATCH --mail-user=<korsakova@apc.in2p3.fr>   # Where to send mail
#SBATCH --mail-type=ALL                        # Mail events (NONE, BEGIN, END, FAIL, ALL)

export PYTHONPATH=$PYTHONPATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow
export PYTHONPATH=$PYTHONPATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments

export PATH=$PATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow
export PATH=$PATH:/sps/lisaf/natalia/github/ai_for_lisa/lisaflow/flow/experiments

source /sps/lisaf/natalia/anaconda3/bin/activate 
conda activate BBH_GBs_rapids_torch

date

# Copy config files or write full path to this files in the way the code is being called
python density_estimation_all_galaxy.py #--config 'configs/gbs/gb_resample.yaml' --config_data 'configs/gbs/gb_log.yaml'

date
