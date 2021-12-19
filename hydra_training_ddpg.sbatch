#!/bin/bash
#
#SBATCH --job-name=hydra_training
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=06:00:00
 
module purge

python3 Capstone-2021/hydra_training_driver_ddpg.py --multirun gamma=0.99,0.999 OUstd=0.1,0.15,0.5

# python3 Capstone-2021/hydra_training_driver_dqn.py --multirun gamma=0.99,0.999

# python3 Capstone-2021/hydra_training_driver_sac.py --multirun gamma=0.9,0.99,0.999 learning_starts=100,300