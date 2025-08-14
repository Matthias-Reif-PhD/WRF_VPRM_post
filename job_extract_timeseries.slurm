#!/bin/bash
#SBATCH --job-name=extract_wrf_domains_mean_timeseries
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=900

module purge
module load Anaconda3/2023.03/miniconda-base-2023.03
eval "$(/bin/conda shell.bash hook)"
conda activate /scratch/c7071034/conda_envs/pyrealm311
srun python extract_timeseries.py