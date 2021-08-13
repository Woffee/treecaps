#!/bin/bash

#SBATCH --job-name=treecaps
#SBATCH --output=sbatch/%j.treecaps.out
#SBATCH --error=sbatch/%j.treecaps.err
##SBATCH  --partition=datasci
#SBATCH  --partition=xye

#SBATCH  --gres=gpu:TitanRtx:1
#SBATCH  --nodes=1
##SBATCH  --ntasks-per-node=2
#SBATCH  --mem=100G

module load cuda
module load singularity

singularity exec --bind /xye_data_nobackup/wenbo/dlvp/data:/data --nv docker://woffee/keras:cflow_joern3.1 sh run.sh

