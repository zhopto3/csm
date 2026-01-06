#!/bin/bash
#SBATCH --job-name=slurm01
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zacharywilliam.hopton@uzh.ch
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM96GB"
#SBATCH --error=./slurm_out/q_sampler.err
#SBATCH --output=./slurm_out/q_sampler.out

# Bash script to convert speech to units with hubert

module load miniforge3

source activate csm

module load cuda/11.8.0


for f in ../asr-surprisal/src/imp_samp/text_pref/*;
do
    echo "Now processing ${f}"
    python3 ./sample_q.py --in_text="$f" --batch_size=128 --samples_per_text=500
done
# python3 ./sample_q.py --in_text="../asr-surprisal/src/imp_samp/text_pref/1x2.txt" --batch_size=256 --samples_per_text=1000