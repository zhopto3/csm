#!/bin/bash
#SBATCH --job-name=slurm01
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zacharywilliam.hopton@uzh.ch
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --error=./slurm_out/q_sampler.err
#SBATCH --output=./slurm_out/q_sampler.out

# Bash script to convert speech to units with hubert

module load miniforge3

source activate csm

module load cuda/11.8.0


# for f in "../asr-surprisal/src/imp_samp/text_pref";
# do
#     python3 ./sample_q.py --input_text="$f"
# done
python3 ./sample_q.py --input_text="../asr-surprisal/src/imp_samp/text_pref/1x1.txt"