#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=end_bert-50k_%j.log           
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=64
#SBATCH --partition=isi
module load git
module load gcc/12.3.0
module load cuda/12.2.1

source /home1/spangher/miniconda3/etc/profile.d/conda.sh
source activate /project/jonmay_1426/spangher/Projects/style-classifier-aria/conda-env 

cd /project/jonmay_1426/spangher/Projects/style-classifier-aria

export OMP_NUM_THREADS=16
python train_bert.py --data_dir datasets/varied-forms-4k-collapsed --epochs 50