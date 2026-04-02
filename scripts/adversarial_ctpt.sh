#!/bin/bash
#SBATCH --job-name=ctpt_pgd
#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/hpc/07/zhang303/C-TPT-main/%x-%j.out
#SBATCH -e /scratch/hpc/07/zhang303/C-TPT-main/%x-%j.err

module add anaconda3/2022.05
source activate ctpt

data_root='/scratch/hpc/07/zhang303/C-TPT-main/data'
testsets=$1

arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_ctpt
lambda_term=50

# PGD params
pgd_eps=0.03137255      # 8/255
pgd_alpha=0.00784314    # 2/255
pgd_steps=10

srun python ./tpt_classification.py ${data_root} \
  --test_sets ${testsets} \
  -a ${arch} \
  -b ${bs} \
  --gpu 0 \
  --tpt \
  --ctx_init ${ctx_init} \
  --run_type ${run_type} \
  --lambda_term ${lambda_term} \
  --eval_pgd \
  --pgd_eps ${pgd_eps} \
  --pgd_alpha ${pgd_alpha} \
  --pgd_steps ${pgd_steps} \
  --pgd_random_start