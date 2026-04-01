
#!/bin/bash
#SBATCH --job-name=ctpt
#SBATCH -p gpu-medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
data_root='put your dataset path here'
testsets=$1
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
run_type=tpt_ctpt
lambda_term=20

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init} --run_type ${run_type} --I_augmix --lambda_term ${lambda_term} \
