#!/bin/bash
#SBATCH -J train_CL
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH -o daft_a100_%x.%j.%N.out
#SBATCH -e daft_a100_%x.%j.%N.err

##########>>>>> -p mcml-dgx-a100-40x8
#########>>>>> --gres=gpu:1 -q mcml
#########>>>>> --mem_per_cpu=100000

ENV_name=msadet
BASE_DIR=/dss/dsshome1/01/ge32fix2
ENV_DIR=ENVS
DATA_DIR=/dss/dsshome1/01/ge32fix2/DATASETS/

srun --mpi=pmi2 \
     --container-mounts=${DATA_DIR}:/mnt/data,${BASE_DIR}:/mnt/code \
     --container-image=${BASE_DIR}/${ENV_DIR}/${ENV_name}.sqsh \
     --container-workdir=/mnt/code \
     /opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreplay_onlyClassLabels

#### sbatch sbatch/train_CL.sbatch
#### squeue -u ge32fix2
#### rm daft_a100_train_CL*