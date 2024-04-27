#!/bin/bash
#SBATCH -J train_CL
#SBATCH -N 1
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:00:10
#SBATCH -o daft_a100_%x.%j.%N.out
#SBATCH -e daft_a100_%x.%j.%N.err

##########>>>>> -p mcml-dgx-a100-40x8
#########>>>>> --gres=gpu:1 -q mcml
#########>>>>> --mem_per_cpu=100000

BASE_DIR=/dss/dsshome1/01/ge32fix2
CODE_DIR=OrganDETR
ENV_DIR=ENVS

srun --mpi=pmi2 \
--container-mounts=${BASE_DIR}/myData:/mnt/data,${BASE_DIR}/${CODE_DIR}:/mnt/code  \
--container-image=${BASE_DIR}/${ENV_DIR}/msadet.sqsh \
--container-workdir=/mnt/code \
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CL


#### sbatch train.sbatch