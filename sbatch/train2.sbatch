#!/bin/bash
#SBATCH -J daft_a100_
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


BASE_DIR=/dss/dsshome1/01/ge32fix2
CODE_DIR=OrganDETR
ENV_DIR=ENVs_mori
ENV_name=daft_mori
DATA_dir=


EXPERIMENT="OASIS_aff_scales02468_128x_128_a100"


srun --mpi=pmi2 \
--container-mounts=${DATA_DIR}:/mnt/data,${BASE_DIR}/${CODE_DIR}:/mnt/code  \
--container-image=${BASE_DIR}/${ENV_DIR}/ai-med+${ENV_name}+latest.sqsh \
--container-workdir=/mnt/code \
	/opt/conda/envs/${ENV_name}/bin/python train.py --task clf --discriminator_net ${discriminator_net} \
  --batchsize 50  --epoch 201  --learning_rate 0.001\
  --train_data /mnt/data/0-train.h5, \
                /mnt/data/1-train.h5, \
                /mnt/data/2-train.h5, \
                /mnt/data/3-train.h5, \
                /mnt/data/4-train.h5 \
  --val_data /mnt/data/0-valid.h5, \
            /mnt/data/1-valid.h5, \
            /mnt/data/2-valid.h5, \
            /mnt/data/3-valid.h5, \
            /mnt/data/4-valid.h5 \
  --test_data /mnt/data/0-test.h5, \
            /mnt/data/1-test.h5, \
            /mnt/data/2-test.h5, \
            /mnt/data/3-test.h5, \
            /mnt/data/4-test.h5


#### sbatch train.sbatch
####--data_dir ${BASE_DIR}/myData/datasets/${DATASET}
Coll