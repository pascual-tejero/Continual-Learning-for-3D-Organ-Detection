##>>>> active session
salloc -p lrz-dgx-a100-80x8 --ntasks=1 --gres=gpu:1 --time=72:00:00
salloc -p mcml-dgx-a100-40x8 --time 96:00:00 --gres gpu:1 -q mcml



## activate the env
ENV_name=msadet
BASE_DIR=/dss/dsshome1/01/ge32fix2
ENV_DIR=ENVS
DATA_DIR=/dss/dsshome1/01/ge32fix2/DATASETS/
DATA_DIR=/dss/dssmcmlfs01/pr62la/pr62la-dss-0002/Mori/PASCUAL/DATASETS/

srun --pty --container-mounts=${DATA_DIR}:/mnt/data,${BASE_DIR}:/mnt/code \
--container-image=${BASE_DIR}/${ENV_DIR}/${ENV_name}.sqsh bash


cd /mnt/code/OrganDETR
ENV_name=msadet
export TRANSOAR_DATA=/mnt/data/


# test no error
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/TEST_NO_ERROR/test_run_no_error_WORD
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/TEST_NO_ERROR/test_run_no_error_ABDOMENCT-1K
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_CLreg
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_CLreg_fewshot
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_CLreg_onlyClassLabels
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_CLreplay
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_CLreplay_fewshot
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_CLreplay_onlyClassLabels
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_mixing_datasets
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_mixing_datasets_fewshot
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/test_run_no_error_mixing_datasets_onlyClassLabels

# train
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_WORD
/opt/conda/envs/${ENV_name}/bin/python scripts/train.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_ABDOMENCT-1K
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreg
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreg_fewshot
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreg_onlyClassLabels
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreplay
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreplay_fewshot
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_CLreplay_onlyClassLabels
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_mixing_datasets
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_mixing_datasets_fewshot
/opt/conda/envs/${ENV_name}/bin/python scripts/train_CL.py --config CL_methods/ABDOMENCT-1K_WORD/msa_def_detr_mixing_datasets_onlyClassLabels


# test
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_WORD_dataset --model_load best
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_ABDOMENCT-1K_dataset --model_load best
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_CLreg_ABDOMENCT-1K_WORD_datasets --model_load best
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_CLreg_ABDOMENCT-1K_WORD_datasets_few_shot --model_load best
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_ABDOMENCT-1K_WORD_datasets_mixing --model_load best
/opt/conda/envs/${ENV_name}/bin/python scripts/test.py --run msa_def_detr_CLreplay_ABDOMENCT-1K_WORD_datasets --model_load best


# Login to LRZ
ssh login.ai.lrz.de -l ge32fix2

# Example copy dataset to LRZ from workstation
scp -r test_dataset ge32fix2@login.ai.lrz.de:/dss/dsshome1/01/ge32fix2/DATASET

# Quit environment
exit

# Tasks submitted
squeue -u ge32fix2 # My tasks
squeue # All tasks from users