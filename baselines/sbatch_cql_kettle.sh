#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=164:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=64G
#SBATCH --job-name="CQL_kettle"
#SBATCH --account=iris

cd /iris/u/khatch/vd5rl/jaxrl2/baselines
source ~/.bashrc
# conda init bash
source /iris/u/khatch/anaconda3/bin/activate
source activate jaxrl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $SLURM_JOB_GPUS
export GPUS=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

which python
which python3
nvidia-smi
pwd
ls -l /usr/local
python3 -u gpu_test.py



XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_trainer.py \
--env_name randomized_kitchen_kettle-v1 \
--tqdm=true \
--eval_episodes 100 \
--eval_interval 10000 \
--debug True \
--seed 3

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u CQL_trainer_walker.py \
--env_name walker_walk \
--tqdm=true \
--eval_episodes 20 \
--project walkertest \
--eval_interval 1000 \
--seed 3

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u IQL_trainer.py \
--env_name randomized_kitchen_kettle-v1 \
--tqdm=true \
--eval_episodes 100 \
--eval_interval 10000 \
--seed 3
