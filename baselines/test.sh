#unset LD_LIBRARY_PATH
#unset LD_PRELOAD
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
#
#echo $SLURM_JOB_GPUS
#export GPUS=$SLURM_JOB_GPUS
export MUJOCO_GL='egl'
export D4RL_DATASET_DIR='/iris/u/kylehsu/data/d4rl'
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export WANDB_MODE=offline
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u IQL_trainer.py \
--env_name randomized_kitchen_kettle-v1 \
--tqdm=true \
--eval_episodes 100 \
--eval_interval 10000 \
--seed 3