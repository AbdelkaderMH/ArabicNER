#!/bin/bash
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total? (<= nodes*56)
#SBATCH --ntasks=1
#SBATCH -A data_sec-6sevvl76uja-default-gpu
#! How much wallclock time will be required?
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# get tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="toubkal.hpc"
port="8464"
# you can chose a number like 8889
omp_threads=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$omp_threads
# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.um6p.ma

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)"

module load Anaconda3
module load CUDA/11.3.1
source ~/.bashrc

conda activate nids
jupyter-lab --no-browser --port=${port} --ip=${node}
