#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/industry/connect/zachloy/log/auro_inf.log
#PBS -l select=1:ncpus=16:ngpus=1:mem=500gb -l walltime=6:00:00
#PBS -P 17001770
#PBS -N aurora_run

# --- Load modules first ---
module purge
module load miniforge3
module load cuda/12.2.2
module load cudnn/12-9.8.0.87

# --- Activate conda environment ---
conda activate /home/users/industry/connect/zachloy/scratch/zachloy/aurora_env

LOG_FILE="/home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/au_run_evaluation_pipeline.log"
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> $LOG_FILE
echo "Job started at $(date)" >> $LOG_FILE

stdbuf -oL -eL python -u /home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora/au_run_evaluation_pipeline.py 1 >> $LOG_FILE 2>&1 &

wait
echo "Job finished at $(date)" >> $LOG_FILE
