#!/bin/bash
set -e
module load miniforge3 # this is for loading conda
module load cuda/12.2.2
module load cudnn/12-9.8.0.87
echo "Creating and activating Aurora virtual environment..."
cd /home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/Aurora

# conda create -p /home/users/industry/connect/zachloy/scratch/zachloy/aurora_env python=3.10 -y
conda env create --prefix /home/users/industry/connect/zachloy/scratch/zachloy/aurora_env --file aurora_env.yml
conda activate /home/users/industry/connect/zachloy/scratch/zachloy/aurora_env

echo "Downloading Aurora checkpoint..."
wget https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.1-finetuned.ckpt

echo "Installing Aurora library via conda..."
# conda install microsoft-aurora -c conda-forge -y
pip install -r /home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/requirements.txt
conda install -c conda-forge esmf esmpy -y

echo "Aurora setup complete!"
