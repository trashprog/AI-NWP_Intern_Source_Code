#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e
module load miniforge3 # this is for loading conda
module load cuda/12.2.2
module load cudnn/12-9.8.0.87
# conda config --add pkgs_dirs /home/users/industry/connect/zachloy/scratch/zachloy/conda_pkgs
# conda clean --all -y

echo "Creating and activating AIFS virtual environment..."
# conda create -p /home/users/industry/connect/zachloy/scratch/zachloy/aifs_env python=3.10 -y
# conda env create --prefix /home/users/industry/connect/zachloy/scratch/zachloy/aifs_env --file aifs_env.yml
conda activate /home/users/industry/connect/zachloy/scratch/zachloy/aifs_env
pip cache purge
echo "environemnt activated"

cd /home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/AIFS # change your path here
# echo "Installing Git Large File System (LFS)..."
# wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
# tar -xvzf git-lfs-linux-amd64-v3.5.1.tar.gz
# export PATH=$HOME/bin:$PATH
# git lfs install

# echo "Downloading AIFS checkpoint..."
# wget https://huggingface.co/ecmwf/aifs-single-1.0/resolve/main/aifs-single-mse-1.0.ckpt

echo "Installing AIFS dependencies..."
pip install --no-cache-dir anemoi-inference[huggingface]==0.4.9 anemoi-models==0.3.1
pip install earthkit-regrid==0.4.0 ecmwf-opendata
pip install torch
pip install flash_attn
pip install -r /home/project/17001770/weather_department/nwp/zach/AI-NWP_Intern_Source_Code/requirements.txt
conda install -c conda-forge esmf esmpy -y

echo "AIFS setup complete!"
