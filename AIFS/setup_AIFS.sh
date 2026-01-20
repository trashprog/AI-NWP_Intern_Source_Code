#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating and activating AIFS virtual environment..."
conda create -n aifs_env python=3.10 -y
conda activate aifs_env

echo "Installing Git Large File System (LFS)..."
wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar -xvzf git-lfs-linux-amd64-v3.5.1.tar.gz
export PATH=$HOME/bin:$PATH
git lfs install

echo "Downloading AIFS checkpoint..."
cd /home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder # change your path
wget https://huggingface.co/ecmwf/aifs-single-1.0/blob/main/aifs-single-mse-1.0.ckpt

echo "Installing AIFS dependencies..."
pip install anemoi-inference[huggingface]==0.4.9 anemoi-models==0.3.1 torch==2.4.0
pip install earthkit-regrid==0.4.0 ecmwf-opendata 
pip install flash_attn

echo "AIFS setup complete!"
