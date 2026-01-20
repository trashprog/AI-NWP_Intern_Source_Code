#!/bin/bash
set -e

echo "Creating and activating Aurora virtual environment..."
conda create -n aurora_env python=3.10 -y
conda activate aurora_env

echo "Downloading Aurora checkpoint..."
cd /home/project/17001770/weather_department/nwp/zach/aurora_folder # change your path do not use mine
wget https://huggingface.co/microsoft/aurora/blob/main/aurora-0.1-finetuned.ckpt

echo "Installing Aurora library via mamba..."
mamba install microsoft-aurora -c conda-forge -y

echo "Aurora setup complete!"
