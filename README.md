This repository is managed by Zachariah Loy Yiqi, AI-NWP intern in CCRS, the files here contain codes needed for input downloading, regridding, plotting, along with running and saving scripts for evaluating and bench marking AI-NWP models **AIFS** and **Aurora**.
<br>
<br>

## Artificial Intelligence Forecasting System (AIFS)
AIFS is based on a GNN encoder–decoder with a sliding-window transformer processor, trained on ECMWF ERA5 reanalysis and operational NWP data, and designed to support multi-level parallelism for high-resolution training. It is ran on 0.25 horizontal degrees resolution


### Setup
#### 1. Create a virtual environment using Anaconda and activate it
```bash
conda create -n aifs_env python=3.10
conda activate aifs_env
```

#### 2. Install Github Large File System (LFS)
```bash
wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
export PATH=$HOME/bin:$PATH
git lfs install
```

#### 3. Install AIFS
```bash
wget https://huggingface.co/ecmwf/aifs-single-1.0/blob/main/aifs-single-mse-1.0.ckpt
pip install anemoi-inference[huggingface]==0.4.9 anemoi-models==0.3.1 torch==2.4.0
pip install earthkit-regrid==0.4.0 ecmwf-opendata 
pip install flash_attn
```


## Aurora
Aurora is a 1.3-billion-parameter flexible 3D Swin Transformer with 3D Perceiver-based encoders and decoders, trained on over a million hours of weather and climate data and fine-tuned using LoRA for specific forecasting tasks. It is ran on 0.1 horizontal degrees resolution.

### Setup
#### 1. Create a virtual environment using Anaconda and activate it
```bash
conda create -n aurora_env python=3.10
conda activate aurora_env
```

#### 2. Download weights and install the Aurora Library
```bash
wget https://huggingface.co/microsoft/aurora/blob/main/aurora-0.1-finetuned.ckpt
mamba install microsoft-aurora -c conda-forge
```




