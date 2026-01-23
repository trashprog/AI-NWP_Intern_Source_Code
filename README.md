# Model Installation Guide

This repository is managed by Zachariah Loy Yiqi, AI-NWP intern in CCRS, the files here contain codes needed for input downloading, regridding, plotting, along with running and saving scripts for evaluating and bench marking AI-NWP models **AIFS** and **Aurora**. AIFS was ran in Aspire2A+ while Aurora was ran in Aspire2A
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

#### 4. Running the pipeline

1. After the data is downloaded, run **aifs_splitdata.py** to split up the pressure and surface files. Then run **aifs_merge2input.py** to merge the files together surface and pressure on each corresponding timestamps, it also handles the two timestamp requirement for AIFS runs.

2. After processing the inputs, run **aifs_run_inference.py** to produce the forecasts

3. After all that is done, assuming the truth data has been downloaded and processed you can run the plotting scripts.
    - **aifs_trwb.py** > computes all RMSE results for traditional regional weatherbench entire year of 2024
    - **aifs_trwb_monsoon.py** > computes all RMSE results for traditional regional weatherbench for three monsoon periods:
        - Inter-Monsoon (Apr, May, Oct, Nov)
        - North-East Monsoon (Dec, Jan, Feb, Mar) 
        - South-West Monsoon (Jun, Jul, Aug, Sep).

    - **aifs_drwb_pl.py** > computes all RMSE results for dynamic regional weatherbench entire year of 2024 for PRESSURE variables only
    - **aifs_drwb_pl_monsoon.py** > computes all RMSE results for dynamic regional weatherbench for the same three monsoon periods, PRESSURE variables only
    - **aifs_drwb_sfc.py** > computes all RMSE results for dynamic regional weatherbench entire year of 2024 for SURFACE variables only
    - **aifs_drwb_sfc_monsoon.py** > computes all RMSE results for dynamic regional weatherbench for the same three monsoon periods, SURFACE variables only
    - **aifs_hke.py** > computes the radical average spectrum for each 48 hour forecast group and outputs them into individual grouped files
    - **aifs_gpm_rmse.py** > computes the rmse for the forecasted precipitation variable against the gpm truth


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

#### 3. Running the pipeline
1. Download the data by running **au_getdata_sfc.sh** and **au_getdata_pl.sh**. Since the downlaods would take super long for the pressure variables, you can also run **au_getdata_pl_noPBS.sh** via nohup, this allows you to continue downloading without worrying about the walltime in normal PBS queues.

2. After the data is downloaded, run **au_splitdata.py**, this splits the files into 6 hourly files.

3. Then run **au_merge2input.py** to merge the pressure and surface files together on each timestamp.

4. After merging, run **au_run_inference.py** to produce the forecasts.

5. Run **au_regrid.py** to regrid the outputs to 0.25 degrees resolution, the weights are provided in the same folder.

6. After all that is done, assuming the truth data has been downloaded and processed you can run the plotting scripts.
    - **au_trwb.py** > computes all RMSE results for traditional regional weatherbench entire year of 2024
    - **au_trwb_monsoon.py** > computes all RMSE results for traditional regional weatherbench for three monsoon periods:
        - Inter-Monsoon (Apr, May, Oct, Nov)
        - North-East Monsoon (Dec, Jan, Feb, Mar) 
        - South-West Monsoon (Jun, Jul, Aug, Sep).

    - **au_drwb_pl.py** > computes all RMSE results for dynamic regional weatherbench entire year of 2024 for PRESSURE variables only
    - **au_drwb_pl_monsoon.py** > computes all RMSE results for dynamic regional weatherbench for the same three monsoon periods, PRESSURE variables only
    - **au_drwb_sfc.py** > computes all RMSE results for dynamic regional weatherbench entire year of 2024 for SURFACE variables only
    - **au_drwb_sfc_monsoon.py** > computes all RMSE results for dynamic regional weatherbench for the same three monsoon periods, SURFACE variables only
    - **au_hke.py** > computes the radical average spectrum for each 48 hour forecast group and outputs them into individual grouped files
    - **au_drwb_sfc_highres.py** > computes all RMSE results for dynamic regional weatherbench entire year of 2024 for SURFACE variables only at 0.1 degrees resolution for the purpose of comparing it with silurian 0.1. 


#### 4. Evaluation test cases
For testing purposes, two scripts can be used to run a full pipeline from running inference -> regridding/processing -> producing rmse results in traditional and dynamic weather bench metrics. To downloaded the test files, go to https://huggingface.co/datasets/DaquaviousDinglenut/ai-nwp-data-assets/tree/main 

Below are where you should move the files to after downlaoding them:
- truth/* -> Truth/truth_files
- Aurora/aurora_inputs -> Aurora/au_input_files
- Aurora/aurora_bilinear_0p25_weights.nc -> Aurora
- aifs_inputs/* -> aifs_input_files

#### 5. Alternate installation
There is a bash script in each model folder titled setup_<model_name>.sh, you can also run them to fully install everything by running the script, do remember to change the paths inside based on where you want to install your environments in.





