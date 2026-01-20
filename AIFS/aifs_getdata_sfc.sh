#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/industry/connect/zachloy/log/
#PBS -l select=1:ncpus=16:ngpus=1:mem=500gb -l walltime=24:00:00
#PBS -P 17001770
#PBS -N aifs_sfc

# Commands start here
module purge
module load miniforge3
conda activate aifs_env #you can change ur env here
export PATH=~/mars/bin/:$PATH
cd /home/users/industry/connect/zachloy/mars/bin

LOG_FILE="/home/users/industry/connect/zachloy/aifs_scripts/aifs_output_sfc.log" # change your path

stdbuf -oL -eL mars <<EOF >> $LOG_FILE 2>&1

	RETRIEVE,
    	CLASS      = OD,
    	TYPE       = AN,
    	STREAM     = OPER,
    	EXPVER     = 0001,
    	LEVTYPE    = SFC,
    	PARAM      = 165/166/168/167/151/235/134/136/172/129/163/160, # 10u, 10v, 2d, 2t, msl, skt, sp, tcw, lsm, z, slor, sdor
    	TIME       = 00/06/12/18,
    	STEP       = 00,
    	DOMAIN     = G,
    	RESOL      = AUTO,
    	AREA       = 90/-180/-90/180,
   	    GRID       = 0.25/0.25,
    	PADDING    = 0,
    	DATE       = 20231231/to/20250101,
    	TARGET     = "/home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder/aifs_sfc.grib" # change output path here
EOF

