#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/industry/connect/zachloy/log/
#PBS -l select=1:ncpus=16:ngpus=1:mem=500gb -l walltime=24:00:00
#PBS -P 17001770
#PBS -N aifs_pl

# Commands start here

module purge
module load miniforge3
conda activate aifs_env #you can change ur env here
export PATH=~/mars/bin/:$PATH
cd /home/users/industry/connect/zachloy/mars/bin

LOG_FILE="/home/users/industry/connect/zachloy/aifs_scripts/aifs_output_pl.log"

# date range
date_ranges=(
"20231231/to/20240229"
"20240301/to/20240531"
"20240601/to/20240831"
"20240901/to/20250101"
)
for i in "${!date_ranges[@]}"; do
  range="${date_ranges[$i]}"
  index=i
  output="/home/project/77010001/ccrs_dwr/nwp/zach/aifs_folder/aifs_pl_${index}.grib"

  echo "Running mars $range" >> $LOG_FILE

  stdbuf -oL -eL mars <<EOF >> $LOG_FILE 2>&1
	RETRIEVE,
		CLASS      = OD,
		TYPE       = AN,
		STREAM     = OPER,
		EXPVER     = 0001,
		LEVTYPE    = PL,
		LEVELIST = 1000/925/850/700/600/500/400/300/250/200/150/100/50,
		PARAM      = 129/130/131/132/133/135,   # z, t, u, v, q, w
		TIME       = 00/06/12/18,
		STEP       = 00,
		DOMAIN     = G,
		RESOL      = AUTO,
		AREA       = 90/-180/-90/180,
		GRID       = 0.25/0.25,
		PADDING    = 0,
		DATE       = $range,
		TARGET     = "$output"
EOF
	echo "completed $range" >> $LOG_FILE
done

