#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/industry/connect/zachloy/log/
#PBS -l select=1:ncpus=16:ngpus=1:mem=500gb -l walltime=24:00:00
#PBS -P 17001770
#PBS -N aurora_sfc

# --- Load environment ---
module purge
module load miniforge3
source /home/app/apps/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate aurora_env                       # Activate your Conda environment
export PATH=~/mars/bin/:$PATH                  # Ensure 'mars' command is in PATH

# --- Go to MARS directory ---
cd /home/users/industry/connect/zachloy/mars/bin

# --- Log file ---
LOG_FILE="/home/users/industry/connect/zachloy/aurora_scripts/aurora_sfc.log"

# --- Monthly date ranges for retrieval ---
# Comment out months that are already done
date_ranges=(
# "20231231/to/20240131" # done
# "20240201/to/20240229" # done
# "20240301/to/20240331" # done
# "20240401/to/20240430" # done
# "20240501/to/20240531" # done
# "20240601/to/20240630" # done
"20240701/to/20240731"
"20240801/to/20240831"
"20240901/to/20240930"
"20241001/to/20241031"
"20241101/to/20241130"
"20241201/to/20241231"
"20250101/to/20250101"
)

# --- Loop over each month ---
for i in "${!date_ranges[@]}"; do
  range="${date_ranges[$i]}"
  
  # Index for file naming
  # The +7  accounts for previous months already done
  index=$((i+7))
  
  output="/home/project/17001770/ccrs_dwr/nwp/zach/aurora_folder/aurora_sfc_month_${index}_highres.grib"

  echo "Running mars $range at $(date)" >> $LOG_FILE

  # --- MARS retrieval command ---
  # Use stdbuf to force line-buffered output to log
  stdbuf -oL -eL mars <<EOF >> $LOG_FILE 2>&1
RETRIEVE,
    CLASS      = OD,
    TYPE       = AN,
    STREAM     = OPER,
    EXPVER     = 0001,
    LEVTYPE    = SFC,
    PARAM      = 167/165/166/151, # 2t, 10u, 10v, msl
    TIME       = 00/06/12/18,
    STEP       = 00,
    DOMAIN     = G,
    RESOL      = AUTO,
    AREA       = 90/-180/-90/180,
    GRID       = 0.1/0.1,
    PADDING    = 0,
    DATE       = $range,
    TARGET     = "$output"
EOF

  echo "Completed $range at $(date)" >> $LOG_FILE
done