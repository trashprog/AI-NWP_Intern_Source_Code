#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/industry/connect/zachloy/log/
#PBS -l select=1:ncpus=1:ngpus=1:mem=20gb -l walltime=6:00:00
#PBS -P 17001770
#PBS -N aurora_pl

module purge
module load miniforge3
source /home/app/apps/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate aurora_env
export PATH=~/mars/bin/:$PATH

# --- Move to MARS script directory ---
cd /home/users/industry/connect/zachloy/mars/bin

# --- Unique log file for this run ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/users/industry/connect/zachloy/aurora_scripts/aurora_missing_${TIMESTAMP}.log"

# --- Function to generate week ranges from start to end ---
generate_weeks() {
  start_date=$1
  end_date=$2
  # Convert to YYYYMMDD integer format
  current_date=$(date -d "$start_date" +%Y%m%d)
  final_date=$(date -d "$end_date" +%Y%m%d)
  
  while [ "$current_date" -le "$final_date" ]; do
    week_start=$current_date
    # Calculate week end as 6 days after week_start
    week_end=$(date -d "$current_date +6 days" +%Y%m%d)
    
    # Don't go past the final_date
    if [ "$week_end" -gt "$final_date" ]; then
      week_end=$final_date
    fi
    
    # Print the range in the format expected by MARS
    echo "$week_start/to/$week_end"
    
    # Move to next week
    current_date=$(date -d "$current_date +7 days" +%Y%m%d)
  done
}

# --- Define the start and end of your data period ---
start="20231231"
end="20250101"

# --- Generate all weekly date ranges within the period ---
# This will produce an array where each index corresponds to a week
mapfile -t date_ranges < <(generate_weeks "$start" "$end")

# --- Specify which weeks are missing (to be downloaded) ---
# Weeks are 1-indexed in your human-readable list
# Example: MISSING_WEEKS=(5 9 10 14 15)
# You can comment/uncomment sets of weeks as needed
# MISSING_WEEKS=(19 20)
# MISSING_WEEKS=(24 25 28 29 30)
# MISSING_WEEKS=(31 32 33 34 35)
# MISSING_WEEKS=(36 37 38 39 40)
# MISSING_WEEKS=(41 42 43 44 45)
# MISSING_WEEKS=(46 47 48 49 50)
# MISSING_WEEKS=(51 52)

# --- Loop through each missing week and download ---
for idx in "${MISSING_WEEKS[@]}"; do
  # Bash arrays are 0-indexed; your week numbers are 1-indexed
  range="${date_ranges[$((10#$idx - 1))]}"
  
  # Output file for this week's data
  output="/home/project/17001770/ccrs_dwr/nwp/zach/aurora_folder/aurora_pl_downloads/aurora_pl_week_$(printf '%02d' "$idx")_highres.grib"

  # Skip if file already exists and is non-empty
  if [ -f "$output" ] && [ -s "$output" ]; then
    echo "Skipping week_${idx} (already complete)" >> "$LOG_FILE"
    continue
  fi

  # Remove file if it exists but is empty (failed previous run)
  if [ -f "$output" ] && [ ! -s "$output" ]; then
    echo "Removing empty file for week_${idx}" >> "$LOG_FILE"
    rm -f "$output"
  fi

  # Log start of download
  echo "Downloading week_${idx} ($range) at $(date)" >> "$LOG_FILE"

  # --- MARS retrieval command ---
  # stdbuf -oL -eL forces line-buffered output to the log
  stdbuf -oL -eL mars <<EOF >> "$LOG_FILE" 2>&1
RETRIEVE,
    CLASS      = OD,
    TYPE       = AN,
    STREAM     = OPER,
    EXPVER     = 0001,
    LEVTYPE    = PL,
    LEVELIST   = 1000/925/850/700/600/500/400/300/250/200/150/100/50,
    PARAM      = 130/131/132/133/129,
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

  # Log completion
  echo "Completed week_${idx} ($range) at $(date)" >> "$LOG_FILE"
done

echo "All selected missing weeks completed at $(date)" >> "$LOG_FILE"