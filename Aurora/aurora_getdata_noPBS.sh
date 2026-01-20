#!/bin/bash
# Script to download ERA5 Pressure Level data for missing weeks.
# This version is designed to run directly via bash (e.g., nohup bash ... &)
# instead of submitting to a PBS queue.

# --- Load environment ---
module purge
module load miniforge3
# You can uncomment if needed: source /home/app/apps/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate aurora_env                 # Activate Conda environment
export PATH=~/mars/bin/:$PATH              # Ensure 'mars' command is in PATH

# --- Move to MARS script directory ---
cd /home/users/industry/connect/zachloy/mars/bin

# --- Unique log file for this run ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/home/users/industry/connect/zachloy/aurora_scripts/aurora_missing_${TIMESTAMP}.log"

# --- Function to generate weekly date ranges ---
generate_weeks() {
  start_date=$1
  end_date=$2
  current_date=$(date -d "$start_date" +%Y%m%d)
  final_date=$(date -d "$end_date" +%Y%m%d)

  while [ "$current_date" -le "$final_date" ]; do
    week_start=$current_date
    week_end=$(date -d "$current_date +6 days" +%Y%m%d)
    
    # Prevent exceeding the final date
    if [ "$week_end" -gt "$final_date" ]; then
      week_end=$final_date
    fi
    
    echo "$week_start/to/$week_end"
    
    # Move to next week
    current_date=$(date -d "$current_date +7 days" +%Y%m%d)
  done
}

# --- Define data period ---
start="20231231"
end="20250101"

# --- Generate array of weekly date ranges ---
mapfile -t date_ranges < <(generate_weeks "$start" "$end")

# --- Specify missing weeks to download ---
# Only these weeks will be retrieved
# Uncomment or change to select weeks
# MISSING_WEEKS=(5 9 10 14 15)
# MISSING_WEEKS=(19 20)
# MISSING_WEEKS=(20 24 25 28 29 30)
# MISSING_WEEKS=(31 32 33 34 35 36 37 38 39 40 53)
# MISSING_WEEKS=(36 37 38 39 40)
MISSING_WEEKS=(32)  # Active week for download
# MISSING_WEEKS=(46 47 48 49 50)
# MISSING_WEEKS=(53)

# --- Loop over missing weeks ---
for idx in "${MISSING_WEEKS[@]}"; do
  # Bash arrays are 0-indexed; weeks are 1-indexed
  range="${date_ranges[$((10#$idx - 1))]}"
  
  # Output file path
  output="/home/project/17001770/weather_department/nwp/intern_sharing/aurora_inputs/pl/aurora_pl_week_$(printf '%02d' "$idx")_highres.grib"

  # Skip if already downloaded
  if [ -f "$output" ] && [ -s "$output" ]; then
    echo "Skipping week_${idx} (already complete)" >> "$LOG_FILE"
    continue
  fi

  # Remove empty file from previous failed attempt
  if [ -f "$output" ] && [ ! -s "$output" ]; then
    echo "Removing empty file for week_${idx}" >> "$LOG_FILE"
    rm -f "$output"
  fi

  # Log start of download
  echo "Downloading week_${idx} ($range) at $(date)" >> "$LOG_FILE"

  # --- Execute MARS retrieval ---
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
