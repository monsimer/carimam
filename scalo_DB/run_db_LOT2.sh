#!/bin/bash
set -e
i=$1  # an int, number of first file to read in files_LOT2_1%.txt

source /applis/environments/conda.sh
conda activate carimam
cd /bettik/PROJECTS/pr-carimam/monsimau/script/scalo_V2_db/
# fichier_time="time_exec.txt"
# START=$(date +%s)

time python Scalogram_comparisons_daubechies_CWT.py $i files_LOT2_1%.txt 100 results_LOT2_1%/

# END=$(date +%s)
# DIFF=$(( $END - $START ))
# echo "$DIFF" >> $fichier_time
rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stdout
rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stderr