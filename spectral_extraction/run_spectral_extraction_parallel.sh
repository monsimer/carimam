#!/bin/bash
set -e
# FOLDER is type of /bettik/PROJECTS/pr-carimam/COMMON/data/LOT2/ANGUILLA_20210128_20210318/20200218_110246UTC_V00OS11.WAV
FOLDER=$1

source /applis/environments/conda.sh
conda activate carimam
cd /bettik/PROJECTS/pr-carimam/monsimau/script/spectral_extraction/

lot="$(cut -d'/' -f7 <<<"$FOLDER")"
session="$(cut -d'/' -f8 <<<"$FOLDER")"
wav="$(cut -d'/' -f9 <<<"$FOLDER")"
OUT_FOLDER="/bettik/PROJECTS/pr-carimam/monsimau/script/spectral_extraction/results/$lot/$session/"
time python spectral_extraction-parallel.py $FOLDER $wav $OUT_FOLDER

rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stdout
rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stderr
