#!/bin/bash
set -e
# This is an example for the lot 2
# This code allows to choose if turn on cpu or gpu
i=$1  # an int

source /applis/environments/cuda_env.sh dahu 10.1
source /applis/environments/conda.sh
conda activate carimam
cd /bettik/PROJECTS/pr-carimam/monsimau/script/CNN/

if [ `oarprint hasgpu` == 'NO' ]
then 
echo "run on cpu"
python forward_UpDimV2_long-parallel-cpu.py /bettik/PROJECTS/pr-carimam/monsimau/script/CNN/file_list_lot2_shuffle.txt --overlap 5 --output_path "/bettik/PROJECTS/pr-carimam/monsimau/script/CNN/results_LOT2/" --first_file $i --number_files 300 --lot "LOT2"
elif [ `oarprint hasgpu` == 'YES' ]
then 
echo "run on gpu"
python forward_UpDimV2_long-parallel-gpu.py /bettik/PROJECTS/pr-carimam/monsimau/script/CNN/file_list_lot2_shuffle.txt --overlap 5 --output_path "/bettik/PROJECTS/pr-carimam/monsimau/script/CNN/results_LOT2/" --first_file $i --number_files 300 --lot "LOT2"
fi

rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stdout
rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stderr