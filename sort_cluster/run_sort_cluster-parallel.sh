#!/bin/bash
set -e
# input looks like "1 LOT2/ANGUILLA_20210128_20210318 mel_BBF"
site=$2
config=$3
# cluster_size=$3
# sample=$4


source /applis/environments/conda.sh
conda activate carimam


time python ./sort_cluster-parallel.py $site $config

rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stdout
rm -f $HOME/OAR.cigri.$CIGRI_CAMPAIGN_ID.$OAR_JOB_ID.stderr