#!/usr/bin/env bash


ROOT_DIR="/Users/ashish/files/research/projects/value_belief_bot_streamlit/vbbr_bot_rproj"
DATA_DIR="${ROOT_DIR}/data/"
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
DATESTAMP=`date +%Y-%m-%d`
RAW_DIR="${DATA_DIR}/raw/${DATESTAMP}"

mkdir ${RAW_DIR}

~/files/scripts/qualtrics_module.py responses SV_0D1D6Va01196rHw --output_dir ${RAW_DIR}
