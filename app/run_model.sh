#!/bin/bash

source /home/natasamortvanski@vhio.org/.bashrc
source activate /home/natasamortvanski@vhio.org/miniconda3/envs/new_env/envs/sc_env_new

python /home/natasamortvanski@vhio.org/CARE/app/predict_model.py /home/natasamortvanski@vhio.org/CARE/app/tmp_files/parsed_data.tsv