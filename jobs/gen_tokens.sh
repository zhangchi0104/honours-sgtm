#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=1:00:00
#PBS -M chi.zhang@student.unsw.edu.au
#PBS -m ae
conda activate honours
cd $HOME/honours-sgtm
python3 gen_tokens.py \
    --out  /srv/scratch/z5211214 \
    --pretrained