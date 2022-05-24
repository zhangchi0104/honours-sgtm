#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=1:00:00
#PBS -M chi.zhang@student.unsw.edu.au

conda activate honours
cd honours-sgtm
python3 gen_tokens.py \
    --out  /srv/scratch/z5211214 \
    --pretrained