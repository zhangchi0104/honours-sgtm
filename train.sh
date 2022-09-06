#!/bin/bash

python3 scripts/train.py \
    --dataset_path=data/agnew/corpus/agnews_cleaned.txt \
    --batch_size=6 \
    --accelerator=cuda \
    --max_epochs=10 \
    --output_dir=./models/agnews