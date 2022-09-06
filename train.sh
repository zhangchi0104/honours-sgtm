#!/bin/bash

python3 scripts/train.py \
    --dataset_path=data/agnews/corpus/agnews_cleaned.txt \
    --batch_size=45 \
    --accelerator=cuda \
    --max_epochs=10 \
    --output_dir=./models/agnews \
    --num_workers=16 \