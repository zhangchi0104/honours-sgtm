#!/bin/bash

WANDB_API_KEY=625102dbe41c4af6314e1be904fe8102adfb1ca5 python3 scripts/train.py \
    --dataset_path=data/agnews/corpus/agnews_cleaned.txt \
    --batch_size=6 \
    --accelerator=cuda \
    --max_epochs=10 \
    --output_dir=./models/agnews \
    --num_workers=16