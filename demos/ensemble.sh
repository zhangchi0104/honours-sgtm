#! /bin/bash
dataset=$1
wg=${2:-0.5}
wl=${3:-0.5}
rho=${4:-0.5}
python3 scripts/ensemble_ranking.py \
    --global_df=./results/$dataset/bert/similarities.pkl \
    --local_df=./results/$dataset/cate/similarities.pkl \
    --output=./ensemble_domo.pkl \
    --local_weight=$wl \
    --global_weight=$wg \
    --rho=$rho \
    --dry_run \
    --verbose
