#！/bin/bash
dataset=${1:-yelp}
python3 scripts/cate_seeds.py gen_seeds \
    --vocab=data/$dataset/vocab/vocab.pkl \
    --seeds=data/$dataset/seeds.json \
    --similarities=results/$dataset/bert/similarities.pkl \
    --out=/dev/null
