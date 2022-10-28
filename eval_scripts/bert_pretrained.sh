# Pretrained BERT will not work for scidoc as the seeds containing phrases
for dataset in "20NewsGroup" "yelp"; do
    # obtain word sets
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    echo "Generating Word Sets For $dataset"
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    mkdir -p results/$dataset/bert-pretrained/
    python3 scripts/similarities.py \
        --output=results/$dataset/bert-pretrained/similarities.pkl \
        --vocab=data/$dataset/vocab/vocab.pkl \
        --seeds=data/$dataset/seeds.json \
        --out_words=results/$dataset/bert-pretrained/word_sets.json \
        bert \
        --weights=bert-base-uncased \
        --tokenizer=bert-base-uncased 

    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "Evaulating pretrainedBERT reults For $dataset"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
    mkdir -p ./results/$dataset/evaluation/
    python3 scripts/evaluation.py \
        --out ./results/$dataset/evaluation/pretrained-bert.json \
        -c ./data/$dataset/cooccurence.csv \
        -v ./data/$dataset/vocab/vocab.pkl \
        results/$dataset/bert-pretrained/word_sets.json

done