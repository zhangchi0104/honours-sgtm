for dataset in "20NewsGroup" "yelp" "scidoc"; do
    # obtain word sets
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    echo "Generating Word Sets For $dataset"
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    python3 scripts/similarities.py \
        --output=results/$dataset/bert/similarities_finetuned.pkl \
        --vocab=data/$dataset/vocab/vocab.pkl \
        --seeds=data/$dataset/seeds.json \
        --out_words=results/$dataset/bert/finetuned_word_sets.json \
        bert \
        --weights=models/$dataset/bert/finetuned-model \
        --tokenizer=models/$dataset/bert/tokenizer

    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "Evaulating finetunedBERT reults For $dataset"
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
    mkdir -p ./results/$dataset/evaluations/
    python3 scripts/evaluation.py \
        --out ./results/$dataset/evaluations/finetuned-bert.json \
        -c ./data/$dataset/cooccurence.csv \
        -v ./data/$dataset/vocab/vocab.pkl \
        results/$dataset/bert/finetuned_word_sets.json
done