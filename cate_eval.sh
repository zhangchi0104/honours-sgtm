for dataset in "20NewsGroup" "yelp" "scidoc"; do
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    echo "Evaulating CatE reults For $dataset"
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    python3 scripts/evaluation.py \
        --out ./results/$dataset/cate/evaluation.json \
        -c ./data/$dataset/cooccurence.csv \
        -v ./data/$dataset/vocab/vocab.pkl \
        ./results/$dataset/cate/word_sets.json
done