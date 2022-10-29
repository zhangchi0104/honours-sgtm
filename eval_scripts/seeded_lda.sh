for dataset in "20NewsGroup" "yelp" "scidoc"; do
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    echo "Evaulating SeededLDA results for $dataset"
    echo "+++++++++++++++++++++++++++++++++++++++++++++"
    result_json=./results/$dataset/seeded_lda/result.json
    python3 scripts/guided_lda.py $dataset
    python3 scripts/evaluation.py \
        --out ./results/$dataset/evaluations/seeded_lda.json \
        -c ./data/$dataset/cooccurence.csv \
        -v ./data/$dataset/vocab/vocab.pkl \
        $result_json > ./results/$dataset/evaluations/seeded_lda.output
done