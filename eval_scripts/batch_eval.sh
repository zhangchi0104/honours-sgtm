for dataset in 'dbpedia'
do
    # compute BERT embeddings

    python3 similarities.py -m bert -d $dataset --output embeddings 

    # compute CaTE embeddings
    python3 similarities.py -m cate -d $dataset --output embeddings 
    # Normal evaluation
    python3 scripts/batch_eval.py -d $dataset  -n ensembe
    # rank reduction
    python3 scripts/rank_reduction.py \
        -s ./results/$dataset/bert/similarities.pkl \
        -e ./results/$dataset/bert/embeddings.pkl \
        -o ./results/$dataset/bert/embeddings_reduced.pkl && \
    # Compute reduced Embeddings
    python3 similarities.py -m bert -d $dataset -e -r
    python3 scripts/batch_eval.py \
        -d $dataset \
        -n evaluation_reduced  \
        -r 
done