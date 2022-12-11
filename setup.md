# Setup Environment with conda
1. Install Anaconda or Miniconda
2. Install the packages
    - **NOTE**: Some packages may not be comaptible with your environment, you might
    want to manually install them as you go
```bash
$ conda env create -f env.yml
$ conda activate honours-sgtm
```
# Processing datasets
Dataset are supposed in raw text. Eeach line represents a document
- Preprocess your dataset with `scripts/clean_text.py`
```bash
$ python3 scripts/clean_text.py -o out_put_path your_dataset
```
- Follow the guide in AutoPhrase to find phrases (joins words by underscore `_`)

# Build vocabulary
- To build vocabulary run
```bash 
# filter out the words with length less than 3 characters
$ python3 scripts/vocabulary.py --min_count 3 -v your/path/to/output/yocab
```
- [Optional] If you are evaluating the results with PMI and NPMI, you can add an extra
argument `-c /path/to/occurrence_matrix`
- Put your seeds as seeds.json in the dataset root

# Train BERT 
- To fine tune BERT you should modify the `train.sh` with the correct hyper parameters
```
$ bash train.sh
``` 
You can run `python3 scripts/train.py --help` for detailsd usages

# Extracting embeddings
**NOTE**: You need to make sure the output directory exists before calling this function
And the files should be in the right locations, you can add `--dry_run` 
to check these locations

- You can run `python3 similarities.py` to extract embeddings
``` bash
# Compute cosine similarities and embeddings for 20NewsGroup Dataset with FIne tuned BERT
$ python3 similarities.py -m bert -d 20NewsGroup --output embeddings

# Compute cosine similarities and dump wordsets for yelp Dataset with rank redeuced 
# embeddings
$ python similarities.py -m bert -d yelp --output words -r 
``` 

# Rank reduction
- Performs rank reduction with UMAP
```bash
$ python3 scripts/rank_reduction.py \ 
    --similarities path/to/similarities.pkl \
    --ndim 80 \ # 10 - 100 
    --embeddings path/to/embeddings.pkl \
    --output path/to/output.pkl
```

# Ensemble ranking
- Combines the score with `scripts/ensemble.py`, This requires users to provide cosine
similarititeis of scores for both local knowledge model and global knowlege model
```bash
$ python3 scripts/ensemble.py -g path/to/global_score.pkl \
    -l path/to/local_score.pkl \
    --global_weight 0.5 \
    --local_weight 0.5 \
    --rho 0.5
```

# Evaluation
- Compute matrics on the results
```bash
python3 scripts/evaluation.py \
    --cooccur path/to/co-occurence.py \
    --out path/to/evaluation.json \
    --vocab path/to/vocabulary.pkl \
    similarities.pkl 
```