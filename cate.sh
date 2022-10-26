#!/bin/bash
DATASET=yelp
CATE_DIR=../../yumeng5/CatE
NOW=$(date +"%Y-%m-%d-%H-%M")
OUT_DIR=./results/$DATASET/cate
mkdir -p $OUT_DIR
cd $CATE_DIR && \
bash ./run.sh && \
cd ../../zhangchi0104/honours-sgtm && \
cp ../../yumeng5/CatE/datasets/$DATASET/emb_seeds* $OUT_DIR/ && \
cp ../../yumeng5/CatE/datasets/$DATASET/res_seeds.txt $OUT_DIR/ 