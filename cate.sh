#!/bin/bash
CATE_DIR=../../yumeng5/CatE
NOW=$(date +"%Y-%m-%d-%H-%M")
OUT_DIR=./results/CatE/$NOW 
mkdir -p $OUT_DIR
cd $CATE_DIR && \
bash ./run.sh && \
cd ../../zhangchi0104/honours-sgtm && \
cp ../../yumeng5/CatE/datasets/agnews/emb_seeds* ./results/$OUT_DIR/ && \
cp ../../yumeng5/CatE/datasets/agnews/res_seeds.txt ./results/$OUT_DIR/ 