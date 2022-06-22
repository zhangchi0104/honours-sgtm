#!/bin/bash
CATE_DIR=../../yumeng5/CatE
cd $CATE_DIR && \
bash ./run.sh && \
cd ../../zhangchi0104/honours-sgtm && \
cp ../../yumeng5/CatE/datasets/agnews/emb_seeds* ./results/CatE/ && \
cp ../../yumeng5/CatE/datasets/agnews/res_seeds.txt ./results/CatE/ 