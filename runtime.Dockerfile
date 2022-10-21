ARG VERSION=22.06

FROM nvcr.io/nvidia/pytorch:${VERSION}-py3

RUN pip3 install --upgrade pip
RUN pip3 install pytorch-lightning \
    wandb \
    transformers 
RUN pip3 install rich
