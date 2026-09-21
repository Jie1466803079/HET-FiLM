# Simple and Efficient Heterogeneous Temporal Graph Neural Network

This repository contains the datasets and source code of SE-HTGNN. 
The code and more datasets will be open-sourced after acceptance.

## Environment

```
python=3.9
pytorch == 1.12.1+cu113 (by default our model is trained on GPU)
pytorch-geometric == 2.3.0+cu113
dgl == 1.1.1
numpy == 1.26.4 
```

## Create a virtual environment
```
conda create -n HTGNN python=3.9 -y
conda activate HTGNN
```
## Install PyTorch and DGL and PyTorch-Geometric
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c dglteam/label/cu113 dgl
pip install torch-geometric
pip install numpy==1.26.4 
```

## File Description

`data/`: The wrap of input data.

`model/`: The wrap of an arbitrary architecture.

`run_covid.py`: The script to test trained models on COVID-19 dataset.


## How to run
To test our model in this paper, we run:
```shell
python run_covid.py
```
