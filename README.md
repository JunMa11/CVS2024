# CVS2024

Solution to the CVS2024 challenge: ConvNext-based multi-label classification

## Environment

- Create virtual environment: `conda create -n autoCVS python=3.12`
- Install [PyTorch](https://pytorch.org/get-started/locally/): `conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`
- `pip install -r requirements.txt`

## Preprocessing

Download the dataset to `data-CVS/videos` and separate the video into frames

```bash
python video2frames.py
```

## Model training

```bash
sh train_b64.sh
```

## Build docker for inference and submission

```bash

```