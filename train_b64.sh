#!/bin/bash

python train_aug_512_final.py --img_size 512 -b 64
python train_aug_512_final.py --img_size 768 -b 32
python train_aug_512_final.py --img_size 1024 -b 16
