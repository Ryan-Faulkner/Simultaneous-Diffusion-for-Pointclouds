#!/bin/bash
mkdir '/data/PreGenImages'
mkdir '/data/PreGenImages/Depth'
mkdir '/data/PreGenImages/Intensity'
mkdir '/data/PreGenImages/Mask'
mkdir '/data/exp'
mkdir '/data/exp/logs'
mkdir '/data/exp/tensorboard'
mkdir '/data/exp/logs/HDVMine'
pip install scikit-learn

cd LiDARGen
#THIS TRAINBASH IS NOT USED - I JUST CALL MAIN.PY DIRECTLY
python main.py --doc "HDVMine" --resume_training --ni
