#!/bin/bash

# Preparing folders and data to process
mkdir -p checkpoints
mkdir -p data

pip install -q gdown
gdown 1uAf4VEbok7CI5W34F7SzsJGU2t3YzN5W -O data/datasets.tar.gz
cd data 
tar -xzvf datasets.tar.gz

