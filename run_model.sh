#!/bin/bash
wget https://ropes-model.s3-us-west-1.amazonaws.com/model.tar.gz
# Copying the input file to /tmp because /orb-dev in read-only.
cp /orb-data/orb_nolabels.jsonl /tmp/input.jsonl
python predictor.py model.tar.gz /tmp/input.jsonl predictions.json
cp predictions.json /results/predictions.json
