#!/bin/bash
wget https://sparc-dataset.s3.us-east-2.amazonaws.com/model.tar.gz
# Copying the input file to /tmp because /orb-dev in read-only.
cp /orb-dev/dev_noanswer.json /tmp/input.json
python predictor.py model.tar.gz /tmp/input.json predictions.json
cp predictions.json /results/predictions.json
