#!/bin/bash
wget https://nabert-model.s3.us-east-2.amazonaws.com/model.tar.gz
cp /orb-devset/all_dev.json /tmp/input.json
# to test without running the model
#python predictor.py model.tar.gz /tmp/input.json predictions.json
cp predictions.json /results/predictions.json