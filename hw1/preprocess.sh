#!/bin/bash -ex

echo "Usage: bash python3 preprocess.py [train] [test]"

train_ark='./data/fbank/train.ark'
test_ark='./data/fbank/test.ark'
lab='./data/train.lab'
train_output_dir='./data/train.tfr'
test_output_dir='./data/test.tfr'

for var in "$@"
do
    if [ "$var" == "train" ]
    then
        python3 preprocess.py \
                --mode  train \
                --ark   $train_ark \
                --lab   $lab \
                --output_dir    $train_output_dir
    elif [ "$var" == "test" ]
    then
        python3 preprocess.py \
                --mode  test \
                --ark   $test_ark \
                --output_dir    $test_output_dir
    fi
done