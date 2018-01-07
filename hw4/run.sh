#!/bin/bash -ex

echo "Usage : ./run.sh [testing_text.txt]"

test_txt=$1

python3 generate.py --test_txt $test_txt



