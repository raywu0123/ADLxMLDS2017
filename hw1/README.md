# ADLxMLDS2017 hw1 - Sequence Labeling


### Directory structure
```
    hw1/
        preprocess.py
        preprocess.sh
        data/
            train.lab
            48_39.map
            48phone_char.map
            mfcc/
                test.ark
                train.ark
            fbank/
                test.ark
                train.ark
                        
```
### Preprocess part only
Run the following command under `/hw1` directory.
The program will generate `all_frames.npy` and `all_labels.npy`
under `./data`
```
./prepocess.sh [train] [test] 
```
### Usage
```
$ ./
```
