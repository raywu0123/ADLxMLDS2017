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
## Usage
### Preprocessing
Run the following command under `/hw1` directory.
The program will generate `trainframes.npy` and `labels.npy`
or `testframes.npy` under `./data`
```
bash go.sh [pretrain] [pretest] 
```
### Training
Run the following command under `/hw1` directory.  
The program will automatically save logs to `./logs`, and also handles crashes.
```
bash go.sh train 
```
### Testing
Run the following command under `/hw1` directory.  
The program will restore from saved logs and generate prediction.
```
bash go.sh test
```

