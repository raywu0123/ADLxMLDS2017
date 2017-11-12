# ADLxMLDS 2017 Fall
## HW1 - Sequence Labeling

### Data Directory structure
```
    hw1/
        data/
            48phone_char.map
            label/
                train.lab
            phones/
                48_39.map
            mfcc/
                test.ark
                train.ark
            fbank/
                test.ark
                train.ark
        ...other files
```
## Usage
### Preprocessing
Run the following command under `hw1` directory.
The program will generate `trainframes.npy` and `labels.npy`
or `testframes.npy` under `./data`
```
bash go.sh [pretrain] [pretest] 
```
### Training
Run the following command under `hw1` directory.  
The program will automatically save logs to `./logs`, and also handles crashes.
```
bash go.sh train 
```
You may tune the parameters in side this script.
### Testing
Run the following command under `hw1` directory.  
The program will restore the model from saved logs and generate prediction.
```
bash go.sh test
```

### Assignment Specificaition
`hw1_rnn.sh`, `hw1_cnn.sh` and `hw1_best.sh` are for submitting the assignment.
The hyper-parameters which matches the stored logs are stored inside the script.
It's not recommended to alter the script.

 

