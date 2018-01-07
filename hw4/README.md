# ADLxMLDS 2017 Fall
## HW4 - Comics Generation


## Usage
### Training
Run the following command under `hw4` directory.  
The program will automatically save logs to `./logs`, and dump images to ./imgs every 200 minibatches.
```
python3 train.py
```

### Inference
Run the following command under `hw4` directory.
The program will generate pictures and save them to `./samples`,
according to the description in `./test_txt.txt`
```
python3 generate.py
```
Description Specification
Every line begins with a non_repeated id, and specifies hair color and/or eye color.
e.g.

```
 10,blue hair red eyes
 23,green eyes black hair
 3,yellow eyes
 4,purple hair
```
### Assignment Specificaition
Usage
```
bash run.sh [testing_text.txt]
```


 

