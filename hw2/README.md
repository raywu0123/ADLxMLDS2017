# ADLxMLDS 2017 Fall
## HW2 - Video Captioning


### Data Directory structure
```
    hw2/
        MLDS_hw2_data/
            peer_review/
                feat/
                video
            testing_data/
                feat/
                video/
            training_data/
                feat_video/
            bleu_eval.py
            peer_review_id.txt
            sample_output_peer_review.txt
            sample_output_testset.txt
            testing_id.txt
            testing_label.json
            training_label.json
        ...other files
```
## Usage
### Preprocessing
Run the following command under `hw2` directory.
This part pretrains a gensim word2vec model on the given caption labels.
The program will generate
`(training/testing)_int_captions.pkl`  
`vocab.txt`  
`wv.npy` under `./preprocess`

```
bash go.sh preprocess
```
### Training
Run the following command under `hw2` directory.  
The program will automatically save logs to `./seq2seq_logs`, and run validation on testing data.
```
bash go.sh train 
```
You may tune the parameters inside this script.
### Testing
Run the following command under `hw2` directory.  
The program will generate test_outputs.txt for all the testing videos.
```
bash go.sh test
```
### Assignment Specificaition
For Special Mission:
```
bash hw2_special.sh data_dir output_file
```
### TODOs
    1. metrics(BLEU@1, perplexity, METEOR...)

 

