#!/bin/bash
for len in 2 3 4 5
do
    for freq in 10 15 20
    do
     	# create pseudo cs corpus concatenate with seame train
     	cat ../pseudo_corpora/phrase_len${len}_freq${freq}.en-zh \
            data/train.nltk_tokenizer.txt \
            ../data/mono.full > data/train.en-zh
        echo "done preparing train data"
        # use GPU 2 for en-zh
        echo "start to train lm"
        python language_model_base.py\
 			--gpus=2\
 			--infile="train.en-zh"\
  			--inference=False\
   			--save_path="save/len${len}_freq${freq}.en-zh"

        echo "done for freq=${freq} len=${len}"
    done
done