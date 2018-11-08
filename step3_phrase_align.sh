#!/bin/bash

mkdir phrases_alignment
mkdir final.alignment

for len in 2 3 4 5
do
 	# create folder for each phrase condition
    mkdir phrases_alignment/phrase_len${len}
    # run mgiza++ alignment
    ~/Projects/mosesdecoder/scripts/training/train-model.perl \
    -external-bin-dir ~/Projects/mosesdecoder/bin/training-tools \
    -root-dir ~/backyard/phrases_alignment/phrases_alignment/phrase_len${len} \
    -corpus ~/backyard/phrases_alignment/phrased_text/clean_len${len}.phrases \
    -f zh -e en -alignment grow-diag-final-and \
    -reordering msd-bidirectional-fe \
    -lm 0:3:/home/grandee/Projects/lm/en-zh.blm.en:1 \
    -mgiza -mgiza-cpus 16

    # unzip the final phrase alignment and extract to final.alignment
    gzip -cd phrases_alignment/phrase_len${len}/giza.en-zh/en-zh.A3.final.gz \
        > final.alignment/phrase_len${len}.en-zh
    gzip -cd phrases_alignment/phrase_len${len}/giza.zh-en/zh-en.A3.final.gz \
        > final.alignment/phrase_len${len}.zh-en

    echo "done for len=${len}"

done
