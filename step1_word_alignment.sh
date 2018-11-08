#!/bin/bash
# First do word alignment and perform the baseline experiment
# the extracted phrase table in the model is used for subsequent expt
~/Projects/mosesdecoder/scripts/training/train-model.perl \
    -external-bin-dir ~/Projects/mosesdecoder/bin/training-tools \
    -root-dir /home/grandee/backyard/phrases_alignment/word_alignment \
    -corpus /home/grandee/backyard/phrases_alignment/data/mono \
    -f zh -e en -alignment grow-diag-final-and \
    -reordering msd-bidirectional-fe \
    -lm 0:3:/home/grandee/Projects/lm/en-zh.blm.en:1 \
    -mgiza -mgiza-cpus 16

# unzip the phrase table and put in word_alignment
gzip -cd word_alignment/model/extract.sorted.gz \
    > word_alignment/extract.sorted
gzip -cd word_alignment/model/extract.inv.sorted.gz \
    > word_alignment/extract.inv

# unzip the word alignment and put in word_alignment
gzip -cd word_alignment/giza.en-zh/en-zh.A3.final.gz \
    > word_alignment/en-zh
gzip -cd word_alignment/giza.zh-en/zh-en.A3.final.gz \
    > word_alignment/zh-en