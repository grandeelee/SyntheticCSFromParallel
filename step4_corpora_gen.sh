#!/bin/bash

# first generate word alignment with naive cs probability
for k in $(seq 1 7)
do 
    python pseudo_cs_gen.py --infile="word_alignment/en-zh" --csprob=0.7\
    		--outfile="pseudo_corpora/tmp.${k}.en-zh"

    python pseudo_cs_gen.py --infile="word_alignment/zh-en" --csprob=0.7\
    		--outfile="pseudo_corpora/tmp.${k}.zh-en"
done
cat pseudo_corpora/tmp.* > pseudo_corpora/tmp.all
python shuffling.py --infile="pseudo_corpora/tmp.all" --outfile="pseudo_corpora/phrase_len0"
rm pseudo_corpora/tmp.*

for len in 2 3 4 5
do
    for k in $(seq 1 7)
    do
     	# create folder for each phrase condition
        python pseudo_cs_gen.py --infile="final.alignment/phrase_len${len}.en-zh" --csprob=0.7\
        --outfile="pseudo_corpora/tmp.${k}.en-zh"

        python pseudo_cs_gen.py --infile="final.alignment/phrase_len${len}.zh-en" --csprob=0.7\
        --outfile="pseudo_corpora/tmp.${k}.zh-en"

    done
    cat pseudo_corpora/tmp.* > pseudo_corpora/tmp.all
    python shuffling.py --infile="pseudo_corpora/tmp.all" --outfile="pseudo_corpora/phrase_len${len}"
    rm pseudo_corpora/tmp.*
    echo "done for len=${len}"
done