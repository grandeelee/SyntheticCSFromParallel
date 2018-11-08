for len in 2 3 4 5
do
	cat data/phrase_len${len} data/adapt_cs_1.0.txt > data/phrase_len${len}_adapt_cs
    python shuffling.py --infile="data/phrase_len${len}_adapt_cs" \
    					--outfile="data/phrase_len${len}_adapt_cs.shuffled"


done
