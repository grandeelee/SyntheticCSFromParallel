# for len in 2 3 4 5
# do

# 	python language_model_base.py\
# 	 --gpus=3\
# 	 --infile="phrase_len${len}_adapt_cs.shuffled"\
# 	  --inference=False\
# 	   --save_path="save/phrase_len${len}_adapt_cs"\
# 	   --adapt_path="adapt_cs_1.0.txt"\
# 	   --adapt=True

# 	echo "finished phrase_len${len}_adapt_cs"

# done


python language_model_base.py\
 --gpus=3\
 --infile="mono"\
  --inference=False\
   --save_path="save/mono"\
   --adapt_path="adapt"\
   --test_path="cs.test"\
   --adapt=False

python language_model_base.py\
 --gpus=3\
 --infile="mono_adapt"\
  --inference=False\
   --save_path="save/mono_adapt"\
   --adapt_path="adapt"\
   --test_path="cs.test"\
   --adapt=True