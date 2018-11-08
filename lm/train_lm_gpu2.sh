# for prob in 1.0 0.8 0.6 0.4 0.2
# do
# 	# cat data/cs_gen_0.7_adapt.shuffled data/cs.train.${prob} > data/cs_gen_adapt_cs_${prob}.txt
# 	# cat data/cs_gen_0.5.txt data/cs.train.${prob} > data/cs_gen_cs_${prob}.txt
# 	python language_model_base.py\
# 	 --gpus=2\
# 	 --infile="cs_gen_adapt_cs_${prob}.shuffled"\
# 	  --inference=False\
# 	   --save_path="save/cs_gen_adapt_cs_${prob}"\
# 	   --adapt_path="adapt_cs_${prob}.shuffled"\
# 	   --adapt=True

# 	echo "finished cs_gen_cs_${prob}"

# done

python language_model_base.py\
 --gpus=2\
 --infile="cs_gen_0.7.txt"\
  --inference=False\
   --save_path="save/cs_gen_0.7"\
   --adapt_path="adapt"\
   --test_path="cs.test"\
   --adapt=False

python language_model_base.py\
 --gpus=2\
 --infile="cs_gen_0.7_adapt.shuffled"\
  --inference=False\
   --save_path="save/cs_gen_0.7_adapt"\
   --adapt_path="adapt"\
   --test_path="cs.test"\
   --adapt=True