from sklearn.model_selection import train_test_split

# with open("cs.train") as f:
# 	text = f.read().split('\n')

# for n in [0.2, 0.4, 0.6, 0.8]:
# 	l = int(n*len(text))
# 	X_train, X_test = train_test_split( text, test_size=0, random_state=42)
# 	newtext = X_train[0:l]
# 	with open("cs.train.{}".format(n), "w") as f:
# 		f.writelines('\n'.join([i for i in newtext]))
#
# X_train, X_test = train_test_split( text, test_size=0.33, random_state=42)
# X_train, X_valid = train_test_split( X_train, test_size=0.5, random_state=42)
#
# with open("cs.train", "w") as f:
# 	f.writelines('\n'.join([i for i in X_train]))
#
# with open("cs.valid", "w") as f:
# 	f.writelines('\n'.join([i for i in X_valid]))
#
# with open("cs.test", "w") as f:



for prob in ['0.2', '0.4', '0.6', '0.8', '1.0']:

	with open("cs_gen_adapt_cs_{}.txt".format(prob)) as f:
		text = f.read().split('\n')

	X_train, X_test = train_test_split( text, test_size=0, random_state=42)

	with open("cs_gen_adapt_cs_{}.shuffled".format(prob), "w") as f:
		f.writelines('\n'.join([i for i in X_train]))


# 	with open("cs_gen_adapt_cs_{}.txt".format(prob)) as f:
# 		text = f.read().split('\n')

# 	X_train, X_test = train_test_split( text, test_size=0, random_state=42)

# 	with open("cs_gen_adapt_cs_{}.shuffled".format(prob), "w") as f:
# 		f.writelines('\n'.join([i for i in X_train]))

# 	with open("cs_gen_cs_{}.txt".format(prob)) as f:
# 		text = f.read().split('\n')

# 	X_train, X_test = train_test_split( text, test_size=0, random_state=42)

# 	with open("cs_gen_cs_{}.shuffled".format(prob), "w") as f:
# 		f.writelines('\n'.join([i for i in X_train]))