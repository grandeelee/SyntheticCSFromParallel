from sklearn.model_selection import train_test_split
import sys
sys.path.append("/home/grandee/Projects/language model/utility")
import grandee_toolbox as grandee


with open("cs.nltk_tokenizer.txt") as f:
	text = f.read().split('\n')

X_train, X_test = train_test_split( text, test_size=0.33, random_state=42)
X_train, X_valid = train_test_split( X_train, test_size=0.5, random_state=42)

text = []
for line in X_train:
	text.append(line.split()[2:])
grandee.write2dlist(text, "cs.train")

text = []
for line in X_valid:
	text.append(line.split()[2:])
grandee.write2dlist(text, "cs.valid")

text = []
for line in X_test:
	text.append(line.split()[2:])
grandee.write2dlist(text, "cs.test")

# a is ma b is sg
text_sg = []
text_ma = []
for line in X_test:
	words = line.split()
	if 'a' in words[0]:
		text_ma.append(words[2:])
	if 'b' in words[0]:
		text_sg.append(words[2:])
grandee.write2dlist(text_sg, "cs.test.sg")
grandee.write2dlist(text_ma, "cs.test.ma")


# with open("adapt") as f:
# 	text = f.read().split('\n')
#
# X_train, X_test = train_test_split( text, test_size=0, random_state=42)
#
# with open("adapt.shuffled", "w") as f:
# 	f.writelines('\n'.join([i for i in X_train]))