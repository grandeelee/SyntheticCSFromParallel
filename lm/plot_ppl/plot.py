from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import re



def ppl_of_cs(sent_file, ent_file):
	sent = np.load(sent_file)
	ent = np.load(ent_file)

	# get english and cn vocab
	vocab = sorted(set(sent))
	vocab_en = re.findall(r"[a-z<>']+", " ".join(vocab))
	vocab_cn = re.findall(r"[\u4e00-\u9fff]+", " ".join(vocab))
	# cn is 1, en is 0, </s> follows previous token
	new_sent = [0 if word in vocab_en else word for word in sent]
	new_sent = [1 if word in vocab_cn else word for word in new_sent]
	# if curr different from prev, and prev and curr not </s> then it is code-switch
	cs_idx = []
	for idx, lid in enumerate(new_sent):
		# ignore first word
		if idx == 0:
			cs_idx.append(0)
			continue
		if not lid == new_sent[idx-1] and not new_sent[idx-1] == '</s>' and not lid == '</s>':
			# this is code-switch point
			cs_idx.append(1)
		else:
			cs_idx.append(0)

	return np.exp(sum(np.multiply(cs_idx,ent))/sum(cs_idx))


sent_file = "monotest_sentence.npy"
ent_file = "monotest_sentence_entropy.npy"
ppl = ppl_of_cs(sent_file, ent_file)
print("ppl of mono on cs points is: {}".format(ppl))

sent_file = "mono_adapttest_sentence.npy"
ent_file = "mono_adapttest_sentence_entropy.npy"
ppl = ppl_of_cs(sent_file, ent_file)
print("ppl of mono_adapt on cs points is: {}".format(ppl))

sent_file = "cs_gen_0.7test_sentence.npy"
ent_file = "cs_gen_0.7test_sentence_entropy.npy"
ppl = ppl_of_cs(sent_file, ent_file)
print("ppl of cs_gen on cs points is: {}".format(ppl))

sent_file = "cs_gen_0.7_adapttest_sentence.npy"
ent_file = "cs_gen_0.7_adapttest_sentence_entropy.npy"
ppl = ppl_of_cs(sent_file, ent_file)
print("ppl of cs_gen_adapt on cs points is: {}".format(ppl))