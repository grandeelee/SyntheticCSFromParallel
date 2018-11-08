from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re


file_big = "mono.vocab"
file_small = "seame.vocab"

with open(file_big , encoding='utf8') as f:
    vocab_big = f.read()
with open(file_small , encoding='utf8') as f:
    vocab_small = f.read()

vocab_big_cn = re.findall(r"[\u4e00-\u9fff]+", vocab_big)
print("There are {} chinese words in {}".format(len(vocab_big_cn), file_big))
vocab_big_en = re.findall(r"[a-zA-Z']+", vocab_big)
print("There are {} english words in {}".format(len(vocab_big_en), file_big))

vocab_small_cn = re.findall(r"[\u4e00-\u9fff]+", vocab_small)
print("There are {} chinese words in {}".format(len(vocab_small_cn), file_small))
vocab_small_en = re.findall(r"[a-zA-Z']+", vocab_small)
print("There are {} english words in {}".format(len(vocab_small_en), file_small))

oov_count_cn = [0 if not word in vocab_big_cn else 1 for word in vocab_small_cn]
oov_count_cn = sum(oov_count_cn)
cn_coverage = oov_count_cn / len(vocab_small_cn)
print("The chinese vocab coverage is {}".format(cn_coverage))

oov_count_en = [0 if not word in vocab_big_en else 1 for word in vocab_small_en]
oov_count_en = sum(oov_count_en)
en_coverage = oov_count_en / len(vocab_small_en)
print("The english vocab coverage is {}".format(en_coverage))

vocab_coverage = (oov_count_en + oov_count_cn) / len(vocab_small_en + vocab_small_cn)
print("The total vocab coverage is {}".format(vocab_coverage))