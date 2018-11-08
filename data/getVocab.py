# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:08:37 2018

@author: adminnus
"""
import re


infile = "parallel.mono_large.full"
#%% just getting vocab
with open(infile , encoding='utf8') as f:
    text = f.read();
    
wordlist = re.findall(r"[a-zA-Z'\d\u4e00-\u9fff]+", text)
print("The total number of token in {} is {}".format(infile, len(wordlist)))
vocab = sorted(set(wordlist))
print("The number of vocab for {} is {}".format(infile, len(vocab)))
print("now writing vocab")
with open('mono.vocab', 'w' , encoding='utf8') as file:
    file.writelines(i + '\n' for i in vocab)

# compute vocab coverage
