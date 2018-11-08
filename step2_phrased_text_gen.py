#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:35:02 2018

@author: grandee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from collections import Counter, defaultdict
import numpy as np

def extraction(readfile, writefile, is_en=True):
    # read in
    with open(readfile, "r") as f:
        text = f.read().split('\n')
    print("finished reading in")
    # remove non english and append phrase
    length_count = []
    new_text = []
    for line in text:
        if is_en:
            sent = re.findall(r"[a-z']+", line)
        else:
            sent = re.findall(r'[\u4e00-\u9fff]+', line)
        if sent:
            new_text.append(sent)
            length_count.append(len(sent))
    print("finished cleaning, printing to file")
    print("max length: %d" %(max(length_count)))
    # write to file
    with open(writefile, 'w') as f:
        f.writelines(' '.join(str(j) for j in i) + '\n' for i in new_text)    

def select_basedon_length(readfile, writefile, length):
    with open(readfile, "r") as f:
        text = f.read().split('\n')
    print("finished reading in")
    new_text = []    
    for line in text:
        line = line.split()
        if len(line) <= length and len(line) >= 2:
            new_text.append(line)
    print("finished cleaning, printing to file")        
    # write to file
    with open(writefile, 'w') as f:
        f.writelines(' '.join(str(j) for j in i) + '\n' for i in new_text)  
        
def select_basedon_abs_freq(readfile, writefile, freq):
    with open(readfile, "r") as f:
        text = f.read().split('\n')
    print("finished reading in")
    counter = Counter(text)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, counts = list(zip(*count_pairs))
    idx = np.sum([1 if i>freq else 0 for i in counts])
    words = words[:idx]
    counts = counts[:idx]
    count_dict = defaultdict(lambda: defaultdict(lambda: 0))
    counter = list(zip(words,counts))
    for item, value in counter:
        first_word = item.split()[0]
        rest_word = ' '.join(item.split()[1:])
        count_dict[first_word][rest_word] = value
        
    print("finished cleaning, printing to file")   
    with open(writefile, 'w') as f:
        f.writelines(i + '\n' for i in words)       
    return count_dict

def sub_phrases(readfile, writefile, phrases_list):
    with open(readfile, "r") as f:
        text = f.read().split('\n')
    new_text = []
    for line in text:
        new_text.append(' '+line+' ')
    text = '\n'.join(new_text)
    with open(phrases_list, "r") as f:
        phrases = f.read().split('\n')
    # replace text with each phrases
    for phrase in phrases:
        text = re.sub(' '+phrase+' ', ' '+'_'.join(phrase.split())+' ', text)
    text = text.split('\n')
    new_text = []
    for line in text:
        new_text.append(line.strip())
    text = '\n'.join(new_text)
    print("finished cleaning, printing to file")  
    with open(writefile, 'w') as f:
        f.writelines(text)

# # extract the ohrases in en from phrase table
# readfile = "word_alignment/extract.inv"
# writefile = "extracted_phrases/en_phrases"
# extraction(readfile, writefile, is_en=True)

# # extract the phrases in cn from phrase table
# readfile = "word_alignment/extract.sorted"
# writefile = "extracted_phrases/cn_phrases"
# extraction(readfile, writefile, is_en=False)

# # phrase selection based on length
# for length in [2,3,4,5]:       
#    readfile = "extracted_phrases/en_phrases"
#    writefile = "extracted_phrases/en_phrases_len{}".format(length)
#    select_basedon_length(readfile, writefile, length)
#    readfile = "extracted_phrases/cn_phrases"
#    writefile = "extracted_phrases/cn_phrases_len{}".format(length)    
#    select_basedon_length(readfile, writefile, length)

# phrase selection based on freq
for freq in [10]:
    for length in [2,3,4,5]:
        readfile = 'extracted_phrases/en_phrases_len{}'.format(length)
        writefile = 'extracted_phrases/en_phrases_len{}_freq{}'.format(length, freq)
        cnt_en = select_basedon_abs_freq(readfile, writefile, freq)
        readfile = 'extracted_phrases/cn_phrases_len{}'.format(length)
        writefile = 'extracted_phrases/cn_phrases_len{}_freq{}'.format(length, freq)
        cnt_cn = select_basedon_abs_freq(readfile, writefile, freq)
        
# insert the selected phrases into the parallel text
for length in [2,3,4,5]:
    phrases_list = 'extracted_phrases/en_phrases_len{}_freq10'.format(length)
    readfile = 'data/mono.en'
    writefile = 'phrased_text/clean_len{}.phrases.en'.format(length)    
    sub_phrases(readfile, writefile, phrases_list)               
    phrases_list = 'extracted_phrases/cn_phrases_len{}_freq10'.format(length)
    readfile = 'data/mono.zh'
    writefile = 'phrased_text/clean_len{}.phrases.zh'.format(length)    
    sub_phrases(readfile, writefile, phrases_list)
