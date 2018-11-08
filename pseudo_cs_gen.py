import sys
sys.path.append("/home/grandee/Projects/language model/utility")
import grandee_toolbox as grandee
import re
import numpy as np
import argparse

# the input is the aligned data (ouput of GIZA++)
# the output is pseudo generated cs text. now the switch is random.
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, help="input file name")
parser.add_argument("--outfile", type=str, help="output file name")
parser.add_argument("--csprob", type=float, help="code switch prob")
# set default values for parser
parser.set_defaults(infile='none', outfile='none', csprob=0.5)
args = parser.parse_args()


raw_text = grandee.read(args.infile).split("\n")
target_text = []
source_text = []
align = []

print("Separating into source target")
for idx, line in enumerate(raw_text):
	if idx % 100000 == 1:
		print("progress: %.3f" %(idx/len(raw_text)))
	if idx % 3 == 2:
		# extract alignment token
		sent_align = np.array(re.findall(r"\{(.*?)\}", line)[1:])
		# delete after extraction
		line = re.sub(r"\((.*?)\)", r"", line)
		# ignore NULL the first item
		line = line.split()[1:]
		# check for consistency
		if len(sent_align) == len(line):
			align.append(sent_align)
			source_text.append(line)
			target_text.append(raw_text[idx-1])

print("Finished separating into source target")

print("Generating pseudo CS text")
# do the switch here and generate final version data
switched_text = []
for idx, (source_sent, target_sent, align_sent) \
		in enumerate(zip(source_text, target_text, align)):
	if idx % 30000 == 1:
		print("progress: %.3f" %(idx/len(source_text)))
	# iterate througth source
	switched_sent = []
	target_sent = target_sent.split()

	for source_word, align_word in zip(source_sent, align_sent):
		# if alignment empty, append source
		alignment = align_word.split()
		if not alignment:
			switched_sent.append(source_word)
		# pseudo code switch, assume 50% chance
		else:
			if np.random.rand(1) > args.csprob:
				switched_sent.append(source_word)
			# 50% chance of switch to any of the alignment
			else:
				target_idx = np.random.choice(np.array(alignment, dtype=int))
				switched_sent.append(target_sent[target_idx-1])

	# append switched sent
	switched_text.append(switched_sent)

print("Finished generating pseudo CS text")
# write to text
grandee.write2dlist(switched_text, args.outfile)
text = grandee.read(args.outfile)
text = re.sub(r'_', ' ', text)
grandee.writestring(text, args.outfile)
