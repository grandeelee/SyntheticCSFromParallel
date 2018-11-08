from sklearn.model_selection import train_test_split
import argparse

# the input is the aligned data (ouput of GIZA++)
# the output is pseudo generated cs text. now the switch is random.
parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, help="input file name")
parser.add_argument("--outfile", type=str, help="output file name")
# set default values for parser
parser.set_defaults(infile='none', outfile='none')
args = parser.parse_args()

def shuffle(infile, outfile):
	with open(infile) as f:
		text = f.read().split('\n')

	X_train, X_test = train_test_split( text, test_size=0, random_state=42)

	with open(outfile, "w") as f:
		f.writelines('\n'.join([i for i in X_train]))

infile = args.infile
outfile = args.outfile
shuffle(infile, outfile)