from argparse import ArgumentParser
from cmath import nan
import csv
from sklearn import neighbors
import pandas as pd
parser = ArgumentParser()
parser.add_argument('--trec', required=True)
parser.add_argument('--save_to', required=True)
args = parser.parse_args()

q_flag_dict={}
with open(args.save_to, 'w') as fout:
	with open(args.trec,'r') as rf:
		new_lines = rf.readlines()
		cur_q, _, p, r, s, _ = new_lines[0].strip().split()
		for i in range(len(new_lines)):
			q, uu, p, r, s, tt = new_lines[i].strip().split()
			if q not in q_flag_dict.keys():
				q_flag_dict[q]=0
			if q==p:
				print(q)
				q_flag_dict[q]=1
			elif q_flag_dict[q]==0:
				fout.write(new_lines[i])
			else:
				fout.write(" ".join([q, uu, p, str(int(r)-1), s, tt])+"\n")