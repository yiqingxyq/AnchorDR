import sys
import json
import re 
import pickle
from tqdm import tqdm
import numpy as np

TOP_K = 5

def main():
    url2anchor_file = sys.argv[1]
    out_file = sys.argv[2]
    stat_dir = sys.argv[3]
    split = sys.argv[4]

    print('Reading url2anchor file')
    with open(url2anchor_file, 'rb') as fin:
        url2anchor = pickle.load(fin)

    print('Writing anchors to file')
    doc_num = 0
    doc_with_anchor_num = 0
    anchor_num = 0
    selected_anchor_num = 0
    with open(out_file, 'w+') as fout:
        for line in sys.stdin:
            if line[:4] != 'Url:':
                continue 

            url = line.strip()[4:]
            fout.write(line)
            doc_num += 1

            # sample TOP_K anchors for each doc 
            if url in url2anchor:
                anchors = url2anchor[url]
                anchor_num += len(anchors)
                doc_with_anchor_num += 1

                if len(anchors) <= TOP_K:
                    selected_anchor_num += len(anchors)
                    for anchor in anchors:
                        fout.write(anchor + '\n')
                    for i in range(TOP_K - len(anchors)):
                        fout.write('[PAD]' + '\n')
                else:
                    selected_anchor_num += TOP_K
                    np.random.shuffle(anchors)
                    anchor_subset = anchors[:TOP_K]
                    for anchor in anchor_subset:
                        fout.write(anchor + '\n')
            else:
                for i in range(TOP_K):
                    fout.write('[PAD]' + '\n')

            fout.write('\n')

    print('finished processing', str(doc_with_anchor_num), '/', str(doc_num), 'docs with anchors')
    print('select', selected_anchor_num, '/', anchor_num, 'anchors')

    with open(stat_dir + '/anchor_'+split+'_sample_data.txt','w') as fout:
        fout.write('doc_with_anchor: ' + str(doc_with_anchor_num) + '\n')
        fout.write('total doc num: ' + str(doc_num) + '\n')
        fout.write('selected anchor num: ' + str(selected_anchor_num) + '\n')
        fout.write('total anchor num (inside train/valid set): ' + str(anchor_num) + '\n')

if __name__ == '__main__':
    main()