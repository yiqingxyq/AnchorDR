import sys
import numpy as np
import os

CQA_datasets = ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress']
datasets = ['arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact', 'trec-covid', 'webis-touche2020']

if __name__ == '__main__':
    model = sys.argv[1]
    result_home_dir = sys.argv[2]
    
    MAX_LEN = np.max([len(x) for x in datasets])
    print('Dataset'+' '*(MAX_LEN-7), '\t', 'MRR@10', '\t', 'NDCG@10')
    for dataset in datasets:

        if not os.path.exists(result_home_dir + '/result/beir/' + dataset + '/' + model + '/result.txt'):
            continue

        with open(result_home_dir + '/result/beir/' + dataset + '/' + model + '/result.txt', 'r') as fin:
            line = fin.readline()
            seg = line.strip().split()
            if len(seg) < 1 or seg[0] != 'recip_rank':
                mrr10 = 0
            else:
                mrr10 = float(seg[2])

            line = fin.readline()
            seg = line.strip().split()
            if len(seg) < 1 or seg[0] != 'ndcg_cut_10':
                ndcg10 = 0
            else:
                ndcg10 = float(seg[2])

            spaces = ' '*(MAX_LEN-len(dataset))
            print(dataset+spaces, '\t', '%.4f'%mrr10, '\t', '%.4f'%ndcg10)

    if os.path.exists(result_home_dir + '/result/beir/cqadupstack/wordpress/' + model + '/result.txt'):
        mrr10s, ndcg10s = [], []
        for cqa_dataset in CQA_datasets:
            with open(result_home_dir + '/result/beir/cqadupstack/' + cqa_dataset + '/' + model + '/result.txt', 'r') as fin:
                line = fin.readline()
                seg = line.strip().split()
                assert seg[0] == 'recip_rank'
                mrr10 = float(seg[2])

                line = fin.readline()
                seg = line.strip().split()
                assert seg[0] == 'ndcg_cut_10'
                ndcg10 = float(seg[2])

                mrr10s.append(mrr10)
                ndcg10s.append(ndcg10)
        
        print('CQADupStack mean:', '\t', '%.4f'%(np.mean(mrr10s)), '\t', '%.4f'%(np.mean(ndcg10s)))
