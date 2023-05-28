import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from utils import *

import numpy as np
import pickle
from datasets import load_dataset, load_from_disk, Dataset

from tabulate import tabulate
from tqdm import tqdm
import random
import sys
import argparse
import os

NUM_EPOCHS=3
THRESH=0.25
MAX_LEN=32

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--neg_num', default=300, type=int, help='num of negs used in training')
    parser.add_argument('--input_queries_file', help='queries to rank and filter', default='url2anchor_step2.pkl')
    parser.add_argument('--output_file', help='the output file', default='url2anchor_final.pkl')
    args = parser.parse_args()

    random.seed(1)
    neg_num = args.neg_num

    # generate training data with k negs in total 
    file_path = 'data/'
    train_data = []
    with open(file_path + 'train_pos.txt', 'r') as fin:
        for line in fin:
            q = line.strip().split(':')[1]
            train_data.append((q,1))

    neg_file = file_path + 'train_neg.txt'
    with open(neg_file, 'r') as fin:
        for i,line in enumerate(fin):
            q = line.strip()
            train_data.append((q,0))
            if i > neg_num:
                break

    random.shuffle(train_data)
    text =  np.array([x[0] for x in train_data])
    labels =  np.array([x[1] for x in train_data])

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case = True
    )

    token_id = []
    attention_masks = []

    for sample in text:
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])

    token_id = torch.cat(token_id, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)

    val_ratio = 0.1
    # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
    batch_size = 8
    val_batch_size = 1024

    # Indices of the train and validation splits stratified by labels
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size = val_ratio,
        shuffle = True,
        stratify = labels,
        random_state = 1
    )

    # Train and validation sets
    train_set = TensorDataset(token_id[train_idx], 
                            attention_masks[train_idx], 
                            labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])

    # # test dataset 
    data_path = '/'.join(args.input_queries_file.split('/')[:-1])
    if os.path.exists(data_path + '/query_set.hf'):
        print('Directly loading anchor dataset..')
        test_set = load_from_disk(data_path + '/query_set.hf')
    else:
        print('Creating anchor dataset..')
        anchors = set()
        url2anchor = pickle.load(open(args.input_queries_file, 'rb'))
        for u in tqdm(url2anchor):
            for a in url2anchor[u]:
                anchors.add(a)
                
        test_text_dataset = Dataset.from_dict({'text':list(anchors)})
        # test_text_dataset = load_dataset('text', data_files=args.input_queries_file)['train']
        test_set = test_text_dataset.map(lambda x: tokenizer(x["text"], return_token_type_ids=False, padding=True, max_length=MAX_LEN, truncation=True), batched=True, batch_size=4096, num_proc=64)
        test_set.save_to_disk(data_path + '/query_set.hf')
        print('Anchor dataset loaded.')

    # Prepare DataLoader
    train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = 100
            )

    test_dataloader = DataLoader(
                test_set,
                sampler = SequentialSampler(test_set),
                batch_size = val_batch_size,
                num_workers=16,
                collate_fn=text_collate_fn,
            )

    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'prajjwal1/bert-mini',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )

    # print(model)

    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr = 5e-5,
                                eps = 1e-08
                                )

    # Run on GPU
    model.cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    epochs = NUM_EPOCHS
    test_logits = []
    best_val_acc = 0
    for epoch in range(epochs):
        
        # ========== Training ==========
        print('========== ', 'Epoch', epoch, ' ==========')
        
        # Set model to training mode
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_labels)
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output.loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        # ========== Validation ==========

        # Set model to evaluation mode
        model.eval()

        # Tracking variables 
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                eval_output = model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask)
            logits = eval_output.logits.detach().cpu().numpy() # for each sample, logits = [score_label0, score_label1]
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)

        print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')


        val_acc = sum(val_accuracy)/len(val_accuracy)
        if val_acc > best_val_acc:
            test_logits = []
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask = batch
                with torch.no_grad():
                    # Forward pass
                    eval_output = model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy() # for each sample, logits = [score_label0, score_label1]
                batch_logits = logits[:,1].tolist()
                test_logits.extend(batch_logits)

            best_val_acc = val_acc
    
    # save rankings to file 
    sorted_idx = np.argsort( -np.array(test_logits) )
    selected_num = int(len(sorted_idx) * THRESH)
    selected_idx = sorted_idx[:selected_num].tolist()
    selected_query_subset = set([test_set[i]['text'] for i in selected_idx])
    # with open(args.output_file, 'wb') as fout:
    #     pickle.dump(selected_query_subset, fout)
    
    new_url2anchor = {}
    for u in url2anchor:
        new_a = [a for a in url2anchor[u] if a in selected_query_subset]
        if len(new_a) > 0:
            new_url2anchor[u] = new_a 
            
    with open(args.output_file, 'wb') as fout:
        pickle.dump(new_url2anchor, fout)