import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tabulate import tabulate
from tqdm import trange
import random

MAX_LEN=32

def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
    '''
    Returns the following metrics:
        - accuracy    = (TP + TN) / N
        - precision   = TP / (TP + FP)
        - recall      = TP / (TP + FN)
        - specificity = TN / (TN + FP)
    '''
    # preds = np.argmax(preds, axis = 1).flatten()

    preds = preds[:,1].flatten()
    labels = labels.flatten()

    b_accuracy, b_precision, b_recall, b_specificity = 0,0,0,0

    scores = sorted(preds)
    # for thresh in range(len(preds)):
    thresh_idx = len(scores)//2
    tmp_preds = preds >= scores[thresh_idx]

    tp = b_tp(tmp_preds, labels)
    tn = b_tn(tmp_preds, labels)
    fp = b_fp(tmp_preds, labels)
    fn = b_fn(tmp_preds, labels)
    tmp_b_accuracy = (tp + tn) / len(labels)
    tmp_b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    tmp_b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    tmp_b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'

    if tmp_b_accuracy > b_accuracy:
        b_accuracy, b_precision, b_recall, b_specificity = tmp_b_accuracy, tmp_b_precision, tmp_b_recall, tmp_b_specificity

    return b_accuracy, b_precision, b_recall, b_specificity


def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - token_type_ids: list of token type ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
                            input_text,
                            add_special_tokens = True,
                            max_length = 32,
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'
                    )

def text_collate_fn(data):
    """
       data: is a list of dicts
    """
    input_ids = torch.cat([torch.tensor(x['input_ids']) for x in data])
    attention_mask = torch.cat([torch.tensor(x['attention_mask']) for x in data])

    return input_ids.reshape(-1,MAX_LEN), attention_mask.reshape(-1,MAX_LEN)