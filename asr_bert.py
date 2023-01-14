# -*- coding: utf-8 -*-
# generate_sst.py
# from __future__ import absolute_import, division, print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig # from pretrained
# from pytorch_transformers import BertTokenizer
from utils import to_gpu, Corpus, batchify, SNLIDataset, collate_snli
from dataset_utils import convert_examples_to_features, load_and_cache_examples, SNLIProcessor
from dataset_utils import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)
from transformers import AdamW, get_linear_schedule_with_warmup


import sys
import time
import math
import time
import random
import argparse
import json
import pickle as pkl
import numpy as np
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from utils import to_gpu, Corpus, SNLIDataset, collate_snli, SNLIDatasetForUniVAETEST, QQPDatasetForUniVAETEST
from models import Baseline_LSTM, Baseline_Embeddings, Baseline_LSTMQQP
import tensorflow as tf


import glob
import logging


import torch.optim as optim

import torch.utils.data
from torch.autograd import Variable


sys.path.append("..") # 导入上级目录模块
# print("系统当前路径" , sys.path)


num_labels = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(device)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def pred_fn_one(x):
    # query baseline classifiers with sentence pairs
   
    model.eval() # 要加 model.eval(), 否则视为 train 过程， batch 必须大于 1
    premise, hypothesis = x # 本身传过来的就是 id 组成的句子
    
    text_a = premise
    text_b = hypothesis

    max_seq_length = 64
    cls_token='[CLS]'
    sep_token='[SEP]'

    pad_token=0
    sequence_a_segment_id=0
    sequence_b_segment_id=1
    cls_token_segment_id=1
    pad_token_segment_id=0
    mask_padding_with_zero=True

    tokens_a = tokenizer.tokenize(text_a)
    # print(tokens_a) # ['i', 'like', 'this', 'movie', 'very', 'much', '.']
    tokens_b = tokenizer.tokenize(text_b)

    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)


    tokens += tokens_b + [sep_token]
    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)


    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
    
    # 将 inputs 转为 PyTorch tensors
    tokens_tensor = torch.tensor([input_ids]) 
    mask_tensors = torch.tensor([input_mask]) 
    segments_tensors = torch.tensor([segment_ids]) 

    # GPU & put everything on cuda
    tokens_tensor = tokens_tensor.to(device) 
    mask_tensors = mask_tensors.to(device) 
    segments_tensors = segments_tensors.to(device) 
    
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensor, attention_mask=mask_tensors, token_type_ids=segments_tensors, )
    logits = outputs.logits
    preds = logits.detach().cpu().numpy()
    pred = np.argmax(preds, axis=1)
     # 3 分类，蕴含，中立，矛盾

    return pred


def asr(data_source, corpus_test):
    # Turn on evaluation mode which disables dropout.
    
    labels = {'tensor(0)': 'entailment', 'tensor(1)': 'neutral', 'tensor(2)': 'contradiction'}
    truenum = 0
    total = 0
    for batch in data_source:
        premise, hypothesis, target, premise_words_ori, hypothesis_words_ori, lengths = batch

        batch_size = premise.size(0)
        total += batch_size
        print(total)
        for i in range(batch_size):

            y_pred = pred_fn_one((premise_words_ori[i], hypothesis_words_ori[i])) # 改动, y_pred = [2]
            # print(y_pred[0] == target[i])
            if y_pred[0] == target[i]:
                truenum += 1

    acc = truenum / total           
    print('acc after attack:', acc)
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description='Generating Natural Adversaries for Text')

    # Path Arguments
    parser.add_argument('--data_path', type=str, default= './data',
                        help='path to data corpus ./data')
    
    # Other
    
    parser.add_argument('--advmode', type=str, 
                        choices=['all', 'fore', 'tail'],
                        default='all',
                        help='which kind of perturbation on latent space.')
    parser.add_argument('--datatype', type=int, default=0,
                        help='performs attack on which dataset,0:snli, 1:qqp')

    parser.add_argument('--modelmode', type=str, 
                        choices=['lstm', 'cnn', 'bert'],
                        default='lstm',
                        help='which kind of model.')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    global args
    args = parse_args()
    
    MODEL_CLASSES = {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
     }
    typedd = {0:'snli', 1:'qqp'}
    
    
    seed = 1111
    data_path = args.data_path
    maxlen = 32
    vocab_size = 11000
    lowercase = True
    voc_file = './vocab.json'
    datatype = args.datatype
    modelmode = args.modelmode
    test_path = 'bestadv' + typedd[datatype] + args.advmode + '.txt'
    # test_path = 'snli3000.txt'
    task_name = typedd[datatype]
    if datatype == 0:
        model_name_or_path = './bert_snli_uncased/checkpoint-25752/'
    elif datatype == 1:
        model_name_or_path = './bert_qqp_uncased/checkpoint-17058-0.902819688350235/'
    
    print(test_path)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.device(0)
    print("using cuda device gpu:" + format(torch.cuda.current_device()))
    torch.cuda.manual_seed(seed)

    ###############################################################################
    # Load data and target classifiers
    ###############################################################################
    global voc_uni
    voc_uni = json.load(open(voc_file, 'r'))
    # print(voc_uni)

    corpus = Corpus(data_path,
                    maxlen=maxlen,  # maximum sentence length
                    vocab_size=vocab_size,  # 词表大小 默认频率前 11000 的
                    lowercase=lowercase,
                    load_vocab=voc_file
                    )

    if datatype == 0:
        print('data set snli')
        corpus_test = SNLIDatasetForUniVAETEST(train=False, test_path=test_path, vocab_size=vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=128,
                                             collate_fn=collate_snli, shuffle=False)
        # BertConfig 在transformer 中的才有 from_pretrained
        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        config = config_class.from_json_file(model_name_or_path+ '/config.json')
        print(config)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)

        model = model_class.from_pretrained(model_name_or_path, num_labels=3)
        model.to(device)
    elif datatype == 1:
        print('data set qqp')
        corpus_test = QQPDatasetForUniVAETEST(train=False, test_path=test_path, vocab_size=vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=128,
                                             collate_fn=collate_snli, shuffle=False)
        # BertConfig 在transformer 中的才有 from_pretrained
        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        config = config_class.from_json_file(model_name_or_path+ '/config.json')
        print(config)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)

        model = model_class.from_pretrained(model_name_or_path, num_labels=2)
        model.to(device)

    test_data = iter(testloader)

    vocab_classifier1 = voc_uni
    print("Loaded data and target classifiers!")

    ###############################################################################
    # Build the models
    ###############################################################################
    ntokens = len(corpus.dictionary.word2idx)
    ntokens = ntokens
    print("Vocabulary Size: {}".format(ntokens))
    
    corpus.dictionary.word2idx = voc_uni


    len_test = len(testloader) #
    print(len_test)

    attack_success_rate = asr(test_data, corpus_test)
    print(attack_success_rate)





    









