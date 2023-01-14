# generate_sst.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from models import Baseline_LSTM, Baseline_Embeddings, Baseline_LSTMQQP, TextCNN
import tensorflow as tf


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


        
def pred_fn_one(x):
    # query baseline classifiers with sentence pairs
    gpu = True
    classifier1.eval() # 要加 model.eval(), 否则视为 train 过程， batch 必须大于 1
    premise, hypothesis = x # 本身传过来的就是 id 组成的句子

    prob_distrib1 = classifier1.forward((premise, hypothesis)) 

    _, predictions = torch.max(prob_distrib1, 1)
     # 3 分类，蕴含，中立，矛盾

    return predictions

def convert_sentence2ids(premise_ori, hypothesis_ori):
    
    vocab = json.load(open('./vocab.json', 'r'))
    maxlen = 32
    # print('h', hypothesis_ori)
    if premise_ori[-1] == '.' or premise_ori[-1] == '?':
        premise_words_ori = premise_ori.strip()[:-1].split(" ") + ["." if premise_ori[-1] == '.' else "?"]
    else:
        premise_words_ori = premise_ori.strip().split(" ")
    if hypothesis_ori[-1] == '.' or hypothesis_ori[-1] == '?':
        hypothesis_words_ori = hypothesis_ori.strip()[:-1].split(" ") + ["." if hypothesis_ori[-1] == '.' else "?"]
    else:
        hypothesis_words_ori = hypothesis_ori.strip().split(" ")
    
    premise_words = ['<sos>'] + premise_words_ori
    premise_words += ['<eos>']
    hypothesis_words = ['<sos>'] + hypothesis_words_ori
    unk_idx = vocab['<oov>']
    hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
    premise_indices = [vocab[w] if w in vocab else unk_idx for w in premise_words]
    if len(premise_indices) < maxlen:
        premise_indices += [0]*(maxlen- len(premise_indices))
    if len(hypothesis_indices) < maxlen:
        hypothesis_indices += [0]*(maxlen - len(hypothesis_indices))
    premise_indices = premise_indices[:maxlen]
    hypothesis_indices = hypothesis_indices[:maxlen] 
    return premise_indices, hypothesis_indices
        
def asr(data_source, corpus_test):
    # Turn on evaluation mode which disables dropout.
    
    truenum = 0
    total = 0
    for batch in data_source:
        premise, hypothesis, target, premise_words_ori, hypothesis_words_ori, lengths = batch

        batch_size = premise.size(0)
        total += batch_size
        print(total)
        for i in range(batch_size):

            y_pred = pred_fn_one((premise[i].unsqueeze(0), hypothesis[i].unsqueeze(0)))

            if y_pred == target[i]:
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
    print(test_path)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
        if modelmode == 'lstm':
            classifier1 = Baseline_LSTM(100, 300, maxlen=maxlen, gpu=False)
            classifier1.load_state_dict(
                torch.load('lstm_univae_acc_0.74.pt' , map_location=torch.device('cpu'))
            )
        elif modelmode == 'cnn':
            classifier1 = TextCNN(100, 300, class_num=3, maxlen=maxlen, gpu=False)
            classifier1.load_state_dict(
                torch.load('textcnnlast_0.64.pt' , map_location=torch.device('cpu'))
            )
            
    elif datatype == 1:
        print('data set qqp')
        corpus_test = QQPDatasetForUniVAETEST(train=False, test_path=test_path, vocab_size=vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=128,
                                             collate_fn=collate_snli, shuffle=False)
        if modelmode == 'lstm':
            classifier1 = Baseline_LSTMQQP(100, 300, maxlen=maxlen, gpu=False)
            classifier1.load_state_dict(
                torch.load('lstm_univae_QQP.pt' , map_location=torch.device('cpu'))
            )
        elif modelmode == 'cnn':
            classifier1 = TextCNN(100, 300, class_num=2, maxlen=maxlen, gpu=False)
            classifier1.load_state_dict(
                torch.load('textcnnlast_0.74QQP.pt' , map_location=torch.device('cpu'))
            )

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
    

    


    









