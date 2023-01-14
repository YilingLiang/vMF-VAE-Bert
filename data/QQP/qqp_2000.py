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
from utils import to_gpu, Corpus, SNLIDataset, collate_snli, SNLIDatasetForUniVAE, QQPDatasetForUniVAE
from models import Baseline_LSTM, Baseline_Embeddings, Baseline_LSTMQQP
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, integerize_shape
from bert4keras.models import build_transformer_model, RoFormer, BERT
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import AutoRegressiveDecoder, text_segmentate
from keras.layers import Input, Dense, Lambda, Concatenate, Layer
from keras.models import Model

      
def pred_fn_one(x):
    # query baseline classifiers with sentence pairs
    gpu = args.cuda
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
        premise_words_ori = premise_ori.strip()[:-1].split(" ")
    else:
        premise_words_ori = premise_ori.strip().split(" ")
    if hypothesis_words_ori[-1] == '.' or hypothesis_words_ori[-1] == '?':
        hypothesis_words_ori = hypothesis_ori.strip()[:-1].split(" ")
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
    
    
def search_qqp_snli(s, y, nsamples=100, left=0., high=1.):
    
    premise_ori, hypothesis_ori = s
    print("Begin generating adversarial examples for: \n" + premise_ori + ' ' + hypothesis_ori + ' ' + str(y.cpu().numpy()))
    x, s = tokenizer.encode(hypothesis_ori, maxlen=maxlen)
    X = sequence_padding([x])
    S = sequence_padding([s])
    Z = encoder.predict([X, S])
    z = Z[0]
    x_adv = []
    distance = []
    nearest_dist = None
    nearest_x_adv = None
    for k in range(nsamples):

        delta_z = np.random.randn(64)
        d = np.random.rand(64) * (high - left) + left  # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=2)

        d_norm = np.divide(d, norm_p)  # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm) 
        delta_z += z
        dist = np.linalg.norm(delta_z - z)
        
        hypo_pre_ori = vec2sent.generate(delta_z)
        
        p, h = convert_sentence2ids(premise_ori, hypo_pre_ori)
        pp = Variable(torch.LongTensor(p))
        hh = Variable(torch.LongTensor(h))
        
        y_pred = pred_fn_one((pp.unsqueeze(0), hh.unsqueeze(0)))
        if y_pred != y:
            temp = premise_ori + '\t' + hypo_pre_ori + '\t' + str(y.cpu().numpy())
            x_adv.append(temp)
            distance.append(dist)
            # print('adv h:', hypo_pre_ori)
            
    if len(x_adv) != 0:    
        nearest_dist = min(distance)
        nearest_x_adv = x_adv[distance.index(nearest_dist)]
            
    if len(x_adv) == 0: print("No adv found for this sentence pair!")
        
    return x_adv, distance, nearest_x_adv, nearest_dist
        

def perturb_single(data_source, epoch, corpus_test, hybrid=False):
    # Turn on evaluation mode which disables dropout.
    labels = {'tensor(0)': 'non-duplicate', 'tensor(1)': 'duplicate'}
    with open("./output/%s/duplicate.txt" % (args.outf), "a") as dup, \
            open("./output/%s/non-duplicate.txt" % (args.outf), "a") as ndup :
        for batch in data_source:
            premise, hypothesis, target, premise_words_ori, hypothesis_words_ori, lengths = batch

            batch_size = premise.size(0)
            for i in range(batch_size):
                
                y_pred = pred_fn_one((premise[i].unsqueeze(0), hypothesis[i].unsqueeze(0)))

                if y_pred != target[i]:
                    continue 
                try:
                    if int(target[i]) == 0:
                        dup.write(premise_words_ori[i] + "\t" + hypothesis_words_ori[i] + "\t" + str(int(target[i])) + '\n')
                    elif int(target[i]) == 1:
                        ndup.write(premise_words_ori[i] + "\t" + hypothesis_words_ori[i] + "\t" + str(int(target[i])) + '\n')
                    dup.flush()
                    ndup.flush()  
                except Exception as e:  # 修改 1
                    print(e)                
    
def parse_args():
    parser = argparse.ArgumentParser(description='Generating Natural Adversaries for Text')

    # Path Arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to data corpus ./data')
    parser.add_argument('--classifier_path', type=str, required=True,
                        help='path to classifier files ./models')

    parser.add_argument('--outf', type=str, default='',
                        help='output directory name')

    # Data Processing Arguments
    parser.add_argument('--vocab_size', type=int, default=11000,
                        help='cut vocabulary down to this size (most frequently seen in training)')
    parser.add_argument('--maxlen', type=int, default=32,
                        help='maximum sentence length')
    parser.add_argument('--lowercase', type=bool, default=True,
                        help='lowercase all text')
    # Other
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='use debug state')


    parser.add_argument('--datatype', type=int, default=0,
                        help='performs which search')

    parser.add_argument('--voc_file', type=str, default='./vocab.json',
                        help='vocab path')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    global args
    args = parse_args()
    print(vars(args))

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.device(0)
        print("using cuda device gpu:" + format(torch.cuda.current_device()))
        torch.cuda.manual_seed(args.seed)

    if args.debug:
        args.outf = "debug"
    else:
        if args.datatype == 0:
            args.outf = 'SNLI' + str(int(time.time()))
        elif args.datatype == 1:
            args.outf = 'QQP' + str(int(time.time()))

    # make output directory if it doesn't already exist
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    if not os.path.isdir('./output/{}'.format(args.outf)):
        os.makedirs('./output/{}'.format(args.outf))
    print("Saving into directory ./output/{0}".format(args.outf))

    cur_dir = './output/{}'.format(args.outf)
    print("Creating new experiment at " + cur_dir)

    ###############################################################################
    # Load data and target classifiers
    ###############################################################################
    global voc_uni
    voc_uni = json.load(open(args.voc_file, 'r'))
    # print(voc_uni)

    corpus = Corpus(args.data_path,
                    maxlen=args.maxlen,  # maximum sentence length
                    vocab_size=args.vocab_size,  # 词表大小 默认频率前 11000 的
                    lowercase=args.lowercase,
                    load_vocab=args.voc_file
                    )

    if args.datatype == 0:
        print('data set snli')
        corpus_test = SNLIDatasetForUniVAE(train=False, vocab_size=args.vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=10,
                                             collate_fn=collate_snli, shuffle=False)
        classifier1 = Baseline_LSTM(100, 300, maxlen=args.maxlen, gpu=False)
        classifier1.load_state_dict(
            torch.load(args.classifier_path + 'lstm_univae_acc_0.74.pt' , map_location=torch.device('cpu'))
        )
    elif args.datatype == 1:
        print('data set qqp')
        corpus_test = QQPDatasetForUniVAE(train=False, vocab_size=args.vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=10,
                                             collate_fn=collate_snli, shuffle=False)
        classifier1 = Baseline_LSTMQQP(100, 300, maxlen=args.maxlen, gpu=False)
        classifier1.load_state_dict(
            torch.load(args.classifier_path + 'lstm_univae_QQP.pt' , map_location=torch.device('cpu'))
        )

    test_data = iter(testloader)

    vocab_classifier1 = voc_uni
    print("Loaded data and target classifiers!")

    ###############################################################################
    # Build the models
    ###############################################################################
    ntokens = len(corpus.dictionary.word2idx)
    args.ntokens = ntokens
    print("Vocabulary Size: {}".format(ntokens))
    
    corpus.dictionary.word2idx = voc_uni


    len_test = len(testloader) #
    print(len_test)

    if args.datatype == 0:
        perturb_single(test_data, 1, corpus_test, hybrid=True)
    elif args.datatype == 1:
        perturb_single(test_data, 1, corpus_test, hybrid=True)

    


    









