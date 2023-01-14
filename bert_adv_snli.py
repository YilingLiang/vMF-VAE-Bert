# -*- coding: utf-8 -*-
# generate_sst.py
# from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


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
from utils import to_gpu, Corpus, SNLIDataset, collate_snli, SNLIDatasetForUniVAE, QQPDatasetForUniVAE, SNLIDatasetForUniVAETEST, QQPDatasetForUniVAETEST
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

import glob
import logging


import torch.optim as optim

import torch.utils.data
from torch.autograd import Variable


# +
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import math
from numpy.linalg import norm
from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

embed = hub.load('./use4')
def sim(u, v):
    sts_encode1 = tf.nn.l2_normalize(embed(tf.constant([u])), axis=1)
    sts_encode2 = tf.nn.l2_normalize(embed(tf.constant([v])), axis=1)
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    """Returns the similarity scores"""
    return scores

web_model = WebBertSimilarity(device='cuda', batch_size=64) # [0:5] 的分数
# -

sys.path.append("..") # 导入上级目录模块
# print("系统当前路径" , sys.path)
task_name = 'snli'
model_name_or_path = './bert_snli_uncased/checkpoint-25752/'

num_labels = 3
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 基本信息
maxlen = 32
batch_size = 64
epochs = 1000
kappa = 32
z_dim = 16
num_latent_layers = 4 # 4

# 模型路径
config_path = './bert_base_model_uncased/bert_config.json'
checkpoint_path = './bert_base_model_uncased/bert_model.ckpt'
dict_path = './bert_base_model_uncased/vocab.txt'

# 建立分词器
tokenizer_univae = Tokenizer(dict_path, do_lower_case=True)

class UniAE_Mask(object):
    """仿UniLM做AE模型
    """

    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def uniae_mask(s, first=True):
                idxs = K.cumsum(s, axis=1)
                mask1 = K.equal(s[:, None, :], s[:, :, None])
                mask2 = idxs[:, None, :] <= idxs[:, :, None]
                mask = K.cast(mask1 & mask2, K.floatx())
                if first:
                    mask = [K.ones_like(mask[..., :1]), mask[..., 1:]]
                    mask = K.concatenate(mask, axis=2)
                return -(1 - mask[:, None]) * 1e12

            self.attention_bias1 = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=uniae_mask,
                arguments={'first': False},
                name='Attention-UniAE1-Mask'
            )

            self.attention_bias2 = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=uniae_mask,
                arguments={'first': True},
                name='Attention-UniAE2-Mask'
            )

            self.attention_bias = [self.attention_bias1, self.attention_bias2]

        if inputs < self.num_hidden_layers - self.num_latent_layers:
            return self.attention_bias[0]
        else:
            return self.attention_bias[1]


class vonMisesFisherSampling(Layer):
    """von Mises Fisher分布重参数
    通过累积概率函数的逆和预计算来实现最简单vMF分布采样
    链接：https://kexue.fm/archives/8404
    """

    def __init__(self, kappa, num_caches=10 ** 7, **kwargs):
        super(vonMisesFisherSampling, self).__init__(**kwargs)
        self.kappa = kappa
        self.num_caches = num_caches

    @integerize_shape
    def build(self, input_shape):
        super(vonMisesFisherSampling, self).build(input_shape)
        self.pw_samples = self.add_weight(
            shape=(self.num_caches,),
            initializer=self.initializer(input_shape[-1]),
            trainable=False,
            name='pw_samples'
        )

    def initializer(self, dims):
        def init(shape, dtype=None):
            x = np.linspace(-1, 1, shape[0] + 2)[1:-1]
            y = self.kappa * x + np.log(1 - x ** 2) * (dims - 3) / 2
            y = np.cumsum(np.exp(y - y.max()))
            return np.interp((x + 1) / 2, y / y[-1], x)

        return init

    def call(self, inputs):
        mu = inputs
        # 采样w
        idxs = K.random_uniform(
            K.shape(mu[..., :1]), 0, self.num_caches, dtype='int32'
        )
        w = K.gather(self.pw_samples, idxs)
        # 采样z
        eps = K.random_normal(K.shape(mu))
        nu = eps - K.sum(eps * mu, axis=1, keepdims=True) * mu
        nu = K.l2_normalize(nu, axis=-1)
        return w * mu + (1 - w ** 2) ** 0.5 * nu

    def get_config(self):
        config = {
            'kappa': self.kappa,
            'num_caches': self.num_caches,
        }
        base_config = super(vonMisesFisherSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class UniVAE(UniAE_Mask, RoFormer):
class UniVAE(UniAE_Mask, RoFormer):
    """RoFormer/BERT + UniAE 做VAE模型
    """

    def __init__(self, *args, **kwargs):
        super(UniVAE, self).__init__(*args, **kwargs)
        self.with_mlm = self.with_mlm or True
        self.num_latent_layers = num_latent_layers
        self.mus = []
        self.mode = 'vae'
        self.z_in = self.apply(
            layer=Input,
            shape=(self.num_latent_layers * z_dim,),
            name='Latent-In'
        )
        self.zs = [None] * (self.num_hidden_layers - self.num_latent_layers)
        self.zs += self.apply(
            inputs=self.z_in,
            layer=Lambda,
            function=lambda x: tf.split(x, self.num_latent_layers, axis=1),
            name='Latent-Split'
        )

    def apply_main_layers(self, inputs, index):
        """在中间层插入隐变量运算
        """
        x = inputs
        if index >= self.num_hidden_layers - self.num_latent_layers:
            z = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='CLS-Pooler-%s' % index
            )
            z = self.apply(
                inputs=z,
                layer=Dense,
                units=z_dim,
                kernel_initializer=self.initializer,
                name='In-Projection-%s' % index
            )
            z = self.apply(
                inputs=z,
                layer=Lambda,
                function=lambda z: K.l2_normalize(z, axis=-1),
                name='L2-Normalization-%s' % index
            )
            if self.mode == 'encoder':
                self.mus.append(z)
            if self.mode == 'vae':
                z = self.apply(
                    inputs=z,
                    layer=vonMisesFisherSampling,
                    kappa=kappa,
                    name='ReParameterization'
                )
            if self.mode == 'decoder':
                z = self.zs[index]
            z = self.apply(
                inputs=z,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Out-Projection-%s' % index
            )
            x = self.apply(
                inputs=[x, z],
                layer=Lambda,
                function=lambda xz: K.
                    concatenate([xz[1][:, None], xz[0][:, 1:]], axis=1),
                mask=lambda x, m: m[0],
                name='Concatenation-%s' % index
            )
        return super(UniVAE, self).apply_main_layers(x, index)

    def build(self, **kwargs):
        super(UniVAE, self).build(**kwargs)
        self.mode = 'encoder'
        output = self.call(self.model.inputs)
        mu = self.apply(inputs=self.mus, layer=Concatenate, axis=1, name='Mu')
        self.encoder = Model(self.model.inputs, mu)
        self.mode = 'decoder'
        output = self.call(self.model.inputs)
        self.decoder = Model(self.model.inputs + [self.z_in], output)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


vae = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=UniVAE,
    return_keras_model=False
)
model = vae.model
encoder = vae.encoder
decoder = vae.decoder
output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)

model.load_weights('./snli_loss_0.28.weights')


class Vector2Sentence(AutoRegressiveDecoder):
    """隐向量解码为句子
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        z = inputs[0]
        token_ids = np.zeros((output_ids.shape[0], maxlen))
        token_ids[:, 0] = tokenizer_univae._token_start_id
        zeros = np.zeros_like(token_ids)
        ones = np.ones_like(output_ids)
        segment_ids = np.concatenate([zeros, ones], axis=1)
        token_ids = np.concatenate([token_ids, output_ids], axis=1)
        return self.last_token(decoder).predict([token_ids, segment_ids, z])

    def generate(self, z, topk=1):
        z = z.reshape((-1, z_dim))
        z /= (z ** 2).sum(axis=1, keepdims=True) ** 0.5
        z = z.reshape(-1)
        output_ids = self.beam_search([z], topk)  # 基于 beam search
        return tokenizer_univae.decode(output_ids)  # id2word


vec2sent = Vector2Sentence(
    start_id=tokenizer_univae._token_start_id,
    end_id=tokenizer_univae._token_end_id,
    maxlen=maxlen
)

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


def search_qqp_snli(s, y, nsamples=100, left=0., high=1.):
    
    premise_ori, hypothesis_ori = s
    print("Begin generating adversarial examples for: \n" + premise_ori + ' ' + hypothesis_ori + ' ' + str(y.cpu().numpy()))
    x, s = tokenizer_univae.encode(hypothesis_ori, maxlen=maxlen)
    X = sequence_padding([x])
    S = sequence_padding([s])
    Z = encoder.predict([X, S])
    z = Z[0]
    x_adv = []
    distance = []
    nearest_dist = None
    nearest_x_adv = None
    for k in range(nsamples):
        if args.advmode == 'all':
            delta_z = np.random.randn(64)
            d = np.random.rand(64) * (high - left) + left  # length range [l, h)
            norm_p = np.linalg.norm(delta_z, ord=2)

            d_norm = np.divide(d, norm_p)  # rescale/normalize factor
            delta_z = np.multiply(delta_z, d_norm) 
            delta_z += z
            dist = np.linalg.norm(delta_z - z)

            hypo_pre_ori = vec2sent.generate(delta_z)
            
        elif args.advmode == 'fore':
            zk = np.random.randn(16)
            
            d = np.random.rand(16) * (high - left) + left
            norm_p = np.linalg.norm(zk, ord=2)
            d_norm = np.divide(d, norm_p)
            zk = np.multiply(zk, d_norm)
            
            z[:16] += zk
            dist = np.linalg.norm(Z[0] - z)
            hypo_pre_ori = vec2sent.generate(z)

        elif args.advmode == 'tail':
            zk = np.random.randn(16)
            
            d = np.random.rand(16) * (high - left) + left
            norm_p = np.linalg.norm(zk, ord=2)
            d_norm = np.divide(d, norm_p)
            zk = np.multiply(zk, d_norm)
            
            z[48:] += zk
            dist = np.linalg.norm(Z[0] - z)
            hypo_pre_ori = vec2sent.generate(z)
        
        y_pred = pred_fn_one((premise_ori, hypo_pre_ori))[0]
        # print(y_pred == y)
        if y_pred != y and sim(hypo_pre_ori, hypothesis_ori) > 0.8 and \
            web_model.predict([(hypo_pre_ori, hypothesis_ori)]) > 0.75:
            
            temp = premise_ori + '\t' + hypo_pre_ori + '\t' + str(y.cpu().numpy())
            x_adv.append(temp)
            distance.append(dist)
            # print('adv h:', hypo_pre_ori)
            
    if len(x_adv) != 0:    
        nearest_dist = min(distance)
        nearest_x_adv = x_adv[distance.index(nearest_dist)]
            
    if len(x_adv) == 0: print("No adv found for this sentence pair!")
        
    return x_adv, distance, nearest_x_adv, nearest_dist

def perturb_snli(data_source, corpus_test, hybrid=False):
    # Turn on evaluation mode which disables dropout.
    
    labels = {'tensor(0)': 'entailment', 'tensor(1)': 'neutral', 'tensor(2)': 'contradiction'}
    advfail = 0
    with open("./output/%s/alladvsnli.txt" % (args.outf), "a") as f, \
            open("./output/%s/bestadvsnli.txt" % (args.outf), "a") as xad, open("./output/%s/advtextsnli.txt" % (args.outf), "a") as g :
        for batch in data_source:
            premise, hypothesis, target, premise_words_ori, hypothesis_words_ori, lengths = batch

            batch_size = premise.size(0)
            for i in range(batch_size):
                f.write("\n================Premise==================\n")
                f.write(premise_words_ori[i] + "\n")  # premise 句子
                f.write("===========Original Hypothesis===========\n")
                f.write(hypothesis_words_ori[i] + "\n") # hypothesis 句子
                f.write("===========Original Label===========\n")
                f.write(labels[str(target[i])] + '\n')

                y_pred = pred_fn_one((premise_words_ori[i], hypothesis_words_ori[i])) # 改动, y_pred = [2]
                # print(y_pred[0] == target[i])
                if y_pred[0] != target[i]:
                    f.write('This sentence pair was classified wrongly! So no adv.\n\n\n')
                    continue # 这段逻辑用于判断句子是否本身是分类正确的，本身分类正确才有对抗样本一说.

                x_adv, distance, nearest_x_adv, nearest_dist = search_qqp_snli((premise_words_ori[i], hypothesis_words_ori[i]),\
                                                       target[i], nsamples=100, left=0., high=args.radius) 
                g.write("##" + premise_words_ori[i] + '\t' + hypothesis_words_ori[i] + '\t' + str(int(target[i])) + '\n')
                xad.write("##" + premise_words_ori[i] + '\t' + hypothesis_words_ori[i] + '\t' + str(int(target[i])) + '\n')
                
                try:
                    f.write("===========Adversarial Hypothesis===========\n")
                    f.write("\n".join(x_adv))
                    f.write('\n')
                    # xad.write("\n".join(x_adv))
                    # xad.write('\n')
                    xad.write(nearest_x_adv + '\n')
                    g.write("\n".join(x_adv))
                    g.write('\n')
                    f.flush()
                    xad.flush()
                    g.flush()
                except Exception as e:  # 修改 1
                    print(e)
                    print(premise_words_ori)
                    print(hypothesis_words_ori)
                    advfail += 1
                    print("no adversary found for this pair!\n")
    print(advfail)


def perturb_qqp(data_source, corpus_test, hybrid=False):
    # Turn on evaluation mode which disables dropout.
    labels = {'tensor(0)': 'non-duplicate', 'tensor(1)': 'duplicate'}
    advfail = 0
    with open("./output/%s/alladvqqp.txt" % (args.outf), "a") as f, \
            open("./output/%s/bestadvqqp.txt" % (args.outf), "a") as xad, open("./output/%s/advtextqqp.txt" % (args.outf), "a") as g :
        for batch in data_source:
            premise, hypothesis, target, premise_words_ori, hypothesis_words_ori, lengths = batch

            batch_size = premise.size(0)
            for i in range(batch_size):
                f.write("\n================Left==================\n")
                f.write(premise_words_ori[i] + "\n")  # premise 句子
                f.write("===========Original Right===========\n")
                f.write(hypothesis_words_ori[i] + "\n") # hypothesis 句子
                f.write("===========Original Label===========\n")
                f.write(labels[str(target[i])] + '\n')

                y_pred = pred_fn_one((premise[i].unsqueeze(0), hypothesis[i].unsqueeze(0)))

                if y_pred != target[i]:
                    f.write('This sentence pair was classified wrongly! So no adv.\n\n\n')
                    continue # 这段逻辑用于判断句子是否本身是分类正确的，本身分类正确才有对抗样本一说.

                x_adv, distance, nearest_x_adv, nearest_dist = search_qqp_snli((premise_words_ori[i], hypothesis_words_ori[i]),\
                                                       target[i], nsamples=50, left=0., high=args.radius) 
                
                try:
                    f.write("===========Adversarial Right===========\n")
                    f.write("\n".join(x_adv))
                    f.write('\n')
                    # xad.write("\n".join(x_adv))
                    # xad.write('\n')
                    xad.write(nearest_x_adv + '\n')
                    g.write("\n".join(x_adv))
                    g.write('\n')
                    f.flush()
                    xad.flush()
                    g.flush()
                except Exception as e:  # 修改 1
                    print(e)
                    print(premise_words_ori)
                    print(hypothesis_words_ori)
                    advfail += 1
                    print("no adversary found for this pair!\n")
    print(advfail)


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
    parser.add_argument('--radius', type=float, default=2.0,
                        help='radius')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='use debug state')

    parser.add_argument('--advmode', type=str, 
                        choices=['all', 'fore', 'tail'],
                        default='latent',
                        help='which kind of perturbation on latent space.')
    parser.add_argument('--datatype', type=int, default=0,
                        help='performs attack on which dataset,0:snli, 1:qqp')
    # 这个部分有待整合
    parser.add_argument('--modeltype', type=str, default='lstm',choices=['lstm', 'textcnn', 'bert'],
                        help='performs attack on which model ')

    parser.add_argument('--voc_file', type=str, default='./vocab.json',
                        help='vocab path')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    global args
    args = parse_args()
    print(vars(args))
    MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
     }

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
            args.outf = 'BERTSNLI' + str(args.advmode) + str(int(time.time())) + '_' + str(args.radius)
        elif args.datatype == 1:
            args.outf = 'BERTQQP' + str(args.advmode) + str(int(time.time())) + '_' + str(args.radius)

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
        corpus_test = SNLIDatasetForUniVAETEST(train=False, vocab_size=args.vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=30,
                                             collate_fn=collate_snli, shuffle=False)
        # BertConfig 在transformer 中的才有 from_pretrained
        config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']

        config = config_class.from_json_file(model_name_or_path+ '/config.json')
        print(config)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)

        model = model_class.from_pretrained(model_name_or_path, num_labels=3)
        model.to(device)
    elif args.datatype == 1:
        print('data set qqp')
        corpus_test = QQPDatasetForUniVAETEST(train=False, vocab_size=args.vocab_size,  # 实际的 snli dataset
                                  reset_vocab=voc_uni
                                  )
        testloader = torch.utils.data.DataLoader(corpus_test, batch_size=30,
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
    args.ntokens = ntokens
    print("Vocabulary Size: {}".format(ntokens))
    
    corpus.dictionary.word2idx = voc_uni


    len_test = len(testloader) #
    print(len_test)

    if args.datatype == 0:
        perturb_snli(test_data, corpus_test, hybrid=True)
    elif args.datatype == 1:
        perturb_qqp(test_data, corpus_test, hybrid=True)

    












