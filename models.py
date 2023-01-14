import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import to_gpu, load_embeddings
import json
import os
import numpy as np



class Baseline_LSTM(nn.Module):
    # BaselineLSTM 用于snli分类的，使用 LSTM 抽取 p 和 h 特征， 拼接经过前馈网络分类 【200->400 bn relu ->100 bn relu ->3 softmax】
    def __init__(self, emb_size, hidden_size, maxlen=10, dropout= 0.0, vocab_size=11004, gpu=False):
        super(Baseline_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = 1
        self.gpu = gpu
        self.maxlen = maxlen
        self.embedding_prem = nn.Embedding(vocab_size+4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size+4, emb_size)
        self.premise_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        print(self.premise_encoder)
        self.hypothesis_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.layers = nn.Sequential()
        layer_sizes = [2*hidden_size, 400, 100]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.add_module("layer" + str(i + 1), layer)
            
            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.add_module("bn" + str(i + 1), bn)

            self.layers.add_module("activation" + str(i + 1), nn.ReLU())

        layer = nn.Linear(layer_sizes[-1], 3)
        self.layers.add_module("layer" + str(len(layer_sizes)), layer)
        
        self.layers.add_module("softmax", nn.Softmax())
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
    
        # Initialize Vocabulary Matrix Weight
        self.embedding_prem.weight.data.uniform_(-initrange, initrange)
        self.embedding_hypo.weight.data.uniform_(-initrange, initrange)
    
        # Initialize Encoder and Decoder Weights
        for p in self.premise_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.hypothesis_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
    
        # Initialize Linear Weight
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)) # (hidden, cell)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        state_prem= self.init_hidden(batch_size)
        state_hypo= self.init_hidden(batch_size)
        premise = self.embedding_prem(premise_indices)
        output_prem, (hidden_prem, _) = self.premise_encoder(premise, state_prem)
        hidden_prem= hidden_prem[-1]
        if hidden_prem.requires_grad:
            hidden_prem.register_hook(self.store_grad_norm)
                
        hypothesis = self.embedding_hypo(hypothesis_indices)
        output_hypo, (hidden_hypo, _) = self.hypothesis_encoder(hypothesis, state_hypo)
        hidden_hypo= hidden_hypo[-1]
        if hidden_hypo.requires_grad:
            hidden_hypo.register_hook(self.store_grad_norm)
            
        concatenated = torch.cat([hidden_prem, hidden_hypo], 1)
        # print(concatenated.shape)
        probs = self.layers(concatenated)
        # print(probs.shape)
        return probs


class Baseline_LSTMQQP(nn.Module):
    # BaselineLSTM 用于snli分类的，使用 LSTM 抽取 p 和 h 特征， 拼接经过前馈网络分类 【200->400 bn relu ->100 bn relu ->3 softmax】
    def __init__(self, emb_size, hidden_size, maxlen=10, dropout= 0.0, vocab_size=11004, gpu=False):
        super(Baseline_LSTMQQP, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = 1
        self.gpu = gpu
        self.maxlen = maxlen
        self.embedding_prem = nn.Embedding(vocab_size+4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size+4, emb_size)
        self.premise_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        print(self.premise_encoder)
        self.hypothesis_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.layers = nn.Sequential()
        layer_sizes = [2*hidden_size, 400, 100]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.add_module("layer" + str(i + 1), layer)
            
            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.add_module("bn" + str(i + 1), bn)

            self.layers.add_module("activation" + str(i + 1), nn.ReLU())

        layer = nn.Linear(layer_sizes[-1], 2)
        self.layers.add_module("layer" + str(len(layer_sizes)), layer)
        
        self.layers.add_module("softmax", nn.Softmax())
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
    
        # Initialize Vocabulary Matrix Weight
        self.embedding_prem.weight.data.uniform_(-initrange, initrange)
        self.embedding_hypo.weight.data.uniform_(-initrange, initrange)
    
        # Initialize Encoder and Decoder Weights
        for p in self.premise_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.hypothesis_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
    
        # Initialize Linear Weight
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)) # (hidden, cell)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        state_prem= self.init_hidden(batch_size)
        state_hypo= self.init_hidden(batch_size)
        premise = self.embedding_prem(premise_indices)
        output_prem, (hidden_prem, _) = self.premise_encoder(premise, state_prem)
        hidden_prem= hidden_prem[-1]
        if hidden_prem.requires_grad:
            hidden_prem.register_hook(self.store_grad_norm)
                
        hypothesis = self.embedding_hypo(hypothesis_indices)
        output_hypo, (hidden_hypo, _) = self.hypothesis_encoder(hypothesis, state_hypo)
        hidden_hypo= hidden_hypo[-1]
        if hidden_hypo.requires_grad:
            hidden_hypo.register_hook(self.store_grad_norm)
            
        concatenated = torch.cat([hidden_prem, hidden_hypo], 1)
        # print(concatenated.shape)
        probs = self.layers(concatenated)
        # print(probs.shape)
        return probs

    
class Baseline_Embeddings(nn.Module):
    # 直接用词向量连接三分类
    def __init__(self, emb_size, vocab_size=11004):
        super(Baseline_Embeddings, self).__init__()
        self.embedding_prem = nn.Embedding(vocab_size, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size*2, 3)
        embeddings_mat = load_embeddings()
        self.embedding_prem.weight.data.copy_(embeddings_mat)
        self.embedding_hypo.weight.data.copy_(embeddings_mat)
        
    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        enc_premise = self.embedding_prem(premise_indices)
        enc_hypothesis = self.embedding_hypo(hypothesis_indices)
        enc_premise = torch.mean(enc_premise,1).squeeze(1)
        enc_hypothesis = torch.mean(enc_hypothesis,1).squeeze(1)
        
        concatenated = torch.cat([enc_premise, enc_hypothesis], 1)
        probs = self.linear(concatenated) # 3 分类
        return probs
class TextCNN(nn.Module):
    """CNN模型"""

    def __init__(self, emb_size, hidden_size, class_num=3, maxlen=64, dropout=0.0, vocab_size=11000, gpu=False):
        super(TextCNN, self).__init__()
        self.hidden_size = hidden_size
        self.gpu = gpu
        self.maxlen = maxlen
        self.embedding_prem = nn.Embedding(vocab_size + 4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size + 4, emb_size)

        self.conv = nn.Sequential(nn.Conv1d(in_channels=emb_size, out_channels=256, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=62))
        self.linear = nn.Linear(256, class_num)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding_prem.weight.data.uniform_(-initrange, initrange)
        self.embedding_hypo.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        premise = self.embedding_prem(premise_indices) # 10 * 32 * 300
        hypothesis = self.embedding_hypo(hypothesis_indices)
        x = torch.cat([premise, hypothesis], 1) # 10 * 64 * 300

        x = x.permute(0, 2, 1)  # 将tensor的维度换位。batch_size, embedding_size, length

        # batch_size,卷积核个数out_channels，(句子长度-kernel_size)/步长+1
        x = self.conv(x)  # Conv1后10*256*62,ReLU后不变,MaxPool1d后10*256*1;

        x = x.view(-1, x.size(1))  # 10*256
        x = F.dropout(x, 0.8)
        x = self.linear(x)  # 10*3 batch_size * class_num
        probs = F.softmax(x,dim=1)
        return probs
class TextCNNL(nn.Module):
    """CNN模型"""

    def __init__(self, emb_size, hidden_size, class_num=3, maxlen=64, dropout=0.0, vocab_size=11000, gpu=False):
        super(TextCNNL, self).__init__()
        self.hidden_size = hidden_size
        self.gpu = gpu
        self.maxlen = maxlen
        self.embedding_prem = nn.Embedding(vocab_size + 4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size + 4, emb_size)

        self.conv = nn.Sequential(nn.Conv1d(in_channels=emb_size, out_channels=256, kernel_size=3),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=62))
        self.fc = nn.Linear(256, 300)
        self.linear = nn.Linear(300, class_num)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding_prem.weight.data.uniform_(-initrange, initrange)
        self.embedding_hypo.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        premise = self.embedding_prem(premise_indices) # 10 * 32 * 300
        hypothesis = self.embedding_hypo(hypothesis_indices)
        x = torch.cat([premise, hypothesis], 1) # 10 * 64 * 300

        x = x.permute(0, 2, 1)  # 将tensor的维度换位。batch_size, embedding_size, length

        # batch_size,卷积核个数out_channels，(句子长度-kernel_size)/步长+1
        x = self.conv(x)  # Conv1后10*256*62,ReLU后不变,MaxPool1d后10*256*1;

        x = x.view(-1, x.size(1))  # 10*256
        x = F.dropout(x, 0.8)
        x = self.fc(x)
        x = F.relu(x)
        x = F.dropout(x, 0.8)
        x = self.linear(x)  # 10*3 batch_size * class_num
        probs = F.softmax(x)
        return probs
class Config(object):

    def __init__(self, class_num=3):
        self.model_name = 'TextCNN'
        self.dropout = 0.2
        self.require_improvement = 1000
        self.num_classes = class_num
        self.embed = 100 # 300
        self.filter_sizes = (3,)
        self.num_filters = 100 # 100 


'''Convolutional Neural Networks for Sentence Classification'''


class WordCNN(nn.Module):
    def __init__(self, emb_size, hidden_size, class_num=3, maxlen=64, dropout=0.0, vocab_size=11000, gpu=False):
        super(WordCNN, self).__init__()
        config = Config(class_num=class_num)

        self.embedding_prem = nn.Embedding(vocab_size + 4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size + 4, emb_size)
        self.embedding = nn.Embedding(vocab_size + 4, emb_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size = (k, emb_size)) for k in config.filter_sizes])

        self.dropout = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.num_filters * len(config.filter_sizes),
                             config.num_filters * len(config.filter_sizes))
        self.fc2 = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        premise = self.embedding_prem(premise_indices)  # 10 * 32 * 100
        hypothesis = self.embedding_hypo(hypothesis_indices)
        x = torch.cat([premise, hypothesis], 1)

        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        probs = F.softmax(out, dim=1)

        return probs