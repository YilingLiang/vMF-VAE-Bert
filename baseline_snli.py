

from models import Baseline_Embeddings, Baseline_LSTM
from utils import to_gpu, Corpus, batchify, SNLIDataset, collate_snli, SNLIDatasetForUniVAE
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import sys
import json
sys.path.append("..") # 导入上级目录模块
# print("系统当前路径" , sys.path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def evaluate_model():
    test_iter = iter(testloader)
    correct = 0
    total = 0
    for batch in test_iter:
        premise, hypothesis, target, _, _, _ = batch
        premise = premise.to(device) # 移动数据到cuda
        hypothesis = hypothesis.to(device) # 或者
        target = target.to(device)
        prob_distrib = baseline_model.forward((premise, hypothesis))
        predictions = np.argmax(prob_distrib.data.cpu().numpy(), 1)
        correct += len(np.where(target.data.cpu().numpy() == predictions)[0])
        total += premise.size(0)
    acc = correct / float(total)
    print("Accuracy:{0}".format(acc))
    return acc

# 一些超参数
data_path = './data/classifier' # 'location of the data corpus'
epochs = 20
batch_size = 64
packed_rep = True # pad all sentences to fixed maxlen

max_len = 32 # max_length of sentence
lr = 1e-5
seed = 1111
beta1 = 0.9
save_path = './'
vocab_size = 11000
attack = False
voc_uni = json.load(open('./vocab.json', 'r')) # reset_vob 导致准确率下降, why ?
# 数据集准备
corpus_train = SNLIDatasetForUniVAE(train=True, vocab_size=vocab_size, path=data_path,
                           reset_vocab=voc_uni, maxlen=max_len
                           )
if attack == False:
    corpus_test = SNLIDatasetForUniVAE(train=False, vocab_size=vocab_size, path=data_path, maxlen=max_len,
                              reset_vocab=voc_uni
                              )
if attack == True:
    corpus_test = SNLI_Outer_Dataset(train=False, vocab_size=vocab_size, path=data_path, maxlen=max_len,
                              reset_vocab=voc_uni
                              )
    print(corpus_test.test_data[:5])
trainloader = torch.utils.data.DataLoader(corpus_train, batch_size=batch_size, collate_fn=collate_snli,
                                          shuffle=True)
train_iter = iter(trainloader)
# print(next(train_iter))
testloader = torch.utils.data.DataLoader(corpus_test, batch_size=batch_size, collate_fn=collate_snli,
                                         shuffle=False)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

baseline_model = Baseline_LSTM(100, 300, maxlen=max_len, gpu=False)
model_type = 'lstm'
baseline_model.load_state_dict(torch.load("./lstmlast.pt"))#  , map_location=torch.device('cpu')

# baseline_model = Baseline_Embeddings(100, vocab_size=11004)# 这里词表大小本来传的是 11000，因为多出四个，所以embedding要 11004
# model_type = 'emb'
# baseline_model.load_state_dict(
#     torch.load("./emb.pt", map_location=torch.device('cpu')))
baseline_model.to(device) # 移动模型到cuda
optimizer = optim.Adam(baseline_model.parameters(),
                       lr=lr,
                       betas=(beta1, 0.999))
criterion = nn.CrossEntropyLoss()

if model_type == 'lstm':
    best_accuracy = 0.73
elif model_type == 'emb':
    best_accuracy = 0.63

train_mode = True
if train_mode == True:
    for epoch in range(0, epochs):
        niter = 0
        loss_total = 0
        while niter < len(trainloader):
            niter += 1
            premise, hypothesis, target = train_iter.next()
            premise = premise.to(device) # 移动数据到cuda
            hypothesis = hypothesis.to(device) # 或者
            target = target.to(device)
            prob_distrib = baseline_model.forward((premise, hypothesis))
            loss = criterion(prob_distrib, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.data
        print(loss_total / float(niter))
        train_iter = iter(trainloader)
        curr_acc = evaluate_model()  # 每个epoch后看一下模型效果
        if curr_acc > best_accuracy:
            print("saving model...")
            with open(save_path + "/" + model_type + '_univae_' + str(curr_acc) + '.pt', 'wb') as f:
                torch.save(baseline_model.state_dict(), f)
            best_accuracy = curr_acc

    with open(save_path + "/" + model_type + 'last_' + str(curr_acc) + '.pt', 'wb') as f:
        torch.save(baseline_model.state_dict(), f)
    print("Best accuracy :{0}".format(best_accuracy))
else:
    evaluate_model()