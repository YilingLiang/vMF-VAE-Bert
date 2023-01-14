
# QQP training baseline codes
from models import Baseline_Embeddings, TextCNN, TextCNNL
from utils import to_gpu, Corpus, batchify, collate_snli, QQPDatasetForUniVAE
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

def f1(tp, tn, fp, fn):
    
    presion = tp / (tp + fp + 0.0001)
    recall = tp / (tp + fn + 0.0001)
    f1 = 2 * presion * recall / (presion + recall + 0.0001)
    return f1

def evaluate_model():
    test_iter = iter(testloader)
    correct = 0
    total = 0
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for batch in test_iter:
        premise, hypothesis, target, _, _, _ = batch
        premise = premise.to(device) # 移动数据到cuda
        hypothesis = hypothesis.to(device) # 或者
        target = target.to(device)
        prob_distrib = baseline_model.forward((premise, hypothesis))
        predictions = np.argmax(prob_distrib.data.cpu().numpy(), 1)
        correct += len(np.where(target.data.cpu().numpy() == predictions)[0])
        correctlist = np.where(target.data.cpu().numpy() == predictions)[0]
        falselist = np.where(target.data.cpu().numpy() != predictions)[0]
        
        p = np.where(target.data.cpu().numpy() == np.array([1]*premise.size(0)))[0]
        # print(p) 
        tp += len(np.intersect1d(correctlist, p))
        fp += len(np.intersect1d(falselist, p))
        n = np.where(target.data.cpu().numpy() == np.array([0]*premise.size(0)))[0]
        # print(n)
        tn += len(np.intersect1d(correctlist, n))
        fn += len(np.intersect1d(falselist, n))
        total += premise.size(0)
    # print(tp + tn)
    assert tp + fp + tn + fn == total
    acc = correct / float(total)
    print(tp, fp, tn, fn)
    print("Accuracy:{0}".format(acc))
    print("f1 score for duplicate:{0}".format(f1(tp, tn, fp, fn)))
    print("f1 score for non-duplicate:{0}".format(f1(tn, tp, fn, fp)))
    return acc

# 一些超参数
data_path = './data/QQP' # 'location of the data corpus'
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
corpus_train = QQPDatasetForUniVAE(train=True, vocab_size=vocab_size, path=data_path,
                           reset_vocab=voc_uni, 
                           maxlen=max_len
                           )

trainloader = torch.utils.data.DataLoader(corpus_train, batch_size=batch_size, collate_fn=collate_snli,
                                          shuffle=True)
train_iter = iter(trainloader)
# print(next(train_iter))
corpus_test = QQPDatasetForUniVAE(train=False, vocab_size=vocab_size, path=data_path, maxlen=max_len,
                              reset_vocab=voc_uni
                              )
testloader = torch.utils.data.DataLoader(corpus_test, batch_size=batch_size, collate_fn=collate_snli,
                                         shuffle=False)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

baseline_model = TextCNNL(100, 300, class_num=2, maxlen=max_len, gpu=False)
model_type = 'textcnn'
baseline_model.load_state_dict(torch.load("./textcnnlast_0.74QQPL.pt"))#  , map_location=torch.device('cpu')

baseline_model.to(device) # 移动模型到cuda
optimizer = optim.Adam(baseline_model.parameters(),
                       lr=lr,
                       betas=(beta1, 0.999))
criterion = nn.CrossEntropyLoss()

best_accuracy = 0.743


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
            with open(save_path + "/" + model_type + '_univae_' + str(curr_acc) + 'QQPL.pt', 'wb') as f:
                torch.save(baseline_model.state_dict(), f)
            best_accuracy = curr_acc

    with open(save_path + "/" + model_type + 'last_' + str(curr_acc)[:4] + 'QQPL.pt', 'wb') as f:
        torch.save(baseline_model.state_dict(), f)
    print("Best accuracy :{0}".format(best_accuracy))
else:
    evaluate_model()